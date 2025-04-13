# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)

import numpy as np


def is_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    return (
        is_param(edge_program, node)
        or is_buffer(edge_program, node)
        or is_lifted_tensor_constant(edge_program, node)
    )


def get_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> torch.Tensor:
    param = None
    if is_param(edge_program, node):
        param = get_param(edge_program, node)
    if is_buffer(edge_program, node):
        param = get_buffer(edge_program, node)
    if is_lifted_tensor_constant(edge_program, node):
        param = get_lifted_tensor_constant(edge_program, node)
    if param is not None:
        # update node.meta["val"] to qualified QNN datatype (e.g. i64 to i32)
        assert isinstance(param, torch.Tensor), "Expect parameter to be tensor"
        param = param.type(node.meta["val"].dtype)
    return param


def set_parameter(
    param: torch.Tensor, node: torch.fx.Node, edge_program: torch.export.ExportedProgram
):
    status = False
    if is_param(edge_program, node):
        edge_program.state_dict[
            edge_program.graph_signature.inputs_to_parameters[node.name]
        ] = param
        status = True
    if is_buffer(edge_program, node):
        buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in edge_program.graph_signature.non_persistent_buffers:
            edge_program.constants[buffer_name] = param
        else:
            edge_program.state_dict[buffer_name] = param
        status = True
    assert status, "Failed to set parameter"


def is_graph_input(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a graph input

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    return tensor.op == "placeholder" and not is_parameter(tensor, edge_program)


def is_graph_output(node: torch.fx.Node) -> bool:
    """
    Check if the given tensor is used as a graph output

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    for user in node.users.keys():
        # getitem node is skiped, check the op_skip_ops.py
        if user.op == "output" or (
            user.target.__name__ == "getitem" and is_graph_output(user)
        ):
            return True
    return False


def is_constant(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a constant

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    # constants should not be treated as input placeholder
    # pay attention to the pytorch design, change this if
    # breakage happened:
    # pytorch/torch/_export/passes/lift_constant_tensor_pass.py
    if is_parameter(tensor, edge_program):
        return tensor.meta["val"].constant is not None

    return False


def deduce_dtype(
    tensor: torch.Tensor, quant_infos: Optional[Dict] = None
) -> torch.dtype:
    if quant_infos:
        quant_range = quant_infos["quant_max"] - quant_infos["quant_min"]
        unsigned = quant_infos["quant_min"] >= 0
        if quant_range <= torch.iinfo(torch.int8).max - torch.iinfo(torch.int8).min:
            return torch.uint8 if unsigned else torch.int8

        elif quant_range <= torch.iinfo(torch.int16).max - torch.iinfo(torch.int16).min:
            return torch.uint16 if unsigned else torch.int16

        return quant_infos["dtype"]

    return tensor.dtype


def parse_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray) -> Tuple:
    assert qweight.dtype == "int32"
    assert qzeros.dtype == "int32"

    bits = 32 // (scales.shape[1] // qzeros.shape[1])
    K = qweight.shape[0] * (32 // bits)
    M = qweight.shape[1]
    group_size = K // scales.shape[0]

    return K, M, bits, group_size


def unpack_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray, gptq_v2: bool = True):
    """
    Unpack GPTQv2
    Return T-MAC biased uint8 weight [0, 2 ** bits), fp16 scales, biased fp16 zeros, bits, group_size
    """
    assert qweight.dtype == "int32"
    assert qzeros.dtype == "int32"
    # TODO: support other pack_dtypes

    K, M, bits, group_size = parse_gptqv2(qweight, scales, qzeros)

    # Detect symmetry
    if bits == 2:
        sym_zero = 0xaaaaaaaa
    elif bits == 4:
        sym_zero = 0x88888888
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    symmetric = not (qzeros - np.uint32(sym_zero).astype(np.int32)).any()

    # Unpack qweight
    qweights = [(qweight >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    w = np.stack(qweights, axis=1).reshape(K, M).T.astype("uint8")

    scales = scales.T

    # Unpack qzeros
    zeros = [(qzeros >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    zeros = np.stack(zeros, axis=-1).reshape(K // group_size, M).T.astype(scales.dtype)
    if not gptq_v2:
        # `zeros = zeros - 1` in AutoGPTQ
        # Not in GPTQModel
        zeros += 1
    zeros = (zeros - (2 ** (bits - 1)))

    return w, scales, zeros, bits, group_size, symmetric


def hvx_preprocess_weights_gptq(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    tile_p: int = 512,
    tile_q: int = 64,
    vec_p: int = 128,
    vec_q: int = 4,
    vec_c: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:

    assert(w.dtype == "uint8")
    assert(scales.dtype == "float16" or scales.dtype == "float32" or scales.dtype == "bfloat16")
    if scales.dtype != "float16":
        scales = scales.astype("float16")
        zeros = zeros.astype("float16") if zeros is not None else None
    # 4 = sizeof(int32/float) / sizeof(uint8)
    assert(vec_p // 4 == vec_c)

    M, K = w.shape

    P = M * bits
    Q = K // g

    # (M, K, bits)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    # (M, K, bits) -> (M, bits, K) -> (M, bits, K) -> (M, bits, K // g, g)
    w = w.transpose(0, 2, 1).reshape(M, bits, Q, g)
    # (M, bits, K // g, g) -> (M, bits, Q)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    # (M, bits, Q) -> (M // vec_p, vec_p, bits, Q) -> (M // vec_p, bits, vec_p, Q) -> (P // vec_p, vec_p, Q)
    w = w.reshape(M // vec_p, vec_p, bits, Q).transpose(0, 2, 1, 3)
    # Interleave even and odd vec_c of w_vec
    # 0, 1 -> even bytes of w_vec -> c_vec_0, c_vec_2 -> c_bitsum_lo
    # 2, 3 ->  odd bytes of w_vec -> c_vec_1, c_vec_3 -> c_bitsum_hi
    # w_vec = w0/w2/w0/w2......w1/w3/w1/w3
    # c_vec_0, c_vec_2 = w0/w0......w1/w1
    # c_vec_1, c_vec_3 = w2/w2......w3/w3
    w = w.reshape(P // vec_p, 2, 2, vec_c, Q).transpose(0, 2, 3, 1, 4)
    w = w.reshape(P // tile_p, tile_p, Q // tile_q, tile_q).transpose(0, 2, 1, 3)
    #             0            1            2                3      4                5
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, vec_p, tile_q // vec_q, vec_q).transpose(0, 1, 2, 4, 5, 3)
    # Pack and interleave: q = 0 -> w_vec_lo_bo, q = 1 -> w_vec_lo_to, q = 2 -> w_vec_hi_bo, q = 3 -> w_vec_hi_to
    # lo -> low 128 bytes, hi -> high 128 bytes, bo -> bot 4 bit in a byte, to -> top 4 bit in a byte
    w = w.reshape(-1, vec_q, vec_p).reshape(-1, vec_q // 2, 2, vec_p).transpose(0, 1, 3, 2)
    w = sum([(w[:, :, :, n] << (n * g)) for n in range(2)])
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, tile_q // vec_q, vec_q // 2, vec_p)
    # Reshape for easy shape inference
    w = np.ascontiguousarray(w).view(np.int32).reshape(M, -1)

    if scales.size >= M:
        group_size = K // scales.shape[1]
        q_group_size = group_size // g
        scales = scales.reshape(P // tile_p, tile_p // bits, Q // tile_q, tile_q // q_group_size).transpose(0, 2, 1, 3)
        #                       0            1            2                        3      4
        scales = scales.reshape(P // tile_p, Q // tile_q, tile_p // bits // vec_p, vec_p, tile_q // q_group_size).transpose(0, 1, 2, 4, 3)
        # s_vec = s0/s0......s1/s1......s2/s2......s3/s3
        # s_vec_lo_lo, s_vec_lo_hi = s0/s0......s1/s1 -> c_vec_0, c_vec_2 -> c_bitsum_lo
        # no need for interleaving
        if zeros is not None:
            zeros = zeros.reshape(P // tile_p, tile_p // bits, Q // tile_q, tile_q // q_group_size).transpose(0, 2, 1, 3)
            zeros = zeros.reshape(P // tile_p, Q // tile_q, tile_p // bits // vec_p, vec_p, tile_q // q_group_size).transpose(0, 1, 2, 4, 3)
            # (c * ls + lb) * s + z * s * lb * 2
            # = (c * ls + lb + z * lb * 2) * s
            # = (c * ls + (z * 2 + 1) * lb) * s
            zeros = zeros * 2 + 1
            scales = np.stack([scales, zeros], axis=-2)
        scales = scales.view(np.int32).reshape(1, -1)
    else:
        # TODO: support BitNet and rename the function
        if zeros is not None:
            scales = np.concatenate([scales, zeros])
    return w, scales
