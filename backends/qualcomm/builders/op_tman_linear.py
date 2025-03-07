# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA, QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import (
    OpTMANLinear,
    QNN_OP_PACKAGE_NAME_TMAN,
)
from .utils import get_parameter


def _get_scratch_size(
    m: int,
    k: int,
    group_size: int,
    bits: int,
) -> int:
    LUT_G = 4
    LUT_SIZE = 16
    ACT_GROUP_SIZE = 256
    # int16
    l_size = k // LUT_G * LUT_SIZE
    # float32
    ls_size = 1 if (ACT_GROUP_SIZE == -1) else (k // ACT_GROUP_SIZE)
    # float32
    lb_size = 1 if (group_size == 0) else (k // group_size)
    # int32
    c_size = m * bits
    return l_size * 2 + ls_size * 4 + lb_size * 4 + c_size * 4


@register_node_visitor
class TMANLinear(NodeVisitor):
    target = ["tman.linear.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        linear_input_tensors = []
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(input_tensor_wrapper)

        qweight_node = node.args[1]
        qweight_tensor = get_parameter(qweight_node, self.edge_program)
        qweight_tensor_wrapper = self.define_tensor(
            qweight_node,
            node,
            qweight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(qweight_tensor_wrapper)

        scales_node = node.args[2]
        scales_tensor = get_parameter(scales_node, self.edge_program)
        scales_tensor_wrapper = self.define_tensor(
            scales_node,
            node,
            scales_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(scales_tensor_wrapper)

        # Is this needed?
        # QNN constraint, topk output_0 requires having the same quant config as input
        node.meta[QCOM_QUANT_ATTRS] = input_node.meta.get(QCOM_QUANT_ATTRS)
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        group_size = cast(int, node.args[3])
        bits = cast(int, node.args[4])
        symmetric = cast(bool, node.args[5])

        k = input_tensor.shape[-1]
        m = output_tensor.shape[-1]

        # do not quantize scratch buffer
        no_quant_encoding, no_quant_configs = (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            {},
        )
        scratch_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_scratch",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            quant_encoding=no_quant_encoding,
            quant_configs=no_quant_configs,
            dims=torch.Size(1, _get_scratch_size(m, k, group_size, bits)),
            tensor=None,  # Unused when is_fake_tensor is True
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        linear_output_tensors = [output_tensor_wrapper, scratch_tensor_wrapper]

        linear_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_TMAN,
            OpTMANLinear.op_name,
        )
        linear_op.AddInputTensors(linear_input_tensors)
        linear_op.AddOutputTensors(linear_output_tensors)

        linear_op.AddScalarParam(
            OpTMANLinear.param_group_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(group_size)},
        )
        linear_op.AddScalarParam(
            OpTMANLinear.param_bits,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(bits)},
        )
        linear_op.AddScalarParam(
            OpTMANLinear.param_symmetric,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(symmetric)},
        )

        return linear_op
