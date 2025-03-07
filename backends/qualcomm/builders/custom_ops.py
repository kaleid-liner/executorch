import torch


@torch.library.custom_op("tman::linear", mutates_args=())
def tman_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
) -> torch.Tensor:
    #  M or output features
    M = qweight.shape[0]
    out_shape = x.shape[:-1] + (M,)
    return x.new_empty(out_shape)


@tman_linear.register_fake
def _(x, qweight, scales, group_size, bits, symmetric):
    M = qweight.shape[0]
    out_shape = x.shape[:-1] + (M,)
    return x.new_empty(out_shape)
