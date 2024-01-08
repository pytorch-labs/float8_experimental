# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable

import torch
import torch.distributed as dist

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# define the e4m3/e5m2 constants
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

FP16_MAX_POS = torch.finfo(torch.float16).max

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12


@torch.no_grad()
def amax_to_scale(amax, float8_dtype, orig_dtype):
    scale = torch.empty((), device=amax.device, dtype=torch.float32)
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    scale.copy_(res)
    return scale


@torch.no_grad()
def amax_history_to_scale(
    amax_history,
    float8_dtype,
    orig_dtype,
    history_to_scale_fn_type,
):
    if history_to_scale_fn_type == "max":
        amax = torch.max(amax_history)
        return amax_to_scale(amax, float8_dtype, orig_dtype)
    raise NotImplementedError()


@torch.no_grad()
def tensor_to_amax(x, distributed_reduction=False):
    amax = torch.max(torch.abs(x))

    # If the user asked for distributed reduction, do it.
    # If the user did not ask for it, assume that it will
    # happen elsewhere.
    if distributed_reduction and dist.is_initialized():
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)

    return amax


@torch.no_grad()
def tensor_to_scale(x, float8_dtype):
    amax = tensor_to_amax(x)
    return amax_to_scale(amax, float8_dtype, x.dtype)


def to_fp8_saturated(x, float8_dtype):
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)


def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1


def get_min_alignment(size: int, alignment_value: int):
    """
    Returns the minimum alignment value that is greater than or equal to the given size.

    Args:
        size: The size of the data to be aligned.
        alignment_value: The alignment value to be used.

    Returns:
        int: The minimum alignment value that is greater than or equal to the given size.
    """
    if size % alignment_value == 0:
        return size
    return (1 + (size // alignment_value)) * alignment_value


def pad_tensor_for_matmul(tensor: torch.Tensor, both: bool = False) -> torch.Tensor:
    """
    Pads a 2D tensor with zeros to ensure that its dimensions are multiples of 16, which is required for H100s.

    Args:
        tensor: The tensor to pad.
        both: Whether to pad both dimensions or just the second dimension.

    Returns:
        torch.Tensor: The padded tensor.
    """
    assert tensor.dim() == 2
    dim1, dim2 = tensor.shape

    # Calculate aligned dimensions
    dim2_aligned = get_min_alignment(dim2, 16)
    dim1_aligned = get_min_alignment(dim1, 16) if both else dim1

    # Check if padding is needed for either dimension
    if dim1 == dim1_aligned and dim2 == dim2_aligned:
        return tensor

    # Calculate padding values for both dimensions
    pad_dim1 = dim1_aligned - dim1
    pad_dim2 = dim2_aligned - dim2

    return torch.nn.functional.pad(tensor, (0, pad_dim2, 0, pad_dim1))
