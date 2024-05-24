# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

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
def amax_to_scale(amax, float8_dtype, orig_dtype, clamp_amax=True):
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == torch.float8_e4m3fn:
        if clamp_amax:
            res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
        else:
            res = E4M3_MAX_POS / amax
    else:  # e5m2
        if clamp_amax:
            res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
        else:
            res = E5M2_MAX_POS / amax

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
def amax_history_to_scale_stack(
    amax_history: torch.Tensor,
    float8_dtype: torch.dtype,
    orig_dtype: torch.dtype,
    history_to_scale_fn_type: str,
) -> torch.Tensor:
    """Takes in a stack of amax_history tensors and returns a scale tensor."""
    if history_to_scale_fn_type == "max":
        amax_stack = torch.max(amax_history, dim=1).values
        return amax_to_scale(amax_stack, float8_dtype, orig_dtype)
    raise NotImplementedError(
        f"Invalid history_to_scale_fn_type, only 'max' is supported. Got: {history_to_scale_fn_type}"
    )


@torch.no_grad()
def tensor_to_amax(x: torch.Tensor, reduce_amax: bool = False) -> torch.Tensor:
    amax = torch.max(torch.abs(x))

    # If the user asked for distributed reduction, do it.
    # If the user did not ask for it, assume that it will
    # happen elsewhere.
    if reduce_amax and dist.is_initialized():
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)

    return amax


@torch.no_grad()
def tensor_to_scale(
    x: torch.Tensor, float8_dtype: torch.dtype, reduce_amax: bool = False
) -> torch.Tensor:
    amax = tensor_to_amax(x, reduce_amax=reduce_amax)
    return amax_to_scale(amax, float8_dtype, x.dtype)


def to_fp8_saturated(x, float8_dtype: torch.dtype):
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


def fp8_tensor_statistics(
    tensor: torch.Tensor, float8_dtype=torch.float8_e4m3fn
) -> Tuple[int, ...]:
    """Calculate FP8 tensor stats"""
    if float8_dtype == torch.float8_e4m3fn:
        FP8_MAX = E4M3_MAX_POS
    else:  # e5m2
        FP8_MAX = E5M2_MAX_POS
    tensor_orig_type = tensor._data.to(dtype=tensor._orig_dtype)
    num_max = (torch.abs(tensor_orig_type) == FP8_MAX).sum().item()
    num_zero = (tensor_orig_type == 0).sum().item()
    return (num_zero, num_max)


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1
