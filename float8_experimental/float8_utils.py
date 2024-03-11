# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# define the e4m3/e5m2 constants
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E4M3_FNUZ_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
E5M2_FNUZ_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max

FP16_MAX_POS = torch.finfo(torch.float16).max

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

IS_AMD = torch.cuda.is_available() and torch.version.hip is not None


@dataclass(frozen=True)
class FP8Dtypes:
    """Defines the fp8 dtypes to be used in forward and backwrad computations"""

    fp8_dtype_fw: torch.dtype = torch.float8_e4m3fn
    fp8_dtype_bw: torch.dtype = torch.float8_e5m2


@torch.no_grad()
def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        res = E4M3_FNUZ_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == torch.float8_e5m2:
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        res = E5M2_FNUZ_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    scale.copy_(res)
    return scale


@torch.no_grad()
def amax_history_to_scale(
    amax_history: torch.Tensor,
    float8_dtype: torch.dtype,
    orig_dtype: torch.dtype,
    history_to_scale_fn_type: Literal["max"],
):
    """Takes in a history of amax values and returns a scale tensor.
    Args:
        amax_history: A tensor containing the history of amax values.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
        history_to_scale_fn_type: The type of function to use to convert the history to a scale.
    """
    if history_to_scale_fn_type == "max":
        amax = torch.max(amax_history)
        return amax_to_scale(amax, float8_dtype, orig_dtype)
    raise NotImplementedError()


@torch.no_grad()
def amax_history_to_scale_stack(
    amax_history: torch.Tensor,
    float8_dtype: torch.dtype,
    orig_dtype: torch.dtype,
    history_to_scale_fn_type: Literal["max"],
) -> torch.Tensor:
    """Takes in a stack of amax_history tensors and returns a scale tensor.
    Args:
        amax_history: A 2D tensor containing a stack of amax histories.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
        history_to_scale_fn_type: The type of function to use to convert the history to a scale.
    """
    if history_to_scale_fn_type == "max":
        amax_stack = torch.max(amax_history, dim=1).values
        return amax_to_scale(amax_stack, float8_dtype, orig_dtype)
    raise NotImplementedError(
        f"Invalid history_to_scale_fn_type, only 'max' is supported. Got: {history_to_scale_fn_type}"
    )


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
def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype):
    """Converts a tensor to a scale tensor.
    Args:
        x: The tensor to calculate the scale for.
        float8_dtype: The float8 dtype.
    """
    amax = tensor_to_amax(x)
    return amax_to_scale(amax, float8_dtype, x.dtype)


def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
    """Converts a tensor to a saturated fp8 tensor.

    Note:
        The default behavior in PyTorch for casting to `float8_e4m3fn`
        and `e5m2` is to not saturate. In this context, we should saturate.
        A common case where we want to saturate is when the history of a
        tensor has a maximum value of `amax1`, and the current amax value
        is `amax2`, where `amax1 < amax2`. This is common when using delayed
        scaling.
    """

    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        x = x.clamp(min=-1 * E4M3_FNUZ_MAX_POS, max=E4M3_FNUZ_MAX_POS)
    elif float8_dtype == torch.float8_e5m2:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        x = x.clamp(min=-1 * E5M2_FNUZ_MAX_POS, max=E5M2_FNUZ_MAX_POS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
    return x.to(float8_dtype)


def compute_error(x: torch.Tensor, y: torch.Tensor):
    """Computes the error between two tensors in dB.

    For more details see:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        x: The original tensor.
        y: The tensor to compare to the original tensor.
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1
