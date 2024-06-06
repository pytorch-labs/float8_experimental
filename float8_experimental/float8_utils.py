# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Tuple

import torch
import torch.distributed as dist

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

IS_AMD = torch.cuda.is_available() and torch.version.hip is not None
FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


@torch.no_grad()
def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype, clamp_amax: bool = True):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
        clamp_amax: default is True. False for FSDP fp8 all-gather since FSDP applied `torch.clamp` during pre-compute after optimizer.step
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype in FP8_TYPES:
        if clamp_amax:
            res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=EPS)
        else:
            res = torch.finfo(float8_dtype).max / amax
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)
    scale.copy_(res)
    return scale


@torch.no_grad()
def amax_history_to_scale(
    amax_history: torch.Tensor,
    float8_dtype: torch.Tensor,
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
    if float8_dtype in FP8_TYPES:
        max_value = torch.finfo(float8_dtype).max
        x = x.clamp(min=-max_value, max=max_value)
        return x.to(float8_dtype)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")


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


def fp8_tensor_statistics(
    tensor: torch.Tensor, float8_dtype=torch.float8_e4m3fn
) -> Tuple[int, ...]:
    """Calculate FP8 tensor stats

    Args:
        tensor: The tensor to calculate stats for.
        float8_dtype: The float8 dtype.

    Returns:
        A tuple containing the number of zeros and the number of max values.
    """
    if float8_dtype in FP8_TYPES:
        FP8_MAX = torch.finfo(float8_dtype).max
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
    tensor_orig_type = tensor._data.to(dtype=tensor._orig_dtype)
    num_max = (torch.abs(tensor_orig_type) == FP8_MAX).sum().item()
    num_zero = (tensor_orig_type == 0).sum().item()
    return (num_zero, num_max)


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1
