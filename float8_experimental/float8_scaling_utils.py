# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

from typing import Optional

import torch

from float8_experimental.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
    ScaledMMConfig,
    tensor_already_casted_to_fp8,
)

from float8_experimental.float8_utils import (
    amax_history_to_scale,
    e4m3_dtype,
    e5m2_dtype,
    tensor_to_amax,
    tensor_to_scale,
)


def cast_to_float8_e4m3_dynamic(
    inpt_tensor: torch.Tensor,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
) -> Float8Tensor:
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = tensor_to_scale(inpt_tensor, e4m3_dtype, reduce_amax)
    return hp_tensor_and_scale_to_float8(
        inpt_tensor,
        scale,
        e4m3_dtype,
        linear_mm_config,
        gemm_input_role,
    )


# TODO(future PR): align name with cast_to_float8_e4m3_dynamic
def cast_to_float8_delayed(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    float8_dtype: torch.dtype,
    amax_buffer: torch.Tensor,
    linear_mm_config: Optional[LinearMMConfig] = None,
    gemm_input_role: Optional[GemmInputRole] = GemmInputRole.INPUT,
):
    amax_buffer.fill_(tensor_to_amax(tensor))
    return hp_tensor_and_scale_to_float8(
        tensor,
        scale,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
    )


def _maybe_initialize_amaxes_scales_for_float8_cast(
    x,
    cur_amax,
    amax_history,
    scale,
    scale_fn_name,
    float8_dtype,
    is_initialized,
    reduce_amax,
):
    """
    If x is about to be cast to `float8` and the amax buffers are not initialized,
    initializes them inplace.
    """
    if is_initialized:
        return
    with torch.no_grad():
        # Note: we need to enable distributed reduction here in order
        # to match numerics between single GPU and multi GPU code for
        # activations and gradients
        new_amax = tensor_to_amax(x, reduce_amax=reduce_amax)
        cur_amax.fill_(new_amax)
        amax_history[0] = new_amax
        new_scale = amax_history_to_scale(
            amax_history, float8_dtype, x.dtype, scale_fn_name
        )
        scale.copy_(new_scale)


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2BwDelayed(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2 with delayed scaling, initialize if needed
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        fp8_amax_grad_output,
        fp8_amax_history_grad_output,
        fp8_scale_grad_output,
        scale_fn_name,
        is_amax_initialized,
        linear_mm_config: LinearMMConfig,
    ):
        ctx.save_for_backward(
            fp8_amax_grad_output, fp8_amax_history_grad_output, fp8_scale_grad_output
        )
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
        ctx.linear_mm_config = linear_mm_config
        return tensor

    @staticmethod
    def backward(ctx, go):
        (
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
        ) = ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        is_amax_initialized = ctx.is_amax_initialized

        _maybe_initialize_amaxes_scales_for_float8_cast(
            go,
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
            scale_fn_name,
            e5m2_dtype,
            is_amax_initialized,
            reduce_amax=True,
        )

        fp8_amax_grad_output.fill_(tensor_to_amax(go))

        res = hp_tensor_and_scale_to_float8(
            go,
            fp8_scale_grad_output,
            e5m2_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        empty_grads = None, None, None, None, None, None
        return res, *empty_grads


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2BwDynamic(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2 with dynamic scaling
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        linear_mm_config: LinearMMConfig,
    ):
        ctx.linear_mm_config = linear_mm_config
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None
        gradY_scale = tensor_to_scale(gradY, e5m2_dtype)
        fp8_tensor = hp_tensor_and_scale_to_float8(
            gradY,
            gradY_scale,
            e5m2_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        return fp8_tensor, None
