# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch

from float8_experimental.config import ScalingGranularity
from float8_experimental.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    tensor_already_casted_to_fp8,
    to_fp8_no_autograd,
)
from float8_experimental.float8_utils import e4m3_dtype, e5m2_dtype, tensor_to_scale


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2Bw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2, initialize if needed
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
        fp8_tensor = to_fp8_no_autograd(
            gradY,
            gradY_scale,
            e5m2_dtype,
            linear_mm_config=ctx.linear_mm_config,
            gemm_input_role=GemmInputRole.GRAD_OUTPUT,
        )
        return fp8_tensor, None


def cast_to_float8_e4m3_dynamic(
    inpt_tensor: torch.Tensor,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    dim: Optional[Union[int, Tuple[int]]] = None,
) -> Float8Tensor:
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = tensor_to_scale(inpt_tensor, e4m3_dtype, reduce_amax, granularity, dim)
    return Float8Tensor.to_float8(
        inpt_tensor,
        scale,
        e4m3_dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )


def cast_to_float8_e5m2_dynamic_bw(
    gradY: torch.Tensor, linear_mm_config: LinearMMConfig
) -> torch.Tensor:
    return NoopFwToFloat8E5M2Bw.apply(gradY, linear_mm_config)
