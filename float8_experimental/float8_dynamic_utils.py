# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

import torch

from float8_experimental.float8_tensor import (
    Float8Tensor,
    ScaledMMConfig,
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
        mm_config: ScaledMMConfig,
    ):
        ctx.mm_config = mm_config
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None
        gradY_scale = tensor_to_scale(gradY, e5m2_dtype)
        fp8_tensor = to_fp8_no_autograd(
            gradY, gradY_scale, e5m2_dtype, mm_config=ctx.mm_config
        )
        return fp8_tensor, None


def cast_to_float8_e4m3_dynamic(
    inpt_tensor: torch.Tensor, mm_config: ScaledMMConfig, reduce_amax: bool = False
) -> Float8Tensor:
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = tensor_to_scale(inpt_tensor, e4m3_dtype, reduce_amax)
    return Float8Tensor.to_float8(inpt_tensor, scale, e4m3_dtype, mm_config=mm_config)


def cast_to_float8_e5m2_dynamic_bw(
    gradY: torch.Tensor, mm_config: ScaledMMConfig
) -> torch.Tensor:
    return NoopFwToFloat8E5M2Bw.apply(gradY, mm_config)
