# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

import torch

from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import tensor_to_scale, to_fp8_saturated


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
        emulate: bool,
    ):
        ctx.emulate = emulate
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        gradY_scale = tensor_to_scale(gradY, torch.float8_e5m2)
        gradY_scaled = gradY * gradY_scale
        bits_fp8 = to_fp8_saturated(gradY_scaled, torch.float8_e5m2)
        return (
            Float8Tensor(bits_fp8, gradY_scale, gradY.dtype, emulate=ctx.emulate),
            None,
        )


def cast_weight_linear(
    x_fp8: Float8Tensor, weight: torch.Tensor, scale: torch.Tensor, bias, emulate: bool
) -> torch.Tensor:
    """Cast weight to fp8_e4m3fn and do linear
    Why a new function for something that can be inlined?
    Because we want to call torch utils checkpoint on this function.
    We always want to recompute the cast of the weight to fp8 since we can, trivially
    fuse this into the transpose/contiguous of the weight during the backwards.

    Args:
        x_fp8 (Float8Tensor): input activation in fp8
        weight (torch.Tensor): weight tensor in higher precision
        scale (torch.Tensor): scale tensor for weight
        bias: bias tensor in higher precision
        emulate (bool): whether to emulate fp8 matmul logic in float32
    """
    w_fp8 = Float8Tensor.to_float8(weight, scale, torch.float8_e4m3fn, emulate=emulate)
    y = torch.nn.functional.linear(x_fp8, w_fp8, bias)
    return y


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def forward(self, x):
        x_fp8 = self.cast_to_float8(x)
        scale = tensor_to_scale(self.weight, torch.float8_e4m3fn)
        y = torch.utils.checkpoint.checkpoint(
            cast_weight_linear,
            x_fp8,
            self.weight,
            scale,
            self.bias,
            self.emulate,
            use_reentrant=False,
        )

        # Cast gradY to float8_e5m2 during backward
        y = self.cast_to_float8e5m2_bw(y)

        return y

    def cast_to_float8(self, inpt_tensor):
        scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn)
        return Float8Tensor.to_float8(
            inpt_tensor, scale, torch.float8_e4m3fn, emulate=self.emulate
        )

    def cast_to_float8e5m2_bw(self, gradY):
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate)

    @classmethod
    def from_float(cls, mod, emulate: bool = False):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        return new_mod
