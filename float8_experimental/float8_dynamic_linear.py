# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

import torch
from float8_experimental.float8_ops import float8_linear

from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import (
    get_maybe_autocast_inputs,
    tensor_to_scale,
    to_fp8_saturated,
)


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


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def forward(self, x):
        # Tried to do this with @custom_fwd/bwd but it didn't work
        temp_x, temp_weight, temp_bias = get_maybe_autocast_inputs(
            x, self.weight, self.bias
        )
        x_fp8 = self.cast_to_float8(temp_x)
        weight_scale = tensor_to_scale(temp_weight, torch.float8_e4m3fn)
        y = float8_linear(
            x_fp8,
            temp_weight,
            None,  # bias
            weight_scale,
            None,
            self.emulate,
            self.recompute_weight_cast,
        )
        # Cast gradY to float8_e5m2 during backward
        y = self.cast_to_float8e5m2_bw(y)

        # TODO We should use addmm above but this fails the single fsdp test:
        # FAILED: _orig_mod.0.fp8_amax_w, 0.2197265625, 0.21875
        # Not immediately clear why the bias being fused in would only effect the numerics
        # for the weight....
        if temp_bias is not None:
            y = y + temp_bias

        return y

    def cast_to_float8(self, inpt_tensor):
        scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn)
        return Float8Tensor.to_float8(
            inpt_tensor, scale, torch.float8_e4m3fn, emulate=self.emulate
        )

    def cast_to_float8e5m2_bw(self, gradY):
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate)

    @classmethod
    def from_float(
        cls, mod, emulate: bool = False, recompute_weight_cast: bool = False
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
            recompute_weight_cast (bool): whether to recompute the weight cast on every
                backwards pass
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        new_mod.recompute_weight_cast = recompute_weight_cast
        return new_mod
