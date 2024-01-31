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
import float8_experimental.config as config


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

def cast_x_to_float8_e4m3fn_pre_hook(module, args):
    """
    Hook to cast the incoming activation to `torch.float8_e4m3fn`
    """
    return module.cast_to_float8(args[0])

def cast_dldy_to_float8_e5m2_backward_pre_hook(module, grad_output):
    """
    Hook to cast the incoming gradient to `torch.float8_e5m2`
    """
    gradY = grad_output[0]
    gradY_scale = tensor_to_scale(gradY, torch.float8_e5m2)
    gradY_scaled = gradY * gradY_scale
    bits_fp8 = to_fp8_saturated(gradY_scaled, torch.float8_e5m2)
    tensor_fp8 = Float8Tensor(bits_fp8, gradY_scale, gradY.dtype, emulate=module.emulate)
    return (tensor_fp8,)

class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_activation_hooks = config.dynamic_use_activation_hooks

    def forward(self, x):
        # cast x to float8_e4m3fn
        if self.use_activation_hooks:
            x_fp8 = x
        else:
            x_fp8 = self.cast_to_float8(x)

        # cast w to float8_e4m3fn
        w_fp8 = self.cast_to_float8(self.weight)

        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)

        # Cast gradY to float8_e5m2 during backward
        if self.use_activation_hooks:
            pass
        else:
            y = self.cast_to_float8e5m2_bw(y)

        return y

    def cast_to_float8(self, inpt_tensor):
        # TODO rename this function to clarify e4m3
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
        new_mod.use_activation_hooks = config.dynamic_use_activation_hooks
        if new_mod.use_activation_hooks:
            # install the hooks
            new_mod.register_forward_pre_hook(cast_x_to_float8_e4m3fn_pre_hook)
            new_mod.register_full_backward_pre_hook(cast_dldy_to_float8_e5m2_backward_pre_hook)
        return new_mod
