# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""
from typing import Optional

import torch

from float8_experimental.float8_tensor import Float8Tensor, to_fp8_no_autograd
from float8_experimental.float8_utils import FP8Dtypes, tensor_to_scale


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2Bw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2, initialize if needed
    """

    @staticmethod
    def forward(ctx, tensor, emulate: bool, fp8_dtype_bw: torch.dtype):
        ctx.emulate = emulate
        ctx.fp8_dtype_bw = fp8_dtype_bw
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        gradY_scale = tensor_to_scale(gradY, ctx.fp8_dtype_bw)
        fp8_tensor = to_fp8_no_autograd(
            gradY, gradY_scale, ctx.fp8_dtype_bw, ctx.emulate
        )
        return fp8_tensor, None, None


def cast_x_to_float8_e4m3fn_pre_hook(module, args):
    """
    Hook to cast the incoming activation to `torch.float8_e4m3fn`
    """
    return module.cast_to_float8_e4m3(args[0])


def cast_grad_to_float8_e5m2_backward_forward_hook(module, input, output):
    """This is a forward hook that sends the output of the model through
    a no-op in the forward but a cast to float8_e5m2 in the backward.

    Args:
        module (nn.Module): the module to cast the output of
        input (Tensor): the input to the module forward call
        output (Tensor): the output of the module forward
    """
    return module.cast_to_float8_e5m2_bw(output)


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def __init__(
        self, use_activation_hooks: bool, fp8_dtype: FP8Dtypes, **super_kwargs
    ):
        """
        Args:
            use_activation_hooks (bool): whether to use activation hooks for casting to and from float8
            fp8_dtype (torch.dtype): the dtype to use for fp8
        """
        super().__init__(**super_kwargs)

        self.use_activation_hooks = use_activation_hooks
        # I want to store the dataclass but I think that will break torch compile
        self.fp8_dtype_fw = fp8_dtype.fp8_dtype_fw
        self.fp8_dtype_bw = fp8_dtype.fp8_dtype_bw
        self.emulate = False

    def forward(self, input):
        # cast x to float8_e4m3fn if not using activation hooks
        x_fp8 = input if self.use_activation_hooks else self.cast_to_float8_e4m3(input)

        # cast w to float8_e4m3fn
        w_fp8 = self.cast_to_float8_e4m3(self.weight)

        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)

        # Cast gradY to float8_e5m2 during backward if not using activation hooks
        if not self.use_activation_hooks:
            y = self.cast_to_float8_e5m2_bw(y)

        return y

    def cast_to_float8_e4m3(self, inpt_tensor: torch.Tensor) -> Float8Tensor:
        """
        This function casts the input tensor to a Float8Tensor
        backed by one of two types depending on the GPU type

        - On Nvidia GPUs, it casts to torch.float8_e4m3fn
        - On AMD Gpus, it casts to torch.float8_e4m3fnuz

        """
        scale = tensor_to_scale(inpt_tensor, self.fp8_dtype_fw)
        return Float8Tensor.to_float8(
            inpt_tensor, scale, self.fp8_dtype_fw, emulate=self.emulate
        )

    def cast_to_float8_e5m2_bw(self, gradY: torch.Tensor) -> torch.Tensor:
        """
        This function is a noop in the forward but casts
        the input tensor to a Float8Tensor during the backwards pass
        backed by one of two types depending on the GPU type

        - On Nvidia GPUs, it casts to torch.float8_e4m3fn
        - On AMD Gpus, it casts to torch.float8_e4m3fnuz

        """
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate, self.fp8_dtype_bw)

    @classmethod
    def from_float(
        cls,
        mod,
        emulate: bool = False,
        use_activation_hooks: bool = False,
        fp8_dtypes: Optional[FP8Dtypes] = None,
    ) -> "Float8DynamicLinear":
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
            use_activation_hooks (bool): whether to use activation hooks for casting to and from float8
        """
        if fp8_dtypes is None:
            fp8_dtypes = FP8Dtypes()
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(use_activation_hooks, fp8_dtypes, **super_kwargs)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        if new_mod.use_activation_hooks:
            # install the hooks
            new_mod.register_forward_pre_hook(cast_x_to_float8_e4m3fn_pre_hook)
            new_mod.register_forward_hook(
                cast_grad_to_float8_e5m2_backward_forward_hook
            )
        return new_mod
