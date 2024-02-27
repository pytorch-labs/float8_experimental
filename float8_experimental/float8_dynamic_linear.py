# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

import functools
from typing import Any, Callable, cast, Optional, Tuple, Union

import torch
import torch.nn as nn

from float8_experimental.float8_tensor import Float8Tensor, to_fp8_no_autograd
from float8_experimental.float8_utils import tensor_to_scale
from torch.utils._pytree import tree_map


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
        fp8_tensor = to_fp8_no_autograd(
            gradY, gradY_scale, torch.float8_e5m2, ctx.emulate
        )
        return fp8_tensor, None


def cast_x_to_float8_e4m3fn_pre_hook(module, args):
    """
    Hook to cast the incoming activation to `torch.float8_e4m3fn`
    """
    return module.cast_to_float8_e4m3fn(args[0])


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

    def __init__(self, use_activation_hooks: bool, **super_kwargs):
        """
        Args:
            use_activation_hooks (bool): whether to use activation hooks for casting to and from float8
        """
        super().__init__(**super_kwargs)

        self.use_activation_hooks = use_activation_hooks

    def forward(self, x):
        # cast x to float8_e4m3fn if not using activation hooks
        x_fp8 = x if self.use_activation_hooks else self.cast_to_float8_e4m3fn(x)

        # cast w to float8_e4m3fn
        w_fp8 = (
            self.weight
            if isinstance(self.weight, Float8Tensor)  # cast by FSDP
            else self.cast_to_float8_e4m3fn(self.weight)
        )

        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)

        # Cast gradY to float8_e5m2 during backward if not using activation hooks
        if not self.use_activation_hooks:
            y = self.cast_to_float8_e5m2_bw(y)

        return y

    def cast_to_float8_e4m3fn(
        self, inpt_tensor: torch.Tensor, reduce_amax: bool = False
    ) -> Float8Tensor:
        scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn, reduce_amax)
        return Float8Tensor.to_float8(
            inpt_tensor, scale, torch.float8_e4m3fn, emulate=self.emulate
        )

    def cast_to_float8_e5m2_bw(self, gradY: torch.Tensor) -> torch.Tensor:
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate)

    @classmethod
    def from_float(
        cls,
        mod,
        emulate: bool = False,
        use_activation_hooks: bool = False,
        use_fp8_all_gather: bool = False,
    ) -> "Float8DynamicLinear":
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
            use_activation_hooks (bool): whether to use activation hooks for casting to and from float8
            use_fp8_all_gather (bool): whether to use fp8 all-gather for FSDP
        """
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(use_activation_hooks, **super_kwargs)
        if use_fp8_all_gather:
            cast_fn = new_mod.cast_to_float8_e4m3fn
            new_mod.weight = nn.Parameter(
                Float8DynamicLinearWeightTensor(mod.weight, cast_fn, emulate)
            )
        else:
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


class Float8DynamicLinearWeightTensor(torch.Tensor):
    def __new__(cls, tensor: torch.Tensor, cast_fn: Callable, emulate: bool):
        return cls._make_subclass(cls, tensor, tensor.requires_grad)

    def __init__(self, tensor: torch.Tensor, cast_fn: Callable, emulate: bool):
        super().__init__()
        self.cast_fn = cast_fn
        self.emulate = emulate

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def wrap(cast_fn: Callable, emulate: bool, o: Any):
            if isinstance(o, torch.Tensor) and not isinstance(o, cls):
                return cls(o, cast_fn, emulate)
            return o

        with torch._C.DisableTorchFunctionSubclass():
            if isinstance(args[0], cls):
                out = func(*args, **kwargs)
                return tree_map(
                    functools.partial(wrap, args[0].cast_fn, args[0].emulate), out
                )
            return func(*args, **kwargs)

    def fsdp_pre_all_gather(self) -> Tuple[Tuple[torch.Tensor, ...], Any]:
        float8_tensor = self.cast_fn(self, reduce_amax=True)
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[Float8Tensor, Tuple[torch.Tensor, ...]], None]:
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            out = cast(Float8Tensor, out)
            assert (
                data.untyped_storage().data_ptr()
                == out._data.untyped_storage().data_ptr()
            )
            out._scale = scale
            return
        return Float8Tensor(data, scale, param_dtype, self.emulate), (data,)
