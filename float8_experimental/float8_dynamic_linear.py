# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from float8_experimental.float8_tensor import (
    Float8Tensor,
    tensor_already_casted_to_fp8,
    to_fp8_no_autograd,
)
from float8_experimental.float8_utils import tensor_to_scale
from torch._prims_common import suggest_memory_format


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
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None
        gradY_scale = tensor_to_scale(gradY, torch.float8_e5m2)
        fp8_tensor = to_fp8_no_autograd(
            gradY, gradY_scale, torch.float8_e5m2, ctx.emulate
        )
        return fp8_tensor, None


def cast_to_float8_e4m3fn(
    inpt_tensor: torch.Tensor, emulate: bool, reduce_amax: bool = False
) -> Float8Tensor:
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn, reduce_amax)
    return Float8Tensor.to_float8(
        inpt_tensor, scale, torch.float8_e4m3fn, emulate=emulate
    )


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def forward(self, x):
        x_fp8 = self.cast_to_float8_e4m3fn(x)
        w_fp8 = (
            self.weight
            if isinstance(self.weight, Float8Tensor)  # cast by FSDP
            else self.cast_to_float8_e4m3fn(self.weight)
        )
        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)
        y = self.cast_to_float8_e5m2_bw(y)
        return y

    def cast_to_float8_e4m3fn(
        self, inpt_tensor: torch.Tensor, reduce_amax: bool = False
    ) -> Float8Tensor:
        return cast_to_float8_e4m3fn(inpt_tensor, self.emulate, reduce_amax)

    def cast_to_float8_e5m2_bw(self, gradY: torch.Tensor) -> torch.Tensor:
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate)

    @classmethod
    def from_float(
        cls,
        mod,
        emulate: bool = False,
        use_fp8_all_gather: bool = False,
    ) -> "Float8DynamicLinear":
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
            use_fp8_all_gather (bool): whether to use fp8 all-gather for FSDP
        """
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(**super_kwargs)
        new_mod.weight = (
            nn.Parameter(Float8DynamicLinearWeightTensor(mod.weight, emulate))
            if use_fp8_all_gather
            else mod.weight
        )
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        return new_mod


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
}


class Float8DynamicLinearWeightTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor, emulate: bool):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, emulate: bool):
        self._tensor = tensor
        self._emulate = emulate

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return args[0]
        emulate = False

        def unwrap(t):
            nonlocal emulate
            emulate |= t._emulate
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            Float8DynamicLinearWeightTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: Float8DynamicLinearWeightTensor(x, emulate), out
        )

    def __tensor_flatten__(self):
        return ["_tensor"], self._emulate

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        emulate = flatten_spec
        return Float8DynamicLinearWeightTensor(inner_tensors["_tensor"], emulate)

    def __repr__(self):
        return f"Float8DynamicLinearWeightTensor(tensor={self._tensor}, emulate={self._emulate})"

    def fsdp_pre_all_gather(self):
        float8_tensor = cast_to_float8_e4m3fn(
            self._tensor, self._emulate, reduce_amax=True
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8Tensor), f"{type(out)}"
            assert (
                data.untyped_storage().data_ptr()
                == out._data.untyped_storage().data_ptr()
            ), f"Expects out's data to be the all-gather output"
            out._scale = scale
            return
        return Float8Tensor(data, scale, param_dtype, self._emulate), (data,)
