# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

from typing import Any, Optional, Tuple

import float8_experimental.config as config

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from float8_experimental.float8_tensor import (
    Float8Tensor,
    merge_mm_configs,
    ScaledMMConfig,
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
        mm_config_arg0: ScaledMMConfig,
        mm_config_arg1: ScaledMMConfig,
    ):
        ctx.mm_config_arg0 = mm_config_arg0
        ctx.mm_config_arg1 = mm_config_arg1
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None
        gradY_scale = tensor_to_scale(gradY, torch.float8_e5m2)
        fp8_tensor = to_fp8_no_autograd(
            gradY, gradY_scale, torch.float8_e5m2, mm_config_arg0=ctx.mm_config_arg0,
            mm_config_arg1=ctx.mm_config_arg1,
        )
        return fp8_tensor, None, None


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def forward(self, x):
        x_fp8 = cast_to_float8_e4m3fn(
            x, mm_config_arg0=self.fwd_gemm_config)
        if isinstance(self.weight, Float8Tensor):  # cast by FSDP
            w_fp8 = self.weight
        else:
            w_fp8 = cast_to_float8_e4m3fn(self.weight)
        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)
        y = cast_to_float8_e5m2_bw(y, self.bwd_gradX_gemm_config, self.bwd_gradW_gemm_config)
        return y

    @classmethod
    def from_float(cls, mod, emulate: bool = False) -> "Float8DynamicLinear":
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
        """
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(**super_kwargs)
        # new_mod.forward_config = ScaledMMConfig(emulate, True if not emulate else False)
        # new_mod.backward_config = ScaledMMConfig(emulate, False)

        new_mod.fwd_gemm_config = ScaledMMConfig(
            emulate=emulate, 
            use_fast_accum=True if not emulate else False)
        new_mod.bwd_gradX_gemm_config = ScaledMMConfig(
            emulate=emulate, use_fast_accum=False)
        new_mod.bwd_gradW_gemm_config = ScaledMMConfig(
            emulate=emulate, use_fast_accum=False)

        if config.enable_fsdp_fp8_all_gather:
            new_mod.weight = nn.Parameter(
                WeightWithDynamicFloat8CastTensor(mod.weight)
            )
        else:
            new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def cast_to_float8_e4m3fn(
    inpt_tensor: torch.Tensor, 
    mm_config_arg0: Optional[ScaledMMConfig]=None, 
    mm_config_arg1: Optional[ScaledMMConfig]=None, 
    reduce_amax: bool = False
) -> Float8Tensor:
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn, reduce_amax)
    return Float8Tensor.to_float8(
        inpt_tensor, scale, torch.float8_e4m3fn, mm_config_arg0=mm_config_arg0,
        mm_config_arg1=mm_config_arg1,
    )


def cast_to_float8_e5m2_bw(
    gradY: torch.Tensor, mm_config_arg0: ScaledMMConfig, mm_config_arg1: ScaledMMConfig,
) -> torch.Tensor:
    return NoopFwToFloat8E5M2Bw.apply(gradY, mm_config_arg0, mm_config_arg1)


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
}


class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
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

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor
            )

        def unwrap(t):
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: WeightWithDynamicFloat8CastTensor(x), out
        )

    def __tensor_flatten__(self):
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicFloat8CastTensor(inner_tensors["_tensor"], None)

    def __repr__(self):
        return f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor})"

    def fsdp_pre_all_gather(self, mesh):
        float8_tensor = cast_to_float8_e4m3fn(
            self._tensor, None, reduce_amax=True
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
            out._scale = scale
            return
        return Float8Tensor(data, scale, param_dtype, mm_config_arg0=None, mm_config_arg1=None), (data,)
