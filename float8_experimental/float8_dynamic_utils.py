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
import torch.utils._pytree as pytree

from float8_experimental.float8_tensor import (
    Float8Tensor,
    merge_mm_configs,
    ScaledMMConfig,
    tensor_already_casted_to_fp8,
    to_fp8_no_autograd,
)
from float8_experimental.float8_utils import e4m3_dtype, e5m2_dtype, tensor_to_scale
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
    def __new__(
        cls,
        tensor: torch.Tensor,
        mm_config: ScaledMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
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

    def __init__(
        self,
        tensor: torch.Tensor,
        mm_config: ScaledMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._mm_config = mm_config
        # for dynamic scaling
        # `precompute_float8_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor, args[0]._mm_config
            )
        mm_config: Optional[ScaledMMConfig] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._mm_config
            else:
                mm_config = merge_mm_configs(mm_config, t._mm_config)
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: WeightWithDynamicFloat8CastTensor(x, mm_config), out
        )

    def __tensor_flatten__(self):
        if self._precomputed_scale:
            return ["_tensor", "_precomputed_scale"], self._mm_config
        else:
            return ["_tensor"], self._mm_config

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        mm_config = flatten_spec
        return WeightWithDynamicFloat8CastTensor(
            inner_tensors["_tensor"],
            mm_config,
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor}, mm_config={self._mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            float8_tensor = Float8Tensor.to_float8(
                self._tensor,
                self._precomputed_scale,
                torch.float8_e4m3fn,
                mm_config=self._mm_config,
            )
        else:
            float8_tensor = cast_to_float8_e4m3_dynamic(
                self._tensor, self._mm_config, reduce_amax=True
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
        return Float8Tensor(data, scale, param_dtype, self._mm_config), (data,)
