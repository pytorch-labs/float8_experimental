# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
from driss_torch import saturated_cast

import torch

from float8_experimental.float8_utils import (
    tensor_to_amax,
    tensor_to_scale,
    to_fp8_saturated,
)

aten = torch.ops.aten


@torch._dynamo.allow_in_graph
class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        scale: float,
        float8_dtype=torch.float8_e4m3fn,
        amax_buffer=None,
        emulate: bool = False,
    ):
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        # tensor_scaled = tensor * scale
        # bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
        bits_fp8 = saturated_cast(tensor, float8_dtype, scale.to(tensor.dtype))

        return Float8Tensor(bits_fp8, scale, tensor.dtype, emulate=emulate)

    @staticmethod
    def backward(ctx, g):
        if isinstance(g, Float8Tensor):
            return g.to_original_precision(), None, None, None, None
        else:
            return g, None, None, None, None


@torch._dynamo.allow_in_graph
class FromFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion from fp8
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return Float8Tensor.to_float8(g), None, None

class Float8Tensor(torch.Tensor):
    """
    A Python-only Float8 tensor subclass.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.
    * `_emulate`: if true using fp32 emulation for the matmuls, helpful
      if you don't have access to h100 hardware.

    Intended usage of this abstraction:
    1. to bundle raw data + fp8 metadata together for easy passing through
       Python PyTorch systems.
    2. Float8-aware user code can use the private fields on these tensors
       to call into float8 operations.
    3. Float8-agnostic user code can use these tensors as is - they will
       convert to original precision in `__torch_dispatch__`.
    """

    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _emulate: bool
    _initialized: bool
    __slots__ = ["_data", "_scale", "_orig_dtype", "_emulate", "_initialized"]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        emulate=False,
    ):
        assert scale.numel() == 1
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self._emulate = emulate
        self._initialized = True
        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, emulate={self._emulate}\nas_orig_prec={self.to_original_precision()}"

    def __tensor_flatten__(self):
        initialized = self._initialized if hasattr(self, "_initialized") else False
        if not initialized:
            self._data = torch.ones(2560, 2560)
            self._scale = torch.ones(1)
            self._orig_dtype = torch.bfloat16
            self._emulate = False
            self._initialized = True
        ctx = {
            "_orig_dtype": self._orig_dtype,
            "_emulate": self._emulate,
            "_initialized": initialized
        }
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return Float8Tensor(
                inner_tensors["_data"],
                inner_tensors["_scale"],
                metadata["_orig_dtype"],
                metadata["_emulate"],
            )

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_float8(tensor, scale, float8_dtype, amax_buffer=None, emulate: bool = False):
        """Converts a higher precision tensor to float8 in a differentiable way.

        Args:
            tensor: the tensor to convert
            scale: the scale to use to convert the tensor
            float8_dtype: the float8 dtype to use
            amax_buffer: a buffer to store the amax value in prior to conversion

        Returns:
            Float8Tensor: a float8 tensor
        """
        return ToFloat8ConstrFunc.apply(
            tensor,
            scale,
            float8_dtype,
            amax_buffer,
            emulate,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 1. tracing through __torch_function__ logic is not supported yet in
        # PT2.0, so we explicitly disallow it here for callsites from user code.
        # 2. We do need to handle a couple of ops in order for
        # TorchDynamo tracing to succeed.

        # Lazy import to avoid circular dependency
        from float8_experimental.float8_ops import FLOAT8_OPS_TABLE

        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


def to_fp8_no_autograd(
    x: torch.Tensor, float8_dtype: torch.dtype, emulate: bool
) -> Float8Tensor:
    """Convert a tensor to float8 without autograd
    This is used in multiple places in the codebase to convert a tensor to float8

    This function will calculate the scale, do the scaling, and then convert to a Float8Tensor
    Args:
        x: the tensor to convert
        scale: the scale to use to convert the tensor
        float8_dtype: the float8 dtype to use
        emulate: whether to emulate the matmuls in fp32
    """
    x_scale = tensor_to_scale(x, float8_dtype)
    x_scaled = x * x_scale
    bits_fp8 = to_fp8_saturated(x_scaled, float8_dtype)
    return Float8Tensor(bits_fp8, x_scale, x.dtype, emulate=emulate)
