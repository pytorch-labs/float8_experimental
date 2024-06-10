# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
from typing import Dict, Optional

import torch

import torch.distributed._functional_collectives as funcol
from float8_experimental.float8_utils import tensor_to_amax, to_fp8_saturated, e4m3_dtype
from torch.distributed._tensor import DTensor

aten = torch.ops.aten


# ScaledMMConfig is a namedtuple that defines the configuration for the scaled_mm in the forward and backward pass.
# emulate: whether to emulate the matmuls in fp32
# use_fast_accum: whether to use the fast-accumulation option for scaled_mm
# fp8_output: whether to output the result of the scaled_mm in fp8
ScaledMMConfig = namedtuple(
    "ScaledMMConfig",
    ["emulate", "use_fast_accum", "fp8_output"],
    defaults=[False, False, False],
)


def merge_mm_configs(
    a_mm_config: ScaledMMConfig, b_mm_config: ScaledMMConfig
) -> ScaledMMConfig:
    """Merges two mm_configs together emulate behavior must match,
    However we want to use_fast_accum in forward and not in backward.
    We do this by populating the fields of the backproping grad. Same applies for fp8_output.

    For both use_fast_accum and fp8_output, if either config is False, the merged config will be False.
    """
    assert (
        a_mm_config.emulate == b_mm_config.emulate
    ), "Both mm_configs must have the same emulate value, but got {} and {}".format(
        a_mm_config.emulate, b_mm_config.emulate
    )
    return ScaledMMConfig(
        emulate=a_mm_config.emulate,
        use_fast_accum=a_mm_config.use_fast_accum and b_mm_config.use_fast_accum,
        fp8_output=a_mm_config.fp8_output and b_mm_config.fp8_output,
    )


def tensor_already_casted_to_fp8(tensor: torch.Tensor) -> bool:
    """
    Check if the tensor is already casted to fp8
    """
    if isinstance(tensor, Float8Tensor):
        return True
    elif isinstance(tensor, DTensor):
        # TODO: shall we stick to public API and directly use tensor.to_local() here?
        return tensor_already_casted_to_fp8(tensor._local_tensor)
    elif isinstance(tensor, funcol.AsyncCollectiveTensor):
        return tensor_already_casted_to_fp8(tensor.elem)

    return False


def to_fp8_no_autograd(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    float8_dtype: torch.dtype,
    mm_config: Optional[ScaledMMConfig],
) -> "Float8Tensor":
    """Convert a tensor to float8 without autograd
    This is used in multiple places in the codebase to convert a tensor to float8

    This function will apply the scaling, and then convert to a Float8Tensor

    Note:
    We will call this function with a DTensor subclass. Ideally this would be an aten OP
    that DTensor could overload to ensure proper semantics. There are some techincal issues
    with that composing with FakeTensor, so we special case here.

    DTensor Invariant: DTensor must always be the outer most tensor subclass

    Args:
        x: the tensor to convert
        scale: the scale to use to convert the tensor
        float8_dtype: the float8 dtype to use
        mm_config: Defines the configuration for the scaled_mm
    """
    x_scaled = x * x_scale
    bits_fp8 = to_fp8_saturated(x_scaled, float8_dtype)

    if isinstance(bits_fp8, DTensor):
        assert isinstance(
            x, DTensor
        ), "Expected Float8 scale to be a DTensor if bits_fp8 is a DTensor"
        bits_mesh = bits_fp8.device_mesh
        bits_placements = bits_fp8.placements
        local_bits = bits_fp8.to_local()
        local_scale = x_scale.to_local()
        inner_float8_tensor = Float8Tensor(
            local_bits, local_scale, x.dtype, mm_config=mm_config
        )
        return DTensor.from_local(
            inner_float8_tensor,
            bits_mesh,
            bits_placements,
            run_check=False,
            shape=bits_fp8.size(),
            stride=bits_fp8.stride(),
        )

    return Float8Tensor(bits_fp8, x_scale, x.dtype, mm_config=mm_config)


@torch._dynamo.allow_in_graph
class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8.
    * forward: convert from high precision to float8
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        float8_dtype=e4m3_dtype,
        amax_buffer: Optional[torch.Tensor] = None,
        mm_config: Optional[ScaledMMConfig] = None,
    ):
        """Autograd enabled wrapper around to_fp8_no_autograd that will also populate the amax buffer.
        Args
            tensor: the tensor to convert
            scale: the scale to use to convert the tensor
            float8_dtype: the float8 dtype either, torch.float8_e4m3fn or torch.float8_e5m2fn
            amax_buffer: an Optional buffer buffer to store the amax value in prior to conversion
            emulate: whether to emulate the matmuls in fp32
        """
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        return to_fp8_no_autograd(tensor, scale, float8_dtype, mm_config=mm_config)

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


@torch._dynamo.allow_in_graph
class FromFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion from fp8.
    * forward: convert from float8 to high precision
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return g, None, None


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
    _mm_config: ScaledMMConfig
    __slots__ = ["_data", "_scale", "_orig_dtype", "_mm_config"]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        mm_config: Optional[ScaledMMConfig],
    ):
        assert (
            scale.numel() == 1
        ), "Scale should contain a single value, but got: {} elements".format(
            scale.numel()
        )

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
        self._mm_config = mm_config if mm_config is not None else ScaledMMConfig()

        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, mm_config={self._mm_config}\nas_orig_prec={self.to_original_precision()}"

    def __tensor_flatten__(self):
        ctx = {
            "_orig_dtype": self._orig_dtype,
            "_mm_config": self._mm_config,
        }
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return Float8Tensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["_orig_dtype"],
            metadata["_mm_config"],
        )

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_float8(
        tensor: torch.Tensor,
        scale: torch.Tensor,
        float8_dtype: torch.dtype,
        amax_buffer: Optional[torch.Tensor] = None,
        mm_config: Optional[ScaledMMConfig] = None,
    ):
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
            tensor, scale, float8_dtype, amax_buffer, mm_config
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 1. tracing through __torch_function__ logic is not supported yet in
        # PT2.0, so we explicitly disallow it here for callsites from user code.
        # 2. We do need to handle a couple of ops in order for
        # TorchDynamo tracing to succeed.

        # Lazy import to avoid circular dependency
        from float8_experimental.float8_ops import FLOAT8_OPS_TABLE

        # All ops in the FLOAT8_OPS_TABLE expect Float8Tensor as inputs
        # And don't support mixed tensor subclasses. This will trigger the handler for
        # the next type in the dispatch list
        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
