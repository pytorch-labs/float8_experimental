# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import enum
from collections import namedtuple
from typing import Dict, Optional

import torch

import torch.distributed._functional_collectives as funcol
from float8_experimental.float8_utils import (
    e4m3_dtype,
    tensor_to_amax,
    to_fp8_saturated,
)
from torch.distributed._tensor import DTensor

aten = torch.ops.aten

#
# A note on configuration of float8 logic in a linear
# TODO(future): move all the configs to separate file
#
# There are three gemms in a forward + backward of a Linear layer:
#
# 1.     x @ w_t   = y     (forward pass)
# 2. dL_dY @ w     = dL_dX (backward pass)
# 3.   x_t @ dL_dY = dL_dW (backward pass)
#
# In the formulas above, there are:
# A. six input tensors (x, x_t, w, w_t, dL_dY, dL_dY_t).
#    - Note that dL_dY_t is implied because of memory format requirements
#      of float8 gemms
# B. three output tensors (y, dL_dX, dL_dW)
#
# We want each input tensor, gemm, and output tensor to be configurable.
# The state of this configuration today is:
#
# i. pairs of input tensors (non-t and t variants) have their scaling
#    configurable via the scaling_type_{x_w_dL_dY} arguments to Float8Linear
# ii. each gemm + output is configurable via ScaledMMConfig, which is not user facing
# iii. LinearMMConfig is a container for the three ScaledMMConfig objects needed
#    to configure all three gemms, also not user facing


# ScaledMMConfig is a namedtuple that defines the configuration for the scaled_mm in the forward and backward pass.
# emulate: whether to emulate the matmuls in fp32
# use_fast_accum: whether to use the fast-accumulation option for scaled_mm
# fp8_output: whether to output the result of the scaled_mm in fp8
# pad_inner_dim: whether to pad the inner dimension of a and b with 0s. This is needed for matmuls not aligned to 16.
ScaledMMConfig = namedtuple(
    "ScaledMMConfig",
    ["emulate", "use_fast_accum", "fp8_output", "pad_inner_dim"],
    defaults=[False, False, False, False],
)

# The object below exists for convenience, to allow Float8Tensor to use
# the right config based on which gemm from `y`, `dL_dX`, `dL_dW` is
# being called.
LinearMMConfig = namedtuple(
    "LinearMMConfig",
    ["y", "dL_dX", "dL_dW"],
    defaults=[
        ScaledMMConfig(False, True, False, False),
        ScaledMMConfig(False, False, False, False),
        ScaledMMConfig(False, False, False, False),
    ],
)


# Given a Float8Tensor, the enum below describes the expected role of this
# tensor in the three gemms present in the fw + bw pass of a Linear layer.
# This is used to choose the right config for a float8 gemm when the
# gemm is performed.
class GemmInputRole(enum.Enum):
    X = "x"
    W = "w"
    DL_DY = "dL_dY"


# choose which scaled_mm_config to use based on gemm inputs
def choose_scaled_mm_config(
    a_role: GemmInputRole,
    a_linear_mm_config: LinearMMConfig,
    b_role: GemmInputRole,
    b_linear_mm_config: LinearMMConfig,
):
    if a_role is GemmInputRole.X and b_role is GemmInputRole.W:
        assert (
            a_linear_mm_config.y == b_linear_mm_config.y
        ), f"linear_mm_config.y mismatch: {a_linear_mm_config.y} vs {b_linear_mm_config.y}"
        return a_linear_mm_config.y
    elif a_role is GemmInputRole.DL_DY and b_role is GemmInputRole.W:
        assert (
            a_linear_mm_config.dL_dX == b_linear_mm_config.dL_dX
        ), f"linear_mm_config.dL_dX mismatch: {a_linear_mm_config.dL_dX} vs {b_linear_mm_config.dL_dX}"
        return a_linear_mm_config.dL_dX
    else:
        assert (
            a_role is GemmInputRole.DL_DY and b_role is GemmInputRole.X
        ), f"unexpected a_role {a_role} and b_role {b_role}"
        assert (
            a_linear_mm_config.dL_dW == b_linear_mm_config.dL_dW
        ), f"linear_mm_config.dL_dW mismatch: {a_linear_mm_config.dL_dW} vs {b_linear_mm_config.dL_dW}"
        return a_linear_mm_config.dL_dW


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
    linear_mm_config: Optional[LinearMMConfig],
    gemm_input_role: Optional[GemmInputRole],
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
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        gemm_input_role: Defines the role of this tensor (x, w or dL_dY) in
          the 3 fwd/bwd gemms of linear
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
            local_bits,
            local_scale,
            x.dtype,
            mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )
        return DTensor.from_local(
            inner_float8_tensor,
            bits_mesh,
            bits_placements,
            run_check=False,
            shape=bits_fp8.size(),
            stride=bits_fp8.stride(),
        )

    return Float8Tensor(
        bits_fp8,
        x_scale,
        x.dtype,
        mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )


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
        linear_mm_config: Optional[LinearMMConfig] = None,
        gemm_input_role: Optional[GemmInputRole] = GemmInputRole.X,
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

        return to_fp8_no_autograd(
            tensor,
            scale,
            float8_dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None


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
    # TODO(before land): change this to _linear_mm_config, wanted to do that after
    # initial review
    _mm_config: LinearMMConfig
    __slots__ = ["_data", "_scale", "_orig_dtype", "_mm_config"]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        mm_config: Optional[LinearMMConfig],
        gemm_input_role: Optional[GemmInputRole] = GemmInputRole.X,
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
        self._mm_config = mm_config if mm_config is not None else LinearMMConfig()
        self._gemm_input_role = gemm_input_role

        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, mm_config={self._mm_config}\ngemm_input_role={self._gemm_input_role}\nas_orig_prec={self.to_original_precision()}"

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
        linear_mm_config: Optional[LinearMMConfig] = None,
        gemm_input_role: Optional[GemmInputRole] = GemmInputRole.X,
    ):
        """Converts a higher precision tensor to float8 in a differentiable way.

        Args:
            tensor: the tensor to convert
            scale: the scale to use to convert the tensor
            float8_dtype: the float8 dtype to use
            amax_buffer: a buffer to store the amax value in prior to conversion
            mm_config: Defines the configuration for the scaled_mm

        Returns:
            Float8Tensor: a float8 tensor
        """
        return ToFloat8ConstrFunc.apply(
            tensor,
            scale,
            float8_dtype,
            amax_buffer,
            linear_mm_config,
            gemm_input_role,
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
