# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch

from float8_experimental.float8_utils import tensor_to_amax, to_fp8_saturated
from torch.library import Library


def mm_float8_emulated(
    m1,  # input 1 data
    s1,  # input 1 scale
    m2,  # input 2 data
    s2,  # input 2 scale
    dtype3,  # output dtype
):
    # naive implementation: dq -> op -> q
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.mm(m1_fp32, m2_fp32)

    return m3_fp32.to(dtype3), tensor_to_amax(m3_fp32)


def cast_to_float8_tensor(
    x: torch.Tensor, x_scale: torch.Tensor, float8_dtype: torch.dtype, emulate: bool
) -> "Float8Tensor":
    """Convert a tensor to float8 without autograd
    This is used in multiple places in the codebase to convert a tensor to float8

    This function will calculate the scale, do the scaling, and then convert to a Float8Tensor
    Args:
        x: the tensor to convert
        x_scale: the scale to use to convert the tensor
        float8_dtype: the float8 dtype to use
        emulate: whether to emulate the matmuls in fp32
    """
    # lazy import to avoid circular dependency
    from float8_experimental.float8_tensor import Float8Tensor
    x_scaled = x * x_scale
    bits_fp8 = to_fp8_saturated(x_scaled, float8_dtype)
    return Float8Tensor(bits_fp8, x_scale, x.dtype, emulate=emulate)


#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

lib.define(
    "mm_float8_emulated(Tensor m1, Tensor s1, Tensor m2, Tensor s2, ScalarType dtype3) -> (Tensor, Tensor)"
)
lib.impl("mm_float8_emulated", mm_float8_emulated, "CPU")
lib.impl("mm_float8_emulated", mm_float8_emulated, "CUDA")

lib.define(
    "cast_to_float8_tensor(Tensor x, Tensor x_scale, ScalarType float8_dtype, bool emulate) -> Tensor"
)
lib.impl("cast_to_float8_tensor", cast_to_float8_tensor, "CPU")
lib.impl("cast_to_float8_tensor", cast_to_float8_tensor, "CUDA")


@torch.library.impl(lib, "mm_float8_emulated", "Meta")
def _mm_float8_emulated_meta(m1, s1, m2, s2, dtype3):
    out = torch.mm(m1.float(), m2.float()).to(dtype3)
    return out, torch.empty(1, device="meta")

@torch.library.impl(lib, "cast_to_float8_tensor", "Meta")
def _cast_to_float8_tensor_meta(x, x_scale, float8_dtype, emulate):
    return torch.empty_like(x, device="meta")
