"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch
from torch.library import Library

from float8_utils import (
    float32_to_float8,
    float8_to_float32,
    E4M3,
    E5M2,
    tensor_to_scale,
)


def mm_float8(m1, s1, flavor1, m2, s2, flavor2, s3, flavor3):
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    m1_fp32 = float8_to_float32(m1, flavor1) / s1
    m2_fp32 = float8_to_float32(m2, flavor2) / s2
    m3_fp32 = torch.mm(m1_fp32, m2_fp32)
    # TODO(future): switch to delayed scaling
    s3.fill_(tensor_to_scale(m3_fp32, flavor3))
    m3_fp32_scaled = m3_fp32 * s3
    return float32_to_float8(m3_fp32_scaled, flavor3)

def add_float8_e5m2(m1, s1, m2, s2, s3):
    # for now this is only implemented for e5m2 because we only care about
    # this for adding gradients
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    # TODO(future): make this more accurate, accuracy is pretty low,
    # can probably just calculate s3 dynamically since this is an edge case
    # unlikely to affect e2e performance
    m1_float32 = float8_to_float32(m1, E5M2) / s1
    m2_float32 = float8_to_float32(m2, E5M2) / s2
    m3_float32 = m1_float32 + m2_float32
    return float32_to_float8(m3_float32 * s3, E5M2)

#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

# For now register on CPU,
# TODO(future) add GPU and test there
lib.define("float32_to_float8(Tensor t, int flavor) -> Tensor")
lib.impl("float32_to_float8", float32_to_float8, "CPU")

lib.define("float8_to_float32(Tensor t, int flavor) -> Tensor")
lib.impl("float8_to_float32", float8_to_float32, "CPU")

lib.define("mm_float8(Tensor m1, Tensor s1, int flavor1, Tensor m2, Tensor s2, int flavor2, Tensor s3, int flavor3) -> Tensor")
lib.impl("mm_float8", mm_float8, "CPU")

lib.define("add_float8_e5m2(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor s3) -> Tensor")
lib.impl("add_float8_e5m2", add_float8_e5m2, "CPU")
