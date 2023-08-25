"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch
from torch.library import Library

from float8_utils import (
    tensor_to_amax,
    to_fp8_saturated,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)

def mm_float8_emulated(
    m1,  # input 1 data
    s1,  # input 1 scale
    m2,  # input 2 data
    s2,  # input 2 scale
    amax3,  # amax buffer of output, updated inplace in this function
    s3,  # output scale
    dtype3,  # output dtype
):
    # naive implementation: dq -> op -> q
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.mm(m1_fp32, m2_fp32)

    amax3.fill_(tensor_to_amax(m3_fp32))
    return m3_fp32.to(dtype3)


# TODO naming of these vars is weird
def addmm_float8_emulated(
    inp1,  # bias (in fp32/fp16/bf16, no fp8 support)
    m1,  # input 1 data
    s1,  # input 1 scale
    m2,  # input 2 data
    s2,  # input 2 scale
    amax3,  # amax buffer of output, updated inplace in this function
    s3,  # output scale
    dtype3,  # output dtype
):
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.addmm(inp1, m1_fp32, m2_fp32)

    amax3.fill_(tensor_to_amax(m3_fp32))
    return m3_fp32.to(dtype3)


#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

lib.define("mm_float8_emulated(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor amax3, Tensor s3, ScalarType dtype3) -> Tensor")
lib.impl("mm_float8_emulated", mm_float8_emulated, "CPU")
lib.impl("mm_float8_emulated", mm_float8_emulated, "CUDA")

lib.define("addmm_float8_emulated(Tensor inp1, Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor amax3, Tensor s3, ScalarType dtype3) -> Tensor")
lib.impl("addmm_float8_emulated", addmm_float8_emulated, "CPU")
lib.impl("addmm_float8_emulated", addmm_float8_emulated, "CUDA")
