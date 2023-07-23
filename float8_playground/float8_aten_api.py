"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch
from torch.library import Library

from float8_utils import (
    tensor_to_scale,
)


def mm_float8(m1, s1, m2, s2, s3, dtype3):
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.mm(m1_fp32, m2_fp32)
    # TODO(future): switch to delayed scaling
    s3.fill_(tensor_to_scale(m3_fp32, dtype3))
    m3_fp32_scaled = m3_fp32 * s3
    if dtype3 == torch.float8_e4m3fn:
        return m3_fp32_scaled.to(torch.float8_e4m3fn)
    else:
        return m3_fp32_scaled.to(torch.float8_e5m2)

def add_float8_e5m2(m1, s1, m2, s2, s3):
    # for now this is only implemented for e5m2 because we only care about
    # this for adding gradients
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    m1_float32 = m1.float() / s1
    m2_float32 = m2.float() / s2
    m3_float32 = m1_float32 + m2_float32
    s3_val = tensor_to_scale(m3_float32, torch.float8_e5m2)
    s3.fill_(s3_val)
    return (m3_float32 * s3).to(torch.float8_e5m2)

# TODO naming of these vars is weird
def addmm_float8(inp1, inp_s1, m1, s1, m2, s2, s3, dtype3):
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    inp1_fp32 = inp1.float() / inp_s1
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.addmm(inp1_fp32, m1_fp32, m2_fp32)
    # TODO(future): switch to delayed scaling
    s3.fill_(tensor_to_scale(m3_fp32, dtype3))
    m3_fp32_scaled = m3_fp32 * s3
    if dtype3 == torch.float8_e4m3fn:
        return m3_fp32_scaled.to(torch.float8_e4m3fn)
    else:
        return m3_fp32_scaled.to(torch.float8_e5m2)


#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

lib.define("mm_float8(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor s3, int dtype3) -> Tensor")
lib.impl("mm_float8", mm_float8, "CPU")
lib.impl("mm_float8", mm_float8, "CUDA")

lib.define("add_float8_e5m2(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor s3) -> Tensor")
lib.impl("add_float8_e5m2", add_float8_e5m2, "CPU")
lib.impl("add_float8_e5m2", add_float8_e5m2, "CUDA")

lib.define("addmm_float8(Tensor inp1, Tensor inp_s1, Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor s3, int dtype3) -> Tensor")
lib.impl("addmm_float8", addmm_float8, "CPU")
lib.impl("addmm_float8", addmm_float8, "CUDA")
