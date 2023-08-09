"""
This file defines the Python functions for float8 which expect inputs
of class `Float8Tensor`. This is a thin wrapper on top of the aten API
to simplify the product code.
"""

import torch
import float8_aten_api

def mm_float8(
    x1,  # input 1
    x2,  # input 2
    amax3,  # output amax, updated inplace in this function
    s3,  # output scale, precomputed
    dtype3,  # output dtype
):
    return torch.ops.aten.mm_float8(
        x1._data, x1._scale,
        x2._data, x2._scale,
        amax3, s3, dtype3)

def addmm_float8(
    inp1,  # addition term (in fp32/fp16/bf16, no fp8 support)
    x1,  # first mm term
    x2,  # second mm term
    amax3,  # output aax, updated inplace in this function
    s3,  # output scale, precomputed
    dtype3,  # output dtype
):
    return torch.ops.aten.addmm_float8(
        inp1,
        x1._data, x1._scale,
        x2._data, x2._scale,
        amax3, s3, dtype3)
