"""
This file defines the Python functions for float8 which expect inputs
of class `Float8Tensor`. This is a thin wrapper on top of the aten API
to simplify the product code.
"""

import torch
from float8_tensor import Float8Tensor
import float8_aten_api

def layout_helper(tensor: torch.Tensor, row_major: bool) -> torch.Tensor:
    """ Cublas requires row_major @ column major tensors"""
    # TODO Figure out a better way of checking for correct layout
    if row_major:
        return tensor.contiguous()
    # We need it to be column major
    if tensor.is_contiguous():
        return tensor.t().contiguous().t()
    return tensor

# [Note] Usage of scales
# The meaning of scale in this library can be found in the definition of the Float8Tensor
# Cublas defines scale to always mean a multiplicative factor for the respective matrices
# For a,b going from fp8 -> fp32 we multiple by the inverse of the scale
# For output going from fp32 -> fp8 we multiply by the scale
def mm_float8(
    a: Float8Tensor,  # input 1
    b: Float8Tensor,  # input 2
    output_amax: torch.Tensor,  # output amax, updated inplace in this function
    output_scale: torch.Tensor,  # output scale, precomputed
    output_dtype: torch.dtype,  # output dtype
    emulate: bool = False,  # whether to emulate the operation using fp32
) -> torch.Tensor:
    if emulate:
        return torch.ops.aten.mm_float8_emulated(
            a._data, a._scale,
            b._data, b._scale,
            output_amax, output_scale, output_dtype)
    temp_a = layout_helper(a._data, row_major=True)
    temp_b = layout_helper(b._data, row_major=False)

    a_inverse_scale = 1 / a._scale
    b_inverse_scale = 1 / b._scale
    output, updated_amax = torch._scaled_mm(
        temp_a,
        temp_b,
        out_dtype=output_dtype,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
    )
    output_amax.fill_(updated_amax)
    return output


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
