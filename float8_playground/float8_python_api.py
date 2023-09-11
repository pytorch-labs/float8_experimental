"""
This file defines the Python functions for float8 which expect inputs
of class `Float8Tensor`. This is a thin wrapper on top of the aten API
to simplify the product code.
"""

import torch
from float8_tensor import Float8Tensor
import float8_aten_api
import warnings
from typing import Optional, Tuple

def layout_helper(tensor: torch.Tensor, row_major: bool) -> torch.Tensor:
    """ Cublas requires row_major @ column major tensors"""
    # TODO Figure out a better way of checking for correct layout
    if row_major:
        return tensor.contiguous()
    # We need it to be column major
    return tensor.t().contiguous().t()


def mm_float8_unwrapped(
        a_data: torch.Tensor,
        a_scale: torch.Tensor,
        b_data: torch.Tensor,
        b_scale: torch.tensor,
        output_scale: torch.Tensor,
        output_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """ This is the unwrapped version of mm_float8, which does not take in Float8Tensors
        as inputs. This is used to standardize the logic between subclassed and non subclassed
        versions of the linear module.
    """
    temp_a = layout_helper(a_data, row_major=True)
    temp_b = layout_helper(b_data, row_major=False)

    a_inverse_scale = 1 / a_scale
    b_inverse_scale = 1 / b_scale
    output, output_amax = torch._scaled_mm(
        temp_a,
        temp_b,
        bias=None,
        out_dtype=output_dtype,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
    )
    return output, output_amax

# [Note] Usage of scales
# The meaning of scale in this library can be found in the definition of the Float8Tensor
# Cublas defines scale to always mean a multiplicative factor for the respective matrices
# For a,b going from fp8 -> fp32 we multiple by the inverse of the scale
# For output going from fp32 -> fp8 we multiply by the scale
def mm_float8(
    a: Float8Tensor,  # input 1
    b: Float8Tensor,  # input 2
    output_scale: torch.Tensor,  # output scale, precomputed
    output_dtype: torch.dtype,  # output dtype
    emulate: bool = False,  # whether to emulate the operation using fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a matrix multiplication of two Float8Tensors `a` and `b`.

    Args:
        a: The first matrix multiplication term.
        b: The second matrix multiplication term.
        output_scale: The output tensor's scale, precomputed.
        output_dtype: The output tensor's dtype.
        emulate: Whether to emulate the operation using fp32.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    if emulate:
        return torch.ops.aten.mm_float8_emulated(
            a._data, a._scale,
            b._data, b._scale,
            output_scale, output_dtype)

    return mm_float8_unwrapped(
        a._data, a._scale,
        b._data, b._scale,
        output_scale, output_dtype
    )
