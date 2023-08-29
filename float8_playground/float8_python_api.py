"""
This file defines the Python functions for float8 which expect inputs
of class `Float8Tensor`. This is a thin wrapper on top of the aten API
to simplify the product code.
"""

import torch
from float8_tensor import Float8Tensor
import float8_aten_api
import warnings
from typing import Optional

def layout_helper(tensor: torch.Tensor, row_major: bool) -> torch.Tensor:
    """ Cublas requires row_major @ column major tensors"""
    # TODO Figure out a better way of checking for correct layout
    if row_major:
        return tensor.contiguous()
    # We need it to be column major
    if tensor.is_contiguous():
        return tensor.t().contiguous().t()
    return tensor


def addmm_float8_unwrapped(
        input_bias: Optional[torch.Tensor],
        a_data: torch.Tensor, 
        a_scale: torch.Tensor, 
        b_data: torch.Tensor, 
        b_scale: torch.tensor, 
        output_amax: torch.Tensor, 
        output_scale: torch.Tensor, 
        output_dtype: torch.dtype) -> torch.Tensor:
    """ This is the unwrapped version of addmm_float8, which does not take in Float8Tensors 
        as inputs. This is used to standardize the logic between subclassed and non subclassed
        versions of the linear module.
    """
    temp_a = layout_helper(a_data, row_major=True)
    temp_b = layout_helper(b_data, row_major=False)

    a_inverse_scale = 1 / a_scale
    b_inverse_scale = 1 / b_scale
    output, updated_amax = torch._scaled_mm(
        temp_a,
        temp_b,
        bias=input_bias,
        out_dtype=output_dtype,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
    )
    output_amax.fill_(updated_amax)
    return output
    
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
    
    return addmm_float8_unwrapped(
        None, # input_bias
        a._data, a._scale,
        b._data, b._scale,
        output_amax, output_scale, output_dtype
    )

# See [Note] Usage of scales
def addmm_float8(
    input_bias: torch.Tensor,
    a: Float8Tensor,
    b: Float8Tensor,
    output_amax: torch.Tensor,
    output_scale: torch.Tensor,
    output_dtype: torch.dtype,
    emulate: bool = False,
) -> torch.Tensor:
    """
    Performs a matrix multiplication of two Float8Tensors `a` and `b`, adds an additional input tensor `input`.

    Args:
        input_bias: The addition term tensor, in fp32/fp16/bf16 format (no fp8 support).
        a: The first matrix multiplication term.
        b: The second matrix multiplication term.
        output_amax: The output tensor's amax, updated inplace in this function.
        output_scale: The output tensor's scale, precomputed.
        output_dtype: The output tensor's dtype.
        emulate: Whether to emulate the operation using fp32.

    Returns:
        torch.Tensor: The result of the matrix multiplication and addition.
    """
    assert input_bias.dtype in {torch.float16, torch.bfloat16}, "addmm_float8 only supports fp16/bf16 bias, you passed in {}".format(
        input_bias.dtype
    )

    if emulate:
        return torch.ops.aten.addmm_float8_emulated(
            input_bias,
            a._data, a._scale,
            b._data, b._scale,
            output_amax, output_scale, output_dtype)

    if input_bias.dtype == torch.float32:
        warnings.warn("addmm_float8 does not support fp32 bias, using fp16 instead")
        input_bias = input_bias.to(torch.float16)

    return addmm_float8_unwrapped(
        input_bias, # input_bias
        a._data, a._scale,
        b._data, b._scale,
        output_amax, output_scale, output_dtype
    )