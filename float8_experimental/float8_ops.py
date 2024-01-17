# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional

import torch

from float8_experimental.float8_python_api import addmm_float8_unwrapped
from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import is_row_major
from torch.utils._pytree import tree_map

aten = torch.ops.aten
FLOAT8_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the float8 op table"""

    def decorator(func):
        for op in aten_ops:
            FLOAT8_OPS_TABLE[op] = func
        return func

    return decorator


@implements(
    [
        aten.view.default,
        aten._unsafe_view.default,
        aten.t.default,
        aten.as_strided.default,
        aten.clone.default,
        aten.detach.default,
    ]
)
def float8_desugar_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return Float8Tensor(new_data, args[0]._scale, args[0]._orig_dtype, args[0]._emulate)


@implements([aten.sum.dim_IntList])
def float8_cast_up_op(aten_op, args, kwargs=None):
    """Be careful with this function, this is a "fallback" op that
    casts the output of the op to the original precision. And performs the op.

    We currently need this to support the backward for admmm bias.
    "addmm" -> out
    "hp_gradBias" <-"sum" <- "identity" <- gradOut <- "hp_gradOut"
    """

    def unwrap(x):
        if isinstance(x, Float8Tensor):
            return x.to_original_precision()
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


def preprocess_addmm(a: Float8Tensor, b: Float8Tensor):
    a_data = a._data
    a_scale = a._scale
    b_data = b._data

    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    b_scale = b._scale
    return a_data, a_scale, b_data, b_scale


def float8_mm_helper(a: Float8Tensor, b: Float8Tensor) -> torch.Tensor:
    """This is a helper function for float8_mm
    Args:
        a: The first matrix multiplication term.
        b: The second matrix multiplication term.
    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    a_data, a_scale, b_data, b_scale = preprocess_addmm(a, b)
    output_dtype = a._orig_dtype
    if a._emulate:
        assert a._emulate == b._emulate
        return torch.ops.aten.mm_float8_emulated(
            a._data, a._scale, b._data, b._scale, output_dtype
        )[0]
    tensor_out, amax = addmm_float8_unwrapped(
        a_data, a_scale, b_data, b_scale, output_dtype, output_scale=None, bias=None
    )
    return tensor_out


@implements([aten.mm.default])
def float8_mm(aten_op, args, kwargs=None):
    assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
    a = args[0]
    b = args[1]
    return float8_mm_helper(a, b)
    a_data, a_scale, b_data, b_scale = preprocess_addmm(a, b)
    output_dtype = a._orig_dtype
    if a._emulate:
        assert a._emulate == b._emulate
        return torch.ops.aten.mm_float8_emulated(
            a._data, a._scale, b._data, b._scale, output_dtype
        )[0]
    tensor_out, amax = addmm_float8_unwrapped(
        a_data, a_scale, b_data, b_scale, output_dtype, output_scale=None, bias=None
    )
    return tensor_out


@implements([aten.addmm.default])
def float8_addmm(aten_op, args, kwargs=None):
    assert (
        isinstance(args[0], torch.Tensor)
        and isinstance(args[1], Float8Tensor)
        and isinstance(args[2], Float8Tensor)
    )
    bias = args[0]
    a = args[1]
    b = args[2]
    a_data, a_scale, b_data, b_scale = preprocess_addmm(a, b)
    output_dtype = a._orig_dtype
    assert bias.dtype == output_dtype, "bias dtype must match output dtype"
    if a._emulate:
        assert a._emulate == b._emulate
        out = torch.ops.aten.mm_float8_emulated(
            a._data, a._scale, b._data, b._scale, output_dtype
        )[0]
        return out + bias
    tensor_out, amax = addmm_float8_unwrapped(
        a_data, a_scale, b_data, b_scale, output_dtype, output_scale=None, bias=bias
    )
    return tensor_out


@implements([aten.is_same_size.default])
def float8_is_same_size(aten_op, args, kwargs=None):
    return args[0].shape == args[1].shape


@implements([aten._to_copy.default])
def autocast_to_copy(aten_op, args, kwargs=None):
    """This gets called when running matmul under autocast
    when the input is a Float8Tensor, presenting as a fp32
    tensor.
    """
    assert isinstance(args[0], Float8Tensor)
    assert (
        len(kwargs) == 1 and "dtype" in kwargs
    ), "Only support dtype kwarg for autocast"
    assert kwargs["dtype"] in {
        torch.float16,
        torch.bfloat16,
    }, "Only support floating point conversion for autocast w/ Float8Tensor"
    return Float8Tensor(
        args[0]._data, args[0]._scale, kwargs["dtype"], args[0]._emulate
    )


class float8_linear(torch.autograd.Function):
    """Custom autograd function for computing torch.nn.Linear on Float8Tensor.

    This is needed for a couple reasons, we want to have fine grained control over the
    recomputation of casted values for backward.
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8: torch.Tensor,
        original_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_amax_buffer: Optional[torch.Tensor],
        emulate: bool,
        recompute_float8_weight: bool,
    ):
        w_fp8 = Float8Tensor.to_float8(
            original_weight,
            weight_scale,
            torch.float8_e4m3fn,
            weight_amax_buffer,
            emulate=emulate,
        )
        if recompute_float8_weight:
            # This should be set to True when using traditional fsdp to avoid saving
            # saving the unsharded weight for backwards
            ctx.save_for_backward(
                x_fp8, original_weight, weight_scale, weight_amax_buffer
            )
        else:
            # Does this interact properly with activation checkpointing?
            ctx.save_for_backward(x_fp8, w_fp8)

        ctx.recompute_float8_weight = recompute_float8_weight
        ctx.emulate = emulate
        x_fp8_reshaped = x_fp8.reshape(-1, x_fp8.size(-1))

        w_fp8_t = w_fp8.t()

        res_bits = float8_mm_helper(x_fp8_reshaped, w_fp8_t)

        res_bits = res_bits.reshape(*x_fp8.shape[:-1], res_bits.size(-1))
        return res_bits

    @staticmethod
    def backward(ctx, go_fp8: torch.Tensor):
        if ctx.recompute_float8_weight:
            x_fp8, original_weight, weight_scale, weight_amax_buffer = ctx.saved_tensors
            w_fp8 = Float8Tensor.to_float8(
                original_weight,
                weight_scale,
                torch.float8_e4m3fn,
                weight_amax_buffer,
                emulate=ctx.emulate,
            )
        else:
            x_fp8, w_fp8 = ctx.saved_tensors

        # calculate dL/dX
        go_fp8_reshaped = go_fp8.reshape(-1, go_fp8.size(-1))
        w_fp8_t_c_t = w_fp8.t().contiguous().t()
        dL_dX = float8_mm_helper(go_fp8_reshaped, w_fp8_t_c_t)
        dL_dX = dL_dX.reshape(*go_fp8.shape[:-1], dL_dX.size(-1))

        # calculate dL/dW
        x_fp8_reshaped_t_c = x_fp8.reshape(-1, x_fp8.size(-1)).t().contiguous()
        go_fp8_reshaped_t_c_t = go_fp8_reshaped.t().contiguous().t()

        dL_dW = float8_mm_helper(x_fp8_reshaped_t_c, go_fp8_reshaped_t_c_t)
        dL_dW = dL_dW.t()

        empty_grads = None, None, None, None, None, None, None, None, None
        return dL_dX, dL_dW, *empty_grads
