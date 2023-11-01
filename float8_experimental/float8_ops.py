from typing import Any, Dict

import torch
from float8_experimental.float8_python_api import mm_float8_unwrapped
from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import is_row_major

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


@implements([aten.mm.default])
def float8_mm(aten_op, args, kwargs=None):
    assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
    a = args[0]
    b = args[1]
    a_data = a._data
    a_scale = a._scale
    b_data = b._data

    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    b_scale = b._scale
    output_dtype = a._orig_dtype
    if a._emulate:
        assert a._emulate == b._emulate
        return torch.ops.aten.mm_float8_emulated(
            a._data, a._scale, b._data, b._scale, output_dtype
        )[0]
    tensor_out, amax = mm_float8_unwrapped(
        a_data, a_scale, b_data, b_scale, output_dtype, output_scale=None
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
    assert (
        kwargs["dtype"] == torch.float16
    ), "Only support floating point conversion for autocast w/ Float8Tensor"
    return Float8Tensor(
        args[0]._data, args[0]._scale, kwargs["dtype"], args[0]._emulate
    )
