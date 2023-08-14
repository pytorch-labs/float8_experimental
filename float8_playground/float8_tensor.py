from enum import Enum
import torch
from torch.utils._pytree import tree_map

from float8_utils import tensor_to_amax

aten = torch.ops.aten


class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8
    """
    @staticmethod
    def forward(
        ctx, 
        tensor, 
        scale: float=None, 
        float8_dtype=torch.float8_e4m3fn, 
        amax_buffer=None,
    ):
        # In TransformerEngine, the casts to float8 are fused with calculating
        # the new amax value. In this codebase, the eager mode code for those 
        # two things is colocated in this function. We expect PT2.0 to fuse it
        # for us.
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        tensor_scaled = tensor * scale
        bits_fp8 = tensor_scaled.to(float8_dtype)
        return Float8Tensor(bits_fp8, scale, tensor.dtype)

    @staticmethod
    def backward(ctx, g):
        if isinstance(g, Float8Tensor):
            return g.to_original_precision(), None, None, None
        else:
            return g, None, None, None


class FromFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion from fp8
    """
    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return Float8Tensor.to_float8(g), None, None


class Float8Tensor(torch.Tensor):
    """
    A Python-only Float8 tensor subclass.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.

    Intended usage of this abstraction:
    1. to bundle raw data + fp8 metadata together for easy passing through 
       Python PyTorch systems.
    2. Float8-aware user code can use the private fields on these tensors
       to call into float8 operations. 
    3. Float8-agnostic user code can use these tensors as is - they will
       convert to original precision in `__torch_dispatch__`.
    """

    def __new__(cls, data, scale, orig_dtype):
        # This is a non-differentiable constructor!
        assert not data.requires_grad
        assert scale.nelement() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype

        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, as_orig_prec={self.to_original_precision()}"

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @classmethod
    def to_float8(cls, tensor, scale, float8_dtype, amax_buffer=None):
        return ToFloat8ConstrFunc.apply(tensor, scale, float8_dtype, amax_buffer)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is aten.view.default:
            orig_tensor, view_args = args
            new_tensor = Float8Tensor(
                orig_tensor._data.view(*view_args), orig_tensor._scale, 
                orig_tensor._orig_dtype)
            return new_tensor
        elif func is aten.t.default:
            orig_tensor, = args
            new_tensor = Float8Tensor(
                orig_tensor._data.t(), orig_tensor._scale,
                orig_tensor._orig_dtype)
            return new_tensor

        # for all ops that get here, fall back to original precision
        def unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.to_original_precision()
            return t

        args = tree_map(unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
