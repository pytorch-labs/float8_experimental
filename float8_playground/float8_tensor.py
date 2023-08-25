from enum import Enum
import torch
from torch.utils._pytree import tree_map, tree_map_only

from float8_utils import (
    tensor_to_amax,
    to_fp8_saturated,
)

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
        bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
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
        return _to_float8(g), None, None


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

    def __new__(cls, data: torch.Tensor, scale: torch.Tensor, orig_dtype: torch.dtype):
        import pdb; pdb.set_trace()
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
        return self

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, orig_dtype: torch.dtype):
        import pdb; pdb.set_trace()
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, as_orig_prec={self.to_original_precision()}"

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @classmethod
    def to_float8(cls, tensor, scale, float8_dtype, amax_buffer=None):
        out = ToFloat8ConstrFunc.apply(tensor, scale, float8_dtype, amax_buffer)
        import pdb; pdb.set_trace()
        return out

    def __tensor_flatten__(self):
        import pdb; pdb.set_trace()
        return ["_data", "_scale"], self._orig_dtype
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype

    @staticmethod
    def __tensor_unflatten__(inner_tensors, orig_dtype):
        assert type(orig_dtype) == torch.dtype
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        return Float8Tensor(data, scale, orig_dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Some ops like mm can remove the subclass-ness of Float8Tensor (and return a plain torch.Tensor)
        # but views should always return another Float8Tensor.
        if func.is_view:
            unwrapped_args = tree_map_only(Float8Tensor, lambda x: x._data, args)
            unwrapped_view = func(*unwrapped_args, **kwargs)
            new_tensor = Float8Tensor(
                unwrapped_view, args[0]._scale,
                args[0]._orig_dtype)
            # TODO: need to use return_and_correct_aliasing once https://github.com/pytorch/pytorch/pull/107915 lands.
            return new_tensor


        # for all ops that get here, fall back to original precision
        def unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.to_original_precision()
            return t

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        # Re-run the function as normal
        # (don't manually call __torch_dispatch__, since this won't give any modes
        #  a chance to run)
        out = func(*args, **kwargs)
        #out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

def _to_float8(tensor, scale, float8_dtype, amax_buffer=None):
    return Float8Tensor.to_float8(tensor, scale, float8_dtype, amax_buffer)

torch._dynamo.allow_in_graph(_to_float8)
