from enum import Enum
import torch
from torch.utils._pytree import tree_map

aten = torch.ops.aten


class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8
    """
    @staticmethod
    def forward(ctx, tensor, scale: float=None, dtype=torch.float8_e4m3fn):
        tensor_scaled = tensor * scale
        bits_fp8 = tensor_scaled.to(dtype)
        return Float8Tensor(bits_fp8, scale, tensor.dtype)

    @staticmethod
    def backward(ctx, g):
        if isinstance(g, Float8Tensor):
            return g.to_original_precision(), None, None
        else:
            return g, None, None


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
    A Python-only FP8 tensor.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.

    The current purpose of this object is 99% to bundle raw data + fp8 metadata
    together for easy passing through PyTorch systems, and 1% to implement
    gradient addition (since that has to happen outside of user code).

    The addition operation is defined inline and uses a naive
    version of stateless scaling. This allows e5m2 gradients to be added.
    TODO(future): verify this is numericaly accurate, optionally replace
    with something better.
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
    def to_float8(cls, tensor, scale, dtype):
        return ToFloat8ConstrFunc.apply(tensor, scale, dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Note: unlike many other subclasses, this subclass's only propagates
        # itself for addition (for gradient addition in backward). For all
        # other ops, it self-converts to original precision. 

        # override addition so we can add e5m2 gradients
        if (
            func is aten.add.Tensor
            and isinstance(args[0], Float8Tensor)
            and isinstance(args[1], Float8Tensor)
        ):
            x1_fp8, x2_fp8 = args[0], args[1]
            assert x1_fp8._data.dtype == torch.float8_e5m2 and x2_fp8._data.dtype == torch.float8_e5m2
            # scale will be filled in by the kernel, not using delayed scaling
            x3_scale = torch.empty(1, device=x1_fp8.device)
            res_bits = torch.ops.aten.add_float8_e5m2(
                x1_fp8._data, x1_fp8._scale,
                x2_fp8._data, x2_fp8._scale,
                x3_scale)
            # TODO(future): handle type promotion if orig dtypes do not match
            # for now, just take the first one
            res = Float8Tensor(res_bits, x3_scale, x1_fp8._orig_dtype)
            return res

        # for all other ops, fall back to original precision
        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.to_original_precision()
            return t

        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
