from enum import Enum
import torch
from torch.utils._pytree import tree_map

from float8_utils import (
    tensor_to_amax,
    amax_to_scale,
)
from float8_python_api import (
    mm_float8,
)

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
        # References to relevant buffers which need to be updated for
        # the scaled_matmul computation. The caller is expected to optionally
        # set these after initializing the tensor subclass.
        self._fp8_buffer_refs = {
            'fp8_amax_y': None,
            # 'fp8_amax_dL_dX': None,
            # 'fp8_amax_dL_dW': None,
            'fw_amax_initialized': None,
            'bw_amax_initialized': None,
        }

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

        if func is aten.mm.default:
            if (
                isinstance(args[0], Float8Tensor) and 
                args[0]._data.dtype == torch.float8_e4m3fn and
                isinstance(args[1], Float8Tensor) and 
                args[1]._data.dtype == torch.float8_e4m3fn
            ):
                # forward, y = torch.mm(x, w)
                x_fp8, w_fp8 = args
                is_fw_amax_initialized = torch.any(w_fp8._fp8_buffer_refs['fw_amax_initialized'])
                fp8_amax_y = w_fp8._fp8_buffer_refs['fp8_amax_y']
                if not is_fw_amax_initialized:
                    # calculate reference amax of output
                    with torch.no_grad():
                        ref_result = torch.mm(x_fp8.to_original_precision(), w_fp8.to_original_precision())
                        fp8_amax_y.fill_(tensor_to_amax(ref_result))

                y_scale = amax_to_scale(fp8_amax_y, torch.float8_e4m3fn)
                res_bits = mm_float8(
                    x_fp8, w_fp8, fp8_amax_y, y_scale, 
                    torch.float8_e4m3fn)

                res = Float8Tensor(res_bits, y_scale, x_fp8._orig_dtype)
                return res
            # TODO(before land): implement the two backward matmuls, currently
            # they take the fallback which leads them to be executed in original
            # precision
            
        elif func is aten.view.default:
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
            new_tensor._fp8_buffer_refs = orig_tensor._fp8_buffer_refs
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
