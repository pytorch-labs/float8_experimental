from enum import Enum
import torch
from torch.utils._pytree import tree_map

from float8_utils import E4M3, E5M2

aten = torch.ops.aten

class Float8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion between fp32 and fp8
    TODO(future): split into two for cleaner code
    """
    @staticmethod
    def forward(ctx, tensor, scale: float=None, flavor=E4M3):
        if isinstance(tensor, Float8Tensor):
            ctx.inp_is_float8 = True
            return torch.ops.aten.float8_to_float32(tensor._data, tensor._flavor) / tensor._scale
        else:
            ctx.inp_is_float8 = False
            tensor_scaled = tensor * scale
            bits_fp8 = torch.ops.aten.float32_to_float8(tensor_scaled, flavor)
            return Float8Tensor(bits_fp8, scale, flavor)

    @staticmethod
    def backward(ctx, g):
        # Assume that we always want to scale the gradients
        # back to full precision. We could do something else
        if isinstance(g, Float8Tensor) and not ctx.inp_is_float8:
            return g.to_float32(), None, None
        elif ctx.inp_is_float8:
            return Float8Tensor.from_float32(g), None, None
        else:
            return g, None, None


class Float8Tensor(torch.Tensor):
    """
    A Python-only FP8 tensor.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_flavor`: either E4M3 or E5M2

    The current purpose of this object is 99% to bundle raw data + fp8 metadata
    together for easy passing through PyTorch systems, and 1% to implement
    gradient addition (since that has to happen outside of user code).

    The addition operation is defined inline and uses a naive
    version of stateless scaling. This allows e5m2 gradients to be added.
    TODO(future): verify this is numericaly accurate, optionally replace
    with something better.

    It would probably make sense to also define fp8 path for data shuffling
    ops like cat, transpose, view, etc inline so we don't have to fall back
    to fp32 for them.
    """

    def __new__(cls, data, scale, flavor):
        # This is a non-differentiable constructor!
        assert not data.requires_grad
        # TODO(future): make bits8 easier to work with and switch to using it
        # assert data.dtype == torch.bits8
        assert scale.dtype == torch.float32
        assert scale.nelement() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=torch.float32,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._flavor = flavor

        return self

    def __repr__(self):
        return f"Float8Tensor(flavor={self._flavor}, scale={self._scale}, as_float32={self.to_float32()}"

    def to_float32(self):
        return Float8ConstrFunc.apply(self)

    @classmethod
    def from_float32(cls, tensor, scale, flavor):
        return Float8ConstrFunc.apply(tensor, scale, flavor)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Note: unlike many other subclasses, this subclass's only propagates
        # itself for addition (for gradient addition in backward). For all
        # other ops, it self-converts to fp32. The user/framework is
        # assumed to take care of defining where fp8 operations occur in the
        # forward pass and how scaling is calculated. In this example, that is
        # done by the `FP8Linear` class.
        # Vasiliy: the main reason I went with ^ is because NVIDIA is
        # doing stateful delayed scaling, and I don't know of a safe
        # way to enable that without either full program capture or punting it
        # to the user. This prototype takes the "punt it to the user" approach.
        # IMO for now let's just write out the scale stuff manually so we can
        # focus on other things, and revisit later if needed.

        # override addition so we can add e5m2 gradients
        if (
            func is aten.add.Tensor
            and isinstance(args[0], Float8Tensor)
            and isinstance(args[1], Float8Tensor)
        ):
            x1_fp8, x2_fp8 = args[0], args[1]
            assert x1_fp8._flavor == E5M2 and x2_fp8._flavor == E5M2
            # naive scale calculation: max of incoming two scales
            x3_scale = torch.max(x1_fp8._scale, x2_fp8._scale)
            res_bits = torch.ops.aten.add_float8_e5m2(
                x1_fp8._data, x1_fp8._scale,
                x2_fp8._data, x2_fp8._scale,
                x3_scale)
            res = Float8Tensor(res_bits, x3_scale, x1_fp8._flavor)
            return res

        # for all other ops, fall back to fp32
        # TODO(future): add support for fp16/bf16

        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.to_float32()
            return t

        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
