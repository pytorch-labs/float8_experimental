"""
A simple manual UEX for a float8 version of `torch.nn.Linear`.

Note: this UEX is not intended for real usage. It merely demonstrates
an example of how features such as casting to and from float8 as well
as stateful scaling can be implemented. For now, we expect framework
owners to implement their own UEX.
"""

import torch

import float8_aten_api

from float8_utils import tensor_to_scale
from float8_tensor import Float8Tensor

class float8_linear(torch.autograd.Function):
    """
    Like F.linear, but with X, W, and Y in float8
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8,
        w_fp8,
        b_fp8,
        fp8_s_out,
        fp8_s_dL_dX,
        fp8_s_dL_dW,
        fp8_s_dL_dY,
    ):
        ctx.save_for_backward(
            x_fp8, w_fp8, b_fp8, fp8_s_dL_dX, fp8_s_dL_dW, fp8_s_dL_dY)
        if b_fp8 is not None:
            res_bits = torch.ops.aten.addmm_float8(
                b_fp8._data, b_fp8._scale,
                x_fp8._data, x_fp8._scale,
                w_fp8._data.t(), w_fp8._scale,
                fp8_s_out, torch.float8_e4m3fn)
        else:
            res_bits = torch.ops.aten.mm_float8(
                x_fp8._data, x_fp8._scale,
                w_fp8._data.t(), w_fp8._scale,
                fp8_s_out, torch.float8_e4m3fn)

        res = Float8Tensor(res_bits, fp8_s_out)
        # scale update would also happen here, for now no-op
        return res

    @staticmethod
    def backward(ctx, go):
        x_fp8, w_fp8, b_fp8, fp8_s_dL_dX, fp8_s_dL_dW, fp8_s_dL_dY = \
            ctx.saved_tensors

        if not isinstance(go, Float8Tensor):
            # TODO(future): switch to delayed scaling
            fp8_s_dL_dY.fill_(tensor_to_scale(go, torch.float8_e5m2))
            go_fp8 = Float8Tensor(
                (go * fp8_s_dL_dY).to(torch.float8_e5m2),
                fp8_s_dL_dY)
        else:
            go_fp8 = go

        dL_dX_bits = torch.ops.aten.mm_float8(
            go_fp8._data, go_fp8._scale,
            w_fp8._data, w_fp8._scale,
            fp8_s_dL_dX, torch.float8_e5m2)
        dL_dX_fp8 = Float8Tensor(dL_dX_bits, fp8_s_dL_dX)

        dL_dW_bits = torch.ops.aten.mm_float8(
            x_fp8._data.t(), x_fp8._scale,
            go_fp8._data, go_fp8._scale,
            fp8_s_dL_dW, torch.float8_e5m2).t()
        dL_dW_fp8 = Float8Tensor(dL_dW_bits, fp8_s_dL_dW)

        # scale update would also happen here, for now no-op
        if b_fp8 is not None:
            return dL_dX_fp8, dL_dW_fp8, go_fp8, None, None, None, None
        else:
            return dL_dX_fp8, dL_dW_fp8, None, None, None, None, None


class Float8Linear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # While this module currently implements just-in-time scaling,
        # the scales are stored in buffers as a placeholder for delayed
        # scaling such as the mechanism described in
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8,
        # or PTQ calibration.
        self.register_buffer('fp8_s_in', torch.tensor(1.0))
        self.register_buffer('fp8_s_weight', torch.tensor(1.0))
        self.register_buffer('fp8_s_bias', torch.tensor(1.0))
        self.register_buffer('fp8_s_out', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dX', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dW', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dY', torch.tensor(1.0))

    def forward(self, x):
        if not isinstance(x, Float8Tensor):
            # TODO(future): switch to delayed scaling
            self.fp8_s_in.fill_(tensor_to_scale(x, torch.float8_e4m3fn))
            x_fp8 = Float8Tensor.from_float32(x, self.fp8_s_in, torch.float8_e4m3fn)
        else:
            x_fp8 = x

        # TODO(future): switch to delayed scaling
        self.fp8_s_weight.fill_(tensor_to_scale(self.weight, torch.float8_e4m3fn))
        w_fp8 = Float8Tensor.from_float32(self.weight, self.fp8_s_weight, torch.float8_e4m3fn)
        maybe_b_fp8 = None
        if self.bias is not None:
            self.fp8_s_bias.fill_(tensor_to_scale(self.bias, torch.float8_e4m3fn))
            maybe_b_fp8 = Float8Tensor.from_float32(self.bias, self.fp8_s_bias, torch.float8_e4m3fn)

        y_fp8 = float8_linear.apply(
            x_fp8, w_fp8, maybe_b_fp8, self.fp8_s_out, self.fp8_s_dL_dX,
            self.fp8_s_dL_dW, self.fp8_s_dL_dY)

        # For now, hardcode returning Float8Tensor (propagate as much as we can).
        # This can be changed to return a different dtype, if needed.
        return y_fp8

    @classmethod
    def from_float(cls, mod):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear
        """
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod
