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
        orig_shape = x_fp8._data.shape
        x_fp8_data_reshaped = x_fp8._data.reshape(-1, orig_shape[-1])
        if b_fp8 is not None:
            res_bits = torch.ops.aten.addmm_float8(
                b_fp8._data, b_fp8._scale,
                x_fp8_data_reshaped, x_fp8._scale,
                w_fp8._data.t(), w_fp8._scale,
                fp8_s_out, torch.float8_e4m3fn)
        else:
            res_bits = torch.ops.aten.mm_float8(
                x_fp8_data_reshaped, x_fp8._scale,
                w_fp8._data.t(), w_fp8._scale,
                fp8_s_out, torch.float8_e4m3fn)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])

        res = Float8Tensor(res_bits, fp8_s_out, x_fp8._orig_dtype)
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
                fp8_s_dL_dY, go.dtype)
        else:
            go_fp8 = go

        go_fp8_orig_shape = go_fp8._data.shape
        go_fp8_data_reshaped = go_fp8._data.reshape(-1, go_fp8_orig_shape[-1])

        dL_dX_bits = torch.ops.aten.mm_float8(
            go_fp8_data_reshaped, go_fp8._scale,
            w_fp8._data, w_fp8._scale,
            fp8_s_dL_dX, torch.float8_e5m2)
        dL_dX_bits = dL_dX_bits.reshape(*go_fp8_orig_shape[:-1], dL_dX_bits.shape[-1])
        dL_dX_fp8 = Float8Tensor(dL_dX_bits, fp8_s_dL_dX, go_fp8._orig_dtype)

        x_fp8_orig_shape = x_fp8._data.shape
        x_fp8_data_reshaped = x_fp8._data.reshape(-1, x_fp8_orig_shape[-1])

        dL_dW_bits = torch.ops.aten.mm_float8(
            x_fp8_data_reshaped.t(), x_fp8._scale,
            go_fp8_data_reshaped, go_fp8._scale,
            fp8_s_dL_dW, torch.float8_e5m2).t()
        dL_dW_fp8 = Float8Tensor(dL_dW_bits, fp8_s_dL_dW, go_fp8._orig_dtype)

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
            # Duplicate the autocast logic for F.linear, so that the output
            # of our module has the right original precision
            if torch.is_autocast_enabled():
                # For now, hardcode to GPU's autocast dtype
                # if we need CPU support in the future, we can add it
                x = x.to(torch.get_autocast_gpu_dtype())

            # TODO(future): switch to delayed scaling
            self.fp8_s_in.fill_(tensor_to_scale(x, torch.float8_e4m3fn))
            x_fp8 = Float8Tensor.to_float8(x, self.fp8_s_in, torch.float8_e4m3fn)
        else:
            x_fp8 = x

        # TODO(future): switch to delayed scaling
        self.fp8_s_weight.fill_(tensor_to_scale(self.weight, torch.float8_e4m3fn))
        w_fp8 = Float8Tensor.to_float8(self.weight, self.fp8_s_weight, torch.float8_e4m3fn)
        maybe_b_fp8 = None
        if self.bias is not None:
            self.fp8_s_bias.fill_(tensor_to_scale(self.bias, torch.float8_e4m3fn))
            maybe_b_fp8 = Float8Tensor.to_float8(self.bias, self.fp8_s_bias, torch.float8_e4m3fn)

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


def swap_linear_with_float8_linear(model):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if isinstance(child, torch.nn.Linear):
            new_child = Float8Linear.from_float(child)
            setattr(model, name, new_child)
        else:
            swap_linear_with_float8_linear(child)
