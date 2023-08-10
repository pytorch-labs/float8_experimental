"""
A simple manual UEX for a float8 version of `torch.nn.Linear`.

Note: this UEX is not intended for real usage. It merely demonstrates
an example of how features such as casting to and from float8 as well
as stateful scaling can be implemented. For now, we expect framework
owners to implement their own UEX.
"""

import torch
import torch.nn.functional as F

from float8_python_api import (
    mm_float8,
    addmm_float8,
)

from float8_utils import (
    tensor_to_amax,
    amax_to_scale,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)
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
        b,
        fp8_amax_y,
        fp8_amax_dL_dX,
        fp8_amax_dL_dW,
        fw_amax_initialized,
        bw_amax_initialized,
    ):
        ctx.save_for_backward(
            x_fp8, w_fp8, b, fp8_amax_dL_dX, fp8_amax_dL_dW,
            bw_amax_initialized)
        orig_shape = x_fp8._data.shape
        x_fp8_reshaped = x_fp8.reshape(-1, orig_shape[-1])
        is_fw_amax_initialized = torch.any(fw_amax_initialized)

        if b is not None:
            if not is_fw_amax_initialized:
                # calculate reference amax of output
                with torch.no_grad():
                    ref_result = torch.addmm(b, x_fp8_reshaped, w_fp8.t())
                    fp8_amax_y.fill_(tensor_to_amax(ref_result))

            y_scale = amax_to_scale(fp8_amax_y, torch.float8_e4m3fn)
            res_bits = addmm_float8(
                b, x_fp8_reshaped, w_fp8.t(), fp8_amax_y, y_scale, 
                torch.float8_e4m3fn)
        else:
            if not is_fw_amax_initialized:
                # calculate reference amax of output
                with torch.no_grad():
                    ref_result = torch.mm(x_fp8_reshaped, w_fp8.t())
                    fp8_amax_y.fill_(tensor_to_amax(ref_result))

            y_scale = amax_to_scale(fp8_amax_y, torch.float8_e4m3fn)
            res_bits = mm_float8(
                x_fp8_reshaped, w_fp8.t(), fp8_amax_y, y_scale, 
                torch.float8_e4m3fn)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])

        res = Float8Tensor(res_bits, y_scale, x_fp8._orig_dtype)
        # scale update would also happen here, for now no-op
        return res

    @staticmethod
    def backward(ctx, go_fp8):
        x_fp8, w_fp8, b_fp8, fp8_amax_dL_dX, fp8_amax_dL_dW, \
            bw_amax_initialized = \
                ctx.saved_tensors
                
        is_bw_amax_initialized = torch.any(bw_amax_initialized)

        go_fp8_orig_shape = go_fp8._data.shape
        go_fp8_reshaped = go_fp8.reshape(-1, go_fp8_orig_shape[-1])

        if not is_bw_amax_initialized:
            # calculate reference amax of output
            with torch.no_grad():
                dL_dX_ref = torch.mm(go_fp8_reshaped, w_fp8)
                fp8_amax_dL_dX.fill_(tensor_to_amax(dL_dX_ref))

        dL_dX_scale = amax_to_scale(fp8_amax_dL_dX, torch.float8_e5m2)
        dL_dX_bits = mm_float8(
            go_fp8_reshaped, w_fp8, fp8_amax_dL_dX, dL_dX_scale, torch.float8_e5m2)
        dL_dX_bits = dL_dX_bits.reshape(*go_fp8_orig_shape[:-1], dL_dX_bits.shape[-1])
        dL_dX_fp8 = Float8Tensor(dL_dX_bits, dL_dX_scale, go_fp8._orig_dtype)

        x_fp8_orig_shape = x_fp8._data.shape
        x_fp8_reshaped = x_fp8.reshape(-1, x_fp8_orig_shape[-1])

        if not is_bw_amax_initialized:
            # calculate reference amax of output
            with torch.no_grad():
                dL_dW_ref = torch.mm(x_fp8_reshaped.t(), go_fp8_reshaped).t()
                fp8_amax_dL_dW.fill_(tensor_to_amax(dL_dW_ref))

        dL_dW_scale = amax_to_scale(fp8_amax_dL_dW, torch.float8_e5m2)
        dL_dW_bits = mm_float8(
            x_fp8_reshaped.t(), go_fp8_reshaped, fp8_amax_dL_dW, 
            dL_dW_scale, torch.float8_e5m2).t()
        dL_dW_fp8 = Float8Tensor(dL_dW_bits, dL_dW_scale, go_fp8._orig_dtype)

        if not is_bw_amax_initialized:
            bw_amax_initialized.fill_(1)

        # scale update would also happen here, for now no-op
        if b_fp8 is not None:
            return dL_dX_fp8, dL_dW_fp8, go_fp8, None, None, None, None, None
        else:
            return dL_dX_fp8, dL_dW_fp8, None, None, None, None, None, None

class _NoOpFwToFloat8E5M2Bw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fp8_amax_dL_dY, bw_amax_initialized):
        ctx.save_for_backward(fp8_amax_dL_dY, bw_amax_initialized)
        return x

    @staticmethod
    def backward(ctx, go):
        fp8_amax_dL_dY, bw_amax_initialized = ctx.saved_tensors
        is_bw_amax_initialized = torch.any(bw_amax_initialized)
        if not isinstance(go, Float8Tensor):
            # TODO(future): switch to windowed delayed scaling
            with torch.no_grad():
                if not is_bw_amax_initialized:
                    fp8_amax_dL_dY.fill_(tensor_to_amax(go))
                dL_dY_scale = amax_to_scale(fp8_amax_dL_dY, torch.float8_e5m2)
                fp8_amax_dL_dY.fill_(tensor_to_amax(go))
            go_fp8 = Float8Tensor(
                (go * dL_dY_scale).to(torch.float8_e5m2),
                dL_dY_scale, go.dtype)
        else:
            go_fp8 = go
        return go_fp8, None, None


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
        self.register_buffer('fp8_amax_x', torch.tensor(E4M3_MAX_POS))
        self.register_buffer('fp8_amax_w', torch.tensor(E4M3_MAX_POS))
        self.register_buffer('fp8_amax_y', torch.tensor(E4M3_MAX_POS))
        self.register_buffer('fp8_amax_dL_dX', torch.tensor(E5M2_MAX_POS))
        self.register_buffer('fp8_amax_dL_dW', torch.tensor(E5M2_MAX_POS))
        self.register_buffer('fp8_amax_dL_dY', torch.tensor(E5M2_MAX_POS))
        self.register_buffer('fw_amax_initialized', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('bw_amax_initialized', torch.tensor([0], dtype=torch.uint8))

    def forward(self, x):
        is_fw_amax_initialized = torch.any(self.fw_amax_initialized)
        if not isinstance(x, Float8Tensor):
            # Duplicate the autocast logic for F.linear, so that the output
            # of our module has the right original precision
            if torch.is_autocast_enabled():
                # For now, hardcode to GPU's autocast dtype
                # if we need CPU support in the future, we can add it
                x = x.to(torch.get_autocast_gpu_dtype())

            # TODO(future): switch to windowed delayed scaling
            if not is_fw_amax_initialized:
                self.fp8_amax_x.fill_(tensor_to_amax(x))
            x_scale = amax_to_scale(self.fp8_amax_x, torch.float8_e4m3fn)
            self.fp8_amax_x.fill_(tensor_to_amax(x))

            x_fp8 = Float8Tensor.to_float8(x, x_scale, torch.float8_e4m3fn)
        else:
            x_fp8 = x

        # TODO(future): switch to windowed delayed scaling
        if not is_fw_amax_initialized:
            self.fp8_amax_w.fill_(tensor_to_amax(self.weight))
        w_scale = amax_to_scale(self.fp8_amax_w, torch.float8_e4m3fn)
        self.fp8_amax_w.fill_(tensor_to_amax(self.weight))

        w_fp8 = Float8Tensor.to_float8(self.weight, w_scale, torch.float8_e4m3fn)

        use_new = True
        if use_new:
            w_fp8._fp8_buffer_refs['fp8_amax_y'] = self.fp8_amax_y
            w_fp8._fp8_buffer_refs['fw_amax_initialized'] = self.fw_amax_initialized
            y_fp8 = F.linear(x_fp8, w_fp8, self.bias)
        else:
            y_fp8 = float8_linear.apply(
                x_fp8, w_fp8, self.bias, self.fp8_amax_y, self.fp8_amax_dL_dX,
                self.fp8_amax_dL_dW, self.fw_amax_initialized, self.bw_amax_initialized)

        if not is_fw_amax_initialized:
            self.fw_amax_initialized.fill_(1)

        # Set up cast to fp8 in bw
        y_fp8 = _NoOpFwToFloat8E5M2Bw.apply(
            y_fp8, self.fp8_amax_dL_dY, self.bw_amax_initialized)

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
