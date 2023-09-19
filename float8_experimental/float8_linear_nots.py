import dataclasses

import torch

from float8_experimental.float8_linear_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    _update_history_with_new_amax,
)

from float8_experimental.float8_utils import (
    tensor_to_amax,
    to_fp8_saturated,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_python_api import mm_float8_unwrapped

def _to_float8_e4m3fn_decomposed(tensor, scale, amax_buffer):
    # In TransformerEngine, the casts to float8 are fused with calculating
    # the new amax value. In this codebase, the eager mode code for those
    # two things is colocated in this function. We expect PT2.0 to fuse it
    # for us.
    if amax_buffer is not None:
        amax_buffer.fill_(tensor_to_amax(tensor))

    tensor_scaled = tensor * scale
    bits_fp8 = to_fp8_saturated(tensor_scaled, torch.float8_e4m3fn)
    return bits_fp8

class float8_linear_no_tensor_subclass(torch.autograd.Function):
    """
    Like `float8_linear`, but without using tensor subclass.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        w,
        fp8_amax_x,
        fp8_amax_history_x,
        fp8_scale_x,
        fp8_amax_w,
        fp8_amax_history_w,
        fp8_scale_w,
        fp8_amax_dL_dY,
        fp8_amax_history_dL_dY,
        fp8_scale_dL_dY,
        is_amax_initialized,
        scale_fn_name,
        emulate: bool,
    ):
        _maybe_initialize_amaxes_scales_for_float8_cast(
            x, fp8_amax_x, fp8_amax_history_x,
            fp8_scale_x, scale_fn_name, torch.float8_e4m3fn,
            is_amax_initialized)
        x_fp8_d = _to_float8_e4m3fn_decomposed(x, fp8_scale_x, fp8_amax_x)
        # r_t_c means reshaped to 2d, then transpose-contiguous
        x_fp8_d_r_t_c = x_fp8_d.reshape(-1, x_fp8_d.shape[-1]).t().contiguous()

        _maybe_initialize_amaxes_scales_for_float8_cast(
            w, fp8_amax_w, fp8_amax_history_w,
            fp8_scale_w, scale_fn_name, torch.float8_e4m3fn,
            is_amax_initialized)
        w_fp8_d = _to_float8_e4m3fn_decomposed(w, fp8_scale_w, fp8_amax_w)
        w_fp8_d_t_c = w_fp8_d.t().contiguous()

        ctx.save_for_backward(
            x_fp8_d, x_fp8_d_r_t_c, fp8_scale_x, w_fp8_d, w_fp8_d_t_c, fp8_scale_w,
            fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY)
        ctx.scale_fn_name = scale_fn_name
        ctx.emulate = emulate
        output_dtype = x.dtype
        ctx.output_dtype = output_dtype
        ctx.is_amax_initialized = is_amax_initialized
        orig_shape = x_fp8_d.shape
        x_fp8_reshaped = x_fp8_d.reshape(-1, orig_shape[-1])

        if emulate:
            res_bits, _output_amax = torch.ops.aten.mm_float8_emulated(
                x_fp8_reshaped, fp8_scale_x,
                w_fp8_d.t(), fp8_scale_w, output_dtype)
        else:
            res_bits, _output_amax = mm_float8_unwrapped(
                x_fp8_reshaped, fp8_scale_x,
                w_fp8_d.t(), fp8_scale_w,
                output_dtype, output_scale=None,
            )
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, go):
        x_fp8_d, x_fp8_d_r_t_c, fp8_scale_x, w_fp8_d, w_fp8_d_t_c, fp8_scale_w, \
            fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY  = \
                ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        emulate = ctx.emulate
        output_dtype = ctx.output_dtype
        is_amax_initialized = ctx.is_amax_initialized

        # cast fp32 to fp8
        _maybe_initialize_amaxes_scales_for_float8_cast(
            go, fp8_amax_dL_dY, fp8_amax_history_dL_dY, 
            fp8_scale_dL_dY, scale_fn_name, torch.float8_e5m2,
            is_amax_initialized)
        fp8_amax_dL_dY.fill_(tensor_to_amax(go))
        go_scaled = go * fp8_scale_dL_dY
        go_fp8_d = to_fp8_saturated(go_scaled, torch.float8_e5m2)

        go_fp8_orig_shape = go_fp8_d.shape
        go_fp8_reshaped = go_fp8_d.reshape(-1, go_fp8_orig_shape[-1])
        go_fp8_reshaped_t_c_t = go_fp8_reshaped.t().contiguous().t()

        #
        # calculate dL/dX, update relevant buffers along the way
        #
        if emulate:
            dL_dX_bits, _dL_dX_amax = torch.ops.aten.mm_float8_emulated(
                go_fp8_reshaped, fp8_scale_dL_dY,
                w_fp8_d_t_c.t(), fp8_scale_w, output_dtype)
        else:
            dL_dX_bits, _dL_dX_amax = mm_float8_unwrapped(
                go_fp8_reshaped, fp8_scale_dL_dY,
                w_fp8_d_t_c.t(), fp8_scale_w, output_dtype, output_scale=None)
        dL_dX_bits = dL_dX_bits.reshape(*go_fp8_orig_shape[:-1], dL_dX_bits.shape[-1])

        x_fp8_orig_shape = x_fp8_d.shape
        x_fp8_reshaped = x_fp8_d.reshape(-1, x_fp8_orig_shape[-1])

        #
        # calculate dL/dW, update relevant buffers along the way
        #
        if emulate:
            dL_dW_bits, _dL_dW_amax = torch.ops.aten.mm_float8_emulated(
                x_fp8_d_r_t_c, fp8_scale_x,
                go_fp8_reshaped_t_c_t, fp8_scale_dL_dY, output_dtype)
            dL_dW_bits = dL_dW_bits.t()
        else:
            dL_dW_bits, _dL_dW_amax = mm_float8_unwrapped(
                x_fp8_d_r_t_c, fp8_scale_x,
                go_fp8_reshaped_t_c_t, fp8_scale_dL_dY,
                output_dtype, output_scale=None)
            dL_dW_bits = dL_dW_bits.t()
        empty_grads = None, None, None, None, None, None, None, None, None, None, None, None
        return dL_dX_bits, dL_dW_bits, *empty_grads

class Float8LinearNoTensorSubclass(Float8Linear):
    """
    Same as `Float8Linear`, but without using tensor subclass. This is a
    temporary class to unblock PT2.0 work on other parts of Float8
    while tensor subclass traceability is landing to PyTorch core
    (ETA 2023-09-15).

    Note: there is no expectation of full feature support or numerical
    correctness, and we plan to delete this class once subclasses are
    traceable in PT2.0.
    """
    def forward(self, x):
        if self.is_amax_initialized and (not self.amax_and_scale_synced):
            raise AssertionError('amaxes and scales not synced, please call `sync_float8_amax_and_scale_history` before forward') 

        is_amax_initialized_this_iteration = self.is_amax_initialized
        self.is_amax_initialized = True
        scale_fn_name = self.recipe.scale_fn_name

        y_fp32 = float8_linear_no_tensor_subclass.apply(
            x, self.weight,
            self.fp8_amax_x, self.fp8_amax_history_x, self.fp8_scale_x,
            self.fp8_amax_w, self.fp8_amax_history_w, self.fp8_scale_w,
            self.fp8_amax_dL_dY, self.fp8_amax_history_dL_dY,
            self.fp8_scale_dL_dY, is_amax_initialized_this_iteration, 
            scale_fn_name, self.emulate)

        if self.bias is not None:
            y_fp32 = y_fp32 + self.bias

        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.amax_and_scale_synced = False

        return y_fp32
