import dataclasses

import torch

from float8_linear_utils import (
    _maybe_initialize_amaxes_for_float8_cast,
    _update_history_with_new_amax,
)

from float8_utils import (
    tensor_to_amax,
    amax_history_to_scale,
    to_fp8_saturated,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)
from float8_linear import Float8Linear
from float8_python_api import addmm_float8_unwrapped

# Note: this class does not take arguments of type `torch.dtype`, as PT2.0
# does not support tracing this: https://gist.github.com/vkuzo/4ca38f74b0be65a62133ba6cc306d5b5
# TODO(future): relax this as needed
# TODO(future): move working around this restriction to `Float8Linear`, once
# there is a better way to test things.
class ToFloat8E4M3FNConstrFuncDecomposed(torch.autograd.Function):
    """
    A differentiable conversion to fp8, returns decomposed representation
    """
    @staticmethod
    def forward(
        ctx,
        tensor,
        scale: float=None,
        amax_buffer=None,
    ):
        # In TransformerEngine, the casts to float8 are fused with calculating
        # the new amax value. In this codebase, the eager mode code for those
        # two things is colocated in this function. We expect PT2.0 to fuse it
        # for us.
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        tensor_scaled = tensor * scale
        bits_fp8 = to_fp8_saturated(tensor_scaled, torch.float8_e4m3fn)
        return bits_fp8

    @staticmethod
    def backward(ctx, g):
        return g.to(torch.float32), None, None, None


class float8_linear_no_tensor_subclass(torch.autograd.Function):
    """
    Like `float8_linear`, but without using tensor subclass.
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8_d,
        x_fp8_scale,
        w_fp8_d,
        w_fp8_scale,
        b,
        fp8_amax_dL_dY,
        fp8_amax_history_dL_dY,
        fp8_noop_amax,
        fp8_noop_scale,
        is_amax_initialized,
        scale_fn_name,
        emulate: bool,
    ):
        ctx.save_for_backward(
            x_fp8_d, x_fp8_scale, w_fp8_d, w_fp8_scale, b,
            fp8_amax_dL_dY, fp8_amax_history_dL_dY,
            fp8_noop_amax, fp8_noop_scale)
        ctx.scale_fn_name = scale_fn_name
        ctx.emulate = emulate
        output_dtype = b.dtype if b is not None else torch.float32
        ctx.output_dtype = output_dtype
        ctx.is_amax_initialized = is_amax_initialized
        orig_shape = x_fp8_d.shape
        x_fp8_reshaped = x_fp8_d.reshape(-1, orig_shape[-1])

        if b is not None:
            if emulate:
                res_bits = torch.ops.aten.addmm_float8_emulated(
                    b, x_fp8_reshaped, x_fp8_scale, w_fp8_d.t(), w_fp8_scale,
                    fp8_noop_amax, fp8_noop_scale, output_dtype)
            else:
                res_bits = addmm_float8_unwrapped(
                    b,
                    x_fp8_reshaped, x_fp8_scale,
                    w_fp8_d.t(), w_fp8_scale,
                    fp8_noop_amax, fp8_noop_scale, output_dtype)
        else:
            if emulate:
                res_bits = torch.ops.aten.mm_float8_emulated(
                    x_fp8_reshaped, x_fp8_scale,
                    w_fp8_d.t(), w_fp8_scale,
                    fp8_noop_amax, fp8_noop_scale,
                    output_dtype)
            else:
                res_bits = addmm_float8_unwrapped(
                    None, # Input bias
                    x_fp8_reshaped, x_fp8_scale,
                    w_fp8_d.t(), w_fp8_scale,
                    fp8_noop_amax, fp8_noop_scale,
                    output_dtype
                )
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, go):
        x_fp8_d, x_fp8_scale, w_fp8_d, w_fp8_scale, \
            b_fp8, fp8_amax_dL_dY, fp8_amax_history_dL_dY, \
            fp8_noop_amax, fp8_noop_scale = \
                ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        emulate = ctx.emulate
        output_dtype = ctx.output_dtype
        is_amax_initialized = ctx.is_amax_initialized

        # cast fp32 to fp8
        _maybe_initialize_amaxes_for_float8_cast(
            go, fp8_amax_dL_dY, fp8_amax_history_dL_dY, is_amax_initialized)
        dL_dY_scale = amax_history_to_scale(
            fp8_amax_history_dL_dY, torch.float8_e5m2, go.dtype,
            scale_fn_name)
        fp8_amax_dL_dY.fill_(tensor_to_amax(go))
        go_scaled = go * dL_dY_scale
        go_fp8_d = to_fp8_saturated(go_scaled, torch.float8_e5m2)

        go_fp8_orig_shape = go_fp8_d.shape
        go_fp8_reshaped = go_fp8_d.reshape(-1, go_fp8_orig_shape[-1])

        #
        # calculate dL/dX, update relevant buffers along the way
        if emulate:
            dL_dX_bits = torch.ops.aten.mm_float8_emulated(
                go_fp8_reshaped, dL_dY_scale,
                w_fp8_d, w_fp8_scale, fp8_noop_amax, fp8_noop_scale, output_dtype)
        else:
            dL_dX_bits = addmm_float8_unwrapped(
                None, # Input bias
                go_fp8_reshaped, dL_dY_scale,
                w_fp8_d, w_fp8_scale, fp8_noop_amax, fp8_noop_scale, output_dtype)
        dL_dX_bits = dL_dX_bits.reshape(*go_fp8_orig_shape[:-1], dL_dX_bits.shape[-1])

        x_fp8_orig_shape = x_fp8_d.shape
        x_fp8_reshaped = x_fp8_d.reshape(-1, x_fp8_orig_shape[-1])

        #
        # calculate dL/dW, update relevant buffers along the way
        #
        if emulate:
            dL_dW_bits = torch.ops.aten.mm_float8_emulated(
                x_fp8_reshaped.t(), x_fp8_scale,
                go_fp8_reshaped, dL_dY_scale,
                fp8_noop_amax, fp8_noop_scale, output_dtype).t()
        else:
            dL_dW_bits = addmm_float8_unwrapped(
                None, # Input bias
                x_fp8_reshaped.t(), x_fp8_scale,
                go_fp8_reshaped, dL_dY_scale,
                fp8_noop_amax, fp8_noop_scale, output_dtype).t()

        empty_grads = None, None, None, None, None, None, None, None, None, None, None, None
        if b_fp8 is not None:
            return dL_dX_bits, None, dL_dW_bits, None, go, *empty_grads
        else:
            return dL_dX_bits, None, dL_dW_bits, None, *empty_grads

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

        _maybe_initialize_amaxes_for_float8_cast(
            x, self.fp8_amax_x, self.fp8_amax_history_x,
            is_amax_initialized_this_iteration)
        x_scale = amax_history_to_scale(
            self.fp8_amax_history_x, torch.float8_e4m3fn, x.dtype,
            scale_fn_name)
        x_fp8_d = ToFloat8E4M3FNConstrFuncDecomposed.apply(
            x, x_scale, self.fp8_amax_x)

        _maybe_initialize_amaxes_for_float8_cast(
            self.weight, self.fp8_amax_w, self.fp8_amax_history_w,
            is_amax_initialized_this_iteration)
        w_scale = amax_history_to_scale(
            self.fp8_amax_history_w, torch.float8_e4m3fn, self.weight.dtype,
            scale_fn_name)
        w_fp8_d = ToFloat8E4M3FNConstrFuncDecomposed.apply(
            self.weight, w_scale, self.fp8_amax_w)

        # TODO This is casting at every forward, we should only do this once
        casted_bias = self.bias.to(x.dtype) if self.bias is not None else None
        y_fp32 = float8_linear_no_tensor_subclass.apply(
            x_fp8_d, x_scale, w_fp8_d, w_scale,
            casted_bias,
            self.fp8_amax_dL_dY, self.fp8_amax_history_dL_dY,
            self.fp8_noop_amax, self.fp8_noop_scale,
            is_amax_initialized_this_iteration, scale_fn_name, self.emulate)

        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.amax_and_scale_synced = False

        return y_fp32
