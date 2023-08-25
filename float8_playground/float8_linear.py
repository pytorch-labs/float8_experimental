"""
A simple manual UEX for a float8 version of `torch.nn.Linear`.

Note: this UEX is not intended for real usage. It merely demonstrates
an example of how features such as casting to and from float8 as well
as stateful scaling can be implemented. For now, we expect framework
owners to implement their own UEX.
"""

import dataclasses

import torch
from typing import Optional

from float8_linear_utils import (
    _maybe_initialize_amaxes_for_float8_cast,
    _update_history_with_new_amax,
)

from float8_python_api import (
    mm_float8,
    addmm_float8,
)

from float8_utils import (
    tensor_to_amax,
    amax_history_to_scale,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)
from float8_tensor import Float8Tensor

class float8_linear(torch.autograd.Function):
    """
    Like F.linear, but with X and W in float8
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8,
        w_fp8,
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
            x_fp8, w_fp8, b,
            fp8_amax_dL_dY, fp8_amax_history_dL_dY,
            fp8_noop_amax, fp8_noop_scale)
        ctx.scale_fn_name = scale_fn_name
        ctx.emulate = emulate
        orig_shape = x_fp8._data.shape
        x_fp8_reshaped = x_fp8.reshape(-1, orig_shape[-1])
        ctx.is_amax_initialized = is_amax_initialized

        if b is not None:
            res_bits = addmm_float8(
                b, x_fp8_reshaped, w_fp8.t(), fp8_noop_amax, fp8_noop_scale,
                output_dtype=x_fp8._orig_dtype, emulate=emulate)
        else:
            res_bits = mm_float8(
                x_fp8_reshaped, w_fp8.t(), fp8_noop_amax, fp8_noop_scale,
                output_dtype=x_fp8._orig_dtype, emulate=emulate)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, go):
        x_fp8, w_fp8, b_fp8, \
            fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_noop_amax, fp8_noop_scale = \
                ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        emulate = ctx.emulate
        is_amax_initialized = ctx.is_amax_initialized

        # cast incoming gradient to fp8
        _maybe_initialize_amaxes_for_float8_cast(
            go, fp8_amax_dL_dY, fp8_amax_history_dL_dY, 
            is_amax_initialized)
        go_scale = amax_history_to_scale(
            fp8_amax_history_dL_dY, torch.float8_e5m2, scale_fn_name)
        go_fp8 = Float8Tensor.to_float8(
            go, go_scale, torch.float8_e5m2, fp8_amax_dL_dY)
        _update_history_with_new_amax(
            fp8_amax_dL_dY, fp8_amax_history_dL_dY)

        go_fp8_orig_shape = go_fp8._data.shape
        go_fp8_reshaped = go_fp8.reshape(-1, go_fp8_orig_shape[-1])

        #
        # calculate dL/dX
        #
        dL_dX = mm_float8(
            go_fp8_reshaped, w_fp8, fp8_noop_amax, fp8_noop_scale, 
            output_dtype=x_fp8._orig_dtype, emulate=emulate)
        dL_dX = dL_dX.reshape(*go_fp8_orig_shape[:-1], dL_dX.shape[-1])

        x_fp8_orig_shape = x_fp8._data.shape
        x_fp8_reshaped = x_fp8.reshape(-1, x_fp8_orig_shape[-1])

        #
        # calculate dL/dW
        #
        dL_dW = mm_float8(
            x_fp8_reshaped.t(), go_fp8_reshaped, fp8_noop_amax,
            fp8_noop_scale, output_dtype=x_fp8._orig_dtype, emulate=emulate).t()

        empty_grads = None, None, None, None, None, None, None, None
        if b_fp8 is not None:
            return dL_dX, dL_dW, go, *empty_grads
        else:
            return dL_dX, dL_dW, *empty_grads

@dataclasses.dataclass
class DelayedScalingRecipe:
    # Controls the history length of amax buffers
    # TODO(future): make default history_len more representative for real usage,
    # current value is for debugging
    history_len = 4

    # Controls the way to calculate current scale from amax history
    # TODO(future): add other functions as needed, hardcoded or user defined
    scale_fn_name = 'max'

class Float8Linear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """
    def __init__(self, *args, **kwargs):
        delayed_scaling_recipe = kwargs.pop('delayed_scaling_recipe', DelayedScalingRecipe())
        super().__init__(*args, **kwargs)

        # TODO(future): have a unique recipe per buffer instead of one per
        # module, saving implementing that until we need it.
        # TODO(future): serialization for recipes
        self.recipe = delayed_scaling_recipe
        history_len = self.recipe.history_len

        self.register_buffer('fp8_amax_x', torch.tensor(E4M3_MAX_POS))
        self.register_buffer('fp8_amax_history_x', torch.zeros(history_len))
        self.register_buffer('fp8_amax_w', torch.tensor(E4M3_MAX_POS))
        self.register_buffer('fp8_amax_history_w', torch.zeros(history_len))
        self.register_buffer('fp8_amax_dL_dY', torch.tensor(E5M2_MAX_POS))
        self.register_buffer('fp8_amax_history_dL_dY', torch.zeros(history_len))
        # The two buffers below are noops, we have them because 
        # torch.scaled_mm requires arguments in case the output of the matmul
        # is float8.
        self.register_buffer('fp8_noop_amax', torch.tensor(float('inf')))
        self.register_buffer('fp8_noop_scale', torch.tensor(1.0))
        # Whether to emulate the fp8 matmul logic in float32
        self.emulate = False

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = False

    def forward(self, x):
        is_amax_initialized_this_iteration = self.is_amax_initialized
        self.is_amax_initialized = True
        scale_fn_name = self.recipe.scale_fn_name

        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            x = x.to(torch.get_autocast_gpu_dtype())

        _maybe_initialize_amaxes_for_float8_cast(
            x, self.fp8_amax_x, self.fp8_amax_history_x, 
            is_amax_initialized_this_iteration)
        x_scale = amax_history_to_scale(
            self.fp8_amax_history_x, torch.float8_e4m3fn, scale_fn_name)
        x_fp8 = Float8Tensor.to_float8(
            x, x_scale, torch.float8_e4m3fn, self.fp8_amax_x)
        _update_history_with_new_amax(
            self.fp8_amax_x, self.fp8_amax_history_x)

        _maybe_initialize_amaxes_for_float8_cast(
            self.weight, self.fp8_amax_w, self.fp8_amax_history_w, 
            is_amax_initialized_this_iteration)
        w_scale = amax_history_to_scale(
            self.fp8_amax_history_w, torch.float8_e4m3fn, scale_fn_name)
        w_fp8 = Float8Tensor.to_float8(
            self.weight, w_scale, torch.float8_e4m3fn, self.fp8_amax_w)
        _update_history_with_new_amax(
            self.fp8_amax_w, self.fp8_amax_history_w)

        y = float8_linear.apply(
            x_fp8, w_fp8, self.bias,
            self.fp8_amax_dL_dY, self.fp8_amax_history_dL_dY,
            self.fp8_noop_amax, self.fp8_noop_scale,
            is_amax_initialized_this_iteration, scale_fn_name, self.emulate)

        return y

    @classmethod
    def from_float(cls, mod, emulate: bool):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
        """
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        # I think its okay to send all params and buffers to device
        new_mod.to(mod.weight.device)
        return new_mod


def swap_linear_with_float8_linear(model, emulate=False):
    """
    Replaces all instances of torch.nn.Linear in the given model with Float8Linear.

    Args:
        model (torch.nn.Module): The model to modify.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in float32.
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if isinstance(child, torch.nn.Linear):
            new_child = Float8Linear.from_float(child, emulate)
            setattr(model, name, new_child)
        else:
            swap_linear_with_float8_linear(child, emulate)
