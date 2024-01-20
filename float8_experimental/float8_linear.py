# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple manual UEX for a float8 version of `torch.nn.Linear`.

Note: this UEX is not intended for real usage. It merely demonstrates
an example of how features such as casting to and from float8 as well
as stateful scaling can be implemented. For now, we expect framework
owners to implement their own UEX.
"""

import dataclasses

from typing import Optional

import float8_experimental.config as config

import torch
from float8_experimental.float8_ops import float8_linear

from float8_experimental.float8_tensor import Float8Tensor

from float8_experimental.float8_utils import (
    amax_history_to_scale,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    get_maybe_autocast_inputs,
    tensor_to_amax,
    to_fp8_saturated,
)


def _maybe_initialize_amaxes_scales_for_float8_cast(
    x,
    cur_amax,
    amax_history,
    scale,
    scale_fn_name,
    float8_dtype,
    is_initialized,
):
    """
    If x is about to be cast to `float8` and the amax buffers are not initialized,
    initializes them inplace.
    """
    if is_initialized:
        return
    with torch.no_grad():
        # Note: we need to enable distributed reduction here in order
        # to match numerics between single GPU and multi GPU code
        new_amax = tensor_to_amax(x, distributed_reduction=True)
        cur_amax.fill_(new_amax)
        amax_history[0] = new_amax
        new_scale = amax_history_to_scale(
            amax_history, float8_dtype, x.dtype, scale_fn_name
        )
        scale.copy_(new_scale)


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2Bw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2, initialize if needed
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        fp8_amax_dL_dY,
        fp8_amax_history_dL_dY,
        fp8_scale_dL_dY,
        scale_fn_name,
        is_amax_initialized,
        emulate: bool,
    ):
        ctx.save_for_backward(fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY)
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
        ctx.emulate = emulate
        return tensor

    @staticmethod
    def backward(ctx, go):
        fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY = ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        is_amax_initialized = ctx.is_amax_initialized

        _maybe_initialize_amaxes_scales_for_float8_cast(
            go,
            fp8_amax_dL_dY,
            fp8_amax_history_dL_dY,
            fp8_scale_dL_dY,
            scale_fn_name,
            torch.float8_e5m2,
            is_amax_initialized,
        )

        fp8_amax_dL_dY.fill_(tensor_to_amax(go))
        go_scaled = go * fp8_scale_dL_dY
        bits_fp8 = to_fp8_saturated(go_scaled, torch.float8_e5m2)
        empty_grads = None, None, None, None, None, None
        res = Float8Tensor(bits_fp8, fp8_scale_dL_dY, go.dtype, emulate=ctx.emulate)
        return res, *empty_grads


@dataclasses.dataclass
class DelayedScalingRecipe:
    # Controls the history length of amax buffers
    history_len = 16

    # Controls the way to calculate current scale from amax history
    # TODO(future): add other functions as needed, hardcoded or user defined
    scale_fn_name = "max"


class Float8LinearMixin(object):
    def __init__(self, *args, **kwargs):
        delayed_scaling_recipe = kwargs.pop(
            "delayed_scaling_recipe", DelayedScalingRecipe()
        )
        # Amax scales should always be kept as float32.
        self.always_float32_buffers = set()
        super().__init__(*args, **kwargs)

        # TODO(future): have a unique recipe per buffer instead of one per
        # module, saving implementing that until we need it.
        # TODO(future): serialization for recipes
        self.recipe = delayed_scaling_recipe
        history_len = self.recipe.history_len

        self.register_always_float32_buffer("fp8_amax_x", torch.tensor(E4M3_MAX_POS))
        self.register_always_float32_buffer(
            "fp8_amax_history_x", torch.zeros(history_len)
        )
        self.register_always_float32_buffer("fp8_scale_x", torch.tensor(1.0))
        self.register_always_float32_buffer("fp8_amax_w", torch.tensor(E4M3_MAX_POS))
        self.register_always_float32_buffer(
            "fp8_amax_history_w", torch.zeros(history_len)
        )
        self.register_always_float32_buffer("fp8_scale_w", torch.tensor(1.0))
        self.register_always_float32_buffer(
            "fp8_amax_dL_dY", torch.tensor(E5M2_MAX_POS)
        )
        self.register_always_float32_buffer(
            "fp8_amax_history_dL_dY", torch.zeros(history_len)
        )
        self.register_always_float32_buffer("fp8_scale_dL_dY", torch.tensor(1.0))
        # Whether to emulate the fp8 matmul logic in float32
        self.emulate = False

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = not config.enable_amax_init

        # Syncing of amaxes and scales happens outside of this function. This
        # flag is here to enforce that the user does not forget to do this.
        self.amax_and_scale_synced = not config.enable_amax_init

        # This is needed to properly handle autocast in the amax/scale
        # update function for torch.float16
        self.last_seen_input_dtype = None

        # If true, this enables TP+SP style distributed comms in TP primitives
        # Note: this is not used in non-TP code.
        self.use_sequence_parallel = False

        # pre_forward and post_forward are currently broken with FSDP
        # and torch.compile, this option can disable them
        self.enable_pre_and_post_forward = config.enable_pre_and_post_forward

        # This flag is used to modify what gets saved for backwards. Its default value
        # is False, this saves the casted weight for backwards. Note that this typically increases memory usage
        # Because both the weight parameter and the casted weight are saved on device. If set to true
        # this will only save the weight parameter and during the backwards pass it will re-cast this weight to fp8.
        # For traditional FSDP this should be set to True in order to not save the un-sharded weight for backwards.
        self.recompute_weight_cast = False

    def register_always_float32_buffer(
        self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True
    ) -> None:
        self.register_buffer(name=name, tensor=tensor, persistent=persistent)
        self.always_float32_buffers.add(name)

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        self.convert_amax_buffer_to_float32()
        return ret

    def convert_amax_buffer_to_float32(self):
        for key in self.always_float32_buffers:
            if self._buffers[key] is not None:
                self._buffers[key] = self._buffers[key].to(torch.float32)

    def cast_x_to_float8(
        self, x: torch.Tensor, is_amax_initialized: bool
    ) -> torch.Tensor:
        scale_fn_name = self.recipe.scale_fn_name
        _maybe_initialize_amaxes_scales_for_float8_cast(
            x,
            self.fp8_amax_x,
            self.fp8_amax_history_x,
            self.fp8_scale_x,
            scale_fn_name,
            torch.float8_e4m3fn,
            is_amax_initialized,
        )
        x_fp8 = Float8Tensor.to_float8(
            x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x, self.emulate
        )
        return x_fp8

    def _maybe_init_amaxes_scales_weight(
        self, w: torch.Tensor, is_amax_initialized: bool
    ):
        scale_fn_name = self.recipe.scale_fn_name
        _maybe_initialize_amaxes_scales_for_float8_cast(
            w,
            self.fp8_amax_w,
            self.fp8_amax_history_w,
            self.fp8_scale_w,
            scale_fn_name,
            torch.float8_e4m3fn,
            is_amax_initialized,
        )

    def cast_w_to_float8(
        self, w: torch.Tensor, is_amax_initialized: bool
    ) -> torch.Tensor:
        scale_fn_name = self.recipe.scale_fn_name
        _maybe_initialize_amaxes_scales_for_float8_cast(
            w,
            self.fp8_amax_w,
            self.fp8_amax_history_w,
            self.fp8_scale_w,
            scale_fn_name,
            torch.float8_e4m3fn,
            is_amax_initialized,
        )

        w_fp8 = Float8Tensor.to_float8(
            w,
            self.fp8_scale_w,
            torch.float8_e4m3fn,
            self.fp8_amax_w,
            self.emulate,
        )
        return w_fp8

    def cast_y_to_float8_in_bw(
        self, y: torch.Tensor, emulate: bool = False
    ) -> torch.Tensor:
        scale_fn_name = self.recipe.scale_fn_name
        y = NoopFwToFloat8E5M2Bw.apply(
            y,
            self.fp8_amax_dL_dY,
            self.fp8_amax_history_dL_dY,
            self.fp8_scale_dL_dY,
            scale_fn_name,
            self.is_amax_initialized,
            emulate,
        )
        return y

    def float8_pre_forward(self, x):
        if not self.enable_pre_and_post_forward:
            return
        if (
            self.is_amax_initialized
            and (not self.amax_and_scale_synced)
            and torch.is_grad_enabled()
        ):
            raise AssertionError(
                "amaxes and scales not synced, please call `sync_float8_amax_and_scale_history` before forward"
            )
        self.last_seen_input_dtype = x.dtype

    def float8_post_forward(self):
        if not self.enable_pre_and_post_forward:
            return
        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.is_amax_initialized = True
        self.amax_and_scale_synced = False


class Float8Linear(Float8LinearMixin, torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def forward(self, x):
        temp_x, temp_weight, temp_bias = get_maybe_autocast_inputs(
            x, self.weight, self.bias
        )
        self.float8_pre_forward(x)

        x_fp8 = self.cast_x_to_float8(temp_x, self.is_amax_initialized)
        self._maybe_init_amaxes_scales_weight(self.weight, self.is_amax_initialized)

        y = float8_linear(
            x_fp8,
            temp_weight,
            None,  # bias
            self.fp8_scale_w,
            self.fp8_amax_w,
            self.emulate,
            self.recompute_weight_cast,
        )

        # Cast gradY to float8_e5m2 during backward
        y = self.cast_y_to_float8_in_bw(y, self.emulate)

        # TODO We should use addmm above but this fails the single fsdp test:
        # FAILED: _orig_mod.0.fp8_amax_w, 0.2197265625, 0.21875
        # Not immediately clear why the bias being fused in would only effect the numerics
        # for the weight....
        if temp_bias is not None:
            y = y + temp_bias

        self.float8_post_forward()
        return y

    @classmethod
    def from_float(
        cls, mod, emulate: bool = False, recompute_weight_cast: bool = False
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
            recompute_weight_cast (bool): whether to recompute the casted weight for backwards
        """
        # TODO Follow up! This is a great idea but we need the mixin base to create real
        # Tensors and the Linear base to create empty params
        # with torch.device("meta"):
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        new_mod.recompute_weight_cast = recompute_weight_cast
        # I think its okay to send all params and buffers to device
        new_mod.to(mod.weight.device)
        return new_mod
