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

import torch

from float8_experimental.float8_linear_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
)

from float8_experimental.float8_tensor import Float8Tensor, ScaledMMConfig

from float8_experimental.float8_utils import (
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    tensor_to_amax,
    to_fp8_saturated,
)


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
        mm_config: ScaledMMConfig,
    ):
        ctx.save_for_backward(fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY)
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
        ctx.mm_config = mm_config
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
        res = Float8Tensor(bits_fp8, fp8_scale_dL_dY, go.dtype, mm_config=ctx.mm_config)
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
        super().__init__(*args, **kwargs)

        # TODO(future): have a unique recipe per buffer instead of one per
        # module, saving implementing that until we need it.
        # TODO(future): serialization for recipes
        self.recipe = delayed_scaling_recipe
        history_len = self.recipe.history_len

        self.register_buffer("fp8_amax_x", torch.tensor(E4M3_MAX_POS))
        self.register_buffer("fp8_amax_history_x", torch.zeros(history_len))
        self.register_buffer("fp8_scale_x", torch.tensor(1.0))
        self.register_buffer("fp8_amax_w", torch.tensor(E4M3_MAX_POS))
        self.register_buffer("fp8_amax_history_w", torch.zeros(history_len))
        self.register_buffer("fp8_scale_w", torch.tensor(1.0))
        self.register_buffer("fp8_amax_dL_dY", torch.tensor(E5M2_MAX_POS))
        self.register_buffer("fp8_amax_history_dL_dY", torch.zeros(history_len))
        self.register_buffer("fp8_scale_dL_dY", torch.tensor(1.0))
        # Whether to emulate the fp8 matmul logic in float32
        self.emulate = False

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = False

        # Syncing of amaxes and scales happens outside of this function. This
        # flag is here to enforce that the user does not forget to do this.
        self.amax_and_scale_synced = False

        # This is needed to properly handle autocast in the amax/scale
        # update function
        self.last_seen_input_dtype = None

        # If true, this enables TP+SP style distributed comms in TP primitives
        # Note: this is not used in non-TP code.
        self.use_sequence_parallel = False

        # Save the Float8Tensor constructor for FSDP.
        # N.B. Do not partially apply the scale into the constructor because
        # buffer Python IDs are not preserved by `nn.Module.to()` and the
        # module could be moved to GPU after this constructor. Instead, FSDP
        # will access the scale when it has ensured that it is on GPU.
        self._float8_tensor_ctor = lambda *args, **kwargs: Float8Tensor(*args, **kwargs)

        # Defines the behavior of the matmul in the forward and backward
        # Forward we use fast_accum, backwards we do not
        self.forward_config = ScaledMMConfig(self.emulate, True if not self.emulate else False)
        self.backward_config = ScaledMMConfig(self.emulate, False)

    def cast_x_to_float8(
        self, x: torch.Tensor, is_amax_initialized: bool
    ) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            x = x.to(autocast_dtype)
            self.bias_dtype = autocast_dtype

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
            x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x, self.forward_config
        )
        return x_fp8

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
            w, self.fp8_scale_w, torch.float8_e4m3fn, self.fp8_amax_w, self.forward_config
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
            self.backward_config,
        )
        return y

    def float8_pre_forward(self, x):
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
        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.is_amax_initialized = True
        self.amax_and_scale_synced = False

    def add_weight_tag(self):
        # We add a tag to the weight nn.Parameter in order to signal
        # To FSDP that this param is a weight
        self.weight._is_fp8_weight = True


class Float8Linear(Float8LinearMixin, torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_weight_tag()

    def forward(self, x):
        self.float8_pre_forward(x)

        x_fp8 = self.cast_x_to_float8(x, self.is_amax_initialized)
        if getattr(self, "_w_fp8", None) is not None:  # FSDP handled the cast
            w_fp8 = self._w_fp8
        else:
            w_fp8 = self.cast_w_to_float8(self.weight, self.is_amax_initialized)

        y = torch.matmul(x_fp8, w_fp8.t())

        # Cast gradY to float8_e5m2 during backward
        y = self.cast_y_to_float8_in_bw(y, self.emulate)

        if self.bias is not None:
            y = y + self.bias.to(self.bias_dtype)

        self.float8_post_forward()
        return y

    @classmethod
    def from_float(cls, mod, emulate: bool = False):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
        """
        # TODO Follow up! This is a great idea but we need the mixin base to create real
        # Tensors and the Linear base to create empty params
        # with torch.device("meta"):
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.forward_config = ScaledMMConfig(emulate, True if not emulate else False)
        new_mod.backward_config = ScaledMMConfig(emulate, False)
        if mod.bias is not None:
            new_mod.bias_dtype = mod.bias.dtype
        # I think its okay to send all params and buffers to device
        new_mod.to(mod.weight.device)
        new_mod.add_weight_tag()
        return new_mod
