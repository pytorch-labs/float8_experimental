"""
A simple manual UEX for a float8 version of `torch.nn.Linear`.

Note: this UEX is not intended for real usage. It merely demonstrates
an example of how features such as casting to and from float8 as well
as stateful scaling can be implemented. For now, we expect framework
owners to implement their own UEX.
"""

import dataclasses

import torch
import torch.distributed as dist

from float8_experimental.float8_linear_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    _update_history_with_new_amax,
)

from float8_experimental.float8_python_api import mm_float8
from float8_experimental.float8_tensor import Float8Tensor, to_float8

from float8_experimental.float8_utils import (
    amax_history_to_scale,
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
    ):
        ctx.save_for_backward(fp8_amax_dL_dY, fp8_amax_history_dL_dY, fp8_scale_dL_dY)
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
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
        empty_grads = None, None, None, None, None
        res = Float8Tensor(bits_fp8, fp8_scale_dL_dY, go.dtype)
        return res, *empty_grads


class float8_linear(torch.autograd.Function):
    """
    Like F.linear, but with X and W in float8
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8,
        w_fp8,
        is_amax_initialized,
        scale_fn_name,
        emulate: bool,
    ):
        ctx.save_for_backward(x_fp8, w_fp8)
        ctx.scale_fn_name = scale_fn_name
        ctx.emulate = emulate
        orig_shape = x_fp8._data.shape
        x_fp8_reshaped = Float8Tensor(
            x_fp8._data.reshape(-1, orig_shape[-1]), x_fp8._scale, x_fp8._orig_dtype
        )
        ctx.is_amax_initialized = is_amax_initialized

        w_fp8_t = Float8Tensor(w_fp8._data.t(), w_fp8._scale, w_fp8._orig_dtype)

        res_bits, _output_amax = mm_float8(
            x_fp8_reshaped, w_fp8_t, output_dtype=x_fp8._orig_dtype, emulate=emulate
        )
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, go_fp8):
        x_fp8, w_fp8 = ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        emulate = ctx.emulate
        is_amax_initialized = ctx.is_amax_initialized

        go_fp8_orig_shape = go_fp8._data.shape
        go_fp8_reshaped = Float8Tensor(
            go_fp8._data.reshape(-1, go_fp8_orig_shape[-1]),
            go_fp8._scale,
            go_fp8._orig_dtype,
        )

        w_fp8_t_c_t = Float8Tensor(
            w_fp8._data.t().contiguous().t(), w_fp8._scale, w_fp8._orig_dtype
        )

        #
        # calculate dL/dX
        #
        dL_dX, _dL_dX_amax = mm_float8(
            go_fp8_reshaped,
            w_fp8_t_c_t,
            output_dtype=x_fp8._orig_dtype,
            emulate=emulate,
        )
        dL_dX = dL_dX.reshape(*go_fp8_orig_shape[:-1], dL_dX.shape[-1])

        x_fp8_orig_shape = x_fp8._data.shape
        x_fp8_reshaped_t_c = Float8Tensor(
            x_fp8._data.reshape(-1, x_fp8_orig_shape[-1]).t().contiguous(),
            x_fp8._scale,
            x_fp8._orig_dtype,
        )

        go_fp8_reshaped_t_c_t = Float8Tensor(
            go_fp8_reshaped._data.t().contiguous().t(),
            go_fp8_reshaped._scale,
            go_fp8_reshaped._orig_dtype,
        )

        #
        # calculate dL/dW
        #
        dL_dW, _dL_dW_amax = mm_float8(
            x_fp8_reshaped_t_c,
            go_fp8_reshaped_t_c_t,
            output_dtype=x_fp8._orig_dtype,
            emulate=emulate,
        )
        dL_dW = dL_dW.t()

        empty_grads = None, None, None, None, None, None, None, None, None
        return dL_dX, dL_dW, *empty_grads


@dataclasses.dataclass
class DelayedScalingRecipe:
    # Controls the history length of amax buffers
    history_len = 16

    # Controls the way to calculate current scale from amax history
    # TODO(future): add other functions as needed, hardcoded or user defined
    scale_fn_name = "max"


class Float8LinearMixin(object):
    def __init__(self, *args, **kwargs):
        delayed_scaling_recipe = kwargs.pop("delayed_scaling_recipe", DelayedScalingRecipe())
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

    def cast_x_to_float8(self, x, is_amax_initialized):
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            x = x.to(torch.get_autocast_gpu_dtype())

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
        x_fp8 = to_float8(x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x)
        return x_fp8

    def cast_w_to_float8(self, w, is_amax_initialized):
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
        w_fp8 = to_float8(w, self.fp8_scale_w, torch.float8_e4m3fn, self.fp8_amax_w)
        return w_fp8

    def cast_y_to_float8_in_bw(self, y):
        scale_fn_name = self.recipe.scale_fn_name
        y = NoopFwToFloat8E5M2Bw.apply(
            y,
            self.fp8_amax_dL_dY,
            self.fp8_amax_history_dL_dY,
            self.fp8_scale_dL_dY,
            scale_fn_name,
            self.is_amax_initialized,
        )
        return y

    def float8_mm(self, x_fp8, w_fp8, is_amax_initialized):
        scale_fn_name = self.recipe.scale_fn_name
        y = float8_linear.apply(x_fp8, w_fp8, is_amax_initialized, scale_fn_name, self.emulate)
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
        y = self.float8_mm(x_fp8, w_fp8, self.is_amax_initialized)
        y = self.cast_y_to_float8_in_bw(y)

        if self.bias is not None:
            y = y + self.bias.to(x_fp8._orig_dtype)

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
        new_mod.emulate = emulate
        # I think its okay to send all params and buffers to device
        new_mod.to(mod.weight.device)
        new_mod.add_weight_tag()
        return new_mod

    def add_weight_tag(self):
        # We add a tag to the weight nn.Parameter in order to signal
        # To FSDP that this param is a weight
        self.weight._is_fp8_weight = True


def swap_linear_with_float8_linear(
    model,
    emulate=False,
    skip_fqn_list=None,
    cur_fqn="",
):
    """
    Replaces all instances of torch.nn.Linear in the given model with Float8Linear.

    Args:
        model (torch.nn.Module): The model to modify.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in float32.
        skip_fqn_list (List[str], optional): If specified, a list of FQNs to skip
        cur_fqn (str, optional): Current fqn, used to implement skip_fqn_list
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        new_fqn = name if cur_fqn == "" else f"{cur_fqn}.{name}"
        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and isinstance(
            child, torch.nn.Linear
        ):
            new_child = Float8Linear.from_float(child, emulate)
            setattr(model, name, new_child)
        else:
            swap_linear_with_float8_linear(child, emulate)


def sync_float8_amax_and_scale_history(model: torch.nn.Module) -> None:
    """
    Manages the float8 amax and scale bookkeeping. In detail, it does the
    following:
    1. in distributed contexts, syncs amax values across workers
    2. adds the `amax` values to history
    3. calculates the scales to be used for next iteration
    4. sets the `amax_and_scale_synced` flag on the Float8Linear modules
       to signal that they have been synced

    TODO(future): design the UX for this (context manager, etc)

    Args:
        model (torch.nn.Module): The model to track amaxes for
    """

    # For now, this is written in a naive way to maximize code readability.
    # TODO(future): benchmark and optimize as needed, we can combine all
    # the reductions into one and probably make the history update faster.

    for name, child in model.named_modules():
        if not isinstance(child, Float8Linear):
            continue

        #
        # 1. in distributed contexts, syncs amax values across workers
        #
        if dist.is_initialized():
            dist.all_reduce(child.fp8_amax_x, op=dist.ReduceOp.MAX)
            dist.all_reduce(child.fp8_amax_w, op=dist.ReduceOp.MAX)
            dist.all_reduce(child.fp8_amax_dL_dY, op=dist.ReduceOp.MAX)

        #
        # 2. adds the `amax` values to history
        #
        _update_history_with_new_amax(child.fp8_amax_x, child.fp8_amax_history_x)
        _update_history_with_new_amax(child.fp8_amax_w, child.fp8_amax_history_w)
        _update_history_with_new_amax(child.fp8_amax_dL_dY, child.fp8_amax_history_dL_dY)

        #
        # 3. calculate the scales
        #
        # TODO what to do with x_dtype
        x_dtype = child.last_seen_input_dtype
        new_scale = amax_history_to_scale(
            child.fp8_amax_history_x,
            torch.float8_e4m3fn,
            x_dtype,
            child.recipe.scale_fn_name,
        )
        child.fp8_scale_x.copy_(new_scale)
        new_scale = amax_history_to_scale(
            child.fp8_amax_history_w,
            torch.float8_e4m3fn,
            x_dtype,
            child.recipe.scale_fn_name,
        )
        child.fp8_scale_w.copy_(new_scale)
        new_scale = amax_history_to_scale(
            child.fp8_amax_history_dL_dY,
            torch.float8_e5m2,
            x_dtype,
            child.recipe.scale_fn_name,
        )
        child.fp8_scale_dL_dY.copy_(new_scale)

        #
        # 4. set a flag to signal amaxes/scales are ready
        #
        child.amax_and_scale_synced = True
