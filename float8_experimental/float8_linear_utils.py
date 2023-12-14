# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
from enum import auto, Enum

import torch
import torch.distributed as dist

from float8_experimental.float8_utils import amax_history_to_scale, tensor_to_amax


class LinearType(Enum):
    DELAYED = auto()
    DYNAMIC = auto()


REQUIRES_SYNC = {LinearType.DELAYED}


def get_float8_linear(
    linear_type: LinearType, linear_ref: torch.nn.Linear, emulate: bool = False
):
    """Returns a Float8Linear module of the given type, initialized from linear_ref.
    Args:
        linear_type: The type of Float8Linear to return.
        linear_ref: The linear module to initialize from.
        emulate: Whether to emulate the fp8 matmul logic in float32.
    """
    from float8_experimental.dynamic_linear import Float8DynamicLinear

    # Lazy import to avoid circular dependency
    from float8_experimental.float8_linear import Float8Linear

    LINEAR_TYPE_MAP = {
        LinearType.DELAYED: Float8Linear,
        LinearType.DYNAMIC: Float8DynamicLinear,
    }
    if linear_type not in LINEAR_TYPE_MAP:
        raise ValueError(f"linear_type must be one of {LINEAR_TYPE_MAP.keys()}")

    return LINEAR_TYPE_MAP[linear_type].from_float(
        copy.deepcopy(linear_ref), emulate=emulate
    )


def linear_requires_sync(linear_type: LinearType):
    """Returns whether the given linear_type requires sync before forward."""
    return linear_type in REQUIRES_SYNC


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


def _update_history_with_new_amax(new_amax, amax_history):
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.
    """
    new_amax_history = torch.roll(amax_history, 1)
    new_amax_history[0] = new_amax
    amax_history.copy_(new_amax_history)


def swap_linear_with_float8_linear(
    model,
    module,
    emulate=False,
    skip_fqn_list=None,
    cur_fqn="",
):
    """
    Replaces all instances of torch.nn.Linear in the given model with module.

    Args:
        model (torch.nn.Module): The model to modify.
        module (Float8Linear): The Float8Linear module to use.
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
            new_child = module.from_float(child, emulate)
            setattr(model, name, new_child)
        else:
            swap_linear_with_float8_linear(child, module, emulate)


def get_float8_layers(model: torch.nn.Module, fp8_classes=None):
    if fp8_classes is None:
        # Lazy import to avoid circular dependency
        from float8_experimental.float8_linear import Float8Linear

        fp8_classes = Float8Linear

    # Get all fp8 layers and tensors
    fp8_layers = [
        child for name, child in model.named_modules() if isinstance(child, fp8_classes)
    ]

    return fp8_layers

def sync_float8_amax_and_scale_history(
    model: torch.nn.Module, fp8_layers, combine_reduction = False
) -> None:
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

    if dist.is_initialized():
        if combine_reduction:
            fp8_amax_x_tensor = torch.tensor(
                [child.fp8_amax_x for child in fp8_layers],
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
            fp8_amax_w_tensor = torch.tensor(
                [child.fp8_amax_w for child in fp8_layers],
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
            fp8_amax_dL_dY_tensor = torch.tensor(
                [child.fp8_amax_dL_dY for child in fp8_layers],
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
            # print("fp8_amax_x_tensor, ", fp8_amax_x_tensor)
            dist.all_reduce(fp8_amax_x_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(fp8_amax_x_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(fp8_amax_x_tensor, op=dist.ReduceOp.MAX)

    for idx in range(len(fp8_layers)):
        child = fp8_layers[idx]
        #
        # 1. in distributed contexts, syncs amax values across workers
        #
        if dist.is_initialized():
            if combine_reduction:
                # TODO: There are errors if no "clone()", need to figure out.
                child.fp8_amax_x = fp8_amax_x_tensor[idx].clone()
                child.fp8_amax_w = fp8_amax_w_tensor[idx].clone()
                child.fp8_amax_dL_dY = fp8_amax_dL_dY_tensor[idx].clone()
            else:
                dist.all_reduce(child.fp8_amax_x, op=dist.ReduceOp.MAX)
                dist.all_reduce(child.fp8_amax_w, op=dist.ReduceOp.MAX)
                dist.all_reduce(child.fp8_amax_dL_dY, op=dist.ReduceOp.MAX)

        #
        # 2. adds the `amax` values to history
        #
        _update_history_with_new_amax(child.fp8_amax_x, child.fp8_amax_history_x)
        _update_history_with_new_amax(child.fp8_amax_w, child.fp8_amax_history_w)
        _update_history_with_new_amax(
            child.fp8_amax_dL_dY, child.fp8_amax_history_dL_dY
        )

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
