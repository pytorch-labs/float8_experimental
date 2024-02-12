# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
from enum import auto, Enum
from typing import List, Optional, Type

import float8_experimental.config as fp8_config

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear

from float8_experimental.float8_utils import amax_history_to_scale


class LinearType(Enum):
    DELAYED = auto()
    DYNAMIC = auto()


REQUIRES_SYNC = {LinearType.DELAYED}


def get_float8_linear(
    linear_type: LinearType,
    linear_ref: torch.nn.Linear,
    emulate: bool = False,
    use_activation_hooks: bool = False,
):
    """Returns a Float8Linear module of the given type, initialized from linear_ref.
    Args:
        linear_type: The type of Float8Linear to return.
        linear_ref: The linear module to initialize from.
        emulate: Whether to emulate the fp8 matmul logic in float32.
        use_activation_hooks: Whether to use activation hooks for dynamic linear.
    """
    LINEAR_TYPE_MAP = {
        LinearType.DELAYED: Float8Linear,
        LinearType.DYNAMIC: Float8DynamicLinear,
    }
    if linear_type not in LINEAR_TYPE_MAP:
        raise ValueError(f"linear_type must be one of {LINEAR_TYPE_MAP.keys()}")
    if use_activation_hooks and linear_type != LinearType.DYNAMIC:
        raise ValueError("use_activation_hooks is only supported for dynamic linear")
    return LINEAR_TYPE_MAP[linear_type].from_float(
        copy.deepcopy(linear_ref),
        emulate=emulate,
        use_activation_hooks=use_activation_hooks,
    )


def linear_requires_sync(linear_type: LinearType):
    """Returns whether the given linear_type requires sync before forward."""
    return linear_type in REQUIRES_SYNC


def _update_history_with_new_amax(new_amax, amax_history):
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.
    """
    new_amax_history = torch.roll(amax_history, 1)
    new_amax_history[0] = new_amax
    amax_history.copy_(new_amax_history)


def swap_linear_with_float8_linear(
    module: nn.Module,
    module_cls: Type[nn.Module],
    emulate: bool = False,
    skip_fqn_list: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replaces all instances of ``torch.nn.Linear`` in ``module`` with instances
    of ``module_cls`` (either ``Float8Linear`` or ``Float8DynamicLinear``).

    Args:
        module (torch.nn.Module): Module to modify.
        module_cls (Union[Type[Float8Linear], Type[Float8DynamicLinear]]): Float8 linear class for the swap.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in fp32.
        skip_fqn_list (List[str], optional): If specified, a list of module FQNs to skip.
            Linear submodules of these skipped modules will also be skipped.
    """
    module_names_to_skip = set(skip_fqn_list or [])
    if isinstance(module, nn.Linear):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return module_cls.from_float(module, emulate)

    # Mark all modules to skip as visited
    root_module = module
    visited_modules = {root_module}
    for module_name, module in root_module.named_modules():
        if module_name in module_names_to_skip:
            visited_modules.add(module)

    # Run a post-order traversal to swap linears
    def post_order_traversal(
        module: nn.Module, module_name: str, parent_module: Optional[nn.Module]
    ):
        nonlocal visited_modules
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                post_order_traversal(child_module, child_module_name, module)
        if isinstance(module, nn.Linear):
            assert (
                parent_module is not None
            ), f"Linear root module should return early: {module}"
            setattr(parent_module, module_name, module_cls.from_float(module, emulate))

    post_order_traversal(root_module, "", None)
    # Without this explicit `del`, this set only gets deleted upon an explicit
    # garbage collection (not from when its refcount hits zero)
    del visited_modules
    return root_module


def get_float8_layers(model: torch.nn.Module):
    """Iterates through the model and returns all the Float8Linear layers.
    Args:
        model (torch.nn.Module): The model to look for Float8Linear layers in.
    """

    # Get all fp8 layers and tensors
    fp8_layers = [
        child
        for name, child in model.named_modules()
        if isinstance(child, Float8Linear)
    ]

    return fp8_layers


def sync_float8_amax_and_scale_history(model: torch.nn.Module, fp8_layers=None) -> None:
    """
    Manages the float8 amax and scale bookkeeping. In detail, it does the
    following:
    1. in distributed contexts, syncs amax values across workers
    2. adds the `amax` values to history
    3. calculates the scales to be used for next iteration
    4. sets the `amax_and_scale_synced` flag on the Float8Linear modules
       to signal that they have been synced

    TODO(future): design the UX for this (context manager, etc)

    PERFORMANCE NOTE:
        When you can it is much more efficient to call te get_float8_layers once a
        the beginning of the training loop and pass the result to this function.
        Because of how this interacts with torch.compile

    Args:
        model (torch.nn.Module): The model to track amaxes for
        fp8_layers (optional): If fp8_layers are provided, fp8_classes are ignored,
            and we loop over all fp8_layers to sync and update amax scale histories.
            Users can use get_float8_layers to get all fp8 layers.
    """

    # For now, this is written in a naive way to maximize code readability.
    # TODO(future): benchmark and optimize as needed, we have combined all
    # the reductions into one and we can probably try other optimizatons to
    # make the history update faster.

    if fp8_layers is None:
        fp8_layers = get_float8_layers(model)

    if dist.is_initialized():
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
        dist.all_reduce(fp8_amax_x_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(fp8_amax_w_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(fp8_amax_dL_dY_tensor, op=dist.ReduceOp.MAX)

    for idx in range(len(fp8_layers)):
        child = fp8_layers[idx]

        #
        # 1. in distributed contexts, syncs amax values across workers
        #
        if dist.is_initialized():
            child.fp8_amax_x = fp8_amax_x_tensor[idx].clone()
            child.fp8_amax_w = fp8_amax_w_tensor[idx].clone()
            child.fp8_amax_dL_dY = fp8_amax_dL_dY_tensor[idx].clone()

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
        # We only update the flag if we know it will be checked by the modules
        if fp8_config.enable_amax_init:
            child.amax_and_scale_synced = True
