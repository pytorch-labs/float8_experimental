# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
from enum import auto, Enum
from typing import Callable, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_linear import Float8Linear, TensorScalingType

from float8_experimental.float8_utils import (
    amax_history_to_scale_stack,
    e4m3_dtype,
    e5m2_dtype,
)
from torch.distributed._functional_collectives import all_reduce, AsyncCollectiveTensor

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def linear_requires_sync(
    scaling_type_x: TensorScalingType = TensorScalingType.DELAYED,
    scaling_type_w: TensorScalingType = TensorScalingType.DELAYED,
    scaling_type_dL_dY: TensorScalingType = TensorScalingType.DELAYED,
):
    """Returns whether the given linear_type requires sync before forward."""
    return any(
        [
            scaling_type_x is TensorScalingType.DELAYED,
            scaling_type_w is TensorScalingType.DELAYED,
            scaling_type_dL_dY is TensorScalingType.DELAYED,
        ]
    )


def _update_history_stack(
    new_amax: torch.Tensor, amax_history_stack: torch.Tensor
) -> torch.Tensor:
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.

    Args:
        new_amax (torch.Tensor): The new amax value to add to the history. (n_amaxes, 1)
        amax_history_stack (torch.Tensor): The history of amax values. (n_amaxes, history_length)
    """
    assert (
        amax_history_stack.dim() == 2
    ), f"Expected amat_history_stack to be 2D, got {amax_history_stack.shape()}"
    assert new_amax.size(0) == amax_history_stack.size(
        0
    ), f"Expected new_amax to have the same size as the first dimension of amax_history_stack, got {new_amax.size(0)} and {amax_history_stack.size(0)}"
    new_amax_history_stack = torch.roll(amax_history_stack, 1, dims=1)
    new_amax_history_stack[:, 0] = new_amax.squeeze(-1)
    amax_history_stack.copy_(new_amax_history_stack)


def filter_out_small_unaligned_layers(size_limit: int) -> Callable[[nn.Linear], bool]:
    """
    Returns a callable that filters out small (dimensions less than the given `size_limit`)
        and unaligned (dimenstions not divisible by 16) layers.
    It can be passed as the `linear_layer_filter` argument to `swap_linear_with_float8_linear`.
    """
    return (
        lambda linear_layer: linear_layer.in_features >= size_limit
        and linear_layer.out_features >= size_limit
        and linear_layer.in_features % 16 == 0
        and linear_layer.out_features % 16 == 0
    )


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    skip_fqn_list: Optional[List[str]] = None,
    linear_layer_filter: Optional[Callable[[nn.Linear], bool]] = None,
) -> Optional[nn.Module]:
    """
    Generic function to swap linear layers in a module with a new type of linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        skip_fqn_list: If specified, a list of module FQNs to skip.
        linear_layer_filter: If specified, only the linear layers
            that pass the filter function will be swapped.
        from_float_kwargs: Additional keyword arguments for from_float_func.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    module_names_to_skip = set(skip_fqn_list or [])

    if isinstance(module, nn.Linear) and (
        linear_layer_filter is None or linear_layer_filter(module)
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return from_float_func(
            module,
        )

    root_module = module
    visited_modules = {root_module}

    for module_name, module in root_module.named_modules():
        if module_name in module_names_to_skip:
            visited_modules.add(module)

    def post_order_traversal(
        module: nn.Module, module_name: str, parent_module: Optional[nn.Module]
    ):
        nonlocal visited_modules
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                post_order_traversal(child_module, child_module_name, module)

        if isinstance(module, nn.Linear) and (
            linear_layer_filter is None or linear_layer_filter(module)
        ):
            assert (
                parent_module is not None
            ), f"Linear root module should return early: {module}"
            new_linear_module = from_float_func(module)
            setattr(parent_module, module_name, new_linear_module)

    post_order_traversal(root_module, "", None)
    # Without this explicit `del`, this set only gets deleted upon an explicit
    # garbage collection (not from when its refcount hits zero)
    del visited_modules
    return root_module


def swap_linear_with_float8_linear(
    module: nn.Module,
    *,
    skip_fqn_list: Optional[List[str]] = None,
    emulate: bool = False,
    linear_layer_filter: Optional[Callable[[nn.Linear], bool]] = None,
    scaling_type_x: TensorScalingType = TensorScalingType.DYNAMIC,
    scaling_type_w: TensorScalingType = TensorScalingType.DYNAMIC,
    scaling_type_dL_dY: TensorScalingType = TensorScalingType.DYNAMIC,
    static_scale_x: Optional[float] = None,
) -> Optional[nn.Module]:
    """
    Swaps `torch.nn.Linear` in `module` with `Float8Linear`.

    Args:
        module: Module to modify.
        skip_fqn_list: If specified, a list of module FQNs to skip.
        emulate: If True, emulation is used instead of hardware accelerated gemm
        linear_layer_filter: If specified, only the linear layers
            that pass the filter function will be swapped.
        scaling_type_x (TensorScalingType): scaling type for `x`
        scaling_type_w (TensorScalingType): scaling type for `w`
        scaling_type_dL_dY (TensorScalingType): scaling type for `dL_dY`
        static_scale_x: static scale for `x`

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from_float = lambda m: Float8Linear.from_float(
        m,
        emulate=emulate,
        scaling_type_x=scaling_type_x,
        scaling_type_w=scaling_type_w,
        scaling_type_dL_dY=scaling_type_dL_dY,
        static_scale_x=static_scale_x,
    )
    return swap_linear_layers(
        module,
        from_float,
        skip_fqn_list=skip_fqn_list,
        linear_layer_filter=linear_layer_filter,
    )


def get_float8_layers(model: torch.nn.Module):
    """Iterates through the model and returns all the Float8Linear layers.
    Args:
        model (torch.nn.Module): The model to look for Float8Linear layers in.
    """

    # Get all fp8 layers and tensors
    fp8_layers = [child for child in model.modules() if isinstance(child, Float8Linear)]
    if not torch._dynamo.is_compiling():
        for layer in fp8_layers:
            for buf in layer.buffers():
                torch._dynamo.mark_static_address(buf, guard=True)
    return fp8_layers


@torch.no_grad()
def sync_float8_amax_and_scale_history(model: torch.nn.Module, fp8_layers=None) -> None:
    """
    Manages the float8 amax and scale bookkeeping. In detail, it does the
    following:
    1. in distributed contexts, syncs amax values across workers for activations and gradients
    2. adds the `amax` values to history
    3. calculates the scales to be used for next iteration
    4. sets the `amax_and_scale_synced` flag on the Float8Linear modules
       to signal that they have been synced

    TODO(future): design the UX for this (context manager, etc)

    PERFORMANCE NOTE:
        When you can, it is much more efficient to call get_float8_layers once at
        the beginning of the training loop and pass the result to this function.
        Because of how this interacts with torch.compile

    Args:
        model (torch.nn.Module): The model to track amaxes for
        fp8_layers (optional): If fp8_layers are provided, fp8_classes are ignored,
            and we loop over all fp8_layers to sync and update amax scale histories.
            Users can use get_float8_layers to get all fp8 layers.
    """
    if fp8_layers is None:
        fp8_layers = get_float8_layers(model)

    if len(fp8_layers) == 0:
        log.warn(
            "Calling sync_float8_amax_and_scale_history on a module with no Float8Linear layers"
        )
        return

    def inner_func():
        """Why do we have this inner_function?

        There are two portions of the outer sync_function that cause graph_breaks:
            1. The `get_float8_layers` call can cause graph breaks if the user did not pass
                in the fp8_layers.
            2. At the end of syncing all the amaxes and scales we set the attr on the module
                signaling that we have synced the amaxes and scales and the next forward can be run.
                # TODO Maybe we should remove this safety check to remove the graph break?

        By having this inner function, we can ensure that although the outer function may cause graph breaks
        the inner function will not.
        """
        # Loop over all fp8 layers and grab the needed tensors
        fp8_amax_x_tensor_list = [None] * len(fp8_layers)
        fp8_amax_w_tensor_list = [None] * len(fp8_layers)
        fp8_amax_dL_dY_tensor_list = [None] * len(fp8_layers)

        fp8_x_amax_history_stack = [None] * len(fp8_layers)
        fp8_w_amax_history_stack = [None] * len(fp8_layers)
        fp8_dL_dY_amax_history_stack = [None] * len(fp8_layers)

        x_dtypes = set()
        scale_fn_recipes = set()

        for idx, child in enumerate(fp8_layers):
            fp8_amax_x_tensor_list[idx] = child.fp8_amax_x
            fp8_amax_w_tensor_list[idx] = child.fp8_amax_w
            fp8_amax_dL_dY_tensor_list[idx] = child.fp8_amax_dL_dY

            fp8_x_amax_history_stack[idx] = child.fp8_amax_history_x
            fp8_w_amax_history_stack[idx] = child.fp8_amax_history_w
            fp8_dL_dY_amax_history_stack[idx] = child.fp8_amax_history_dL_dY

            x_dtypes.add(child.last_seen_input_dtype)
            scale_fn_recipes.add(child.recipe.scale_fn_name)

        # TODO This way to get the activation dtype is not ideal
        if len(x_dtypes) != 1:
            raise ValueError(
                f"All layers must have the same last seen input_dtype, got {x_dtypes}"
            )
        x_dtype = next(iter(x_dtypes))

        if len(scale_fn_recipes) != 1:
            raise ValueError(
                f"All layers must have the same scale_fn recipe, got {scale_fn_recipes}"
            )
        scale_fn_recipe = next(iter(scale_fn_recipes))

        assert (
            len(fp8_amax_x_tensor_list)
            == len(fp8_amax_w_tensor_list)
            == len(fp8_amax_dL_dY_tensor_list)
        ), "Mismatched lengths of amax tensors."

        if dist.is_initialized():
            # Combine all the amax tensors into one tensor and reduce it
            # Note: do not reduce the weight values, because FSDP already ensures
            # the weight values on all ranks are the same after all-gather.
            all_amax_tensors = torch.cat(
                fp8_amax_x_tensor_list + fp8_amax_dL_dY_tensor_list
            )
            all_reduced_amax_tensor = all_reduce(
                all_amax_tensors, "MAX", list(range(dist.get_world_size()))
            )
            if isinstance(all_reduced_amax_tensor, AsyncCollectiveTensor):
                all_reduced_amax_tensor = all_reduced_amax_tensor.wait()

            (
                reduced_fp8_amax_tensor,
                reduced_fp8_amax_dL_dY_tensor,
            ) = torch.split(all_reduced_amax_tensor, len(fp8_amax_x_tensor_list))

            for idx, child in enumerate(fp8_layers):
                child.fp8_amax_x.copy_(reduced_fp8_amax_tensor[idx])
                child.fp8_amax_dL_dY.copy_(reduced_fp8_amax_dL_dY_tensor[idx])

        # We create two stacked tensor groups, one for the amax history and one for the current scales
        fp8_amax_x_tensors = torch.vstack(fp8_amax_x_tensor_list)
        fp8_amax_w_tensors = torch.vstack(fp8_amax_w_tensor_list)
        fp8_amax_dL_dY_tensors = torch.vstack(fp8_amax_dL_dY_tensor_list)

        fp8_x_amax_history_stack = torch.vstack(fp8_x_amax_history_stack)
        fp8_w_amax_history_stack = torch.vstack(fp8_w_amax_history_stack)
        fp8_dL_dY_amax_history_stack = torch.vstack(fp8_dL_dY_amax_history_stack)

        # Update the history stacks with the new amax values
        _update_history_stack(fp8_amax_x_tensors, fp8_x_amax_history_stack)
        _update_history_stack(fp8_amax_w_tensors, fp8_w_amax_history_stack)
        _update_history_stack(fp8_amax_dL_dY_tensors, fp8_dL_dY_amax_history_stack)

        # Calculate the new scales from the updated history stacks
        new_x_scales = amax_history_to_scale_stack(
            fp8_x_amax_history_stack, e4m3_dtype, x_dtype, scale_fn_recipe
        )
        new_w_scales = amax_history_to_scale_stack(
            fp8_w_amax_history_stack, e4m3_dtype, x_dtype, scale_fn_recipe
        )
        new_dL_dY_scales = amax_history_to_scale_stack(
            fp8_dL_dY_amax_history_stack, e5m2_dtype, x_dtype, scale_fn_recipe
        )

        # Iterate through the layers and update the scales
        for idx, child in enumerate(fp8_layers):
            child.fp8_scale_x.copy_(new_x_scales[idx])
            child.fp8_scale_w.copy_(new_w_scales[idx])
            child.fp8_scale_dL_dY.copy_(new_dL_dY_scales[idx])

    # This allows for the compile to succede on the inner func and fail on the graph breaks
    # at the beginning and and of syncing
    inner_func()

    for child in fp8_layers:
        # Set a flag to signal amaxes/scales are ready
        child.amax_and_scale_synced = True
