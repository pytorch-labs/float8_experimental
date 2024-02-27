import contextlib

from typing import List, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental import config
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


def init_transformer_with_fp8(
    module_cls: Type,
    *,
    checkpoint_activations: bool = False,
    use_activation_hooks: Optional[bool] = None,
    use_fp8_all_gather: bool = False,
):
    torch.manual_seed(42)
    args = ModelArgs(
        n_layers=3,
        dim=768,
        n_heads=12,
        dropout_p=0.0,
        weight_tying=False,
        checkpoint_activations=checkpoint_activations,
    )
    module = Transformer(args)
    # Only dynamic linear supports activation hooks
    use_hooks = use_activation_hooks or (module_cls is Float8DynamicLinear)
    return swap_linear_with_float8_linear(
        module, module_cls, emulate=True, use_activation_hooks=use_hooks
    )


@contextlib.contextmanager
def enable_amax_init(enable: bool):
    prev_value = config.enable_amax_init
    config.enable_amax_init = enable
    try:
        yield
    finally:
        config.enable_amax_init = prev_value


@contextlib.contextmanager
def enable_pre_and_post_forward(enable: bool):
    prev_value = config.enable_pre_and_post_forward
    config.enable_pre_and_post_forward = enable
    try:
        yield
    finally:
        config.enable_pre_and_post_forward = prev_value


def check_parity_no_mp(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    module_cls: Type,
):
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model:
                for param in model.parameters():
                    dist.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())
            if module_cls is Float8Linear:
                sync_float8_amax_and_scale_history(model)
            optim.step()
        test_cls.assertEqual(losses[0], losses[1])


def check_parity_bf16_mp(
    test_cls,
    ref_model: nn.Module,
    ref_model_bf16: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    module_cls: Type,
):
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in (
            (ref_model_bf16, ref_optim),
            (fsdp_model, fsdp_optim),
        ):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model_bf16:
                for param_bf16, param_fp32 in zip(
                    ref_model_bf16.parameters(), ref_model.parameters()
                ):
                    dist.all_reduce(param_bf16.grad)
                    param_bf16.grad.div_(dist.get_world_size())
                    param_fp32.grad = param_bf16.grad.float()
                    param_bf16.grad = None
            if module_cls is Float8Linear:
                sync_float8_amax_and_scale_history(model)
            optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
        test_cls.assertEqual(losses[0], losses[1])
