from typing import Any, List, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.testing._internal.common_fsdp import MLP
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class TestFloat8Common:
    def broadcast_module(self, module: nn.Module) -> None:
        # Broadcast for multi-threaded process group tests since seed is per
        # process, not per thread
        for param in module.parameters():
            dist.broadcast(param, src=0)

    def init_single_module(self) -> nn.Module:
        torch.manual_seed(42)
        module = nn.Linear(16, 16, device="cuda")
        self.broadcast_module(module)
        return module

    def init_multi_module(self) -> nn.Module:
        torch.manual_seed(42)
        module = nn.Sequential(*[MLP(16, device="cuda") for _ in range(3)])
        self.broadcast_module(module)
        return module

    def init_transformer(self, weight_tying: bool = False) -> nn.Module:
        torch.manual_seed(42)
        args = ModelArgs(
            n_layers=3, dim=768, n_heads=12, dropout_p=0.0, weight_tying=weight_tying
        )
        module = Transformer(args).cuda()
        self.broadcast_module(module)
        return module

    def get_local_inp(self, dtype: torch.dtype = torch.float32):
        torch.manual_seed(42)
        global_inp = torch.randn((16 * self.world_size, 16), device="cuda", dtype=dtype)
        dist.broadcast(global_inp, src=0)
        return global_inp.view(self.world_size, -1)[self.rank].view(16, 16)

    def swap_linear_with_dynamic(self, module: nn.Module, **kwargs: Any) -> nn.Module:
        if "use_activation_hooks" in kwargs:
            assert kwargs["use_activation_hooks"] is True
            del kwargs["use_activation_hooks"]
        # Always use activation hooks since we need it to compose with DTensor
        # tensor parallelism
        return swap_linear_with_float8_linear(
            module, Float8DynamicLinear, use_activation_hooks=True, **kwargs
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
    use_activation_hooks = use_activation_hooks or (module_cls is Float8DynamicLinear)
    return swap_linear_with_float8_linear(
        module,
        module_cls,
        use_activation_hooks=use_activation_hooks,
        use_fp8_all_gather=use_fp8_all_gather,
    )


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
