import copy
import gc
import itertools
import threading
import unittest
from typing import Any, List, Type

import torch
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import (
    Float8DynamicLinear,
    Float8DynamicLinearWeightTensor,
)
from float8_experimental.float8_linear import Float8Linear, Float8LinearWeightTensor
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_1d_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_all_gather,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFloat8Common:
    def _broadcast_module(self, module: nn.Module) -> None:
        # Broadcast for multi-threaded process group tests since seed is per
        # process, not per thread
        for param in module.parameters():
            dist.broadcast(param, src=0)

    def _init_single_module(self) -> nn.Module:
        torch.manual_seed(42)
        module = nn.Linear(16, 16, device="cuda")
        self._broadcast_module(module)
        return module

    def _init_multi_module(self) -> nn.Module:
        torch.manual_seed(42)
        module = nn.Sequential(*[MLP(16, device="cuda") for _ in range(3)])
        self._broadcast_module(module)
        return module

    def _init_transformer(self, weight_tying: bool) -> nn.Module:
        torch.manual_seed(42)
        args = ModelArgs(
            n_layers=3, dim=768, n_heads=12, dropout_p=0.0, weight_tying=weight_tying
        )
        module = Transformer(args).cuda()
        self._broadcast_module(module)
        return module

    def _get_local_inp(self, dtype: torch.dtype = torch.float32):
        torch.manual_seed(42)
        global_inp = torch.randn((2 * self.world_size, 16), device="cuda", dtype=dtype)
        dist.broadcast(global_inp, src=0)
        return global_inp.view(self.world_size, -1)[self.rank].view(2, 16)

    def _check_parity_fp32(
        self,
        ref_module: nn.Module,
        ref_optim: torch.optim.Optimizer,
        module: nn.Module,
        optim: torch.optim.Optimizer,
        inp: torch.Tensor,
        module_cls: Type,
    ):
        for iter_idx in range(10):
            if self.rank == 0:
                print(f"Iter {iter_idx}")
            losses: List[torch.Tensor] = []
            for _module, _optim in ((ref_module, ref_optim), (module, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_module(inp).sum())
                losses[-1].backward()
                if _module is ref_module:  # manually reduce for data parallelism
                    for param in ref_module.parameters():
                        dist.all_reduce(param.grad)
                        param.grad.div_(self.world_size)
                if module_cls is Float8Linear:
                    sync_float8_amax_and_scale_history(_module)
                _optim.step()
            self.assertEqual(losses[0], losses[1])
            check_1d_sharded_parity(self, ref_module, module)

    def _check_parity_bf16_mp(
        self,
        ref_module_fp32: nn.Module,
        ref_module_bf16: nn.Module,
        ref_optim: torch.optim.Optimizer,
        module: nn.Module,
        optim: torch.optim.Optimizer,
        inp: torch.Tensor,
    ):
        for iter_idx in range(10):
            if self.rank == 0:
                print(f"Iter {iter_idx}")
            losses: List[torch.Tensor] = []
            for _module in (ref_module_bf16, module):
                losses.append(_module(inp).sum())
                losses[-1].backward()
            self.assertEqual(losses[0], losses[1])
            for param_fp32, param_bf16 in zip(
                ref_module_fp32.parameters(), ref_module_bf16.parameters()
            ):
                dist.all_reduce(param_bf16.grad)
                param_bf16.grad.div_(self.world_size)
                param_fp32.grad = param_bf16.grad.float()
                param_bf16.grad = None
            ref_optim.step()
            optim.step()
            check_1d_sharded_parity(self, ref_module_fp32, module)
            for param_fp32, param_bf16 in zip(
                ref_module_fp32.parameters(), ref_module_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
            ref_optim.zero_grad()
            optim.zero_grad()


class TestFloat8MultiProcess(FSDPTest, TestFloat8Common):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity(self):
        for use_fp8_all_gather, module_cls in itertools.product(
            [False, True], [Float8DynamicLinear, Float8Linear]
        ):
            self._test_transformer_fp32_parity(
                use_fp8_all_gather=use_fp8_all_gather,
                module_cls=module_cls,
            )

    def _test_transformer_fp32_parity(self, use_fp8_all_gather: bool, module_cls: Type):
        if self.rank == 0:
            print(module_cls, use_fp8_all_gather)
        use_activation_hooks = module_cls is Float8DynamicLinear
        # NOTE: Weight-tying does not compose with fp8 all-gather because the
        # embedding weight and output linear weight are tied but only the
        # latter uses fp8 compute. With fp8 all-gather, FSDP would pre-cast to
        # fp8 for that tied weight, incorrectly using fp8 for the embedding.
        weight_tying = not use_fp8_all_gather
        module = self._init_transformer(weight_tying=weight_tying)
        ref_module = copy.deepcopy(module)
        ref_module = swap_linear_with_float8_linear(
            ref_module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
        )
        ref_module = ref_module.cuda()
        module = swap_linear_with_float8_linear(
            module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=use_fp8_all_gather,
        )
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                fully_shard(submodule)
        fully_shard(module)
        ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = torch.randint(
            0, ref_module.tok_embeddings.weight.size(0), (1, 4), device="cuda"
        )
        self._check_parity_fp32(
            ref_module, ref_optim, module, optim, local_inp, module_cls
        )

    @skip_if_lt_x_gpu(2)
    def test_transformer_memory(self):
        """Tests peak active memory in the forward and backward passes."""
        for use_fp8_all_gather, module_cls in itertools.product(
            [False, True], [Float8DynamicLinear, Float8Linear]
        ):
            self._test_transformer_memory(module_cls, use_fp8_all_gather)

    def _test_transformer_memory(self, module_cls: Type, use_fp8_all_gather: bool):
        """Based on test_fully_shard_memory.py"""
        if self.rank == 0:
            print(module_cls, use_fp8_all_gather)
        torch.manual_seed(42)
        # Pre-run a linear forward (gemm and bias) and backward (gemm) to
        # allocate the cuBLAS workspaces before measuring the memory usage
        # since the workspace size can differ between hardwares
        lin = torch.nn.Linear(768, 768, device="cuda")
        inp = torch.randn(1, 768, device="cuda")
        lin(inp).sum().backward()
        torch.cuda.empty_cache()
        base_mem_mb = self._get_peak_active_memory_mb()
        vocab_size = 32
        model_args = ModelArgs(
            vocab_size=vocab_size,
            n_layers=3,
            dim=768,
            n_heads=12,
            weight_tying=False,
        )
        model = Transformer(model_args)
        model = swap_linear_with_float8_linear(
            model,
            module_cls,
            emulate=True,
            use_activation_hooks=(module_cls is Float8DynamicLinear),
            use_fp8_all_gather=use_fp8_all_gather,
        )
        model_unsharded_numel = sum(p.numel() for p in model.parameters())
        model_sharded_numel = (model_unsharded_numel + 1) // 2
        block_lin_weight_numel = 0
        block_other_numel = 0
        for module in model.layers[0].modules():
            for param in module.parameters(recurse=False):
                if isinstance(module, nn.Linear):
                    block_lin_weight_numel += param.numel()
                else:
                    block_other_numel += param.numel()
        non_block_numel = round(
            sum(p.numel() for p in model.tok_embeddings.parameters())
            + sum(p.numel() for p in model.pos_embeddings.parameters())
            + sum(p.numel() for p in model.norm.parameters())
            + sum(p.numel() for p in model.output.parameters())
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)

        # Init: Each module is moved to GPU before sharding parameters
        peak_mem_mb = self._get_peak_active_memory_mb()
        curr_mem_mb = self._get_curr_active_memory_mb()
        init_mem_mb = (
            (model_sharded_numel + block_lin_weight_numel + block_other_numel) * 4 / 1e6
        )
        # Allow for some buffer for the peak memory since original parameters
        # are not freed until a `fully_shard` call returns
        buffer_mb = 4
        self.assertLessEqual(peak_mem_mb - base_mem_mb, init_mem_mb + buffer_mb)
        self.assertLessEqual(curr_mem_mb - base_mem_mb, init_mem_mb)

        # Use a small input to minimize activation memory usage
        inp = torch.randint(0, vocab_size, (1, 4), device="cuda")

        # Forward:
        loss = model(inp)
        mem_mb = self._get_peak_active_memory_mb()
        # Allow for some buffer for fragmentation/activations (where this
        # number is kept much smaller than the actual memory usage, which is on
        # the order of 100-200+ MB)
        buffer_mb = 16
        if use_fp8_all_gather:
            # Non-block parameters (fp32), 3x block non-linear-weight
            # parameters (fp32) and block linear-weight parameters (fp8)
            # (current all-gather, copy-out, and next all-gather), and other
            expected_mem_mb = (
                (non_block_numel * 4)
                + 3 * (block_lin_weight_numel + block_other_numel * 4)
            ) / 1e6 + buffer_mb
        else:
            # Non-block parameters (fp32), 3x block parameters (fp32)
            # (current all-gather, copy-out, and next all-gather), Nx block
            # linear-weight parameters (fp8) for N blocks (saved by autograd),
            # and other
            expected_mem_mb = (
                (non_block_numel + 3 * (block_lin_weight_numel + block_other_numel)) * 4
                + model_args.n_layers * block_lin_weight_numel
            ) / 1e6 + buffer_mb
        # Sharded parameters
        expected_mem_mb += model_sharded_numel * 4 / 1e6
        self.assertLessEqual(mem_mb, expected_mem_mb + base_mem_mb)

        # Backward:
        loss.sum().backward()
        mem_mb = self._get_peak_active_memory_mb()
        if use_fp8_all_gather:
            # Non-block parameters (fp32), 2x block non-linear weight
            # parameters (fp32) and block linear-weight parameters (fp8)
            # (current copy-out and next all-gather), 1x block gradients (fp32)
            expected_mem_mb = (
                (non_block_numel * 4)
                + 2 * (block_lin_weight_numel + block_other_numel * 4)
                + 1 * (block_lin_weight_numel + block_other_numel) * 4
            ) / 1e6 + buffer_mb
        else:
            # Non-block parameters (fp32), 3x block parameters (fp32) (current
            # copy-out, next all-gather, current gradients)
            expected_mem_mb = (
                non_block_numel + 3 * (block_lin_weight_numel + block_other_numel) * 4
            ) * 4 / 1e6 + buffer_mb
        # 2x sharded parameters/gradients
        expected_mem_mb += 2 * model_sharded_numel * 4 / 1e6
        self.assertLessEqual(mem_mb, expected_mem_mb + base_mem_mb)

    def _get_peak_active_memory_mb(self) -> int:
        mem_stats = torch.cuda.memory_stats()
        return round(mem_stats["active_bytes.all.peak"] / 1e6)

    def _get_curr_active_memory_mb(self) -> int:
        mem_stats = torch.cuda.memory_stats()
        return round(mem_stats["active_bytes.all.current"] / 1e6)


class TestFloat8MultiThread(FSDPTestMultiThread, TestFloat8Common):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_weight_subclass_delayed(self):
        self._test_weight_subclass(Float8Linear)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_weight_subclass_dynamic(self):
        self._test_weight_subclass(Float8DynamicLinear)

    def _test_weight_subclass(self, module_cls: Type):
        tensor_cls = (
            Float8DynamicLinearWeightTensor
            if module_cls is Float8DynamicLinear
            else Float8LinearWeightTensor
        )
        use_activation_hooks = module_cls is Float8DynamicLinear
        # Check for a single FSDP paramter group
        module_fp32 = self._init_single_module()
        module = swap_linear_with_float8_linear(
            module_fp32,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=True,
        )
        self.assertIsInstance(module.weight, tensor_cls)
        fully_shard(module)
        for param_name, param in module.named_parameters():
            self.assertIsInstance(param, DTensor)
            if "weight" in param_name:
                self.assertIsInstance(param._local_tensor, tensor_cls)

        # Check for multiple FSDP paramter groups
        module = self._init_multi_module()
        module = swap_linear_with_float8_linear(
            module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=True,
        )
        for param_name, param in module.named_parameters():
            if "weight" in param_name:
                self.assertIsInstance(param, tensor_cls)
        for mlp in module:
            fully_shard(mlp)
        fully_shard(module)
        for param_name, param in module.named_parameters():
            self.assertIsInstance(param, DTensor)
            if "weight" in param_name:
                self.assertIsInstance(param._local_tensor, tensor_cls)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fp8_fp32_all_gather_delayed_comm_size(self):
        self._test_fp8_fp32_all_gather_comm_size(Float8Linear)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fp8_fp32_all_gather_dynamic_comm_size(self):
        self._test_fp8_fp32_all_gather_comm_size(Float8DynamicLinear)

    def _test_fp8_fp32_all_gather_comm_size(self, module_cls: Type):
        use_activation_hooks = module_cls is Float8DynamicLinear
        orig_all_gather = dist.all_gather_into_tensor
        all_gather_sizes: List[int] = []
        lock = threading.Lock()

        def all_gather(*args: Any, **kwargs: Any):
            nonlocal all_gather_sizes
            if len(args) > 0:
                output = args[0]
            elif "output_tensor" in kwargs:
                output = kwargs["output_tensor"]
            else:
                raise AssertionError(
                    f"Cannot get all-gather output from\nargs: {args}\nkwargs: {kwargs}"
                )
            with lock:
                all_gather_sizes.append(output.numel() * output.itemsize)
            return orig_all_gather(*args, **kwargs)

        def get_expected_all_gather_size(module: nn.Module):
            size = 0
            for param_name, param in module.named_parameters():
                bytes_per_numel = 1 if "weight" in param_name else param.itemsize
                size += param.numel() * bytes_per_numel
            return size

        # - Check for a single FSDP parameter group
        module_fp32 = self._init_single_module()
        ref_module = copy.deepcopy(module_fp32)
        module = swap_linear_with_float8_linear(
            module_fp32,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=True,
        )
        fully_shard(module)
        local_inp = self._get_local_inp()
        expected_all_gather_size = get_expected_all_gather_size(ref_module)
        with patch_all_gather(all_gather):
            out = module(local_inp)
        # For MPTG, one rank runs all all-gathers, each of the same size
        if all_gather_sizes:
            self.assertEqual(len(all_gather_sizes), self.world_size)
            self.assertEqual(
                all_gather_sizes, [expected_all_gather_size] * self.world_size
            )
        all_gather_sizes.clear()
        # Force-reshard the module to check the backward all-gather
        module.reshard()
        with patch_all_gather(all_gather):
            out.sum().backward()
        if all_gather_sizes:
            self.assertEqual(len(all_gather_sizes), self.world_size)
            self.assertEqual(
                all_gather_sizes, [expected_all_gather_size] * self.world_size
            )
        all_gather_sizes.clear()

        # - Check for multiple FSDP parameter groups
        module = self._init_multi_module()
        ref_module = copy.deepcopy(module)
        module = swap_linear_with_float8_linear(
            module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=True,
        )
        for submodule in module:
            fully_shard(submodule)
        fully_shard(module)
        expected_all_gather_sizes = (
            get_expected_all_gather_size(submodule) for submodule in module
        )
        with patch_all_gather(all_gather):
            out = module(local_inp)
        if all_gather_sizes:
            self.assertEqual(len(all_gather_sizes), self.world_size * len(module))
            self.assertEqual(
                all_gather_sizes,
                [s for s in expected_all_gather_sizes for _ in range(self.world_size)],
            )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_single_module_parity(self):
        for use_fp8_all_gather, module_cls in itertools.product(
            [False, True], [Float8DynamicLinear, Float8Linear]
        ):
            self._test_single_module_fp32_parity(
                use_fp8_all_gather=use_fp8_all_gather,
                module_cls=module_cls,
            )

    def _test_single_module_fp32_parity(
        self, use_fp8_all_gather: bool, module_cls: Type
    ):
        if self.rank == 0:
            print(module_cls, use_fp8_all_gather)
        use_activation_hooks = module_cls is Float8DynamicLinear
        module_fp32 = self._init_single_module()
        ref_module = swap_linear_with_float8_linear(
            copy.deepcopy(module_fp32),
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
        )
        ref_module = ref_module.cuda()
        module = swap_linear_with_float8_linear(
            module_fp32,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=use_fp8_all_gather,
        )
        fully_shard(module)
        ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = self._get_local_inp()
        self._check_parity_fp32(
            ref_module, ref_optim, module, optim, local_inp, module_cls
        )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_multi_module_parity(self):
        for use_fp8_all_gather, module_cls in itertools.product(
            [False, True], [Float8DynamicLinear, Float8Linear]
        ):
            self._test_multi_module_fp32_parity(
                use_fp8_all_gather=use_fp8_all_gather,
                module_cls=module_cls,
            )

    def _test_multi_module_fp32_parity(
        self, use_fp8_all_gather: bool, module_cls: Type
    ):
        use_activation_hooks = module_cls is Float8DynamicLinear
        module = self._init_multi_module()
        ref_module = copy.deepcopy(module)
        ref_module = swap_linear_with_float8_linear(
            ref_module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
        )
        ref_module = ref_module.cuda()
        module = swap_linear_with_float8_linear(
            module,
            module_cls,
            emulate=True,
            use_activation_hooks=use_activation_hooks,
            use_fp8_all_gather=use_fp8_all_gather,
        )
        for submodule in module:
            fully_shard(submodule)
        fully_shard(module)
        ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = self._get_local_inp()
        self._check_parity_fp32(
            ref_module, ref_optim, module, optim, local_inp, module_cls
        )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fp8_bf16_all_gather_dynamic_single_parity(self):
        # TODO: This test fails because for the reference module, the amax and
        # scale are computed with respect to the bf16 parameters, whereas for
        # FSDP, they are computed with respect to the fp32 parameters.
        return
        self._test_bf16_dynamic_single_parity(use_fp8_all_gather=True)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_bf16_all_gather_dynamic_single_parity(self):
        self._test_bf16_dynamic_single_parity(use_fp8_all_gather=False)

    def _test_bf16_dynamic_single_parity(self, use_fp8_all_gather: bool):
        torch.manual_seed(42)
        module_fp32 = nn.Linear(16, 16, device="cuda")
        for param in module_fp32.parameters():
            dist.broadcast(param, src=0)
        ref_module_bf16 = copy.deepcopy(module_fp32).to(torch.bfloat16)
        ref_module_bf16 = swap_linear_with_float8_linear(
            ref_module_bf16,
            Float8DynamicLinear,
            emulate=True,
            use_activation_hooks=True,
        )
        ref_module_fp32 = copy.deepcopy(module_fp32)
        module = swap_linear_with_float8_linear(
            module_fp32,
            Float8DynamicLinear,
            emulate=True,
            use_activation_hooks=True,
            use_fp8_all_gather=use_fp8_all_gather,
        )
        fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
        ref_optim = torch.optim.Adam(ref_module_fp32.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = self._get_local_inp(torch.bfloat16)
        self._check_parity_bf16_mp(
            ref_module_fp32, ref_module_bf16, ref_optim, module, optim, local_inp
        )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_bf16_all_gather_dynamic_multi_parity(self):
        module = self._init_multi_module()
        ref_module_bf16 = copy.deepcopy(module).to(torch.bfloat16)
        ref_module_bf16 = swap_linear_with_float8_linear(
            ref_module_bf16,
            Float8DynamicLinear,
            emulate=True,
            use_activation_hooks=True,
        )
        ref_module_fp32 = copy.deepcopy(module).cuda()
        module = swap_linear_with_float8_linear(
            module, Float8DynamicLinear, emulate=True, use_activation_hooks=True
        )
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        for mlp in module:
            fully_shard(mlp, mp_policy=mp_policy)
        fully_shard(module, mp_policy=mp_policy)
        ref_optim = torch.optim.Adam(ref_module_fp32.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = self._get_local_inp(torch.bfloat16)
        self._check_parity_bf16_mp(
            ref_module_fp32, ref_module_bf16, ref_optim, module, optim, local_inp
        )


if __name__ == "__main__":
    run_tests()
