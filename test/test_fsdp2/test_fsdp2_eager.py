import copy
import threading
import unittest
from typing import Any, List

import torch
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear_utils import swap_linear_with_float8_linear
from test_fsdp2_common import check_parity_bf16_mp, check_parity_no_mp, TestFloat8Common
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    patch_all_gather,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFloat8MultiProcess(FSDPTest, TestFloat8Common):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_dynamic(self):
        for use_fp8_all_gather in [False, True]:
            self._test_transformer_parity_dynamic(use_fp8_all_gather)

    def _test_transformer_parity_dynamic(self, use_fp8_all_gather: bool):
        # NOTE: Weight-tying does not compose with fp8 all-gather because the
        # embedding weight and output linear weight are tied but only the
        # latter uses fp8 compute. With fp8 all-gather, FSDP would pre-cast to
        # fp8 for that tied weight, incorrectly using fp8 for the embedding.
        weight_tying = not use_fp8_all_gather
        module = self.init_transformer(weight_tying=weight_tying)
        ref_module = copy.deepcopy(module)
        ref_module = self.swap_linear_with_dynamic(ref_module).cuda()
        module = self.swap_linear_with_dynamic(
            module, use_fp8_all_gather=use_fp8_all_gather
        )
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                fully_shard(submodule)
        fully_shard(module)
        ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
        local_inp = torch.randint(
            0, ref_module.tok_embeddings.weight.size(0), (16, 16), device="cuda"
        )
        check_parity_no_mp(
            self, ref_module, ref_optim, module, optim, local_inp, Float8DynamicLinear
        )

    @skip_if_lt_x_gpu(2)
    def test_transformer_memory(self):
        """Tests peak active memory in the forward and backward passes."""
        for use_fp8_all_gather in [False, True]:
            self._test_transformer_memory(use_fp8_all_gather)

    def _test_transformer_memory(self, use_fp8_all_gather: bool):
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
        # Emulate the fp8 matmul to bypass the scaled matmul op's divisibility
        # requirement to use a smaller activation size
        model = self.swap_linear_with_dynamic(
            model, use_fp8_all_gather=use_fp8_all_gather, emulate=True
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
    def test_fsdp_extensions(self):
        from torch.distributed._composable.fsdp import FSDPTensorExtensions

        # Check for a single FSDP paramter group
        module_fp32 = self.init_single_module()
        module = self.swap_linear_with_dynamic(module_fp32, use_fp8_all_gather=True)
        self.assertTrue(hasattr(module, "fsdp_extensions"))
        self.assertTrue(callable(module.fsdp_extensions))
        fsdp_extensions = module.fsdp_extensions()
        self.assertIsInstance(fsdp_extensions.get("weight"), FSDPTensorExtensions)
        fully_shard(module)
        self.assertTrue(hasattr(module, "fsdp_extensions"))
        self.assertTrue(callable(module.fsdp_extensions))
        fsdp_extensions = module.fsdp_extensions()
        self.assertIsInstance(fsdp_extensions.get("weight"), FSDPTensorExtensions)

        # Check for multiple FSDP paramter groups
        module = self.init_multi_module()
        module = self.swap_linear_with_dynamic(module, use_fp8_all_gather=True)
        for submodule in module.modules():
            if not isinstance(submodule, Float8DynamicLinear):
                continue
            self.assertTrue(hasattr(submodule, "fsdp_extensions"))
            self.assertTrue(callable(submodule.fsdp_extensions))
            fsdp_extensions = submodule.fsdp_extensions()
            self.assertIsInstance(fsdp_extensions.get("weight"), FSDPTensorExtensions)
        for mlp in module:
            fully_shard(mlp)
        fully_shard(module)
        for submodule in module.modules():
            if not isinstance(submodule, Float8DynamicLinear):
                continue
            self.assertTrue(hasattr(submodule, "fsdp_extensions"))
            self.assertTrue(callable(submodule.fsdp_extensions))
            fsdp_extensions = submodule.fsdp_extensions()
            self.assertIsInstance(fsdp_extensions.get("weight"), FSDPTensorExtensions)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fp8_fp32_all_gather_dynamic_comm_size(self):
        """
        Tests that fp8 all-gather with dynamic scaling communicates the
        expected number of bytes.
        """
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
        module_fp32 = self.init_single_module()
        ref_module = copy.deepcopy(module_fp32)
        module = self.swap_linear_with_dynamic(module_fp32, use_fp8_all_gather=True)
        fully_shard(module)
        local_inp = self.get_local_inp()
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
        module = self.init_multi_module()
        ref_module = copy.deepcopy(module)
        module = self.swap_linear_with_dynamic(module, use_fp8_all_gather=True)
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
    def test_fp32_fp8_single_module_parity(self):
        """
        Tests numeric parity for fp32 parameters with fp8 computation with a
        single module/FSDP communication group.
        """
        for use_fp8_all_gather in [False, True]:
            module_fp32 = self.init_single_module()
            ref_module = self.swap_linear_with_dynamic(copy.deepcopy(module_fp32))
            ref_module = ref_module.cuda()
            module = self.swap_linear_with_dynamic(
                module_fp32, use_fp8_all_gather=use_fp8_all_gather
            )
            fully_shard(module)
            ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
            optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
            local_inp = self.get_local_inp()
            check_parity_no_mp(
                self,
                ref_module,
                ref_optim,
                module,
                optim,
                local_inp,
                Float8DynamicLinear,
            )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fp32_fp8_multi_module_parity(self):
        """
        Tests numeric parity for fp32 parameters with fp8 computation with
        multiple modules/FSDP communication groups.
        """
        for use_fp8_all_gather in [False, True]:
            module = self.init_multi_module()
            ref_module = copy.deepcopy(module)
            ref_module = self.swap_linear_with_dynamic(ref_module).cuda()
            module = self.swap_linear_with_dynamic(
                module, use_fp8_all_gather=use_fp8_all_gather
            )
            for submodule in module:
                fully_shard(submodule)
            fully_shard(module)
            ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
            optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)
            local_inp = self.get_local_inp()
            check_parity_no_mp(
                self,
                ref_module,
                ref_optim,
                module,
                optim,
                local_inp,
                Float8DynamicLinear,
            )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_bf16_mp_fp8_dynamic_multi_parity(self):
        """
        Tests numeric parity for fp32 parameters with FSDP's bf16 mixed
        precision and fp8 computation with multiple modules/FSDP communication
        groups. Parameters are all-gathered in bf16 before being cast to fp8.
        """
        # NOTE: We cannot test easily with fp8 all-gather because then the scale
        # is computed using the fp32 sharded parameters, not the bf16 unsharded
        # parameters, changing the numerics.
        module = self.init_multi_module()
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
        check_parity_bf16_mp(
            self,
            ref_module_fp32,
            ref_module_bf16,
            torch.optim.Adam(ref_module_fp32.parameters(), lr=1e-2),
            module,
            torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True),
            self.get_local_inp(torch.bfloat16),
            Float8DynamicLinear,
        )


if __name__ == "__main__":
    run_tests()
