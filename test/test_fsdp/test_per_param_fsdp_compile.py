import contextlib
import copy
from typing import Type

import torch
import torch._dynamo.testing
import torch.distributed as dist
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from test_fsdp_common import (
    check_parity_bf16_mp,
    check_parity_no_mp,
    enable_amax_init,
    enable_pre_and_post_forward,
    init_transformer_with_fp8,
)
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import TransformerBlock
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestFloat8CompileFakePG(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_compile_submodule_dynamic(self):
        for use_fp8_all_gather in (False, True):
            self._test_compile_submodule_dynamic(use_fp8_all_gather)

    def _test_compile_submodule_dynamic(self, use_fp8_all_gather: bool):
        module = init_transformer_with_fp8(
            Float8DynamicLinear, use_fp8_all_gather=use_fp8_all_gather
        )

        # Compile each transformer block separately
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
                num_compiled_fns += 1
                fully_shard(submodule)
        fully_shard(module)
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        out = module(local_inp)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, num_compiled_fns)

        # Compile the output projection
        module.output.forward = torch.compile(module.output.forward, backend=cnt)
        # in float8_mm
        # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_dynamic(self):
        for use_fp8_all_gather in (False, True):
            self._test_compile_root_dynamic(use_fp8_all_gather)

    def _test_compile_root_dynamic(self, use_fp8_all_gather: bool):
        module = init_transformer_with_fp8(
            Float8DynamicLinear, use_fp8_all_gather=use_fp8_all_gather
        )

        # Compile the root module
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                fully_shard(submodule)
        fully_shard(module)
        module = torch.compile(module, backend=cnt)
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        # in forward
        #   h = x + self.attention(self.attention_norm(x))
        # in float8_mm
        #   assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_submodule_delayed(self):
        for use_fp8_all_gather in (False, True):
            self._test_compile_submodule_delayed(use_fp8_all_gather)

    def _test_compile_submodule_delayed(self, use_fp8_all_gather: bool):
        module = init_transformer_with_fp8(
            Float8Linear, use_fp8_all_gather=use_fp8_all_gather
        )

        # Compile each transformer block separately
        torch._dynamo.config.cache_size_limit = 16
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
                fully_shard(submodule)
        fully_shard(module)
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        # Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'
        with self.assertRaises(RuntimeError):
            module(local_inp)

        # Compile each transformer block separately with amax init disabled
        with enable_amax_init(False):
            module = init_transformer_with_fp8(
                Float8Linear, use_fp8_all_gather=use_fp8_all_gather
            )
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
                    fully_shard(submodule)
            fully_shard(module)
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, 18)  # TODO!
            # self.assertEqual(cnt.frame_count, num_compiled_fns)

        # Compile each transformer block separately with amax init disabled and
        # pre/post-forward disabled
        with enable_amax_init(False), enable_pre_and_post_forward(False):
            module = init_transformer_with_fp8(
                Float8Linear, use_fp8_all_gather=use_fp8_all_gather
            )
            # TODO: Inductor backend errors for `use_fp8_all_gather=True`!
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
                    fully_shard(submodule)
            fully_shard(module)
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, num_compiled_fns)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_delayed(self):
        for use_fp8_all_gather in (False, True):
            self._test_compile_root_delayed(use_fp8_all_gather)

    def _test_compile_root_delayed(self, use_fp8_all_gather: bool):
        module_cls = Float8Linear
        with enable_amax_init(False):
            module = init_transformer_with_fp8(
                module_cls, use_fp8_all_gather=use_fp8_all_gather
            )
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    fully_shard(submodule)
            fully_shard(module)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            # Error for `use_fp8_all_gather=True`:
            # fsdp_param.all_gather_inputs for fsdp_param in fsdp_params
            # all_gather_inputs, self._all_gather_metadata = fsdp_pre_all_gather(
            # attrs, _ = type(x).__tensor_flatten__(x)
            # AttributeError("'Float8Tensor' object has no attribute '_data'")
            with self.assertRaises(
                RuntimeError
            ) if use_fp8_all_gather else contextlib.nullcontext():
                module(local_inp)
        if not use_fp8_all_gather:
            self.assertEqual(cnt.frame_count, 19)  # TODO!


class TestFloat8CompileNCCLPG(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_no_mp(self):
        """
        Test numeric parity against manual data parallelism without using
        FSDP's mixed precision.
        """
        self.run_subtests(
            {
                "module_cls": [Float8Linear, Float8DynamicLinear],
                "checkpoint_activations": [False, True],
            },
            self._test_transformer_parity_no_mp,
        )

    def _test_transformer_parity_no_mp(
        self, module_cls: Type, checkpoint_activations: bool
    ):
        model = init_transformer_with_fp8(
            module_cls, checkpoint_activations=checkpoint_activations
        )
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fsdp_model = fully_shard(model)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2, foreach=True)
        check_parity_no_mp(
            self,
            ref_model,
            ref_optim,
            fsdp_model,
            fsdp_optim,
            torch.randint(0, 16, (1, 4), device="cuda"),
            module_cls,
        )

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_bf16_mp(self):
        """
        Test numeric parity against manual data parallelism using FSDP's bf16
        mixed precision.
        """
        self.run_subtests(
            {
                "module_cls": [Float8Linear, Float8DynamicLinear],
                "checkpoint_activations": [False, True],
            },
            self._test_transformer_parity_bf16_mp,
        )

    def _test_transformer_parity_bf16_mp(
        self, module_cls: Type, checkpoint_activations: bool
    ):
        model = init_transformer_with_fp8(
            module_cls, checkpoint_activations=checkpoint_activations
        )
        ref_model = copy.deepcopy(model).cuda()  # used for optimizer
        ref_model_bf16 = copy.deepcopy(ref_model).to(
            torch.bfloat16
        )  # used for forward/backward
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, mp_policy=mp_policy)
        fsdp_model = fully_shard(model, mp_policy=mp_policy)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2, foreach=True)
        check_parity_bf16_mp(
            self,
            ref_model,
            ref_model_bf16,
            ref_optim,
            fsdp_model,
            fsdp_optim,
            torch.randint(0, 16, (1, 4), device="cuda"),
            module_cls,
        )


if __name__ == "__main__":
    run_tests()
