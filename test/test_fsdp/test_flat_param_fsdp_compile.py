import contextlib
import copy
from typing import List, Type

import torch
import torch._dynamo.testing
import torch.distributed as dist
from float8_experimental import config
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestFloat8CompileCommon:
    def _init_transformer_with_fp8(
        self, module_cls: Type, checkpoint_activations: bool = False
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
        use_hooks = module_cls is Float8DynamicLinear
        return swap_linear_with_float8_linear(
            module, module_cls, emulate=True, use_activation_hooks=use_hooks
        )

    @contextlib.contextmanager
    def enable_amax_init(self, enable: bool):
        prev_value = config.enable_amax_init
        config.enable_amax_init = enable
        try:
            yield
        finally:
            config.enable_amax_init = prev_value

    @contextlib.contextmanager
    def enable_pre_and_post_forward(self, enable: bool):
        prev_value = config.enable_pre_and_post_forward
        config.enable_pre_and_post_forward = enable
        try:
            yield
        finally:
            config.enable_pre_and_post_forward = prev_value


class TestFloat8CompileFakePG(
    TestFloat8CompileCommon, torch._dynamo.test_case.TestCase
):
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
        module = self._init_transformer_with_fp8(Float8DynamicLinear)

        # Compile each transformer block separately
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
                num_compiled_fns += 1
        module = FSDP(
            module,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            use_orig_params=True,
            device_id=dist.get_rank(),
        )
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
        module = self._init_transformer_with_fp8(Float8DynamicLinear)

        # Compile the root module
        module = FSDP(
            module,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            use_orig_params=True,
            device_id=dist.get_rank(),
        )
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        module = torch.compile(module, backend=cnt)
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        # in forward
        #   h = layer(h)
        # in float8_mm
        #   assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_submodule_delayed(self):
        module = self._init_transformer_with_fp8(Float8Linear)

        # Compile each transformer block separately
        torch._dynamo.config.cache_size_limit = 16
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
        module = FSDP(
            module,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            use_orig_params=True,
            device_id=dist.get_rank(),
        )
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        # Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'
        # with self.assertRaises(RuntimeError):
        with self.assertRaises(RuntimeError):
            module(local_inp)

        # Compile each transformer block separately with amax init disabled
        with self.enable_amax_init(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = FSDP(
                module,
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
                use_orig_params=True,
                device_id=dist.get_rank(),
            )
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, 18)  # TODO!

        # Compile each transformer block separately with amax init disabled and
        # pre/post-forward disabled
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = FSDP(
                module,
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
                use_orig_params=True,
                device_id=dist.get_rank(),
            )
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, num_compiled_fns)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_delayed(self):
        with self.enable_amax_init(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            module = FSDP(
                module,
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
                use_orig_params=True,
                device_id=dist.get_rank(),
            )
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        self.assertEqual(cnt.frame_count, 19)  # TODO!

        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            module = FSDP(
                module,
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
                use_orig_params=True,
                device_id=dist.get_rank(),
            )
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        self.assertEqual(cnt.frame_count, 19)  # TODO!


class TestFloat8CompileNCCLPG(TestFloat8CompileCommon, FSDPTest):
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
        model = self._init_transformer_with_fp8(module_cls, checkpoint_activations)
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            use_orig_params=True,
            device_id=self.rank,
        )
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(
            False
        ) if module_cls is Float8Linear else contextlib.nullcontext():
            for iter_idx in range(10):
                losses: List[torch.Tensor] = []
                for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
                    optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    losses.append(model(local_inp).sum())
                    losses[-1].backward()
                    if model is ref_model:
                        for param in model.parameters():
                            dist.all_reduce(param.grad)
                            param.grad.div_(self.world_size)
                    if module_cls is Float8Linear:
                        sync_float8_amax_and_scale_history(model)
                    optim.step()
                self.assertEqual(losses[0], losses[1])

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
        model = self._init_transformer_with_fp8(module_cls, checkpoint_activations)
        ref_model = copy.deepcopy(model).cuda()  # used for optimizer
        ref_model_bf16 = copy.deepcopy(ref_model).to(
            torch.bfloat16
        )  # used for forward/backward
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            use_orig_params=True,
            device_id=self.rank,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        )
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(
            False
        ) if module_cls is Float8Linear else contextlib.nullcontext():
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
                            param_bf16.grad.div_(self.world_size)
                            param_fp32.grad = param_bf16.grad.float()
                            param_bf16.grad = None
                    if module_cls is Float8Linear:
                        sync_float8_amax_and_scale_history(model)
                    optim.step()
                    for param_fp32, param_bf16 in zip(
                        ref_model.parameters(), ref_model_bf16.parameters()
                    ):
                        param_bf16.detach().copy_(param_fp32)
                self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
