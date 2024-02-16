import contextlib
import functools
from typing import List, Optional, Type

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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

# Synctactic sugar: require `use_orig_params=True` for compile
FSDP = functools.partial(FSDP, use_orig_params=True)
# Increase cache size limit for running all unit tests together
torch._dynamo.config.cache_size_limit = 16


class TestFloat8CompileCommon:
    def _init_transformer_with_fp8(
        self,
        module_cls: Type,
        *,
        checkpoint_activations: bool = False,
        use_activation_hooks: Optional[bool] = None,
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

    def apply_fsdp(self, transformer: Transformer):
        return FSDP(
            transformer,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            device_id=dist.get_rank(),
        )


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
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")

        # Compile each transformer block forward
        module = self._init_transformer_with_fp8(Float8DynamicLinear)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
                num_compiled_fns += 1
        module = self.apply_fsdp(module)
        out = module(local_inp)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, num_compiled_fns)

        # Compile the output projection
        module.output.forward = torch.compile(module.output.forward, backend=cnt)
        # in float8_mm
        # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

        # Compile each transformer block module
        module = self._init_transformer_with_fp8(Float8DynamicLinear)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.compile(backend=cnt)
                num_compiled_fns += 1
        module = self.apply_fsdp(module)
        # in float8_mm
        # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_dynamic(self):
        # Compile the root module
        module = self._init_transformer_with_fp8(Float8DynamicLinear)
        module = self.apply_fsdp(module)
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
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")

        # Compile each transformer block forward
        module = self._init_transformer_with_fp8(Float8Linear)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
        module = self.apply_fsdp(module)
        module(local_inp).sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        # TODO: We get one graph per `Float8Linear` in a transformer block
        # (-1 because output projection is not compiled).
        self.assertEqual(cnt.frame_count, num_float8_linears - 1)

        # Compile each transformer block forward with amax init disabled
        with self.enable_amax_init(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = self.apply_fsdp(module)
            module(local_inp).sum().backward()
            num_float8_linears = sum(
                1 for m in module.modules() if isinstance(m, Float8Linear)
            )
            # TODO: We get one graph per `Float8Linear` in a transformer block
            # (-1 because output projection is not compiled).
            self.assertEqual(cnt.frame_count, num_float8_linears - 1)

        # Compile each transformer block forward with amax init disabled and
        # pre/post-forward disabled
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = self.apply_fsdp(module)
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, num_compiled_fns)

        # Compile each transformer block module with amax init disabled and
        # pre/post-forward disabled
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.compile(backend=cnt)
            module = self.apply_fsdp(module)
            module(local_inp).sum().backward()
            num_float8_linears = sum(
                1 for m in module.modules() if isinstance(m, Float8Linear)
            )
            # TODO: We get one graph per `Float8Linear` in a transformer block
            # (-1 because output projection is not compiled).
            self.assertEqual(cnt.frame_count, num_float8_linears - 1)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_delayed(self):
        with self.enable_amax_init(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            module = self.apply_fsdp(module)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        self.assertEqual(cnt.frame_count, num_float8_linears)  # TODO!

        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            module = self._init_transformer_with_fp8(Float8Linear)
            module = self.apply_fsdp(module)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        self.assertEqual(cnt.frame_count, num_float8_linears)  # TODO!


class TestFloat8CompileNCCLPG(TestFloat8CompileCommon, FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    def _test_parity(
        self,
        ref_model: torch.nn.Module,
        ref_optim: torch.optim.Optimizer,
        fsdp_model: torch.nn.Module,
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
                if model is ref_model:  # manual data parallelism
                    for param in model.parameters():
                        dist.all_reduce(param.grad)
                        param.grad.div_(self.world_size)
                if module_cls is Float8Linear:
                    sync_float8_amax_and_scale_history(model)
                optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_delayed_no_mp(self):
        module_cls, backend = Float8Linear, "inductor"
        with self.enable_amax_init(False), self.enable_pre_and_post_forward(False):
            model = self._init_transformer_with_fp8(module_cls)
        with self.enable_amax_init(False):
            ref_model = self._init_transformer_with_fp8(module_cls).cuda()
        # NOTE: We compile the ref model in the same way as the FSDP model for
        # numeric parity. Compiling the full ref model or running the full ref
        # model in eager both show differences 5+ iterations in.
        for module in ref_model.modules():
            if isinstance(module, TransformerBlock):
                module.forward = torch.compile(module.forward, backend=backend)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
        num_compiled_fns = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                # TODO: For `Float8Linear`, compiling the module gives one
                # graph per `Float8Linear` instead of per `TransformerBlock`.
                module.forward = torch.compile(module.forward, backend=cnt)
                num_compiled_fns += 1
        fsdp_model = self.apply_fsdp(model)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        self._test_parity(
            ref_model, ref_optim, fsdp_model, fsdp_optim, local_inp, module_cls
        )
        self.assertEqual(cnt.frame_count, num_compiled_fns)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_dynamic_no_mp(self):
        self.run_subtests(
            {"use_activation_hooks": [False, True]},
            self._test_transformer_parity_dynamic_no_mp,
        )

    def _test_transformer_parity_dynamic_no_mp(self, use_activation_hooks: bool):
        module_cls, backend = Float8DynamicLinear, "inductor"
        model = self._init_transformer_with_fp8(
            module_cls, use_activation_hooks=use_activation_hooks
        )
        ref_model = self._init_transformer_with_fp8(
            module_cls, use_activation_hooks=use_activation_hooks
        ).cuda()
        # NOTE: We compile the ref model in the same way as the FSDP model for
        # numeric parity. Compiling the full ref model or running the full ref
        # model in eager both show differences 5+ iterations in.
        for module in ref_model.modules():
            if isinstance(module, TransformerBlock):
                module.forward = torch.compile(module.forward, backend=backend)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
        num_compiled_fns = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                # TODO: For `Float8DynamicLinear`, compiling the module errors
                # for both using and not using activation hooks.
                # in float8_mm
                # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
                # module.compile(backend=cnt)
                module.forward = torch.compile(module.forward, backend=cnt)
                num_compiled_fns += 1
        fsdp_model = self.apply_fsdp(model)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        self._test_parity(
            ref_model, ref_optim, fsdp_model, fsdp_optim, local_inp, module_cls
        )
        self.assertEqual(cnt.frame_count, num_compiled_fns)


if __name__ == "__main__":
    run_tests()
