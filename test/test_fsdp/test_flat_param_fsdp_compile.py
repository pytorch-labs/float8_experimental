import torch
import torch._dynamo.testing
import torch.distributed as dist
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from test_fsdp_common import (
    apply_fsdp,
    check_parity_no_mp,
    enable_amax_init,
    enable_pre_and_post_forward,
    init_transformer_with_fp8,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import TransformerBlock
from torch.testing._internal.distributed.fake_pg import FakeStore

# Increase cache size limit for running all unit tests together
torch._dynamo.config.cache_size_limit = 16


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
        local_inp = torch.randint(0, 16, (1, 4), device="cuda")

        # Compile each transformer block forward
        module = init_transformer_with_fp8(Float8DynamicLinear)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
                num_compiled_fns += 1
        module = apply_fsdp(module)
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
        module = init_transformer_with_fp8(Float8DynamicLinear)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        num_compiled_fns = 0
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.compile(backend=cnt)
                num_compiled_fns += 1
        module = apply_fsdp(module)
        # in float8_mm
        # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
        with self.assertRaises(RuntimeError):
            module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_dynamic(self):
        # Compile the root module
        module = init_transformer_with_fp8(Float8DynamicLinear)
        module = apply_fsdp(module)
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
        module = init_transformer_with_fp8(Float8Linear)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        for submodule in module.modules():
            if isinstance(submodule, TransformerBlock):
                submodule.forward = torch.compile(submodule.forward, backend=cnt)
        module = apply_fsdp(module)
        module(local_inp).sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        # TODO: We get one graph per `Float8Linear` in a transformer block
        # (-1 because output projection is not compiled).
        self.assertEqual(cnt.frame_count, num_float8_linears - 1)

        # Compile each transformer block forward with amax init disabled
        with enable_amax_init(False):
            module = init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = apply_fsdp(module)
            module(local_inp).sum().backward()
            num_float8_linears = sum(
                1 for m in module.modules() if isinstance(m, Float8Linear)
            )
            # TODO: We get one graph per `Float8Linear` in a transformer block
            # (-1 because output projection is not compiled).
            self.assertEqual(cnt.frame_count, num_float8_linears - 1)

        # Compile each transformer block forward with amax init disabled and
        # pre/post-forward disabled
        with enable_amax_init(False), enable_pre_and_post_forward(False):
            module = init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
            module = apply_fsdp(module)
            module(local_inp).sum().backward()
            self.assertEqual(cnt.frame_count, num_compiled_fns)

        # Compile each transformer block module with amax init disabled and
        # pre/post-forward disabled
        with enable_amax_init(False), enable_pre_and_post_forward(False):
            module = init_transformer_with_fp8(Float8Linear)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.compile(backend=cnt)
            module = apply_fsdp(module)
            module(local_inp).sum().backward()
            num_float8_linears = sum(
                1 for m in module.modules() if isinstance(m, Float8Linear)
            )
            # TODO: We get one graph per `Float8Linear` in a transformer block
            # (-1 because output projection is not compiled).
            self.assertEqual(cnt.frame_count, num_float8_linears - 1)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_delayed(self):
        with enable_amax_init(False):
            module = init_transformer_with_fp8(Float8Linear)
            module = apply_fsdp(module)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        self.assertEqual(cnt.frame_count, num_float8_linears)  # TODO!

        with enable_amax_init(False), enable_pre_and_post_forward(False):
            module = init_transformer_with_fp8(Float8Linear)
            module = apply_fsdp(module)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            module = torch.compile(module, backend=cnt)
            local_inp = torch.randint(0, 16, (1, 4), device="cuda")
            out = module(local_inp)
            out.sum().backward()
        num_float8_linears = sum(
            1 for m in module.modules() if isinstance(m, Float8Linear)
        )
        self.assertEqual(cnt.frame_count, num_float8_linears)  # TODO!


class TestFloat8CompileNCCLPG(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity_delayed_no_mp(self):
        module_cls, backend = Float8Linear, "inductor"
        with enable_amax_init(False), enable_pre_and_post_forward(False):
            model = init_transformer_with_fp8(module_cls)
        with enable_amax_init(False):
            ref_model = init_transformer_with_fp8(module_cls).cuda()
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
        fsdp_model = apply_fsdp(model)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        check_parity_no_mp(
            self, ref_model, ref_optim, fsdp_model, fsdp_optim, local_inp, module_cls
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
        model = init_transformer_with_fp8(
            module_cls, use_activation_hooks=use_activation_hooks
        )
        ref_model = init_transformer_with_fp8(
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
        fsdp_model = apply_fsdp(model)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        local_inp = torch.randint(0, 16, (1, 4), device="cuda")
        check_parity_no_mp(
            self, ref_model, ref_optim, fsdp_model, fsdp_optim, local_inp, module_cls
        )
        self.assertEqual(cnt.frame_count, num_compiled_fns)


if __name__ == "__main__":
    run_tests()
