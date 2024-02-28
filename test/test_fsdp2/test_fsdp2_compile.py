import torch
import torch._dynamo.testing
import torch.distributed as dist
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from test_fsdp2_common import init_transformer_with_fp8
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
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
        for use_fp8_all_gather in [False, True]:
            module = init_transformer_with_fp8(
                Float8DynamicLinear, use_fp8_all_gather=use_fp8_all_gather
            )

            # Compile each transformer block forward separately
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            num_compiled_fns = 0
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.forward = torch.compile(submodule.forward, backend=cnt)
                    num_compiled_fns += 1
                    fully_shard(submodule)
            fully_shard(module)
            local_inp = torch.randint(0, 16, (16, 16), device="cuda")
            out = module(local_inp)
            out.sum().backward()
            self.assertEqual(cnt.frame_count, num_compiled_fns)

            # Compile the output projection
            module.output.forward = torch.compile(module.output.forward, backend=cnt)
            # in float8_mm
            # assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
            with self.assertRaises(RuntimeError):
                module(local_inp)

            # Compile each transformer block module separately
            module = init_transformer_with_fp8(
                Float8DynamicLinear, use_fp8_all_gather=use_fp8_all_gather
            )
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            for submodule in module.modules():
                if isinstance(submodule, TransformerBlock):
                    submodule.compile(backend=cnt)
                    fully_shard(submodule)
            fully_shard(module)

            # in backward
            # Expecting  both Float8Tensor for mm inputs but found
            # <class 'torch._subclasses.functional_tensor.FunctionalTensor'> and
            # <class 'float8_experimental.float8_tensor.Float8Tensor'>
            with self.assertRaises(RuntimeError):
                module(local_inp)

    @skip_if_lt_x_gpu(2)
    def test_compile_root_dynamic(self):
        for use_fp8_all_gather in [False, True]:
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
            local_inp = torch.randint(0, 16, (16, 16), device="cuda")
            # in forward
            #   h = x + self.attention(self.attention_norm(x))
            # in float8_mm
            #   assert isinstance(args[0], Float8Tensor) and isinstance(args[1], Float8Tensor)
            with self.assertRaises(RuntimeError):
                module(local_inp)


if __name__ == "__main__":
    run_tests()
