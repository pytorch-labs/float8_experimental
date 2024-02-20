# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random
import unittest

import pytest

import torch
import torch.nn as nn
from float8_experimental.float8_linear_utils import get_float8_linear, LinearType
from float8_experimental.float8_tensor import Float8Tensor

from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import CompileCounterWithBackend

# Setting to unblock for calling contiguous in backwards
is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def _test_compile_base(
    backend: str,
    fullgraph: bool,
    emulate: bool,
    linear_type: LinearType,
    dtype: torch.dtype,
    use_activation_hooks: bool,
):
    random.seed(0)
    torch.manual_seed(0)
    x_shape = (16, 16)
    linear_dtype = torch.bfloat16

    x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
    m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)

    m_fp8 = get_float8_linear(linear_type, m_ref, emulate, use_activation_hooks)

    m_fp8 = torch.compile(m_fp8, backend=backend, fullgraph=fullgraph)
    m_ref = torch.compile(m_ref, backend=backend, fullgraph=fullgraph)
    y_fp8 = m_fp8(x)
    y_fp8.sum().backward()
    y_ref = m_ref(x)
    y_ref.sum().backward()
    torch.testing.assert_close(y_fp8, y_ref, atol=9.5e-2, rtol=9.5e-2)
    torch.testing.assert_close(
        m_fp8.weight.grad, m_ref.weight.grad, atol=2e-1, rtol=2e-1
    )
    torch.testing.assert_close(m_fp8.bias.grad, m_ref.bias.grad, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
@pytest.mark.parametrize("emulate", [False, True] if is_H100 else [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_activation_hooks", [False, True])
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_eager_only(
    fullgraph,
    emulate: bool,
    linear_type: bool,
    dtype: torch.dtype,
    use_activation_hooks: bool,
):
    if linear_type == LinearType.DELAYED and use_activation_hooks:
        pytest.skip("use_activation_hooks is only supported for dynamic linear")
    torch._dynamo.reset()
    _test_compile_base(
        "eager", fullgraph, emulate, linear_type, dtype, use_activation_hooks
    )


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False, True] if is_H100 else [True])
@pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_activation_hooks", [False, True])
# TODO this shouldn't fail but multiple fake modes
@pytest.mark.usefixtures("x_fail_activation_hooks")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_aot_eager(
    fullgraph,
    emulate: bool,
    linear_type: bool,
    dtype: torch.dtype,
    use_activation_hooks: bool,
):
    if linear_type == LinearType.DELAYED and use_activation_hooks:
        pytest.skip("use_activation_hooks is only supported for dynamic linear")
    torch._dynamo.reset()
    _test_compile_base(
        "aot_eager", fullgraph, emulate, linear_type, dtype, use_activation_hooks
    )


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False])
@pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
@pytest.mark.parametrize("use_activation_hooks", [False, True])
# TODO this shouldn't fail but multiple fake modes
@pytest.mark.usefixtures("x_fail_activation_hooks")
@unittest.skipIf(not torch.cuda.is_available() or not is_H100, "CUDA not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_inductor(
    fullgraph,
    emulate: bool,
    linear_type: bool,
    dtype: torch.dtype,
    use_activation_hooks: bool,
):
    if linear_type == LinearType.DELAYED and use_activation_hooks:
        pytest.skip("use_activation_hooks is only supported for dynamic linear")
    torch._dynamo.reset()
    _test_compile_base(
        "inductor", fullgraph, emulate, linear_type, dtype, use_activation_hooks
    )


class TestGraphBreaks(DynamoTestCase):
    class MockLinear(torch.nn.Module):
        def __init__(self, graph_break: bool):
            super().__init__()
            self.register_buffer("fp8_amax_x", torch.tensor(1.0))
            self.register_buffer("fp8_scale_x", torch.tensor(1.0))
            self.graph_break = graph_break

        def forward(self, x):
            x_fp8 = Float8Tensor.to_float8(
                x,
                self.fp8_scale_x,
                torch.float8_e4m3fn,
                self.fp8_amax_x,
                emulate=True,  # TODO: I set this to True so that people on A100 can test, but once fix is in, set to False
            )
            if self.graph_break:
                torch._dynamo.graph_break()
                x_hp = x_fp8.to_original_precision()
                return x_hp
            return x_fp8

    def test_float8_with_graph_break_in_the_middle(self):
        """Test that having Float8Tensor object at the boundary of a subgraph"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=True).cuda()
        compiled_mod = copy.deepcopy(mod)
        compiled_mod = torch.compile(compiled_mod, backend=cnts)
        x = torch.randn(16, 16, device="cuda")
        y_eager = mod(x)
        y_compiled = compiled_mod(x)
        self.assertEqual(cnts.frame_count, 2, "Compiled graph should have 2 frames!")
        torch.testing.assert_close(y_eager, y_compiled)

    def test_float8_graph_input(self):
        """Test that having Float8Tensor object as a graph input"""

        def to_float(x):
            return x.to_original_precision()

        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).cuda()
        x = torch.randn(2, 2, device="cuda")
        compiled_to_float = torch.compile(to_float, backend=cnts)
        y = mod(x)
        y2_eager = to_float(y)
        y2_compiled = compiled_to_float(y)
        self.assertEqual(
            cnts.frame_count,
            1,
            "to_float was not compiled into 1 frame and likely encountered a skip!",
        )
        torch.testing.assert_close(y2_eager, y2_compiled)

    def test_float8_graph_output(self):
        """Test that having Float8Tensor object as a graph output works"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).cuda()
        compiled_mod = torch.compile(mod, backend=cnts)
        x = torch.randn(16, 16, device="cuda")
        y_compiled = compiled_mod(x)

        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")
        tensors, ctx = y_compiled.__tensor_flatten__()
        for tensor in tensors:
            assert not isinstance(
                getattr(y_compiled, tensor), torch._subclasses.fake_tensor.FakeTensor
            ), "Float8Tensor should not contain any FakeTensors!"
        assert isinstance(
            y_compiled._orig_dtype, torch.dtype
        ), "Float8Tensor._orig_dtype should be a dtype but got {}".format(
            type(y_compiled._orig_dtype)
        )
        assert isinstance(
            y_compiled._emulate, bool
        ), "Float8Tensor._emulate should be a bool but got {}".format(
            type(y_compiled._emulate)
        )


if __name__ == "__main__":
    pytest.main([__file__])
