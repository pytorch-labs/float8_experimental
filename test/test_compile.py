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

# Setting to unblock for calling contiguous in backwards
is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def _test_compile_base(
    backend: str,
    fullgraph: bool,
    emulate: bool,
    linear_type: LinearType,
    dtype: torch.dtype,
):
    random.seed(0)
    torch.manual_seed(0)
    x_shape = (16, 16)
    linear_dtype = torch.bfloat16

    x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
    m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)

    m_fp8 = get_float8_linear(linear_type, m_ref, emulate=emulate)

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
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_eager_only(fullgraph, emulate: bool, linear_type: bool, dtype: torch.dtype):
    torch._dynamo.reset()
    _test_compile_base("eager", fullgraph, emulate, linear_type, dtype)


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False, True] if is_H100 else [True])
@pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_aot_eager(fullgraph, emulate: bool, linear_type: bool, dtype: torch.dtype):
    torch._dynamo.reset()
    _test_compile_base("aot_eager", fullgraph, emulate, linear_type, dtype)


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False])
@pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
@unittest.skipIf(not torch.cuda.is_available() or not is_H100, "CUDA not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_inductor(fullgraph, emulate: bool, linear_type: bool, dtype: torch.dtype):
    torch._dynamo.reset()
    _test_compile_base("inductor", fullgraph, emulate, linear_type, dtype)


class TestGraphBreaks:
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

    @pytest.mark.xfail(reason="TODO: Fix this test, see TODO in MockLinear")
    def test_float8_with_graph_break_in_the_middle(self):
        """Test that having Float8Tensor object at the boundary of a subgraph"""
        mod = self.MockLinear(graph_break=True).cuda()
        compiled_mod = copy.deepcopy(mod)
        compiled_mod = torch.compile(compiled_mod)
        x = torch.randn(16, 16, device="cuda")
        y_eager = mod(x)
        y_compiled = compiled_mod(x)
        torch.testing.assert_close(y_eager, y_compiled)

    def test_float8_graph_input(self):
        """Test that having Float8Tensor object as a graph input"""

        def to_float(x):
            return x.to_original_precision()

        to_float = torch.compile(to_float)

        mod = self.MockLinear(graph_break=False).cuda()
        x = torch.randn(2, 2, device="cuda")
        compiled_to_float = torch.compile(to_float)
        y = mod(x)
        y2_eager = to_float(y)
        y2_compiled = compiled_to_float(y)
        torch.testing.assert_close(y2_eager, y2_compiled)

    @pytest.mark.xfail(reason="TODO: Fix this test, see TODO in MockLinear")
    def test_float8_graph_output(self):
        """Test that having Float8Tensor object as a graph output works"""
        mod = self.MockLinear(graph_break=False).cuda()
        compiled_mod = torch.compile(mod)
        x = torch.randn(16, 16, device="cuda")
        y_compiled = compiled_mod(x)

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
