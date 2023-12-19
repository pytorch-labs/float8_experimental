# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
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

def test_float8_with_graph_break_in_the_middle():
    # test that having Float8Tensor object at the boundary of a subgraph
    # works

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fp8_amax_x", torch.tensor(1.0))
            self.register_buffer("fp8_scale_x", torch.tensor(1.0))

        def forward(self, x):
            x_fp8 = Float8Tensor.to_float8(
                x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x, 
                emulate=False,
            )

            # graph break
            print('foo')
            x_hp = x_fp8.to_original_precision()
            return x_hp

    m = M().cuda()
    m = torch.compile(m)
    x = torch.randn(16, 16, device='cuda')
    y = m(x)

def test_float8_graph_input():
    # test that having Float8Tensor object as a graph input works
    # works fine!

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fp8_amax_x", torch.tensor(1.0))
            self.register_buffer("fp8_scale_x", torch.tensor(1.0))

        def forward(self, x):
            x_fp8 = Float8Tensor.to_float8(
                x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x, 
                emulate=False,
            )

            return x_fp8

    def to_float(x):
        return x.to_original_precision()

    to_float = torch.compile(to_float)

    m = M().cuda()
    x = torch.randn(2, 2, device='cuda')
    y = m(x)
    print(1, y)
    y2 = to_float(y)
    print(2, y2)

def test_float8_graph_output():
    # test that having Float8Tensor object as a graph output works
    # silently incorrect - `y` has fake tensors!

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fp8_amax_x", torch.tensor(1.0))
            self.register_buffer("fp8_scale_x", torch.tensor(1.0))

        def forward(self, x):
            x_fp8 = Float8Tensor.to_float8(
                x, self.fp8_scale_x, torch.float8_e4m3fn, self.fp8_amax_x, 
                emulate=False,
            )

            return x_fp8

    m = M().cuda()
    m = torch.compile(m)
    x = torch.randn(16, 16, device='cuda')
    y = m(x)
    print('y', y)



if __name__ == "__main__":
    pytest.main([__file__])
