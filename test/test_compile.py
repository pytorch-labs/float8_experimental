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


if __name__ == "__main__":
    pytest.main([__file__])
