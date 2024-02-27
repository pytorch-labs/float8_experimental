from typing import Tuple

import pytest
import torch
from float8_experimental.float8_utils import amax_to_scale, tensor_to_amax
from float8_experimental.fused_kernels.fused_casting_kernels import (
    abs_max,
    abs_max_to_scale,
)


@pytest.mark.parametrize(
    "shape",
    [
        (16384, 1024),
        (16384, 8192),
        (16384, 3584),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_abs_max(shape: Tuple[int], dtype: torch.dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    max_abs = abs_max(x)
    assert torch.allclose(max_abs, tensor_to_amax(x))


@pytest.mark.parametrize("numel", [2**i for i in range(10, 20)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_amax_to_scale(numel: int, dtype: torch.dtype, fp8_type: torch.dtype):
    x = torch.randn(numel, dtype=dtype, device="cuda")
    max_abs = abs_max(x)
    fused = abs_max_to_scale(max_abs, fp8_type, dtype == torch.float16)
    eager = amax_to_scale(max_abs, fp8_type, dtype)
    assert torch.allclose(fused, eager)
