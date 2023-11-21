# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import itertools
import random
import unittest
import warnings
from enum import Enum

import pytest

import torch
import torch.nn as nn
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    get_float8_linear,
    linear_requires_sync,
    LinearType,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_python_api import mm_float8
from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import (
    amax_to_scale,
    compute_error,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    FP16_MAX_POS,
    tensor_to_scale,
)

random.seed(0)
torch.manual_seed(0)


class TestFloat8Tensor(unittest.TestCase):
    def test_preserves_dtype(self):
        # hp means high precision, lp means low precision
        hp_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        lp_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for hp_dtype, lp_dtype in itertools.product(hp_dtypes, lp_dtypes):
            x1_hp = torch.randn(4, 4, dtype=hp_dtype)
            x1_s = tensor_to_scale(x1_hp, lp_dtype)
            x2_lp = Float8Tensor.to_float8(x1_hp, x1_s, lp_dtype)
            x3_hp = x2_lp.to_original_precision()
            self.assertTrue(x3_hp.dtype == hp_dtype)


class TestFloat8Linear:
    def _test_linear_impl(self, x, m_ref, linear_type: LinearType, emulate: bool):
        m_fp8 = get_float8_linear(linear_type, m_ref, emulate)
        for _ in range(2):
            if linear_requires_sync(linear_type):
                sync_float8_amax_and_scale_history(m_fp8)
            y_fp8 = m_fp8(x)
            y_fp8.sum().backward()
            y_ref = m_ref(x)
            y_ref.sum().backward()

        assert y_ref.shape == y_fp8.shape

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)
        # verify sqnr is reasonable
        assert y_sqnr >= 18.0, f"{y_sqnr} is too low"
        assert g_sqnr >= 17.0, f"{g_sqnr} is too low"
        if m_ref.bias is not None:
            torch.testing.assert_close(m_ref.bias.grad, m_fp8.bias.grad)

        # verify all of the amax buffers got updated
        if linear_requires_sync(linear_type):
            amax_buffer_names = [
                "fp8_amax_x",
                "fp8_amax_w",
                "fp8_amax_dL_dY",
            ]
            for buffer_name in amax_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                for init_val in (E4M3_MAX_POS, E5M2_MAX_POS):
                    assert torch.ne(
                        buffer_value, torch.tensor(init_val)
                    ), f"{buffer_name} not filled, current value {buffer_value}"

            # verify all of the amax history buffers got updated
            amax_history_buffer_names = [
                "fp8_amax_history_x",
                "fp8_amax_history_w",
                "fp8_amax_history_dL_dY",
            ]
            for buffer_name in amax_history_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                assert torch.max(buffer_value) > 0.0, f"{buffer_name} not filled"

            # verify all of the scale buffers got updated
            scale_buffer_names = [
                "fp8_scale_x",
                "fp8_scale_w",
                "fp8_scale_dL_dY",
            ]
            for buffer_name in scale_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                assert torch.ne(
                    buffer_value, torch.tensor(1.0)
                ), f"{buffer_name} not filled, current value {buffer_value}"

            # verify initialization flags got updated
            assert m_fp8.is_amax_initialized == True

    @pytest.mark.parametrize("emulate", [True, False])
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    def test_linear_nobias(self, x_shape, linear_type: LinearType, emulate: bool):
        if not emulate:
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available")
                pytest.skip()
            elif torch.cuda.get_device_capability() < (9, 0):
                warnings.warn(
                    f"CUDA capability {torch.cuda.get_device_capability()} < (9.0)"
                )
                pytest.skip()

        x = torch.randn(*x_shape, device="cuda")
        m_ref = nn.Linear(16, 32, bias=False, device="cuda")
        self._test_linear_impl(x, m_ref, linear_type, emulate)

    @pytest.mark.parametrize("emulate", [True, False])
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_linear_bias(
        self, x_shape, linear_type: LinearType, emulate: bool, linear_dtype: torch.dtype
    ):
        if not emulate:
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available")
                pytest.skip()
            elif torch.cuda.get_device_capability() < (9, 0):
                warnings.warn(
                    f"CUDA capability {torch.cuda.get_device_capability()} < (9.0)"
                )
                pytest.skip()

        x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
        m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)
        self._test_linear_impl(x, m_ref, linear_type, emulate)

        m = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        m = Float8Linear.from_float(m, emulate)

        # autocast off
        x = torch.randn(16, 32, device="cuda", dtype=linear_dtype)
        sync_float8_amax_and_scale_history(m)
        y = m(x)
        assert y.dtype == linear_dtype, f"y.dtype is {y.dtype}, expected {linear_dtype}"

        # autocast on
        with torch.autocast("cuda"):
            sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

        with torch.autocast("cuda", dtype=torch.bfloat16):
            sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert (
            y.dtype == torch.bfloat16
        ), f"y.dtype is {y.dtype}, expected {torch.bfloat16}"

    def test_linear_float8_weight_tag(self):
        m_ref = nn.Linear(16, 32, bias=False, device="cuda")
        m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref))
        assert m_fp8.weight._is_fp8_weight

    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_type_cast(self, linear_type: LinearType, linear_dtype: torch.dtype):
        emulate = (
            not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0)
        )
        x_shape = (16, 16)

        x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
        m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)
        self._test_linear_impl(x, m_ref, linear_type, emulate)

        m = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        m = Float8Linear.from_float(m, emulate)

        # Cast the module to dtype
        m = m.to(dtype=linear_dtype)

        # autocast off
        x = torch.randn(16, 32, device="cuda", dtype=linear_dtype)
        sync_float8_amax_and_scale_history(m)
        y = m(x)
        assert y.dtype == linear_dtype, f"y.dtype is {y.dtype}, expected {linear_dtype}"

        # autocast on
        with torch.autocast("cuda"):
            sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

        with torch.autocast("cuda", dtype=torch.bfloat16):
            sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert (
            y.dtype == torch.bfloat16
        ), f"y.dtype is {y.dtype}, expected {torch.bfloat16}"


class TestScaledMM:
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "CUDA not available",
    )
    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_scaled_mm_vs_emulated(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = torch.float8_e4m3fn
        output_dtype = base_dtype
        compare_type = torch.float32

        a = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        b = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()

        a_fp8 = Float8Tensor.to_float8(a, a_scale, input_dtype)
        b_fp8 = Float8Tensor.to_float8(b, b_scale, input_dtype)

        out_scaled_mm, output_amax_scaled = mm_float8(
            a_fp8, b_fp8, output_dtype=output_dtype, emulate=False
        )
        out_emulated, output_amax_emulated = mm_float8(
            a_fp8, b_fp8, output_dtype=output_dtype, emulate=True
        )

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_emulated = out_emulated.to(compare_type)

            out_scaled_mm = out_scaled_mm / amax_to_scale(
                output_amax_scaled, input_dtype
            )
            out_emulated = out_emulated / amax_to_scale(
                output_amax_emulated, input_dtype
            )

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)


class TestNumerics:
    @pytest.mark.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_small_amax_float16(self, float8_dtype):
        # If we calculate scale naively with FP8_MAX_POS / amax,
        # the result may not be representable in fp16. Verify that
        # the way we calculate scales actually works for tensors with
        # small values.
        #
        #   naive_s = fp8_max_pos / (amax + eps)
        #
        # failing case:
        #
        #   fp8_max_pos / (amax + eps) >= fp16_max_pos, or
        #
        #   amax + eps >= fp8_max_pos / fp16_max_pos

        float8_max_pos = (
            E4M3_MAX_POS if float8_dtype is torch.float8_e4m3fn else E5M2_MAX_POS
        )

        target_amax = float8_max_pos / (FP16_MAX_POS + 1e-12)
        x = torch.tensor([target_amax], dtype=torch.float16, device="cuda")
        scale = tensor_to_scale(x, float8_dtype)
        assert not torch.any(torch.isinf(scale))


if __name__ == "__main__":
    pytest.main([__file__])
