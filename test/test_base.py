# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import random
import unittest
import warnings

import pytest

import torch
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    filter_out_small_unaligned_layers,
    get_float8_linear,
    linear_requires_sync,
    LinearType,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_python_api import addmm_float8_unwrapped
from float8_experimental.float8_tensor import (
    Float8Tensor,
    merge_mm_configs,
    ScaledMMConfig,
)
from float8_experimental.float8_utils import (
    amax_to_scale,
    compute_error,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    FP16_MAX_POS,
    fp8_tensor_statistics,
    tensor_to_scale,
)

random.seed(0)
torch.manual_seed(0)

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


class TestFloat8Tensor(unittest.TestCase):
    def test_preserves_dtype(self) -> None:
        # hp means high precision, lp means low precision
        hp_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        lp_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for hp_dtype, lp_dtype in itertools.product(hp_dtypes, lp_dtypes):
            x1_hp = torch.randn(4, 4, dtype=hp_dtype)
            x1_s = tensor_to_scale(x1_hp, lp_dtype)
            x2_lp = Float8Tensor.to_float8(x1_hp, x1_s, lp_dtype)
            x3_hp = x2_lp.to_original_precision()
            self.assertTrue(x3_hp.dtype == hp_dtype)

    def test_differentiable_casts(self) -> None:
        lp_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for f8_dtype in lp_dtypes:
            x = torch.randn(1).requires_grad_()
            grad = torch.randn(1)
            x_s = tensor_to_scale(x, f8_dtype)
            x_f8 = Float8Tensor.to_float8(x, x_s, f8_dtype)
            x_f8_hp = x_f8.to_original_precision()
            x_f8_hp.backward(grad)
            # the gradient should be unchanged through both casts
            torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)


class TestFloat8Linear:
    def _test_linear_impl(
        self,
        x,
        m_ref,
        linear_type: LinearType,
        emulate: bool,
    ):
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
            assert m_fp8.is_amax_initialized, "Amax was not properly initialized"

    @pytest.mark.parametrize("emulate", [True, False] if is_H100 else [True])
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_linear_nobias(
        self,
        x_shape,
        linear_type: LinearType,
        emulate: bool,
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

        x = torch.randn(*x_shape, device="cuda")
        m_ref = nn.Linear(16, 32, bias=False, device="cuda")
        self._test_linear_impl(x, m_ref, linear_type, emulate)

    @pytest.mark.parametrize("emulate", [True, False] if is_H100 else [True])
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_linear_bias(
        self,
        x_shape,
        linear_type: LinearType,
        emulate: bool,
        linear_dtype: torch.dtype,
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

    @pytest.mark.parametrize("emulate", [True, False] if is_H100 else [True])
    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_autocast_outputs(
        self,
        linear_type: LinearType,
        emulate: bool,
        linear_dtype: torch.dtype,
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

        m_ref = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        m = get_float8_linear(linear_type, m_ref, emulate)

        # autocast off
        x = torch.randn(16, 32, device="cuda", dtype=linear_dtype)
        if linear_requires_sync(linear_type):
            sync_float8_amax_and_scale_history(m)
        y = m(x)
        assert y.dtype == linear_dtype, f"y.dtype is {y.dtype}, expected {linear_dtype}"

        # autocast on
        with torch.autocast("cuda"):
            if linear_requires_sync(linear_type):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if linear_requires_sync(linear_type):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert (
            y.dtype == torch.bfloat16
        ), f"y.dtype is {y.dtype}, expected {torch.bfloat16}"

    @pytest.mark.parametrize("linear_type", [LinearType.DELAYED, LinearType.DYNAMIC])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("emulate", [True, False] if is_H100 else [True])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_type_cast(
        self, linear_type: LinearType, linear_dtype: torch.dtype, emulate: bool
    ):
        emulate = (
            not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0)
        )

        m = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        m = get_float8_linear(linear_type, m, emulate)

        # Cast the module to dtype
        m = m.to(dtype=linear_dtype)
        if linear_requires_sync(linear_type):
            # Check amax buffer types
            for key in [
                "fp8_amax_x",
                "fp8_amax_history_x",
                "fp8_scale_x",
                "fp8_amax_w",
                "fp8_amax_history_w",
                "fp8_scale_w",
                "fp8_amax_dL_dY",
                "fp8_amax_history_dL_dY",
                "fp8_scale_dL_dY",
            ]:
                assert (
                    m._buffers[key].dtype == torch.float32
                ), f"{key}.dtype is {m._buffers[key].dtype}, expected torch.float32"

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
        not is_H100,
        "CUDA not available",
    )
    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("use_fast_accum", [True, False])
    def test_scaled_mm_vs_emulated(self, base_dtype, use_fast_accum):
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

        out_scaled_mm, output_amax_scaled = addmm_float8_unwrapped(
            a_fp8._data,
            a_fp8._scale,
            b_fp8._data,
            b_fp8._scale,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        out_emulated, output_amax_emulated = torch.ops.aten.mm_float8_emulated(
            a_fp8._data, a_fp8._scale, b_fp8._data, b_fp8._scale, output_dtype
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

    @unittest.skipIf(not is_H100, "CUDA not available")
    def test_different_configs_error(self):
        x_fp32 = torch.randn(16, 16, device="cuda")
        x_scale = torch.tensor(1.0, device="cuda")
        fp8_dtype = torch.float8_e4m3fn
        a = Float8Tensor.to_float8(x_fp32, x_scale, fp8_dtype)
        b = Float8Tensor.to_float8(
            x_fp32, x_scale, fp8_dtype, mm_config=ScaledMMConfig(True)
        )
        with pytest.raises(
            AssertionError,
            match="Both mm_configs must have the same emulate value, but got False and True",
        ):
            a @ b

    def test_merge_configs(sel):
        a = ScaledMMConfig(False, True, True)
        b = ScaledMMConfig(True, False, False)
        with pytest.raises(
            AssertionError,
            match="Both mm_configs must have the same emulate value, but got False and True",
        ):
            merge_mm_configs(a, b)
        a = ScaledMMConfig(False, True, True)
        b = ScaledMMConfig(False, False, False)
        c = merge_mm_configs(a, b)
        assert c.emulate is False
        assert c.use_fast_accum is False
        assert c.fp8_output is False

        a = ScaledMMConfig(False, True, False)
        b = ScaledMMConfig(False, True, False)
        c = merge_mm_configs(a, b)
        assert c.emulate is False
        assert c.use_fast_accum is True
        assert c.fp8_output is False


class TestNumerics:
    @pytest.mark.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
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


class TestFloat8LinearUtils(unittest.TestCase):
    def test_swap_root_linear(self):
        for module_cls, emulate in itertools.product(
            [Float8Linear, Float8DynamicLinear], [True, False]
        ):
            module = nn.Linear(3, 3)
            module = swap_linear_with_float8_linear(module, module_cls, emulate=emulate)
            self.assertIsInstance(module, module_cls)
            self.assertEqual(module.forward_config.emulate, emulate)
            self.assertEqual(module.backward_config.emulate, emulate)

    def test_swap_root_linear_with_children_raises(self):
        for module_cls, emulate in itertools.product(
            [Float8Linear, Float8DynamicLinear], [True, False]
        ):
            module = nn.Linear(3, 3)
            module.child = nn.Sequential(nn.Linear(3, 3))
            with self.assertRaisesRegex(
                AssertionError,
                "Does not support a root nn.Linear with children",
            ):
                swap_linear_with_float8_linear(module, module_cls, emulate=emulate)

    def test_swap_submodule_linears(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, dim)

        for module_cls, emulate in itertools.product(
            [Float8Linear, Float8DynamicLinear], [True, False]
        ):
            model = nn.Sequential(MLP(3), nn.Linear(3, 3), MLP(3))
            model = swap_linear_with_float8_linear(model, module_cls, emulate=emulate)
            self.assertIsInstance(model[0].lin1, module_cls)
            self.assertIsInstance(model[0].lin2, module_cls)
            self.assertIsInstance(model[1], module_cls)
            self.assertIsInstance(model[2].lin1, module_cls)
            self.assertIsInstance(model[2].lin2, module_cls)

    def test_swap_linears_with_filters(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, 4 * dim)

        for module_cls, emulate in itertools.product(
            [Float8Linear, Float8DynamicLinear], [True, False]
        ):
            model = nn.Sequential(MLP(8), nn.Linear(32, 32), MLP(40))
            # filter out the linear layers whose shape is smaller than 32 or non-divisible by 16.
            model = swap_linear_with_float8_linear(
                model,
                module_cls,
                emulate=emulate,
                linear_layer_filter=filter_out_small_unaligned_layers(32),
            )
            # in_features=8, out_features=32, 8 is less than 32.
            self.assertNotIsInstance(model[0].lin1, module_cls)
            # in_features=32, out_features=32,
            self.assertIsInstance(model[0].lin2, module_cls)
            # in_features=32, out_features=32,
            self.assertIsInstance(model[1], module_cls)
            # in_features=40, out_features=160, 40 is not divisible by 16.
            self.assertNotIsInstance(model[2].lin1, module_cls)
            # in_features=160, out_features=160,
            self.assertIsInstance(model[2].lin2, module_cls)

    def test_swap_submodule_linears_with_skip(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, dim)

        for module_cls, emulate in itertools.product(
            [Float8Linear, Float8DynamicLinear], [True, False]
        ):
            model = nn.Sequential(MLP(3), nn.Linear(3, 3), MLP(3))
            skip_fqn_list = ["2", "0.lin2"]
            model = swap_linear_with_float8_linear(
                model, module_cls, emulate=emulate, skip_fqn_list=skip_fqn_list
            )
            self.assertIsInstance(model[0].lin1, module_cls)
            self.assertNotIsInstance(model[0].lin2, module_cls)
            self.assertIsInstance(model[0].lin2, nn.Linear)
            self.assertIsInstance(model[1], module_cls)
            self.assertNotIsInstance(model[2].lin2, module_cls)
            self.assertNotIsInstance(model[2].lin2, module_cls)
            self.assertIsInstance(model[2].lin1, nn.Linear)
            self.assertIsInstance(model[2].lin2, nn.Linear)

    def test_fp8_tensor_statistics(self):
        hp_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        lp_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for hp_dtype, lp_dtype in itertools.product(hp_dtypes, lp_dtypes):
            x1_hp = torch.ones(4, 4, dtype=hp_dtype)
            tensor_len = x1_hp.numel()

            # Overflow caused by a too large scaling factor
            s_overflow = torch.tensor(1e9)
            fp8_overflow = Float8Tensor.to_float8(x1_hp, s_overflow, lp_dtype)
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_overflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (0, tensor_len))

            # Underflow caused by a too small scaling factor
            s_underflow = torch.tensor(1e-9)
            fp8_underflow = Float8Tensor.to_float8(x1_hp, s_underflow, lp_dtype)
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_underflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (tensor_len, 0))

            # Both overflow and underflow
            x2_hp = torch.cat((x1_hp * 1e9, x1_hp * 1.0, x1_hp * 1e-9), 0)
            fp8_over_underflow = Float8Tensor.to_float8(
                x2_hp, torch.tensor(1.0), lp_dtype
            )
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_over_underflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (tensor_len, tensor_len))


if __name__ == "__main__":
    pytest.main([__file__])
