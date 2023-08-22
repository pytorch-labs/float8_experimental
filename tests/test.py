import copy
import itertools
import random
import unittest
import pytest
import warnings

import torch
import torch.nn as nn
from torch._dynamo.testing import (
    EagerAndRecordGraphs, 
    CompileCounterWithBackend,
)

# set up float8 path
import context

from float8_utils import (
    compute_error,
    tensor_to_scale,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    amax_to_scale
)
from float8_python_api import mm_float8, addmm_float8
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear
from float8_linear_nots import Float8LinearNoTensorSubclass

random.seed(0)
torch.manual_seed(0)

class TestFloat8Tensor(unittest.TestCase):
    def test_grad_add(self):
        x1_fp32 = torch.randn(4, 4, device='cuda')
        x1_s = tensor_to_scale(x1_fp32, torch.float8_e5m2)
        x2_fp32 = torch.randn(4, 4, device='cuda')
        x2_s = tensor_to_scale(x2_fp32, torch.float8_e5m2)
        x1_fp8 = Float8Tensor.to_float8(x1_fp32, x1_s, torch.float8_e5m2)
        x2_fp8 = Float8Tensor.to_float8(x2_fp32, x2_s, torch.float8_e5m2)
        x3_fp32 = x1_fp8 + x2_fp8
        x3_fp32_ref = x1_fp32 + x2_fp32
        sqnr = compute_error(x3_fp32_ref, x3_fp32)
        self.assertTrue(sqnr >= 20.0)

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

    def test_reshape(self):
        x1_fp32 = torch.randn(4, 4, device='cuda')
        x1_s = tensor_to_scale(x1_fp32, torch.float8_e4m3fn)
        x1_fp8 = Float8Tensor.to_float8(x1_fp32, x1_s, torch.float8_e4m3fn)
        new_shape = (2, -1)
        x2_fp32 = x1_fp32.reshape(*new_shape)
        x2_fp8 = x1_fp8.reshape(*new_shape)
        self.assertTrue(x2_fp8.shape == x2_fp32.shape)
        self.assertTrue(type(x2_fp8) == Float8Tensor)

    def test_transpose(self):
        x1_fp32 = torch.randn(4, 4, device='cuda')
        x1_s = tensor_to_scale(x1_fp32, torch.float8_e4m3fn)
        x1_fp8 = Float8Tensor.to_float8(x1_fp32, x1_s, torch.float8_e4m3fn)
        x2_fp32 = x1_fp32.t()
        x2_fp8 = x1_fp8.t()
        self.assertTrue(x2_fp8.shape == x2_fp32.shape)
        self.assertTrue(type(x2_fp8) == Float8Tensor)


class TestFloat8Linear:
    def _test_linear_impl(self, x, m_ref, use_no_tensor_subclass: bool, emulate: bool):

        if not use_no_tensor_subclass:
            m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref), emulate)
        else:
            m_fp8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(m_ref), emulate)

        y_fp8 = m_fp8(x)
        y_fp8.sum().backward()
        y_ref = m_ref(x)
        y_ref.sum().backward()

        assert (y_ref.shape == y_fp8.shape)

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)
        # verify sqnr is reasonable
        assert y_sqnr >= 18.0, f'{y_sqnr} is too low'
        assert g_sqnr >= 17.0, f'{g_sqnr} is too low'
        if m_ref.bias is not None:
            torch.testing.assert_close(m_ref.bias.grad, m_fp8.bias.grad)

        # verify all of the amax buffers got updated
        amax_buffer_names = [
            'fp8_amax_x',
            'fp8_amax_w',
            'fp8_amax_dL_dY',
        ]
        for buffer_name in amax_buffer_names:
            buffer_value = getattr(m_fp8, buffer_name)
            for init_val in (E4M3_MAX_POS, E5M2_MAX_POS):
                assert torch.ne(buffer_value, torch.tensor(init_val)), f"{buffer_name} not filled, current value {buffer_value}"

        # verify all of the amax history buffers got updated
        amax_history_buffer_names = [
            'fp8_amax_history_x',
            'fp8_amax_history_w',
            'fp8_amax_history_dL_dY',
        ]
        for buffer_name in amax_history_buffer_names:
            buffer_value = getattr(m_fp8, buffer_name)
            assert torch.max(buffer_value) > 0.0, f"{buffer_name} not filled"

        # verify initialization flags got updated
        assert (m_fp8.is_amax_initialized == True)

    @pytest.mark.parametrize("emulate", [True, False])
    @pytest.mark.parametrize("x_shape", [(16, 16),(2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("use_no_ts", [True, False])
    def test_linear_nobias(self, x_shape, use_no_ts: bool, emulate: bool):
        if not emulate:
            if not torch.cuda.is_available():
                warnings.warn('CUDA not available')
                pytest.skip()
            elif torch.cuda.get_device_capability() < (9, 0):
                warnings.warn(f'CUDA capability {torch.cuda.get_device_capability()} < (9.0)')
                pytest.skip()
            elif use_no_ts:
                warnings.warn('use_no_ts does not support real compute yet')
                pytest.skip()

        x = torch.randn(*x_shape, device='cuda')
        m_ref = nn.Linear(16, 32, bias=False, device='cuda')
        self._test_linear_impl(x, m_ref, use_no_ts, emulate)

    @pytest.mark.parametrize("emulate", [True, False])
    @pytest.mark.parametrize("x_shape", [(16, 16),(2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("use_no_ts", [True, False])
    def test_linear_bias(self, x_shape, use_no_ts: bool, emulate: bool):
        if not emulate:
            if not torch.cuda.is_available():
                warnings.warn('CUDA not available')
                pytest.skip()
            elif torch.cuda.get_device_capability() < (9, 0):
                warnings.warn(f'CUDA capability {torch.cuda.get_device_capability()} < (9.0)')
                pytest.skip()
            elif use_no_ts:
                warnings.warn('use_no_ts does not support real compute yet')
                pytest.skip()
            elif not use_no_ts:
                warnings.warn('real compute with bias needs fixing, skip for now')
                pytest.skip()

        x = torch.randn(*x_shape, device='cuda')
        m_ref = nn.Linear(16, 32, bias=True, device='cuda')
        self._test_linear_impl(x, m_ref, use_no_ts, emulate)

        m = nn.Linear(32, 16, device='cuda')
        m = Float8Linear.from_float(m, emulate)

        # autocast off
        x = torch.randn(16, 32, device='cuda')
        y = m(x)
        assert y.dtype == torch.float, f"y.dtype is {y.dtype}, expected {torch.float}"

        # autocast on
        with torch.autocast('cuda'):
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

    def _test_pt2_impl(self, use_no_tensor_subclass):

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        m = nn.Linear(4, 4, device='cpu', bias=False)
        x = torch.randn(4, 4, device='cpu')
        # TODO(future): switch back to tensor subclass based UX once the PT
        # support is there
        if use_no_tensor_subclass:
            m = Float8LinearNoTensorSubclass.from_float(m, emulate=True)
        else:
            m = Float8Linear.from_float(m)
        m = torch.compile(m, backend=cnt, fullgraph=True)
        # verify things don't crash
        m(x)
        # TODO(future): inspect the graph programmaticaly
        for gm in backend.graphs:
            # print('gm', gm)
            pass

    def test_pt2_nots(self):
        self._test_pt2_impl(use_no_tensor_subclass=True)

    @unittest.skip("PT2.0 tracing subclasses does not work yet")
    def test_pt2_ts(self):
        self._test_pt2_impl(use_no_tensor_subclass=False)

class TestScaledMM:
    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "CUDA not available")
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, bias, base_dtype):
        torch.manual_seed(42)
        input_dtype = torch.float8_e4m3fn
        output_dtype = torch.float8_e4m3fn
        compare_type = torch.float32

        a = torch.randn(16, 16, device='cuda', dtype=base_dtype)
        b = torch.randn(32, 16, device='cuda', dtype=base_dtype).t()
        if bias:
            input_bias = torch.randn(32, device='cuda', dtype=base_dtype).to(torch.float16)
            input_bias = torch.zeros_like(input_bias)
            out_ref = torch.addmm(input_bias.to(a.dtype), a, b)
        else:
            out_ref = torch.matmul(a, b)

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()
        out_scale = tensor_to_scale(out_ref, output_dtype).float()

        a_fp8 = Float8Tensor.to_float8(a, a_scale, input_dtype)
        b_fp8 = Float8Tensor.to_float8(b, b_scale, input_dtype)

        output_amax_scaled = torch.tensor(0, device='cuda')
        output_amax_emulated = torch.tensor(0, device='cuda')

        if bias:
            out_scaled_mm = addmm_float8(input_bias, a_fp8, b_fp8, output_amax_scaled, out_scale, output_dtype=output_dtype, emulate=False)
            out_emulated = addmm_float8(input_bias, a_fp8, b_fp8, output_amax_emulated, out_scale, output_dtype=output_dtype, emulate=True)
        else:
            out_scaled_mm = mm_float8(a_fp8, b_fp8, output_amax_scaled, out_scale, output_dtype=output_dtype, emulate=False)
            out_emulated = mm_float8(a_fp8, b_fp8, output_amax_emulated, out_scale, output_dtype=output_dtype, emulate=True)

        out_scaled_mm = out_scaled_mm.to(compare_type)
        out_emulated = out_emulated.to(compare_type)

        out_scaled_mm = out_scaled_mm / amax_to_scale(output_amax_scaled, input_dtype)
        out_emulated = out_emulated / amax_to_scale(output_amax_emulated, input_dtype)
        
        if base_dtype==torch.float16:
            atol, rtol = 2e-2, 2e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__])
