import copy
import itertools
import random
import unittest
#pip install parameterized
from parameterized import parameterized

import torch
import torch.nn as nn

# set up float8 path
import context

from float8_utils import (
    compute_error,
    tensor_to_scale,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear
from float8_linear_nots import Float8LinearNoTensorSubclass

random.seed(0)
torch.manual_seed(0)

class Float8TensorUnitTest(unittest.TestCase):
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


class Float8LinearUnitTest(unittest.TestCase):
    def _test_linear_impl(self, x, m_ref, use_no_tensor_subclass=False):

        if not use_no_tensor_subclass:
            m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref))
        else:
            m_fp8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(m_ref))

        y_fp8 = m_fp8(x)
        y_fp8.sum().backward()
        y_ref = m_ref(x)
        y_ref.sum().backward()

        self.assertTrue(y_ref.shape == y_fp8.shape)

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)
        # verify sqnr is reasonable
        self.assertTrue(y_sqnr >= 18.0, f'{y_sqnr} is too low')
        self.assertTrue(g_sqnr >= 17.0, f'{g_sqnr} is too low')
        if m_ref.bias is not None:
            torch.testing.assert_close(m_ref.bias.grad, m_fp8.bias.grad)

        # verify all of the amax buffers got updated
        amax_buffer_names = [
            'fp8_amax_x',
            'fp8_amax_w',
            'fp8_amax_y',
            'fp8_amax_dL_dX',
            'fp8_amax_dL_dW',
            'fp8_amax_dL_dY',
        ]
        for buffer_name in amax_buffer_names:
            buffer_value = getattr(m_fp8, buffer_name)
            for init_val in (E4M3_MAX_POS, E5M2_MAX_POS):
                self.assertTrue(
                    torch.ne(buffer_value, torch.tensor(init_val)),
                    f"{buffer_name} not filled, current value {buffer_value}")

        # verify all of the amax history buffers got updated
        amax_history_buffer_names = [
            'fp8_amax_history_x',
            'fp8_amax_history_w',
            'fp8_amax_history_y',
            'fp8_amax_history_dL_dX',
            'fp8_amax_history_dL_dW',
            'fp8_amax_history_dL_dY',
        ]
        for buffer_name in amax_history_buffer_names:
            buffer_value = getattr(m_fp8, buffer_name)
            self.assertTrue(torch.max(buffer_value) > 0.0, f"{buffer_name} not filled")

        # verify initialization buffers got updated
        self.assertTrue(m_fp8.fw_amax_initialized[0] == 1)
        self.assertTrue(m_fp8.bw_amax_initialized[0] == 1)

    @parameterized.expand([(True,), (False,)])
    def test_linear_nobias(self, emulate: bool):
        x_shapes = ((16, 16),(2, 16, 16), (3, 2, 16, 16))
        for x_shape in x_shapes:
            for use_no_ts in (True, False):
                x = torch.randn(*x_shape, device='cuda')
                m_ref = nn.Linear(3, 4, bias=False, device='cuda')
                self._test_linear_impl(x, m_ref, use_no_tensor_subclass=use_no_ts)

    @parameterized.expand([(True,), (False,)])
    def test_linear_bias(self, emulate: bool):
        x_shapes = ((16, 16),(2, 16, 16), (3, 2, 16, 16))
        for x_shape in x_shapes:
            for use_no_ts in (True, False):
                x = torch.randn(*x_shape, device='cuda')
                m_ref = nn.Linear(3, 4, bias=True, device='cuda')
                self._test_linear_impl(x, m_ref, use_no_tensor_subclass=use_no_ts)


        m = nn.Linear(4, 4, device='cuda')
        m = Float8Linear.from_float(m, emulate)

        # autocast off
        x = torch.randn(4, 4, device='cuda')
        y = m(x)
        self.assertTrue(y._orig_dtype == torch.float)

        # autocast on
        with torch.autocast('cuda'):
            y = m(x)
        self.assertTrue(y._orig_dtype == torch.half)

    def _test_pt2_impl(self, use_no_tensor_subclass):
        from torch._dynamo.testing import EagerAndRecordGraphs, CompileCounterWithBackend

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        m = nn.Linear(4, 4, device='cpu', bias=False)
        x = torch.randn(4, 4, device='cpu')
        # TODO(future): switch back to tensor subclass based UX once the PT
        # support is there
        if use_no_tensor_subclass:
            m = Float8LinearNoTensorSubclass.from_float(m)
        else:
            m = Float8Linear.from_float(m)
        m = torch.compile(m, backend='eager')
        # verify things don't crash
        m(x)
        # logs with TORCH_LOGS="aot": https://gist.github.com/vkuzo/e4e372bd88085a0ac2de46d553a3c0d8
        # TODO(future): inspect the graph programmaticaly

    def test_pt2_nots(self):
        self._test_pt2_impl(use_no_tensor_subclass=True)

    @unittest.skip("PT2.0 tracing subclasses does not work yet")
    def test_pt2_ts(self):
        self._test_pt2_impl(use_no_tensor_subclass=False)


if __name__ == '__main__':
    unittest.main()
