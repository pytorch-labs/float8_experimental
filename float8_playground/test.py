import copy
import random
import unittest

import torch
import torch.nn as nn

from float8_utils import (
    compute_error,
    tensor_to_scale,
)
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear

random.seed(0)
torch.manual_seed(0)

class Float8TensorUnitTest(unittest.TestCase):
    def test_add(self):
        x1_fp32 = torch.randn(4, 4)
        x1_s = tensor_to_scale(x1_fp32, torch.float8_e5m2)
        x2_fp32 = torch.randn(4, 4)
        x2_s = tensor_to_scale(x2_fp32, torch.float8_e5m2)
        x1_fp8 = Float8Tensor.from_float32(x1_fp32, x1_s, torch.float8_e5m2)
        x2_fp8 = Float8Tensor.from_float32(x2_fp32, x2_s, torch.float8_e5m2)
        x3_fp8 = x1_fp8 + x2_fp8
        x3_fp32 = x3_fp8.to_float32()
        x3_fp32_ref = x1_fp32 + x2_fp32
        sqnr = compute_error(x3_fp32_ref, x3_fp32)
        # TODO(future): make this more accurate, accuracy is pretty low
        self.assertTrue(sqnr >= 10.0)

class Float8LinearUnitTest(unittest.TestCase):
    def _test_linear_impl(self, x, m_ref):

        m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref))

        y_fp8 = m_fp8(x)
        y_fp8.sum().backward()
        y_ref = m_ref(x)
        y_ref.sum().backward()

        self.assertTrue(y_ref.shape == y_fp8.shape)

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)

        # verify sqnr is reasonable
        self.assertTrue(y_sqnr >= 24.0)
        self.assertTrue(g_sqnr >= 24.0)
        if m_ref.bias is not None:
            torch.testing.assert_close(m_ref.bias.grad, m_fp8.bias.grad)

        # verify all of the scales got updated
        buffer_names = [
            'fp8_s_in',
            'fp8_s_weight',
            'fp8_s_out',
            'fp8_s_dL_dX',
            'fp8_s_dL_dW',
            'fp8_s_dL_dY',
        ]
        if m_ref.bias is not None:
            buffer_names.append('fp8_s_bias')
        for buffer_name in buffer_names:
            buffer_value = getattr(m_fp8, buffer_name)
            self.assertTrue(
                torch.ne(buffer_value, torch.tensor(1.0)),
                f"{buffer_name} not filled")

    def test_linear_nobias(self):
        x_shapes = ((2, 3), (4, 2, 3), (5, 4, 2, 3))
        for x_shape in x_shapes:
            x = torch.randn(*x_shape)
            m_ref = nn.Linear(3, 4, bias=False)
            self._test_linear_impl(x, m_ref)

    def test_linear_bias(self):
        x_shapes = ((2, 3), (4, 2, 3), (5, 4, 2, 3))
        for x_shape in x_shapes:
            x = torch.randn(*x_shape)
            m_ref = nn.Linear(3, 4, bias=True)
            self._test_linear_impl(x, m_ref)


if __name__ == '__main__':
    unittest.main()
