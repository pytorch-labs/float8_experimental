import copy
import random
import unittest

import torch
import torch.nn as nn

from float8_utils import (
    float32_to_float8,
    float8_to_float32,
    E4M3,
    E5M2,
    compute_error,
    tensor_to_scale,
)
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear

random.seed(0)
torch.manual_seed(0)

class Float8CastsUnitTest(unittest.TestCase):
    """
    Test the casts between fp32 and fp8 (e4m3 and e5m2)
    """

    def _compare_many_exact(self, flavor, x_fp32, comp_name):
        x_fp8 = float32_to_float8(x_fp32, flavor)
        x_fp8_fp32 = float8_to_float32(x_fp8, flavor)
        torch.testing.assert_close(x_fp32, x_fp8_fp32)

    def _compare_many_approx(self, flavor, x_fp32, comp_name):
        if flavor == E4M3:
            sqnr_target = 25.0
        else:  # e5m2
            sqnr_target = 23.0

        x_fp8 = float32_to_float8(x_fp32, flavor)
        x_fp8_fp32 = float8_to_float32(x_fp8, flavor)

        # sign should always be the same
        torch.testing.assert_close(
            torch.sign(x_fp32),
            torch.sign(x_fp8_fp32),
            atol=0, rtol=0)

        # for now just measure that sqnr is somewhat reasonable
        # TODO(future): make this significantly more robust, this is about
        # 2/10 on the scale of "robust enough"
        sqnr = compute_error(x_fp32, x_fp8_fp32)
        assert sqnr >= sqnr_target


    def _compare_one(self, flavor, bits_str, expected_fp32, comp_name):
        fp8_bits_ref = torch.tensor([int(bits_str, 2)], dtype=torch.uint8)

        fp32_tensor = torch.tensor([expected_fp32], dtype=torch.float)
        fp8_bits = float32_to_float8(fp32_tensor, flavor)
        torch.testing.assert_close(fp8_bits, fp8_bits_ref, atol=0, rtol=0)

        fp32_from_fp8_tensor = float8_to_float32(fp8_bits, flavor)
        torch.testing.assert_close(fp32_tensor, fp32_from_fp8_tensor, atol=0, rtol=0)

    def test_e4m3_numerics_single(self):
        # ensure that our format matches https://arxiv.org/pdf/2209.05433.pdf, Table 1

        flavor = E4M3
        # e4m3 does not support infinity
        self._compare_one(flavor, "00000000", 0.0, "zero")
        self._compare_one(flavor, "10000000", -0.0, "neg_zero")
        self._compare_one(flavor, "01111110", 448.0, "max_normal")
        self._compare_one(flavor, "11111110", -448.0, "neg_max_normal")
        self._compare_one(flavor, "00001000", 2 ** -6, "min_normal")
        self._compare_one(flavor, "10001000", -1 * (2 ** -6), "neg_min_normal")
        self._compare_one(flavor, "00000111", 0.875 * (2 ** -6), "max_subnorm")
        self._compare_one(flavor, "10000111", -0.875 * (2 ** -6), "neg_max_subnorm")
        self._compare_one(flavor, "00000001", 2 ** -9, "min_subnorm")
        self._compare_one(flavor, "10000001", -1 * (2 ** -9), "neg_min_subnorm")

    def test_e5m2_numerics_single(self):
        flavor = E5M2
        # e5m2 infinity (below) is off by one, TODO(future PR) debug or just move
        # to NVIDIA's intrinsic casts
        # _compare_one(flavor, "01111100", float("inf"), "inf")
        # _compare_one(flavor, "11111100", -1 * float("inf"), "neg_inf")
        self._compare_one(flavor, "00000000", 0.0, "zero")
        self._compare_one(flavor, "10000000", -0.0, "neg_zero")
        self._compare_one(flavor, "01111011", 57344.0, "max_normal")
        self._compare_one(flavor, "11111011", -57344.0, "neg_max_normal")
        self._compare_one(flavor, "00000100", 2 ** -14, "min_normal")
        self._compare_one(flavor, "10000100", -1 * (2 ** -14), "neg_min_normal")
        self._compare_one(flavor, "00000011", 0.75 * (2 ** -14), "max_subnorm")
        self._compare_one(flavor, "10000011", -0.75 * (2 ** -14), "neg_max_subnorm")
        self._compare_one(flavor, "00000001", 2 ** -16, "min_subnorm")
        self._compare_one(flavor, "10000001", -1 * (2 ** -16), "neg_min_subnorm")

    def test_e4m3_numerics_multiple(self):
        # test special cases
        x = torch.tensor([
            0.0,
            -0.0,
            448.0,
            -448.0,
            2 ** -6,
            -1 * (2 ** 6),
            0.875 * (2 ** 6),
            -0.875 * (2 ** 6),
            2 ** -9,
            -1 * (2 ** -9),
        ])
        self._compare_many_exact(E4M3, x, 'special_cases')

        # test normal values + shapes
        for _ in range(10):
            x = torch.randn(1, 2, 3, 4) * random.uniform(0.1, 300.0)
            x.clamp_(min=-448.0, max=448.0)
            self._compare_many_approx(E4M3, x, 'normal_cases')

    def test_e5m2_numerics_multiple(self):
        # test special cases
        x = torch.tensor([
            0.0,
            -0.0,
            57344.0,
            -57344.0,
            2 ** -14,
            -1 * (2 ** -14),
            0.75 * (2 ** -14),
            -0.75 * (2 ** -14),
            2 ** -16,
            -1 * (2 ** -16),
        ])
        self._compare_many_exact(E5M2, x, 'special_cases')

        # test normal values + shapes
        for _ in range(10):
            x = torch.randn(1, 2, 3, 4) * random.uniform(0.1, 30000.0)
            x.clamp_(min=-57344.0, max=57344.0)
            self._compare_many_approx(E5M2, x, 'normal_cases')

class Float8TensorUnitTest(unittest.TestCase):
    def test_add(self):
        x1_fp32 = torch.randn(4, 4)
        x1_s = tensor_to_scale(x1_fp32, E5M2)
        x2_fp32 = torch.randn(4, 4)
        x2_s = tensor_to_scale(x2_fp32, E5M2)
        x1_fp8 = Float8Tensor.from_float32(x1_fp32, x1_s, E5M2)
        x2_fp8 = Float8Tensor.from_float32(x2_fp32, x2_s, E5M2)
        x3_fp8 = x1_fp8 + x2_fp8
        x3_fp32 = x3_fp8.to_float32()
        x3_fp32_ref = x1_fp32 + x2_fp32
        sqnr = compute_error(x3_fp32_ref, x3_fp32)
        # TODO(future): make this more accurate, accuracy is pretty low
        self.assertTrue(sqnr >= 10.0)

class Float8LinearUnitTest(unittest.TestCase):

    def test_e2e(self):
        m_ref = nn.Linear(4, 4, bias=False)
        m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref))

        x = torch.randn(4, 4)

        y_fp8 = m_fp8(x)
        y_fp8.sum().backward()
        y_ref = m_ref(x)
        y_ref.sum().backward()

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)

        # verify sqnr is reasonable
        self.assertTrue(y_sqnr >= 27.0)
        self.assertTrue(g_sqnr >= 27.0)

        # verify all of the scales got updated
        for buffer_name in (
            'fp8_s_in',
            'fp8_s_weight',
            'fp8_s_out',
            'fp8_s_dL_dX',
            'fp8_s_dL_dW',
            'fp8_s_dL_dY',
        ):
            buffer_value = getattr(m_fp8, buffer_name)
            self.assertTrue(
                torch.ne(buffer_value, torch.tensor(1.0)),
                f"{buffer_name} not filled")


if __name__ == '__main__':
    unittest.main()
