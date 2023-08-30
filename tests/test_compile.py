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
    amax_to_scale,
    output_dtype_to_addmm_bias_dtype
)
from float8_python_api import mm_float8, addmm_float8
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear
from float8_linear_nots import Float8LinearNoTensorSubclass



def main():
    random.seed(0)
    torch.manual_seed(0)
    emulate = False
    use_no_tensor_subclass = True
    x_shape =(16, 16)
    linear_dtype = torch.bfloat16

    compile = False


    x = torch.randn(*x_shape, device='cuda', dtype=linear_dtype)
    m_ref = nn.Linear(16, 32, bias=True, device='cuda', dtype=linear_dtype)

    if not use_no_tensor_subclass:
        m_fp8 = Float8Linear.from_float(copy.deepcopy(m_ref), emulate)
    else:
        m_fp8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(m_ref), emulate)
    if compile:
        m_fp8 = torch.compile(m_fp8, backend="inductor", dynamic=False)
        m_ref = torch.compile(m_ref, backend="inductor", dynamic=False)
    y_fp8 = m_fp8(x)
    y_fp8.sum().backward()
    y_ref = m_ref(x)
    y_ref.sum().backward()

    print(y_ref[-1])
    print(y_fp8[-1])



if __name__ == '__main__':
    main()