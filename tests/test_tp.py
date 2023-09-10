"""
Test numerics of manually defined float16 TP vs float8 TP of toy models
"""

import copy
import os

import torch
import torch.nn as nn

from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear, 
    RowParallelLinear,
)
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

# set up float8 path
import context

from tp_linear import (
    swap_tp_linear_with_float8_linear,
)

from float8_utils import (
    compute_error,
)

from distributed_utils import (
    _AllGatherFwSplitBw,    
)

# copied from https://github.com/facebookresearch/llama/blob/main/example.py
def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def test_column_parallel_linear():
    M, K, N = 128, 64, 256
    m_ref = nn.Sequential(ColumnParallelLinear(K, N)).cuda()
    m = copy.deepcopy(m_ref)
    swap_tp_linear_with_float8_linear(m)

    x = torch.randn(M, K, device='cuda')
    y_ref = m_ref(x)
    y = m(x)

    y_ref.sum().backward()
    y.sum().backward()

    sqnr_y = compute_error(y_ref, y)
    sqnr_w_grad = compute_error(m_ref[0].weight.grad, m[0].weight.grad)

    assert sqnr_y >= 20.0, f'sqnr_y {sqnr_y} is too low'
    assert sqnr_w_grad >= 20.0, f'sqnr_w_grad {sqnr_w_grad} is too low'

def test_row_parallel_linear():
    M, K, N = 128, 64, 256
    m_ref = nn.Sequential(RowParallelLinear(K, N)).cuda()
    m = copy.deepcopy(m_ref)
    swap_tp_linear_with_float8_linear(m)

    x = torch.randn(M, K, device='cuda')
    y_ref = m_ref(x)
    y = m(x)

    y_ref.sum().backward()
    y.sum().backward()

    sqnr_y = compute_error(y_ref, y)
    sqnr_w_grad = compute_error(m_ref[0].weight.grad, m[0].weight.grad)

    assert sqnr_y >= 20.0, f'sqnr_y {sqnr_y} is too low'
    assert sqnr_w_grad >= 20.0, f'sqnr_w_grad {sqnr_w_grad} is too low'

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.act = nn.ReLU()
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x

def test_ffn():
    M, dim, hidden_dim = 32, 32, 64
    m_ref = FFN(dim, hidden_dim).cuda()
    m = copy.deepcopy(m_ref)
    swap_tp_linear_with_float8_linear(m)

    x = torch.randn(M, dim, device='cuda')
    y_ref = m_ref(x)
    y = m(x)

    y_ref.sum().backward()
    y.sum().backward()

    sqnr_y = compute_error(y_ref, y)
    sqnr_w1_grad = compute_error(m_ref.w1.weight.grad, m.w1.weight.grad)
    sqnr_w2_grad = compute_error(m_ref.w2.weight.grad, m.w2.weight.grad)

    assert sqnr_y >= 20.0, f'sqnr_y {sqnr_y} is too low'
    assert sqnr_w1_grad >= 14.0, f'sqnr_w1_grad {sqnr_w1_grad} is too low'
    assert sqnr_w2_grad >= 30.0, f'sqnr_w2_grad {sqnr_w2_grad} is too low'


def test_ffn_sp(local_rank, world_size):
    # for this test:
    # baseline: float8 FFN with TP but no SP
    # experiment: float8 FFN with TP and SP
    # we expect the numerics to match exactly

    M, dim, hidden_dim = 32, 32, 64
    m_ref = FFN(dim, hidden_dim).cuda()
    swap_tp_linear_with_float8_linear(m_ref)

    m = copy.deepcopy(m_ref)
    # TODO(future): nicer API for setting this
    m.w1.use_sequence_parallel = True
    m.w2.use_sequence_parallel = True

    x_ref = torch.randn(M, dim, device='cuda')
    y_ref = m_ref(x_ref)
    y_ref.sum().backward()

    # for SP, we need to split the input before passing it to the module
    # TODO(before land): just use scatter
    # TODO(before land): can use _split_along_first_dim
    assert x_ref.shape[0] % world_size == 0
    x_list = torch.split(x_ref, x_ref.shape[0] // world_size)
    x = x_list[local_rank]
     
    y_local = m(x)
    y_gathered = _AllGatherFwSplitBw.apply(y_local)
    y_gathered.sum().backward()

    torch.testing.assert_close(y_ref, y_gathered)
    torch.testing.assert_close(m_ref.w1.weight.grad, m.w1.weight.grad)
    torch.testing.assert_close(m_ref.w2.weight.grad, m.w2.weight.grad)
    

if __name__ == '__main__':
    local_rank, world_size = setup_model_parallel()
    test_column_parallel_linear()
    test_row_parallel_linear()
    test_ffn()
    test_ffn_sp(local_rank, world_size)
