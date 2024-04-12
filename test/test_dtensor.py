# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs float8 TP of toy models
"""

import copy
import os

import torch
import torch.nn as nn

from float8_experimental.float8_dynamic_linear import (
    Float8DynamicLinear,
    NoopFwToFloat8E5M2Bw,
)
from float8_experimental.float8_linear_utils import swap_linear_with_float8_linear
from float8_experimental.float8_tensor import Float8Tensor, ScaledMMConfig
from float8_experimental.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
)
from float8_experimental.float8_utils import tensor_to_scale
from torch.distributed._tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from tqdm import tqdm


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    return device_mesh


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


def test_scaled_mm(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = torch.float8_e4m3fn
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)
    y_fp32 = torch.eye(size, device=device).t()

    placement_combs = (
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
        (Shard(1), Shard(0)),
    )
    expected_dt_out_shape = (
        (size * world_size, size),
        (size, size * world_size),
        (size, size),
    )
    for idx, (lhs_placement, rhs_placement) in enumerate(placement_combs):
        x_scale = tensor_to_scale(x_fp32, fp8_dtype).float()
        y_scale = tensor_to_scale(y_fp32, fp8_dtype).float()

        x_fp8 = Float8Tensor.to_float8(x_fp32, x_scale, fp8_dtype)
        y_fp8 = Float8Tensor.to_float8(y_fp32, y_scale, fp8_dtype)

        dist_x_fp8 = DTensor.from_local(x_fp8, mesh, [lhs_placement], run_check=False)
        dist_y_fp8 = DTensor.from_local(y_fp8, mesh, [rhs_placement], run_check=False)

        assert isinstance(dist_x_fp8.to_local(), Float8Tensor)
        assert isinstance(dist_y_fp8.to_local(), Float8Tensor)
        assert dist_x_fp8.to_local()._orig_dtype == torch.float32
        out_fp8 = torch.mm(dist_x_fp8, dist_y_fp8)
        local_fp8_out = out_fp8.to_local()
        assert out_fp8.shape == expected_dt_out_shape[idx], (idx, local_fp8_out.shape)

        # after mm the out dtype should be fp32
        assert local_fp8_out.dtype == torch.float32


def test_fp8_redistribute(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = torch.float8_e4m3fn
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)

    x_scale = tensor_to_scale(x_fp32, fp8_dtype).float()

    x_fp8 = Float8Tensor.to_float8(x_fp32, x_scale, fp8_dtype)

    dist_x_fp8 = DTensor.from_local(x_fp8, mesh, [Shard(0)], run_check=False)
    out_dist = dist_x_fp8.redistribute(placements=[Replicate()])
    assert out_dist.shape == (size * world_size, size)
    assert out_dist.placements == (Replicate(),)
    out_local = out_dist.to_local()
    # after allgather the out shape should be replicate
    assert out_local.shape == (size * world_size, size)
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    if isinstance(out_local, AsyncCollectiveTensor):
        out_local = out_local.wait()

    assert isinstance(out_local, Float8Tensor)
    assert out_local._data.dtype == fp8_dtype


def test_dtensor_cast_to_fp8(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = torch.float8_e4m3fn

    x_fp32 = torch.rand(size, size, device=device)
    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])

    dist_x_scale = tensor_to_scale(dist_x_fp32, fp8_dtype).float()
    assert isinstance(dist_x_scale, DTensor)

    dist_x_fp8 = Float8Tensor.to_float8(dist_x_fp32, dist_x_scale, fp8_dtype)
    assert isinstance(dist_x_fp8, DTensor)


def test_dtensor_fp8_autograd(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = torch.float8_e4m3fn

    x_fp32 = torch.rand(size, size, device=device, requires_grad=True)
    local_weight = torch.rand(2 * size, size, device=device, requires_grad=True)
    target = torch.rand(size, 2 * size, device=device)

    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])
    dist_x_scale = tensor_to_scale(dist_x_fp32, fp8_dtype).float()

    dist_wight_fp32 = distribute_tensor(local_weight, mesh, [Shard(0)])
    dist_weight_scale = tensor_to_scale(dist_wight_fp32, fp8_dtype).float()
    dist_target = distribute_tensor(target, mesh, [Shard(0)])

    dist_x_fp8 = Float8Tensor.to_float8(dist_x_fp32, dist_x_scale, fp8_dtype)
    dist_weight_fp8 = Float8Tensor.to_float8(
        dist_wight_fp32, dist_weight_scale, fp8_dtype
    )

    out = torch.nn.functional.linear(dist_x_fp8, dist_weight_fp8)
    out = NoopFwToFloat8E5M2Bw.apply(out, ScaledMMConfig())
    assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
    loss = torch.sum(torch.abs(out - dist_target))
    loss.backward()


def test_fp8_mlp_tensor_parallelism(mesh: DeviceMesh, size=16):
    device = mesh.device_type

    toy_model = ToyModel().to(device)
    toy_model_fp8 = swap_linear_with_float8_linear(
        toy_model, Float8DynamicLinear, emulate=True
    )

    tp_model = copy.deepcopy(toy_model)
    tp_model = swap_linear_with_float8_linear(
        tp_model, Float8DynamicLinear, emulate=True
    )
    sp_model = copy.deepcopy(toy_model)
    sp_model = swap_linear_with_float8_linear(
        sp_model, Float8DynamicLinear, emulate=True
    )

    # vanilla TP
    tp_model = parallelize_module(
        tp_model,
        mesh,
        {
            "in_proj": Float8ColwiseParallel(),
            "out_proj": Float8RowwiseParallel(),
        },
    )

    # "sequence parallel" mlp computation
    sp_model = parallelize_module(
        sp_model,
        mesh,
        {
            "in_proj": Float8ColwiseParallel(input_layouts=Shard(0)),
            "out_proj": Float8RowwiseParallel(
                output_layouts=Shard(0), use_local_output=False
            ),
        },
    )

    x_fp32 = torch.rand(size * 2, size, device=device, requires_grad=False)
    x_fp32_tp_input = x_fp32.clone()
    x_fp32_sp_input = distribute_tensor(x_fp32.clone(), mesh, [Shard(0)])

    tp_out = tp_model(x_fp32_tp_input)
    tp_out.sum().backward()
    sp_out = sp_model(x_fp32_sp_input)
    sp_out.sum().backward()
    global_out = toy_model_fp8(x_fp32)
    global_out.sum().backward()
    torch.testing.assert_close(tp_out, global_out)
    torch.testing.assert_close(sp_out.full_tensor(), global_out)
    torch.testing.assert_close(
        tp_model.in_proj.weight.grad, sp_model.in_proj.weight.grad
    )
    torch.testing.assert_close(
        tp_model.out_proj.weight.grad, sp_model.out_proj.weight.grad
    )


if __name__ == "__main__":
    # float8 only works on CUDA H100 so we only test cuda and we follow
    # other test files to not use TestCase but instead just add the test
    # cases in the main func.
    device_mesh = setup_distributed()
    tests = [
        test_scaled_mm,
        test_fp8_redistribute,
        test_dtensor_cast_to_fp8,
        test_dtensor_fp8_autograd,
        test_fp8_mlp_tensor_parallelism,
    ]

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e
