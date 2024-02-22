# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs float8 TP of toy models
"""

import os

import torch
import torch.nn as nn

from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import tensor_to_scale
from torch.distributed._tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    return device_mesh


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

    dist_x_fp8 = Float8Tensor.to_float8(x_fp32, dist_x_scale, fp8_dtype)
    assert isinstance(dist_x_fp8, DTensor)


if __name__ == "__main__":
    # float8 only works on CUDA H100 so we only test cuda and we follow
    # other test files to not use TestCase but instead just add the test
    # cases in the main func.
    device_mesh = setup_distributed()
    test_scaled_mm(device_mesh)
    test_fp8_redistribute(device_mesh)
    test_dtensor_cast_to_fp8(device_mesh)
