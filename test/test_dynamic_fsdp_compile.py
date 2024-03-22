# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke test of autocast + torch.compile + FSDP + Float8DynamicLinear
"""

import os
import warnings

import fire

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from float8_experimental import config
from float8_experimental.dynamic_linear.dynamic_float8_linear import Float8DynamicLinear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

torch.manual_seed(0)

B, M, K, N = 8, 8, 32, 32
lr = 0.01
N_ITER = 1


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_PORT"] = "12356"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, is_fp8, emulate, base_dtype=torch.float32):
    m = nn.Sequential(
        nn.Linear(K, N, dtype=base_dtype),
        nn.ReLU(),
    )
    swap_linear_with_float8_linear(m, Float8DynamicLinear, emulate=emulate)
    return m


# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    (emulate,) = args

    model = get_model(K, N, is_fp8=True, emulate=emulate, base_dtype=torch.bfloat16).to(
        rank
    )
    print(model)

    # To compile FSDP, we need use_orig_params to True
    model = FSDP(model, use_orig_params=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)
    input_local = torch.randn(B, M, K, N, device="cuda")

    model = torch.compile(model)

    for iter in range(N_ITER):
        optimizer.zero_grad()
        with torch.autocast("cuda"):
            y_local = model(input_local)
        y_local.sum().backward()
        optimizer.step()

    print("done!")
    cleanup()


def run():
    emulate = False
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, running in emulation_mode")
        emulate = True
    elif torch.cuda.get_device_capability() < (9, 0):
        warnings.warn(
            f"CUDA capability {torch.cuda.get_device_capability()} < (9.0), running in emulation mode"
        )
        emulate = True

    WORLD_SIZE = torch.cuda.device_count()
    args = (emulate,)
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    fire.Fire(run)
