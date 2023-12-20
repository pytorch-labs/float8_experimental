# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests of FSDP + compile + Float8Linear
"""

import os
import warnings

import fire

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from float8_experimental.float8_linear_utils import (
    sync_float8_amax_and_scale_history,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from fsdp_utils import setup, cleanup, get_model

torch.manual_seed(0)

B, M, K, N = 8, 8, 32, 32
lr = 0.01
N_ITER = 3


def test_no_compile(world_size, emulate, base_dtype, rank, ref_input_local):
    model = get_model(K, N, is_fp8=True, emulate=emulate, base_dtype=base_dtype).to(
        rank
    )
    model = FSDP(model, use_orig_params=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)

    for _ in range(N_ITER):
        optimizer.zero_grad()
        y_local = model(ref_input_local)
        y_local.sum().backward()
        sync_float8_amax_and_scale_history(model)
        optimizer.step()

    dist.barrier()

def test_fsdp_then_compile_with_workaround(world_size, emulate, base_dtype, rank, ref_input_local):
    model = get_model(K, N, is_fp8=True, emulate=emulate, base_dtype=base_dtype).to(
        rank
    )
    model = FSDP(model, use_orig_params=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)
    sync_func = torch.compile(sync_float8_amax_and_scale_history)

    for _ in range(N_ITER):
        optimizer.zero_grad()
        y_local = model(ref_input_local)
        y_local.sum().backward()
        sync_func(model)
        optimizer.step()

        if _ == 0:
            # right now things only work if we compile after the first iteration
            # otherwise, we get https://gist.github.com/vkuzo/665e27a4d362f3999ad9a9e786acbe02
            # TODO(future): fix this
            model = torch.compile(model)

    dist.barrier()

def test_compile_then_fsdp(world_size, emulate, base_dtype, rank, ref_input_local):
    model = get_model(K, N, is_fp8=True, emulate=emulate, base_dtype=base_dtype).to(
        rank
    )
    model = torch.compile(model)
    model = FSDP(model, use_orig_params=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)
    sync_func = torch.compile(sync_float8_amax_and_scale_history)

    for _ in range(N_ITER):
        optimizer.zero_grad()
        y_local = model(ref_input_local)
        y_local.sum().backward()
        sync_func(model)
        optimizer.step()

    dist.barrier()


# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    emulate, = args
    base_dtype = torch.bfloat16
    ref_input_global = torch.randn(B, M, K).cuda().to(base_dtype)
    # basic distributed data sampling
    bsz_global = ref_input_global.shape[0]
    assert B % world_size == 0
    bsz_local_start = int(rank / world_size * B)
    bsz_local_end = int((rank + 1) / world_size * B)
    ref_input_local = ref_input_global[bsz_local_start:bsz_local_end].to(rank)

    test_args = world_size, emulate, base_dtype, rank, ref_input_local

    test_no_compile(*test_args)
    # TODO(future): remove the workaround
    test_fsdp_then_compile_with_workaround(*test_args)
    # TOOD(future): unbreak this if needed
    # test_compile_then_fsdp(*test_args)
    # fails with https://gist.github.com/vkuzo/d7c65a073ebf47d64aa5b1a56df171c6

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
