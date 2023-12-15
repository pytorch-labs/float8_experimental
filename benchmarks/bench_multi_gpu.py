# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Callable, List, Optional, Tuple

import fire

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.benchmark as benchmark
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    get_float8_layers,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

# Check if transformer_engine is installed
transformer_engine_installed = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    transformer_engine_installed = True
except ImportError:
    print("transformer_engine not installed and we won't compare against this")


torch.manual_seed(0)

# TODO: Add more shapes for the benchmark
B, M, K, N = 32, 32, 32, 32
lr = 0.01


def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.blocked_autorange().median * 1e6


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, is_fp8, is_te, base_dtype=torch.float32):
    modules = [
        nn.Linear(K, N, dtype=base_dtype)
        if not is_te
        else te.Linear(K, N, params_dtype=base_dtype),
        nn.ReLU(),
    ]
    N_LAYERS = 20
    # N linear layers
    for _ in range(N_LAYERS - 1):
        if is_te:
            modules.append(te.Linear(N, N, params_dtype=base_dtype))
        else:
            modules.append(nn.Linear(N, N, dtype=base_dtype))
        modules.append(nn.ReLU())
    m = nn.Sequential(*modules)
    if is_fp8:
        assert not is_te, "`is_fp8` (using pytorch fp8) can't be used with `is_te`"
        swap_linear_with_float8_linear(m, Float8Linear, emulate=False)
    return m


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    base_dtype, input_global, compile = args

    # basic distributed data sampling
    bsz_global = input_global.shape[0]
    assert B % world_size == 0
    bsz_local_start = int(rank / world_size * B)
    bsz_local_end = int((rank + 1) / world_size * B)
    input_tensor = input_global[bsz_local_start:bsz_local_end].to(rank)

    fp8_model = get_model(K, N, is_fp8=True, is_te=False, base_dtype=base_dtype).to(
        rank
    )
    fp8_optimizer = torch.optim.SGD(fp8_model.parameters(), lr=lr * world_size)

    if compile:
        # TODO: Need to fix issues with compile
        fp8_model = torch.compile(fp8_model)
        compiled_sync_float8_amax_and_scale_history = torch.compile(
            sync_float8_amax_and_scale_history
        )

    fp8_model = FSDP(fp8_model, use_orig_params=compile)
    if rank == 0:
        print(fp8_model)

    fp8_layers = get_float8_layers(fp8_model)
    print(fp8_layers)

    def float8_forw_backward():
        fp8_optimizer.zero_grad()
        if compile:
            compiled_sync_float8_amax_and_scale_history(
                fp8_model, fp8_layers=fp8_layers, combine_reduction=True
            )
        else:
            sync_float8_amax_and_scale_history(
                fp8_model, fp8_layers=fp8_layers, combine_reduction=True
            )
        y_local = fp8_model(input_tensor)
        y_local.sum().backward()
        fp8_optimizer.step()
        return y_local

    ref_model = get_model(K, N, is_fp8=False, is_te=False, base_dtype=base_dtype).to(
        rank
    )
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr * world_size)
    if compile:
        ref_model = torch.compile(ref_model)
    ref_model = FSDP(ref_model, use_orig_params=compile)

    def ref_forw_backward():
        ref_optimizer.zero_grad()
        ref_model(input_tensor).sum().backward()
        ref_optimizer.step()

    if transformer_engine_installed:
        te_model = FSDP(
            get_model(K, N, is_fp8=False, is_te=True, base_dtype=base_dtype).to(rank),
            use_orig_params=compile,
        )
        fp8_format = recipe.Format.HYBRID
        fp8_recipe = recipe.DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        # Compiling TE_linear fails but they are already compiling under the hood
        # if transformer_engine_installed:
        #     te_forw_backward = torch.compile(te_forw_backward)
        if rank == 0:
            print(te_model)

        te_optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr * world_size)

        def te_forw_backward():
            te_optimizer.zero_grad()
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                y = te_model(input_tensor)
            y.sum().backward()
            te_optimizer.step()

    def run_n_iterations(n, fn):
        for _ in range(n):
            fn()
        # make sure training is done on all ranks
        dist.barrier()

    # warmup
    run_n_iterations(50, ref_forw_backward)
    run_n_iterations(50, float8_forw_backward)
    if transformer_engine_installed:
        run_n_iterations(50, te_forw_backward)

    N_ITER = 50
    ref_time = (
        benchmark_torch_function_in_microseconds(
            run_n_iterations, N_ITER, ref_forw_backward
        )
        * 1e-6
        / N_ITER
    )
    float8_time = (
        benchmark_torch_function_in_microseconds(
            run_n_iterations, N_ITER, float8_forw_backward
        )
        * 1e-6
        / N_ITER
    )
    if transformer_engine_installed:
        te_time_sec = (
            benchmark_torch_function_in_microseconds(
                run_n_iterations, N_ITER, te_forw_backward
            )
            * 1e-6
            / N_ITER
        )
    else:
        te_time_sec = None

    if rank == 0:
        print("ref_time", ref_time)
        print("float8_time", float8_time)
        print("te_time_sec", te_time_sec)
        print("float8 speedup", ref_time / float8_time)
        if transformer_engine_installed:
            print("te speedup", ref_time / te_time_sec)

    cleanup()


def run():
    compile = True
    base_dtype = torch.bfloat16
    print(f"{base_dtype = }")
    print(f"{compile = }")
    # generate input data
    ref_input = torch.randn(B, M, K).cuda().to(base_dtype)
    # run fsdp model
    WORLD_SIZE = torch.cuda.device_count()
    print(f"{WORLD_SIZE = }")
    args = (base_dtype, ref_input, compile)
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    fire.Fire(run)
