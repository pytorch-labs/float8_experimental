# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# This is a performance-only benchmark script of lit-gpt LLaMa 7b training
# Code was taken from https://github.com/drisspg/lit-gpt/blob/fp8_train/pretrain/fp8_openweb.py
# and slimmed down.

import os
import sys

import torch
from jsonargparse import CLI

# hack to import lit_gpt, assumes the following dir setup:
# ../
#   /float8_experimental
#   /lit_gpt
lit_gpt_path = "/".join(os.getcwd().split("/")[:-1] + ["lit-gpt"])
print(lit_gpt_path)
sys.path.insert(0, lit_gpt_path)

lit_gpt_installed = False
try:
    from lit_gpt.model import Config, GPT
    from lit_gpt.utils import chunked_cross_entropy
except ImportError:
    print("lit_gpt not installed, terminating")
    sys.exit(0)

import csv
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from float8_experimental.dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear

# from lit_gpt.model import GPT, Config
# from lit_gpt.utils import chunked_cross_entropy

# Float8 imports
from float8_experimental.float8_linear_utils import (
    linear_requires_sync,
    LinearType,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)

LINEAR_TYPE_MAP = {
    LinearType.DELAYED: Float8Linear,
    LinearType.DYNAMIC: Float8DynamicLinear,
}

instruction_tuning = True
eval_interval = 500
save_interval = 10000
eval_iters = 100
log_interval = 500
# change this value to force a maximum sequence length
override_max_seq_length = None

OVERFIT = False
COMPILE = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 128 if not OVERFIT else 1
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 10
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
model_name = "Llama-2-7b-hf"
name = "openwebtext"
device = torch.device("cuda")


def get_profile_context(
    profile: bool,
    fp8_linear_type: LinearType,
    profile_path: Optional[str] = None,
):
    if profile:
        assert profile_path is not None

    def trace_handler(prof):
        dtype_str = fp8_linear_type if fp8_linear_type else "bf16"
        prof.export_chrome_trace(profile_path)
        print(f"Wrote profile to: {profile_path}")

    if profile:
        context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=3, warmup=1, active=2, repeat=1),
            record_shapes=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        )
        return context
    else:
        return nullcontext()


def main(
    compile: bool = False,
    fp8_linear_type: Optional[str] = None,
    profile: bool = False,
    profile_path: Optional[str] = None,
    # if specified, set `n` encoder blocks to do nothing in order to
    # simplify looking at traces and inductor code
    skip_n_encoder_blocks: Optional[int] = None,
):
    config = Config.from_name(model_name)

    print("Initializing the model")
    with device:
        model = GPT(config).to(torch.bfloat16)
        model.apply(model._init_weights)

    if skip_n_encoder_blocks is not None:
        # slim down the model for easier debugging
        class DoNothing(torch.nn.Module):
            def forward(self, x0, x1, x2, x3, x4):
                return x0

        l = len(model.transformer.h)
        assert skip_n_encoder_blocks <= l
        for i in range(l - skip_n_encoder_blocks, l):
            model.transformer.h[i] = DoNothing()

    if fp8_linear_type is not None:
        fp8_linear_type = LinearType[fp8_linear_type.upper()]
    if fp8_linear_type is not None:
        fp8_module = LINEAR_TYPE_MAP[fp8_linear_type]
        swap_linear_with_float8_linear(model, fp8_module)

    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    global COMPILE
    COMPILE = compile
    if compile:
        model = torch.compile(model)

    model.train()
    profile_context = get_profile_context(profile, fp8_linear_type, profile_path)

    # Sanity check
    dtype_str = fp8_linear_type if fp8_linear_type else "bf16"
    sync_func = (
        torch.compile(sync_float8_amax_and_scale_history)
        if COMPILE
        else sync_float8_amax_and_scale_history
    )

    warmup_iters = 2
    start_time = None

    # create fake data
    input_ids = torch.randint(0, 1000, (1, 4096), device=device, dtype=torch.int64)
    targets = torch.randint(0, 1000, (1, 4096), device=device, dtype=torch.int64)

    with profile_context as p:
        for iter_num in range(max_iters):

            if iter_num == warmup_iters:
                start_time = time.perf_counter()

            lr = learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # Determine if this is correct location
            if linear_requires_sync(fp8_linear_type):
                sync_func(model)

            t0 = time.perf_counter()

            is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            # Scale the loss by grad_accumulation iters
            (loss / gradient_accumulation_iters).backward()

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

            dt = time.perf_counter() - t0
            # print('iter', iter_num, 'dt', dt)

            if profile:
                p.step()

            torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    print("total_time", total_time)
    print("time per iter", total_time / (max_iters - warmup_iters))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Example usage:
    # python pretrain/fp8_openweb.py --fp8_linear_type "dynamic" --compile True
    CLI(main)
