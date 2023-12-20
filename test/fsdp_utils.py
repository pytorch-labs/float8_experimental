# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, is_fp8, emulate, base_dtype=torch.float32):
    m = nn.Sequential(
        nn.Linear(K, N, dtype=base_dtype),
        nn.Linear(N, N, dtype=base_dtype),
    )
    if is_fp8:
        swap_linear_with_float8_linear(m, Float8Linear, emulate=emulate)
    return m

