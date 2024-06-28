# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fire

import torch
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    get_float8_linear,
    linear_requires_sync,
    LinearType,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.profiler import profile, ProfilerActivity, record_function


class LNLinear(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


@dataclass
class ProfileConfig:
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: Optional[str] = None


def profile_function(
    config: ProfileConfig, func: Callable, *args, **kwargs
) -> torch.profiler.profile:
    """Profile a torch function and save the result to a file"""
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    activities = [ProfilerActivity.CPU]
    if config.cuda:
        activities.append(ProfilerActivity.CUDA)

    if config.warmup_iters >= 0:
        for _ in range(config.warmup_iters):
            func(*args, **kwargs)
    if config.sync:
        torch.cuda.synchronize()
    name_context = (
        nullcontext() if config.name is None else record_function(config.name)
    )
    profile_memory = config.memory_profile_path is not None
    with profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=profile_memory,
        with_stack=profile_memory,
        **config.extra_kwargs,
    ) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)
                if config.sync:
                    torch.cuda.synchronize()

    if config.file_path is not None:
        prof.export_chrome_trace(config.file_path)

    if config.file_path is None:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return prof


@dataclass(frozen=True)
class ModelParams:
    M: int
    K: int
    N: int
    ref_dtype: torch.dtype
    layer_norm: bool = True


def main(
    profile_path_prefix: Path,
    compile: bool = True,
    linear_type: str = "dynamic",
    use_layer_norm: bool = False,
):
    params = ModelParams(
        M=4 * 4096,
        K=8192,
        N=7168,
        ref_dtype=torch.bfloat16,
        layer_norm=use_layer_norm,
    )
    print(f"Compile is set to          | {compile}")
    print(f"Using Linear type:         | {linear_type}")
    print(f"Use layer norm is set to   | {params.layer_norm}")

    device = "cuda"
    if params.layer_norm:
        m_ref = LNLinear(params.K, params.N)
    else:
        m_ref = torch.nn.Sequential(
            torch.nn.Linear(params.K, params.N, bias=False),
        )
    m_ref = m_ref.to(device).to(params.ref_dtype)

    linear_type = LinearType[linear_type.upper()]
    linear_cls = (
        Float8Linear if linear_type is LinearType.DELAYED else Float8DynamicLinear
    )

    m_float8 = copy.deepcopy(m_ref)
    swap_linear_with_float8_linear(m_float8, linear_cls)

    input_tensor = torch.randn(
        params.M, params.K, device="cuda", dtype=params.ref_dtype, requires_grad=True
    )

    def ref_forw_backward(x):
        out = m_ref(x)
        out.sum().backward()

    def float8_forw(x):
        out = m_float8(x)
        return out

    def float8_forw_backward_wrapper(x):
        # sync_float8_amax_and_scale_history is not full graph torch
        # compile friendly, so we add a high level wrapper to allow
        # inspection of the fw+bw torch.compile without the scale
        # syncing code
        # TODO(future): make this better
        if linear_requires_sync(linear_type):
            with record_function("scale_amax_and_scales"):
                sync_float8_amax_and_scale_history(m_float8)
        out = float8_forw(x)

        # out.sum().backward() is also not torch.compile fullgraph
        # friendly
        with record_function("backward"):
            out.sum().backward()

    if compile:
        ref_forw_backward = torch.compile(ref_forw_backward)
        float8_forw = torch.compile(float8_forw, fullgraph=True)

    for _ in range(5):
        ref_forw_backward(input_tensor)
        float8_forw_backward_wrapper(input_tensor)

    # Profile Reference Model
    ref_suffix = f"_ref_compile_{compile}.json"
    profile_config = ProfileConfig(
        profile_path_prefix + ref_suffix, ref_suffix, iters=5, warmup_iters=5, sync=True
    )
    profile_function(profile_config, ref_forw_backward, input_tensor)

    # Profile Float8 Model
    float8_suffix = f"_float8_compile_{compile}_{linear_type}.json"
    profile_config = ProfileConfig(
        profile_path_prefix + float8_suffix,
        float8_suffix,
        iters=5,
        warmup_iters=5,
        sync=True,
    )
    profile_function(profile_config, float8_forw_backward_wrapper, input_tensor)


def invoke_main() -> None:
    # Example usage: python benchmarks/profile_linear_float8.py benchmarks/data/profiles/current_profile --compile=True --linear_type="dynamic"
    fire.Fire(main)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
