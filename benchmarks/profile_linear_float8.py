# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fire

import torch
from float8_experimental.float8_linear_utils import (
    get_float8_linear,
    linear_requires_sync,
    LinearType,
    sync_float8_amax_and_scale_history,
)
from torch.profiler import profile, ProfilerActivity, record_function


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


# Check if transformer_engine is installed
transformer_engine_installed = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    transformer_engine_installed = True
except ImportError:
    print("transformer_engine not installed and we won't compare against this")


@dataclass(frozen=True)
class LinearParams:
    M: int
    K: int
    N: int
    input_bias: bool
    ref_dtype: torch.dtype
    layer_norm: bool = True
    torch_compile: Optional[bool] = False


def main(profile_path: Path, compile: bool, linear_type: str):
    profile_path = Path(profile_path)
    assert profile_path.is_dir(), f"Path {profile_path} must be a directory"
    params = LinearParams(
        M=4 * 4096,
        K=8192,
        N=7168,
        input_bias=False,
        ref_dtype=torch.bfloat16,
        layer_norm=True,
        torch_compile=compile,
    )
    print(f"Compile is set to          | {compile}")
    print(f"Using Linear type:         | {linear_type}")
    print(f"Use layer norm is set to   | {params.layer_norm}")
    linear_ref = torch.nn.Linear(
        params.K,
        params.N,
        bias=params.input_bias,
        device="cuda",
        dtype=params.ref_dtype,
    )
    linear_type = LinearType[linear_type.upper()]
    linear_float8 = get_float8_linear(linear_type, linear_ref)

    input_tensor = torch.randn(
        params.M, params.K, device="cuda", dtype=params.ref_dtype, requires_grad=True
    )

    if params.layer_norm:
        ln = torch.nn.LayerNorm(
            params.K, elementwise_affine=False, device="cuda", dtype=params.ref_dtype
        )

    def ref_forw_backward(x):
        if params.layer_norm:
            with record_function("layer_norm"):
                x = ln(x)
        with record_function("forward"):
            out = linear_ref(x)
        with record_function("backward"):
            out.sum().backward()

    def float8_forw_backward(x):
        if params.layer_norm:
            with record_function("layer_norm"):
                x = ln(x)
        with record_function("forward"):
            out = linear_float8(x)
        return out

    def float8_forw_backward_wrapper(x):
        # sync_float8_amax_and_scale_history is not full graph torch
        # compile friendly, so we add a high level wrapper to allow
        # inspection of the fw+bw torch.compile without the scale
        # syncing code
        # TODO(future): make this better
        if linear_requires_sync(linear_type):
            with record_function("scale_amax_and_scales"):
                sync_float8_amax_and_scale_history(linear_float8)
        out = float8_forw_backward(x)

        # out.sum().backward() is also not torch.compile fullgraph
        # friendly
        with record_function("backward"):
            out.sum().backward()

    if transformer_engine_installed:
        # Create an FP8 recipe. Note: All input args are optional.
        fp8_recipe = recipe.DelayedScaling(
            margin=0, interval=1, fp8_format=recipe.Format.E4M3
        )
        te_linear = te.Linear(params.K, params.N, bias=params.input_bias).to(
            device="cuda", dtype=params.ref_dtype
        )

        def te_forw_backward(x):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                with record_function("forward"):
                    out = te_linear(x)
                with record_function("backward"):
                    out.sum().backward()

    if params.torch_compile:
        ref_forw_backward = torch.compile(ref_forw_backward)
        float8_forw_backward = torch.compile(float8_forw_backward, fullgraph=True)
        # Compiling TE_linear fails but they are already compiling under the hood
        # if transformer_engine_installed:
        #     te_forw_backward = torch.compile(te_forw_backward)

    for _ in range(5):
        ref_forw_backward(input_tensor)
        float8_forw_backward_wrapper(input_tensor)
        if transformer_engine_installed:
            te_forw_backward(input_tensor)

    # Profile Reference Linear
    ref_string = f"linear_ref_dtype_{params.ref_dtype}_M_{params.M}_K_{params.K}_N_{params.N}_input_bias_{params.input_bias}_compile_{params.torch_compile}.json"
    profile_config = ProfileConfig(
        str(profile_path / ref_string), ref_string, iters=5, warmup_iters=5, sync=True
    )
    profile_function(profile_config, ref_forw_backward, input_tensor)

    # # Profile Float8 Linear
    float8_string = f"linear_float8_M_{params.M}_K_{params.K}_N_{params.N}_input_bias_{params.input_bias}_compile_{params.torch_compile}_{linear_type}.json"
    profile_config = ProfileConfig(
        str(profile_path / float8_string),
        float8_string,
        iters=5,
        warmup_iters=5,
        sync=True,
    )
    profile_function(profile_config, float8_forw_backward_wrapper, input_tensor)

    te_string = f"linear_transformer_engine_M_{params.M}_K_{params.K}_N_{params.N}_input_bias_{params.input_bias}.json"
    if transformer_engine_installed:
        profile_config = ProfileConfig(
            str(profile_path / te_string), te_string, iters=5, warmup_iters=5, sync=True
        )
        profile_function(profile_config, te_forw_backward, input_tensor)


if __name__ == "__main__":
    # Example usage: python benchmarks/profile_linear_float8.py benchmarks/data/profiles --compile=True --linear_type="dynamic"
    fire.Fire(main)
