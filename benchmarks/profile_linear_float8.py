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
import torch.nn as nn
import torch.nn.functional as F
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


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py
class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class NormFFNResidualNorm(nn.Module):
    """
    A fragment representing the end of TransformerBlock n and the start
    of TransformerBlock n + 1, intended to include the fusions relevant
    to float8 gemms in the FFN module in forward and backward.
    """

    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.attn_norm = RMSNorm(dim)

    def forward(self, h):
        # end of transformer block n
        x = self.ffn_norm(h)
        x = self.ffn(x)
        x = h + x
        # start of transformer block n + 1
        x = self.attn_norm(x)
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


def main(
    profile_path_prefix: Path,
    compile: bool = True,
    linear_type: str = "dynamic",
    model_type: str = "linear",
):
    assert model_type in ("linear", "ln_linear", "norm_ffn_norm"), "unsupported"

    print(f"Compile is set to          | {compile}")
    print(f"Using Linear type:         | {linear_type}")
    print(f"model_type is set to       | {model_type}")

    device = "cuda"
    ref_dtype = torch.bfloat16
    if model_type == "ln_linear":
        M, K, N = 4 * 4096, 8192, 7168
        m_ref = LNLinear(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )
    elif model_type == "norm_ffn_norm":
        m_ref = NormFFNResidualNorm(
            dim=4096,
            hidden_dim=16384,
            multiple_of=1024,
            ffn_dim_multiplier=1.3,
        )
        input_tensor = torch.randn(
            1, 8192, 4096, device=device, dtype=ref_dtype
        ).requires_grad_()
    else:
        M, K, N = 4 * 4096, 8192, 7168
        m_ref = torch.nn.Sequential(
            torch.nn.Linear(K, N, bias=False),
        )
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )

    m_ref = m_ref.to(device).to(ref_dtype)

    linear_type = LinearType[linear_type.upper()]
    linear_cls = (
        Float8Linear if linear_type is LinearType.DELAYED else Float8DynamicLinear
    )

    m_float8 = copy.deepcopy(m_ref)
    swap_linear_with_float8_linear(m_float8, linear_cls)

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
    ref_suffix = f"_{model_type}_ref_compile_{compile}.json"
    profile_config = ProfileConfig(
        profile_path_prefix + ref_suffix, ref_suffix, iters=5, warmup_iters=5, sync=True
    )
    profile_function(profile_config, ref_forw_backward, input_tensor)

    # Profile Float8 Model
    float8_suffix = f"_{model_type}_float8_compile_{compile}_{linear_type}.json"
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
