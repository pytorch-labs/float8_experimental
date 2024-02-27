# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

import torch
import torch.utils.benchmark as benchmark
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
import float8_experimental.config as fp8_config
from tqdm import tqdm

# estimating TOPs for matmuls in fp32, fp16, fp8
# assuming A * B = C, with A being M * K, B being K * N, C being M * N

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 1979e12
h100_peak_tops_float8_tc = 3958e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}


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


@dataclass
class Experiment:
    name: str
    shape: Tuple[int, int, int]
    ref_time_sec: float
    float8_time_sec: float
    dtype: torch.dtype
    use_fused_cast: bool
    float_8_dtype: Optional[torch.dtype] = torch.float8_e4m3fn

    # 3 Times since we are calculating forward backward
    @property
    def ref_tops_sec(self):
        M, K, N = self.shape
        return float(3 * (2 * M * K * N)) / self.ref_time_sec

    @property
    def ref_pct_top_peak(self):
        return self.ref_tops_sec / dtype_to_peak_tops[self.dtype]

    @property
    def float8_tops_sec(self):
        M, K, N = self.shape
        return float(3 * (2 * M * K * N)) / self.float8_time_sec

    @property
    def float8_pct_top_peak(self):
        return self.float8_tops_sec / dtype_to_peak_tops[self.float_8_dtype]


def main(
    sweep_path: Path,
    n_limit: Optional[int] = None,
):
    device = "cuda"

    # LLaMa 2 70B single-node weight shapes
    # assumes fused attn.wqkv and ffn.w13
    name_to_shapes_70b = {
        "attn.wqkv": (8192, 1280),
        "attn.w0": (1024, 8192),
        "ffn.w13": (8192, 7168),
        "ffn.w2": (3584, 8192),
    }
    input_bias = False
    ref_dtypes = [torch.bfloat16, torch.float32]
    experiment_list: List[Experiment] = []
    fused_casts = [True, False]
    for idx, (dtype, (name, (K, N)), fuse_cast) in enumerate(
        tqdm(list(product(ref_dtypes, name_to_shapes_70b.items(), fused_casts)))
    ):
        fp8_config.use_fused_cast = fuse_cast
        if n_limit is not None and idx >= n_limit:
            break
        linear_ref = torch.nn.Linear(K, N, bias=input_bias).to(
            device=device, dtype=dtype
        )

        linear_float8 = Float8DynamicLinear.from_float(
            copy.deepcopy(linear_ref), emulate=False
        )

        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        input_tensor = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
        ref_forw_backward = lambda: linear_ref(input_tensor).sum().backward()

        def float8_forw_backward():
            linear_float8(input_tensor).sum().backward()

        def n_times(n, fn, *args, **kwargs):
            def wrapper(*args, **kwargs):
                for _ in range(n):
                    fn(*args, **kwargs)

            return wrapper

        REPEAT_N = 100

        ref_forw_backward = n_times(REPEAT_N, ref_forw_backward)
        float8_forw_backward = n_times(REPEAT_N, float8_forw_backward)

        for _ in range(5):
            ref_forw_backward()
            float8_forw_backward()

        ref_time = (
            benchmark_torch_function_in_microseconds(ref_forw_backward)
            * 1e-6
            / REPEAT_N
        )
        float8_time = (
            benchmark_torch_function_in_microseconds(float8_forw_backward)
            * 1e-6
            / REPEAT_N
        )
        experiment = Experiment(
            name,
            (M, K, N),
            ref_time,
            float8_time,
            dtype,
            fuse_cast
        )
        print(experiment)
        print("float8 speedup", experiment.ref_time_sec / experiment.float8_time_sec)
        experiment_list.append(experiment)
        torch._dynamo.reset()

    headers = [
        "name",
        "M",
        "K",
        "N",
        "ref_dtype",
        "fuse_cast",
        "fp8_dtype",
        "ref_time_sec",
        "pt_fp8_time_sec",
        "ref_tops_sec",
        "ref_pct_top_peak",
        "pt_fp8_tops_sec",
        "pt_fp8_pct_top_peak",
    ]
    data = []
    for experiment in experiment_list:
        data.append(
            [
                experiment.name,
                experiment.shape[0],
                experiment.shape[1],
                experiment.shape[2],
                experiment.dtype,
                experiment.use_fused_cast,
                experiment.float_8_dtype,
                experiment.ref_time_sec,
                experiment.float8_time_sec,
                experiment.ref_tops_sec,
                experiment.ref_pct_top_peak,
                experiment.float8_tops_sec,
                experiment.float8_pct_top_peak,
            ]
        )

    data_pd = pd.DataFrame(data, columns=headers)
    data_pd["pt_fp8_speedup"] = data_pd["ref_time_sec"] / data_pd["pt_fp8_time_sec"]
    data_pd["shape"] = (
        "("
        + data_pd["M"].astype(str)
        + ", "
        + data_pd["K"].astype(str)
        + ", "
        + data_pd["N"].astype(str)
        + ")"
    )

    data_pd_simple = data_pd[
        [
            "shape",
            "ref_dtype",
            "fuse_cast",
            "ref_time_sec",
            "pt_fp8_time_sec",
            "pt_fp8_speedup",
        ]
    ]
    print(data_pd_simple)

    sweep_path = sweep_path.with_suffix(".csv")
    with open(sweep_path, mode="w") as file:
        data_pd.to_csv(sweep_path)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-n", "--n_limit", type=int, required=False)
    args = parser.parse_args()
    output_path = Path(args.output_path)
    main(output_path, args.n_limit)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
