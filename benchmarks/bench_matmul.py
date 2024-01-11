# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import csv
import itertools
from typing import Optional

import bench_constants as bc

import fire
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean


def do_benchmarks(tops, peak_tops, f, *args, **kwargs):
    time_sec = benchmark_fn_in_sec(f, *args, **kwargs)
    tops_sec = float(tops) / time_sec
    pct_top_peak = tops_sec / peak_tops
    return time_sec, tops_sec, pct_top_peak


@torch.inference_mode()
def run(
    llama_model_size: Optional[str] = "70B",
    n_limit: Optional[int] = None,
    output_path: Optional[str] = None,
):
    print("model size", llama_model_size)
    device = "cuda"

    headers = ("name", "shape", "dtype", "ref_time_s", "fp8_time_s", "fp8_speedup")
    results = []

    name_to_shapes = bc.name_to_shapes[llama_model_size]
    if llama_model_size == "70B":
        # common distributed setup, single GPU numbers
        bsz_and_seq_len = ((4, 4096),)
    else:
        # debug single gpu setup
        bsz_and_seq_len = ((1, 4096),)
    dtypes = (torch.bfloat16,)

    for idx, (dtype, (name, (K, N))) in enumerate(
        itertools.product(dtypes, name_to_shapes.items())
    ):
        if n_limit is not None and idx >= n_limit:
            break

        # source: Xiao Sun, these are realistic for LLaMa 70B training
        bsz, seq_len = 4, 4096

        M = bsz * seq_len
        print("M, K, N:", M, K, N)
        tops = 2 * M * N * K
        print(f"tops: {tops:.2E}")

        # raw torch.mm
        A = torch.randn(M, K, device=device, dtype=dtype)
        m_ref = nn.Sequential(nn.Linear(K, N, dtype=dtype, device=device, bias=False))
        ref_time_sec, ref_tops_sec, ref_pct_top_peak = do_benchmarks(
            tops, bc.dtype_to_peak_tops[dtype], m_ref, A
        )
        print(
            f"{dtype} time_sec {ref_time_sec:.2E}, tops/sec {ref_tops_sec:.2E}, pct_peak {ref_pct_top_peak:.3f}"
        )

        del A

        # raw float8 matmul (upper bound for what we can achive in eager mode)
        # TODO(future): add e5m2
        d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, dtype
        A = torch.zeros(M, K, device=device, dtype=d1)
        B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()

        def do_matmul(A, B):
            return torch._scaled_mm(A, B, out_dtype=d3, use_fast_accum=False)

        fp8_time_sec, fp8_tops_sec, fp8_pct_top_peak = do_benchmarks(
            tops, bc.dtype_to_peak_tops[d1], do_matmul, A, B
        )
        print(
            f"fp8 time_sec {fp8_time_sec:.2E}, tops/sec {fp8_tops_sec:.2E}, pct_peak {fp8_pct_top_peak:.3f}"
        )

        del A, B

        results.append(
            [
                name,
                (M, K, N),
                dtype,
                ref_time_sec,
                fp8_time_sec,
                ref_time_sec / fp8_time_sec,
            ]
        )

    data_pd = pd.DataFrame(results, columns=headers)
    print(data_pd)
    if output_path is not None:
        with open(output_path, mode="w") as file:
            data_pd.to_csv(output_path)


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()  # pragma: no cover
