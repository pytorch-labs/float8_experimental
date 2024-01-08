from dataclasses import dataclass
from typing import Optional

import fire

import torch
import torch.utils.benchmark as benchmark
from float8_experimental.float8_utils import pad_tensor_for_matmul
from tabulate import tabulate

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


def benchmark_fn_in_usec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


def get_tops_info(tops, time, peak_tops):
    time_sec = time / 1e6
    tops_sec = float(tops) / time_sec
    pct_top_peak = tops_sec / peak_tops
    return tops_sec, pct_top_peak


def do_fp8_matmul(A, B, fp8_dtype, out_dtype):
    A_fp8 = A.to(fp8_dtype)
    B_fp8 = B.to(fp8_dtype).t()  # view

    A_pad = pad_tensor_for_matmul(A_fp8)  # mem copy
    B_pad = pad_tensor_for_matmul(B_fp8, both=True).contiguous().t()  # mem copy

    return torch._scaled_mm(A_pad, B_pad, out_dtype=out_dtype)[0][
        : A.shape[0], : B.shape[1]
    ]


def do_fp8_pad_first_matmul(A, B, fp8_dtype, out_dtype):
    A_pad = pad_tensor_for_matmul(A)  # mem copy
    B_pad = pad_tensor_for_matmul(B, both=True)  # mem copy

    A_pad = A_pad.to(fp8_dtype)  # mem copy
    B_pad = B_pad.to(fp8_dtype)  # mem copy

    B_pad = B_pad.t().contiguous().t()  # mem copy

    return torch._scaled_mm(A_pad, B_pad, out_dtype=out_dtype)[0][
        : A.shape[0], : B.shape[1]
    ]


def do_hp_matmul(A, B):
    return torch.matmul(A, B)


@dataclass
class Experiment_config:
    M: int
    K: int
    N: int
    output_dtype: torch.dtype
    fp8_dtype: torch.dtype

    def __iter__(self):
        return iter((self.M, self.K, self.N, self.output_dtype, self.fp8_dtype))


def gen_configs():
    shapes = [(8192, 2500, 5000), (4096, 10, 4096)]
    output_dtype = torch.float32
    fp8_dtype = torch.float8_e4m3fn
    return [Experiment_config(*shape, output_dtype, fp8_dtype) for shape in shapes]


@torch.no_grad()
def run(compile: bool = False, n_limit: Optional[int] = None):
    device = "cuda"
    experiments = gen_configs()
    results = []
    tops_table = []
    tops_headers = [
        "Shape",
        "Ref Dtype",
        "Ref Tops",
        "FP8 Tops",
        "Ref % Peak",
        "FP8 % Peak",
    ]
    for experiment in experiments:
        M, K, N, output_dtype, fp8_dtype = experiment
        tops = 2 * M * N * K

        A_base = torch.rand(M, K, device=device, dtype=output_dtype)
        B_base = torch.rand(K, N, device=device, dtype=output_dtype)

        hp_func = torch.compile(do_hp_matmul) if compile else do_hp_matmul
        fp8_func = torch.compile(do_fp8_pad_first_matmul) if compile else do_fp8_matmul

        ref_time = benchmark_fn_in_usec(hp_func, A_base, B_base)
        fp8_time = benchmark_fn_in_usec(
            fp8_func, A_base, B_base, fp8_dtype, output_dtype
        )

        ref_tops_sec, ref_pct_top_peak = get_tops_info(
            tops, ref_time, dtype_to_peak_tops[output_dtype]
        )
        fp8_tops_sec, fp8_pct_top_peak = get_tops_info(
            tops, fp8_time, dtype_to_peak_tops[fp8_dtype]
        )
        tops_table.append(
            [
                f"({M}x{K}x{N})",
                f"{output_dtype}",
                f"{ref_tops_sec:.2E}",
                f"{fp8_tops_sec:.2E}",
                f"{ref_pct_top_peak:.3f}",
                f"{fp8_pct_top_peak:.3f}",
            ]
        )
        results.append(
            [(M, K, N), output_dtype, ref_time, fp8_time, ref_time / fp8_time]
        )

    print("TOPs".center(80, "*"))
    print(tabulate(tops_table, headers=tops_headers))
    print("Speed Results".center(80, "*"))
    headers = ["Shape", "Ref Dtype", "Ref Time", "FP8 Time", "Speedup"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    fire.Fire(run)
