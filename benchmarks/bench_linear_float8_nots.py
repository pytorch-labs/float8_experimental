import argparse
import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple, Callable

import torch
import torch.utils.benchmark as benchmark
from tqdm import tqdm

from float8_experimental.float8_linear import sync_float8_amax_and_scale_history
from float8_experimental.float8_linear_nots import Float8LinearNoTensorSubclass

# Check if transformer_engine is installed
transformer_engine_installed = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    transformer_engine_installed = True
except ImportError:
    print("transformer_engine not installed and we won't compare against this")

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
		func: Callable, *args, **kwargs,
) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)", 
        globals={"args": args, "kwargs": kwargs, "func": func}
    )
    return t0.blocked_autorange().median * 1e6

@dataclass
class Experiment:
    name: str
    shape: Tuple[int, int, int]
    ref_time_sec: float
    float8_time_sec: float
    dtype: torch.dtype
    compiled: bool = False
    float_8_dtype: Optional[torch.dtype] = torch.float8_e4m3fn
    te_time_sec: Optional[float] = None

    # 3 Times since we are calculating forward backward
    @property
    def ref_tops_sec(self):
        M, K, N = self.shape
        return float(3*(2* M * K * N)) / self.ref_time_sec
    @property
    def ref_pct_top_peak(self):
        return self.ref_tops_sec / dtype_to_peak_tops[self.dtype]
    @property
    def float8_tops_sec(self):
        M, K, N = self.shape
        return float(3*(2* M * K * N)) / self.float8_time_sec
    @property
    def float8_pct_top_peak(self):
        return self.float8_tops_sec / dtype_to_peak_tops[self.float_8_dtype]

    @property
    def te_tops_sec(self):
        M, K, N = self.shape
        if self.te_time_sec is not None:
            return float(3*(2* M * K * N)) / self.te_time_sec
        else:
            return None

    @property
    def te_pct_top_peak(self):
        if self.te_tops_sec is not None:
            return self.te_tops_sec / dtype_to_peak_tops[self.float_8_dtype]
        else:
            return None

def main(sweep_path: Path, compile: bool, n_limit: Optional[int] = None):
    device = 'cuda'

    # LLaMa 2 70B single-node weight shapes
    # assumes fused attn.wqkv and ffn.w13
    # source: https://fburl.com/gsheet/g8onr7rh
    name_to_shapes_70b = {
        'attn.wqkv': (8192, 1280),
        'attn.w0': (1024, 8192),
        'ffn.w13': (8192, 7168),
        'ffn.w2': (3584, 8192),
    }
    input_bias = False
    ref_dtypes = [torch.bfloat16, torch.float16]
    experiment_list: List[Experiment] = []
    for idx, (dtype, (name, (K, N))) in \
            enumerate(tqdm(list(product(ref_dtypes, name_to_shapes_70b.items())))):
        if n_limit is not None and idx >= n_limit:
            break
        linear_ref = torch.nn.Linear(K, N, bias=input_bias).to(device=device, dtype=dtype)
        linear_float8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(linear_ref), emulate=False)
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        input_tensor = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
        ref_forw_backward = lambda : linear_ref(input_tensor).sum().backward()
        def float8_forw_backward():
            sync_float8_amax_and_scale_history(linear_float8)
            linear_float8(input_tensor).sum().backward()

        if transformer_engine_installed:
            # Create an FP8 recipe. Note: All input args are optional.
            fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
            te_linear = te.Linear(K, N, bias=input_bias).to(device=device, dtype=dtype)
            def te_forw_backward():
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    y = te_linear(input_tensor)
                y.sum().backward()

        if compile:
            ref_forw_backward = torch.compile(ref_forw_backward)
            float8_forw_backward = torch.compile(float8_forw_backward)
            # Compiling TE_linear fails but they are already compiling under the hood
            # if transformer_engine_installed:
            #     te_forw_backward = torch.compile(te_forw_backward)

        for _ in range(5):
            ref_forw_backward()
            float8_forw_backward()
            if transformer_engine_installed:
                te_forw_backward()

        ref_time = benchmark_torch_function_in_microseconds(ref_forw_backward)*1e-6
        float8_time = benchmark_torch_function_in_microseconds(float8_forw_backward)*1e-6
        if transformer_engine_installed:
            te_time_sec = benchmark_torch_function_in_microseconds(te_forw_backward)*1e-6
        else:
            te_time_sec = None
        experiment = Experiment(name, (M, K, N), ref_time, float8_time, dtype, compile, te_time_sec=te_time_sec)
        print(experiment)
        print('float8 speedup', experiment.ref_time_sec / experiment.float8_time_sec)
        if transformer_engine_installed:
            print('te speedup', experiment.ref_time_sec / experiment.te_time_sec)
        experiment_list.append(experiment)
        torch._dynamo.reset()

    headers = [
        "name",
        "M",
        "K",
        "N",
        "ref_dtype",
        "compiled",
        "fp8_dtype",
        "ref_time_sec",
        "pt_fp8_time_sec",
        "te_fp8_time_sec",
        "ref_tops_sec",
        "ref_pct_top_peak",
        "pt_fp8_tops_sec",
        "pt_fp8_pct_top_peak",
        "te_fp8_tops_sec",
        "te_fp8_pct_top_peak",
    ]
    data = []
    for experiment in experiment_list:
        data.append([
            experiment.name,
            experiment.shape[0],
            experiment.shape[1],
            experiment.shape[2],
            experiment.dtype,
            experiment.compiled,
            experiment.float_8_dtype,
            experiment.ref_time_sec,
            experiment.float8_time_sec,
            experiment.te_time_sec,
            experiment.ref_tops_sec,
            experiment.ref_pct_top_peak,
            experiment.float8_tops_sec,
            experiment.float8_pct_top_peak,
            experiment.te_tops_sec,
            experiment.te_pct_top_peak,
        ])

    data_pd = pd.DataFrame(data, columns=headers)
    data_pd['pt_fp8_speedup'] = data_pd['ref_time_sec'] / data_pd['pt_fp8_time_sec']
    if transformer_engine_installed:
        data_pd['te_fp8_speedup'] = data_pd['ref_time_sec'] / data_pd['te_fp8_time_sec']
    else:
        data_pd['te_fp8_speedup'] = -1.0
    data_pd['shape'] = (
        '(' + data_pd['M'].astype(str) + ', ' + 
        data_pd['K'].astype(str) + ', ' + data_pd['N'].astype(str) + ')'
    )

    data_pd_simple = data_pd[[
        'name', 'shape', 'ref_dtype', 'compiled', 'ref_time_sec', 
        'pt_fp8_time_sec', 'te_fp8_time_sec', 'pt_fp8_speedup', 'te_fp8_speedup']]
    print(data_pd_simple)

    with open(sweep_path, mode="w") as file:
        data_pd.to_csv(sweep_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('-n', '--n_limit', type=int, required=False)
    args = parser.parse_args()
    output_path = Path(args.output_path)
    main(output_path, args.compile, args.n_limit)
