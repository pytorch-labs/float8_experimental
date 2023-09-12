import argparse
import copy
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

import context
from float8_linear import sync_float8_amax_and_scale_history
from float8_linear_nots import Float8LinearNoTensorSubclass
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds

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

@dataclass
class Experiment:
    shape: Tuple[int, int, int]
    ref_time_sec: float
    float8_time_sec: float
    dtype: torch.dtype
    compiled: bool = False
    float_8_dtype: Optional[torch.dtype] = torch.float8_e4m3fn
    te_time: Optional[float] = None

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
        if self.te_time is not None:
            return float(3*(2* M * K * N)) / self.te_time
        else:
            return None

    @property
    def te_pct_top_peak(self):
        if self.te_tops_sec is not None:
            return self.te_tops_sec / dtype_to_peak_tops[self.float_8_dtype]
        else:
            return None

def main(sweep_path: Path, compile: bool):
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
    ref_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    experiment_list: List[Experiment] = []
    for (K, N), dtype in tqdm(list(product(name_to_shapes_70b.values(), ref_dtypes))):
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
                    te_linear(input_tensor).sum().backward()

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
            te_time = benchmark_torch_function_in_microseconds(te_forw_backward)*1e-6
        else:
            te_time = None
        experiment = Experiment((M, K, N), ref_time, float8_time, dtype, compile, te_time=te_time)
        experiment_list.append(experiment)

    # Update sweep path to have .csv suffix
    sweep_path = sweep_path.with_suffix(".csv")
    with open(sweep_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "M",
                "K",
                "N",
                "dtype",
                "compiled",
                "float_8_dtype",
                "ref_time_sec",
                "float8_time_sec",
                "te_time_sec",
                "ref_tops_sec",
                "ref_pct_top_peak",
                "float8_tops_sec",
                "float8_pct_top_peak",
                "te_tops_sec",
                "te_pct_top_peak",
            ]
        )
        for experiment in experiment_list:
            writer.writerow(
                [
                    experiment.shape[0],
                    experiment.shape[1],
                    experiment.shape[2],
                    experiment.dtype,
                    experiment.compiled,
                    experiment.float_8_dtype,
                    experiment.ref_time_sec,
                    experiment.float8_time_sec,
                    experiment.te_time,
                    experiment.ref_tops_sec,
                    experiment.ref_pct_top_peak,
                    experiment.float8_tops_sec,
                    experiment.float8_pct_top_peak,
                    experiment.te_tops_sec,
                    experiment.te_pct_top_peak,
                ]
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    output_path = Path(args.output_path)
    main(output_path, args.compile)
