import torch
from dataclasses import dataclass
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds
from typing import Tuple, Optional, List
import context
from itertools import product
import copy
from tqdm import tqdm
from pathlib import Path
import argparse
from float8_linear_nots import Float8LinearNoTensorSubclass
import csv
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
    float_8_dtype: Optional[torch.dtype] = torch.float8_e4m3fn

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

def main(sweep_path: Path):
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
    for (K, N), dtype in tqdm(product(name_to_shapes_70b.values(), ref_dtypes)):
        linear_ref = torch.nn.Linear(K, N, bias=input_bias).to(device=device, dtype=dtype)
        linear_float8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(linear_ref), emulate=False)
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        input_tensor = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
        ref_forw_backward = lambda : linear_ref(input_tensor).sum().backward()
        float8_forw_backward = lambda : linear_float8(input_tensor).sum().backward()
        ref_time = benchmark_torch_function_in_microseconds(ref_forw_backward)*1e-6
        float8_time = benchmark_torch_function_in_microseconds(float8_forw_backward)*1e-6
        experiment = Experiment((M, K, N), ref_time, float8_time, dtype)
        experiment_list.append(experiment)

    # Update sweep path to have .csv suffix
    sweep_path = sweep_path.with_suffix('.csv')
    with open(sweep_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['M', 'K', 'N', 'dtype', 'float_8_dtype', 'ref_time_sec', 'float8_time_sec', 'ref_tops_sec', 'ref_pct_top_peak', 'float8_tops_sec', 'float8_pct_top_peak'])
        for experiment in experiment_list:
            writer.writerow([experiment.shape[0], experiment.shape[1], experiment.shape[2], experiment.dtype, experiment.float_8_dtype, experiment.ref_time_sec, experiment.float8_time_sec, experiment.ref_tops_sec, experiment.ref_pct_top_peak, experiment.float8_tops_sec, experiment.float8_pct_top_peak])



if __name__ == '__main__':
    # argparse for the sweep path
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_path', type=str, required=True)
    args = parser.parse_args()
    sweep_path = Path(args.sweep_path)
    main(sweep_path)