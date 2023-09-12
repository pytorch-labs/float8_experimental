import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import context
from float8_linear import sync_float8_amax_and_scale_history
from float8_linear_nots import Float8LinearNoTensorSubclass
from transformer_nuggets.utils import ProfileConfig, profile_function


@dataclass(frozen=True)
class LinearParams:
    M: int
    K: int
    N: int
    input_bias: bool
    ref_dtype: torch.dtype
    torch_compile: Optional[bool] = False


def main(profile_path: Path):
    assert profile_path.is_dir(), f"Path {profile_path} must be a directory"
    params = LinearParams(
        M=4*4096,
        K=8192,
        N=7168,
        input_bias=False,
        ref_dtype=torch.float16,
        torch_compile=True
        )

    linear_ref = torch.nn.Linear(params.K, params.N, bias=params.input_bias, device='cuda', dtype=params.ref_dtype)
    linear_float8 = Float8LinearNoTensorSubclass.from_float(copy.deepcopy(linear_ref), emulate=False)
    input_tensor = torch.randn(params.M, params.K, device='cuda', dtype=params.ref_dtype, requires_grad=True)

    ref_forw_backward = lambda : linear_ref(input_tensor).sum().backward()

    def float8_forw_backward():
            sync_float8_amax_and_scale_history(linear_float8)
            linear_float8(input_tensor).sum().backward()

    if params.torch_compile:
            ref_forw_backward = torch.compile(ref_forw_backward)
            float8_forw_backward = torch.compile(float8_forw_backward)
            for _ in range(5):
                ref_forw_backward()
                float8_forw_backward()

    # Profile Reference Linear
    ref_string = f"linear_ref_dtype_{params.ref_dtype}_M_{params.M}_K_{params.K}_N_{params.N}_input_bias_{params.input_bias}.json"
    profile_config = ProfileConfig(
        str(profile_path/ref_string),
        ref_string,
        iters=5,
        warmup_iters=5,
        sync=True
    )
    profile_function(profile_config, ref_forw_backward)

    # # Profile Float8 Linear
    float8_string = f"linear_float8_M_{params.M}_K_{params.K}_N_{params.N}_input_bias_{params.input_bias}_compile_{params.torch_compile}.json"
    profile_config = ProfileConfig(
        str(profile_path/float8_string),
        float8_string,
        iters=5,
        warmup_iters=5,
        sync=True
    )
    profile_function(profile_config, float8_forw_backward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, required=True,help="Path to save folder")
    args = parser.parse_args()
    output_path = Path(args.output_path)
    main(output_path)
