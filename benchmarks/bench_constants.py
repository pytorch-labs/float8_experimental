# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

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

name_to_shapes = {
    # LLaMa 2 70B single-node weight shapes
    # assumes fused attn.wqkv and ffn.w13
    # source: https://fburl.com/gsheet/g8onr7rh
    "70B": {
        "attn.wqkv": (8192, 1280),
        "attn.w0": (1024, 8192),
        "ffn.w13": (8192, 7168),
        "ffn.w2": (3584, 8192),
    },
    # source: LLaMa 2 7B def, unfused ffn
    "7B": {
        "attn.wqkv": (4096, 12288),
        "attn.w0": (4096, 4096),
        "ffn.w1_or_w3": (4096, 11008),
        "ffn.w2": (11008, 4096),
    },
    # source: LLaMa 2 13B def, unfused ffn
    "13B": {
        "attn.wqkv": (5120, 15360),
        "attn.w0": (5120, 5120),
        "ffn.w1_or_w3": (5120, 13824),
        "ffn.w2": (13824, 5120),
    },
}
