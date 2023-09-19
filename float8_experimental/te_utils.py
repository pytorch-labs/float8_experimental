"""
Temporary wrapper of parts of TransformerEngine, to test performance
with TE kernels
"""

import torch

from float8_experimental.float8_utils import (
    tensor_to_amax,
    to_fp8_saturated,
)

# TODO guard against TE not available
# import transformer_engine as te
import transformer_engine.pytorch.cpp_extensions as tex

def te_fp8_cast_transpose_fused(
    x: torch.Tensor,
    scale: torch.Tensor,
    amax_history: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,  # float8_e4m3fn or float8_e5m2
):
    fp8_meta_tensor = tex.FP8TensorMeta()
    # reference: https://github.com/NVIDIA/TransformerEngine/blob/eb64ec2a26b1091a78af424deb7d5204bee17cd2/transformer_engine/pytorch/module/base.py#L291
    fp8_meta_tensor.scale = scale
    fp8_meta_tensor.amax_history = amax_history

    # The TE kernel requires scale_inv, but the PyTorch codebase currently
    # does not have this. For now, create it inline. This will be a performance
    # regression against TE.
    # TODO(future): guard against division by zero
    fp8_meta_tensor.scale_inv = 1.0 / scale

    # this is just used for the index of scale/amax_history/scale_inv tensors
    # reference: https://github.com/NVIDIA/TransformerEngine/blob/eb64ec2a26b1091a78af424deb7d5204bee17cd2/transformer_engine/pytorch/cpp_extensions/transpose.py#L36
    fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT

    otype = tex.DType.kFloat8E4M3 if out_dtype is torch.float8_e4m3fn \
        else tex.DType.kFloat8E5M2
    
    # y_t_c means y_transpose_contiguous
    y, y_t_c = tex.fp8_cast_transpose_fused(
        x,
        fp8_meta_tensor,
        fp8_tensor,
        otype,
    )

    # The TE kernel does not update amax, but the PyTorch codebase currently
    # uses a separate buffer for amax. For now, update it manually.
    # This will be a performance regression against TE.
    amax.copy_(amax_history[0])

    return y.view(out_dtype), y_t_c.view(out_dtype)

def pt_fp8_cast_transpose(
    x: torch.Tensor,
    scale: torch.Tensor,
    amax_history: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,  # float8_e4m3fn or float8_e5m2
):
    new_amax = tensor_to_amax(x)
    amax.copy_(new_amax)
    x_scaled = x * scale
    y = to_fp8_saturated(x_scaled, out_dtype)
    y_t_c = y.t().contiguous()
    return y, y_t_c
