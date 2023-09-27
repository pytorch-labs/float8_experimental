from typing import Callable
import torch
import torch.distributed as dist

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0

FP16_MAX_POS = 65504.0

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

@torch.no_grad()
def amax_to_scale(amax, float8_dtype, orig_dtype):
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS) 
    return res

@torch.no_grad()
def amax_history_to_scale(
    amax_history, 
    float8_dtype, 
    orig_dtype,
    history_to_scale_fn_type,
):
    if history_to_scale_fn_type == 'max':
        amax = torch.max(amax_history)
        return amax_to_scale(amax, float8_dtype, orig_dtype)
    raise NotImplementedError()

@torch.no_grad()
def tensor_to_amax(x, distributed_reduction=False):
    amax = torch.max(torch.abs(x))

    # If the user asked for distributed reduction, do it.
    # If the user did not ask for it, assume that it will
    # happen elsewhere.
    if distributed_reduction and dist.is_initialized():
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)

    return amax

@torch.no_grad()
def tensor_to_scale(x, float8_dtype):
    amax = tensor_to_amax(x)
    return amax_to_scale(amax, float8_dtype, x.dtype)

def to_fp8_saturated(x, float8_dtype):
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1*E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1*E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)

def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)
