import torch

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

def amax_to_scale(amax, dtype):
    if dtype == torch.float8_e4m3fn:
        return E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        return E5M2_MAX_POS / torch.clamp(amax, min=EPS)

def tensor_to_scale(x, dtype):
    amax = torch.max(torch.abs(x))
    return amax_to_scale(amax, dtype)

def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)
