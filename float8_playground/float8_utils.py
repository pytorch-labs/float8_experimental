import torch

# This file reproduces the emulated fp8 <-> fp32 casts from
# https://github.com/pytorch/FBGEMM/pull/974/files with plain PyTorch.
# This implements the fp8 format spec from https://arxiv.org/pdf/2209.05433.pdf
#
# TODO(future PR): hook up to NVIDIA's casts on gpu, and
# import the fbgemm emulator directly on cpu. We'll also need to ensure
# the two are aligned.
#
# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# define the e4m3/e5m2 constants
E4M3_EBITS = 4
E4M3_EXP_BIAS = 7
E4M3_MAX_POS = 448.0

E5M2_EBITS = 5
E5M2_EXP_BIAS = 15
E5M2_MAX_POS = 57344.0

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

# enum without using an enum, for brevity
# TODO(future): make this into an enum if needed
E4M3 = 0
E5M2 = 1


def _float_to_hfp8(
    val_fp: torch.Tensor,  # fp32 values
    ebits: int,  # exponent bits, mbits = 8 - ebits
    exponent_bias: int,  # exponent bias to use in the fp8 encoding
    max_pos: float,  # maximum positive number for fp8 encoding
):
    mbits = 7 - ebits

    val_out = val_fp.clone().detach()

    # S{1}, E{8}, M{23}
    sign_bit = (val_out.view(torch.int32) & 0x80000000).view(torch.float)

    # set sign bit to 0
    # 0{1}, E{8}, M{23}
    val_out = (val_out.view(torch.int32) & 0x7FFFFFFF).view(torch.float)

    # ensure abs(val_out) <= max_pos)
    val_out = torch.clamp(val_out, max=max_pos)

    smallest_normal = torch.zeros_like(val_fp).to(torch.int32)
    smallest_normal = ((smallest_normal + 127 - exponent_bias + 1) << 23).view(torch.float)

    # normal and denormal paths split below, record
    # which element need which path
    is_normal_mask = torch.ge(val_out, smallest_normal)

    #
    # normal path
    #

    # Use round to nearest even. We make use of the standard rounding mechanism
    # in FP32 rather than rounding the mantissa and handling tie-to-even and
    # incrementing exponent We want to round of 23-mbits of the FP32 value
    # val_in This can be done by adding a power of 2 exactly 23-mbits larger
    # than the exponent of val_in This forces val_in to be moved to the right
    # and rounding exact at the location corresponding to having mbits of
    # explicit mantissa left
    n_bouncer = ((val_out.view(torch.int32) & 0xFF800000) + ((23 - mbits) << 23)).view(torch.float)
    n_val_out = (n_bouncer + val_out) - n_bouncer

    # adding the bouncer rounds off bits, and subtracting bouncer
    # leaves the desired value, albeit in FP32 encoding
    # All we need is to change the exponent encoding to using "bias"
    n_val_out_i = (n_val_out.view(torch.int32) - ((127 - exponent_bias) << 23)) << (8 - ebits)
    n_val_out_i = (n_val_out_i | sign_bit.view(torch.int32)) >> 24
    n_val_out = n_val_out_i.view(torch.float)

    #
    # denormal path
    #

    # When the value is in the denormal range, IEEE numbers essentially becomes
    # a fixed point number. The lsb is the smallest non-zero number
    # 2^(1-bias-mbits) Hence, we define the bouncer so that its lsb is this
    # smallest non-zero number Adding the input to this bouncer forces rounding
    # to occur appropriately Also, in this situation, after adding the bouncer,
    # the 8 least significant bits of the sum is already the HFP8 encoding of
    # the desired result. Just need to restore the sign bit
    # bouncer.I = (127 + (23 + (1 - exponent_bias - mbits))) << 23;
    # val_out.F = bouncer.F + val_out.F;
    # val_out.I = val_out.I | (sign_bit >> 24);

    dn_bouncer = ((torch.zeros_like(val_out).view(torch.int32) + 127 + (23 + (1 - exponent_bias - mbits))) << 23).view(torch.float)
    dn_val_out = dn_bouncer + val_out
    dn_val_out_i = dn_val_out.view(torch.int32) | (sign_bit.view(torch.int32) >> 24)
    dn_val_out = dn_val_out_i.view(torch.float)

    #
    # combine normal and denormal paths
    #
    val_out = torch.where(is_normal_mask, n_val_out, dn_val_out)
    # take the 8 least significant bits
    orig_shape = val_fp.shape
    val_out = val_out.view(torch.uint8)
    val_out = val_out.reshape(-1, 4)
    val_out = torch.tensor_split(val_out, 4, dim=-1)[0]
    val_out = val_out.reshape(orig_shape)
    return val_out


def _hfp8_to_float(
    hfp8_val: torch.Tensor,
    ebits: int,
    exponent_bias: int,
):
    assert hfp8_val.dtype == torch.uint8

    sign_i = (hfp8_val & 0x80).to(torch.int32) << 24

    val_out_i = (hfp8_val & 0x7F).to(torch.int32) << (24 - (8 - ebits))
    # so that the mantissa bits start at the mantissa bit positions of FP32
    # encoding

    # Let the hfp8 mantissa bits correspond to the value frac, 0 <= frac < 1
    # So if the hfp8 value is a normal number, it's value is 2^e x (1+frac)
    # where e is its (true, unbiased) exponent
    # If the hfp8 value is denormal, the value is 2^(1-bias) x frac

    # However, the bit pattern in the 8-bit exponent field of val_out.F
    # is bias+e when hfp8 is normal, and 0 when hfp8 is subnormal.
    # So, as an FP32 value, when hfp8 is normal, val_out.F represents the value
    # of 2^(bias+e-127) * (1+frac)
    # And when hfp8 is subnormal, val_out.F is also subnormal, and represents the
    # value of 2^(-126) * frac In either case, val_out.F corresponds to
    # 2^(bias-127) * (value of hfp8 input) Thus, if we multiply val_out.F by
    # 2^(127-bias), we obtain the hfp8 value as an FP32 number

    # multiplier.I = (127 + (127 - exponent_bias))
    #     << 23; // multiplier.F is 2^(127-bias)
    # val_out.F *= multiplier.F;
    # val_out.I |= sign.I;
    # return val_out.F;

    multiplier_i = (torch.zeros_like(hfp8_val).to(torch.int32) + 127 + (127 - exponent_bias)) << 23  # multiplier_f is 2^(127-bias)
    val_out_f = val_out_i.view(torch.float)
    val_out_f *= multiplier_i.view(torch.float)
    val_out_f = (val_out_f.view(torch.int32) | sign_i).view(torch.float)
    return val_out_f

def float32_to_float8(x, flavor):
    if flavor == E4M3:
        return _float_to_hfp8(x, E4M3_EBITS, E4M3_EXP_BIAS, E4M3_MAX_POS)
    else:  # e5m2
        return _float_to_hfp8(x, E5M2_EBITS, E5M2_EXP_BIAS, E5M2_MAX_POS)

def float8_to_float32(x, flavor):
    if flavor == E4M3:
        return _hfp8_to_float(x, E4M3_EBITS, E4M3_EXP_BIAS)
    else:  # e5m2
        return _hfp8_to_float(x, E5M2_EBITS, E5M2_EXP_BIAS)

def amax_to_scale(amax, flavor):
    if flavor == E4M3:
        return E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        return E5M2_MAX_POS / torch.clamp(amax, min=EPS)

def tensor_to_scale(x, flavor):
    amax = torch.max(torch.abs(x))
    return amax_to_scale(amax, flavor)

def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)
