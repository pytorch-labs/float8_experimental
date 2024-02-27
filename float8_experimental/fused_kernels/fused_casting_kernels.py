import torch
import triton
import triton.language as tl
from triton import next_power_of_2


E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
FP16_MAX_POS = torch.finfo(torch.float16).max
EPS = 1e-12


@triton.jit
def abs_max_kernel(x_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    offset_base = tl.arange(0, BLOCK_SIZE)[None, :]
    acc = tl.full([1, BLOCK_SIZE], -float("inf"), tl.float32)
    for offset in range(0, n_elements, BLOCK_SIZE):
        index = offset + offset_base
        mask = index < n_elements
        x = tl.load(x_ptr + index, mask, eviction_policy="evict_first", other=0.0)
        x_broadcast = tl.broadcast_to(x, [1, BLOCK_SIZE])
        x_abs = tl.abs(x_broadcast)
        acc = tl.maximum(acc, x_abs)
    out = tl.max(acc, 1)[:, None]
    tl.store(out_ptr + (tl.full([1, 1], 0, tl.int32)), out.to(tl.float32))


def abs_max(x: torch.Tensor) -> torch.Tensor:
    "Calculates the global max of the absolute values of a tensor"
    output = torch.empty((), device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 1024
    abs_max_kernel[grid](x, output, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def abs_max_to_scale_kernel_e4m3(x_ptr, out_ptr, clamp_float16):
    abs_max = tl.load(x_ptr).to(tl.float32)
    clamped = E4M3_MAX_POS / tl.clamp(abs_max, min=EPS, max=float("inf"))
    if clamp_float16:
        clamped = tl.clamp(clamped, min=EPS, max=FP16_MAX_POS)
    tl.store(out_ptr, clamped)


@triton.jit
def abs_max_to_scale_kernel_e5m2(x_ptr, out_ptr, clamp_float16):
    abs_max = tl.load(x_ptr)
    clamped = E5M2_MAX_POS / tl.clamp(abs_max, min=EPS, max=float("inf"))
    if clamp_float16:
        clamped = tl.clamp(clamped, min=EPS, max=FP16_MAX_POS)
    tl.store(out_ptr, clamped)


def abs_max_to_scale(
    x: torch.Tensor, fp8_dtype: torch.dtype, clamp_float16: bool
) -> torch.Tensor:
    assert x.numel() == 1, "Expected a single value, but got: {} elements".format(
        x.numel()
    )
    assert x.dtype == torch.float32, "Expected a float32 tensor, but got: {}".format(
        x.dtype
    )
    output = torch.empty((), device=x.device, dtype=torch.float32)
    grid = lambda meta: (1,)
    if fp8_dtype == torch.float8_e4m3fn:
        abs_max_to_scale_kernel_e4m3[grid](x, output, clamp_float16)
    else:
        abs_max_to_scale_kernel_e5m2[grid](x, output, clamp_float16)
    return output
