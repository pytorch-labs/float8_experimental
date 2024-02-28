import torch
import triton
import triton.language as tl
from triton import next_power_of_2


E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
FP16_MAX_POS = torch.finfo(torch.float16).max
EPS = 1e-12


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def maximum(a, b):
    mask = a > b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def abs_max_kernel(
    x_ptr,
    out_ptr,
    x_numel: int,
    r_numel: int,
    X_BLOCK_SIZE: tl.constexpr,
    R_BLOCK_SIZE: tl.constexpr,
):
    x_offset = tl.program_id(0) * X_BLOCK_SIZE
    x_index = x_offset + tl.arange(0, X_BLOCK_SIZE)[:, None]
    x_mask = x_index < x_numel
    reduction_base = tl.arange(0, R_BLOCK_SIZE)[None, :]
    acc = tl.full([X_BLOCK_SIZE, R_BLOCK_SIZE], -float("inf"), tl.float32)
    for r_offset in range(0, r_numel, R_BLOCK_SIZE):
        r_index = r_offset + reduction_base
        r_mask = r_index < r_numel
        values = tl.load(
            x_ptr + (r_index + (r_numel * x_index)),
            x_mask & r_mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        x_abs = tl.abs(values)
        x_abs_broadcasted = tl.broadcast_to(x_abs, [X_BLOCK_SIZE, R_BLOCK_SIZE])
        acc_mask = maximum(acc, x_abs_broadcasted)
        acc = tl.where(x_mask, acc_mask, acc)
    out = tl.max(acc, 1)[:, None]
    tl.store(out_ptr + x_index, out.to(tl.float32), x_mask)


def abs_max(x: torch.Tensor) -> torch.Tensor:
    "Calculates the global max of the absolute values of a tensor"
    output = torch.empty((512, 1), device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (meta["X_BLOCK_SIZE"],)
    X_BLOCK_SIZE = 1
    R_BLOCK_SIZE = 1024
    r_numel = n_elements // 512
    abs_max_kernel[grid](
        x,
        output,
        x_numel=512,
        r_numel=r_numel,
        X_BLOCK_SIZE=X_BLOCK_SIZE,
        R_BLOCK_SIZE=R_BLOCK_SIZE,
    )
    return output.max()


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
