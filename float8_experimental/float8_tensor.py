from typing import Dict, Optional
from dataclasses import dataclass

import torch
from float8_experimental.float8_utils import tensor_to_amax, to_fp8_saturated

aten = torch.ops.aten


@dataclass(frozen=True)
class ScaledMMConfig:
    emulate: bool = False
    use_fast_accum: bool = False
    fp8_output: bool = False

    def __post_init__(self):
        if self.use_fast_accum:
            assert not self.emulate, "fast_accum only works with real compute"

    def merge(self, other: "ScaledMMConfig") -> "ScaledMMConfig":
        """Merges two configs together emulate behavior must match,
        However we want to use_fast_accum in forward and not in backward.
        We do this by populating the fields of the backproping grad. Same applies for fp8_output.
        """
        assert isinstance(other, ScaledMMConfig)
        assert self.emulate == other.emulate
        return ScaledMMConfig(
            emulate=self.emulate,
            use_fast_accum=self.use_fast_accum and other.use_fast_accum,
            fp8_output=self.fp8_output and other.fp8_output,
        )


class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        scale: float,
        float8_dtype=torch.float8_e4m3fn,
        amax_buffer=None,
        mm_config: Optional[ScaledMMConfig] = None,
    ):
        # In TransformerEngine, the casts to float8 are fused with calculating
        # the new amax value. In this codebase, the eager mode code for those
        # two things is colocated in this function. We expect PT2.0 to fuse it
        # for us.
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        tensor_scaled = tensor * scale
        bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
        return Float8Tensor(bits_fp8, scale, tensor.dtype, mm_config=mm_config)

    @staticmethod
    def backward(ctx, g):
        if isinstance(g, Float8Tensor):
            return g.to_original_precision(), None, None, None, None
        else:
            return g, None, None, None, None


class FromFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion from fp8
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return Float8Tensor.to_float8(g), None, None


class Float8Tensor(torch.Tensor):
    """
    A Python-only Float8 tensor subclass.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.
    * `_emulate`: if true using fp32 emulation for the matmuls, helpful
      if you don't have access to h100 hardware.

    Intended usage of this abstraction:
    1. to bundle raw data + fp8 metadata together for easy passing through
       Python PyTorch systems.
    2. Float8-aware user code can use the private fields on these tensors
       to call into float8 operations.
    3. Float8-agnostic user code can use these tensors as is - they will
       convert to original precision in `__torch_dispatch__`.
    """

    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _mm_config: ScaledMMConfig
    __slots__ = ["_data", "_scale", "_orig_dtype", "_mm_config"]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        mm_config: Optional[ScaledMMConfig] = None,
    ):
        assert scale.numel() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self._mm_config = mm_config if mm_config is not None else ScaledMMConfig()

        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, mm_config={self._mm_config}\nas_orig_prec={self.to_original_precision()}"

    def __tensor_flatten__(self):
        ctx = {
            "_orig_dtype": self._orig_dtype,
            "_mm_config": self._mm_config,
        }
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata):
        assert len(inner_tensors) == 2
        return Float8Tensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["_orig_dtype"],
            metadata["_mm_config"],
        )

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_float8(
        tensor,
        scale,
        float8_dtype,
        amax_buffer=None,
        mm_config: Optional[ScaledMMConfig] = None,
    ):
        """Converts a higher precision tensor to float8 in a differentiable way.

        Args:
            tensor: the tensor to convert
            scale: the scale to use to convert the tensor
            float8_dtype: the float8 dtype to use
            amax_buffer: a buffer to store the amax value in prior to conversion

        Returns:
            Float8Tensor: a float8 tensor
        """
        return ToFloat8ConstrFunc.apply(
            tensor, scale, float8_dtype, amax_buffer, mm_config
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 1. tracing through __torch_function__ logic is not supported yet in
        # PT2.0, so we explicitly disallow it here for callsites from user code.
        # 2. We do need to handle a couple of ops in order for
        # TorchDynamo tracing to succeed.

        # Lazy import to avoid circular dependency
        from float8_experimental.float8_ops import FLOAT8_OPS_TABLE

        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


# In order for dynamo to successfuly trace our tensor subclass, we need
# to be able to represent it in the graph.
torch._dynamo.allow_in_graph(Float8Tensor)
