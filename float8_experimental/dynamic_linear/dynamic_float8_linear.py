"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

import torch

from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import (amax_to_scale, tensor_to_amax,
                                              to_fp8_saturated)


class NoopFwToFloat8E5M2Bw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2, initialize if needed
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        emulate: bool,
    ):
        ctx.emulate = emulate
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        gradY_scale = amax_to_scale(
            tensor_to_amax(gradY), torch.float8_e5m2, gradY.dtype
        )
        gradY_scaled = gradY * gradY_scale
        bits_fp8 = to_fp8_saturated(gradY_scaled, torch.float8_e5m2)
        return (
            Float8Tensor(bits_fp8, gradY_scale, gradY.dtype, emulate=ctx.emulate),
            None,
        )


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_weight_tag()

    def forward(self, x):
        x_fp8 = self.cast_to_float8(x)
        if getattr(self, "_w_fp8", None) is not None:  # FSDP handled the cast
            w_fp8 = self._w_fp8
        else:
            w_fp8 = self.cast_to_float8(self.weight)

        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)

        # Cast gradY to float8_e5m2 during backward
        y = self.cast_to_float8e5m2_bw(y)

        return y

    def add_weight_tag(self):
        # We add a tag to the weight nn.Parameter in order to signal
        # To FSDP that this param is a weight
        self.weight._is_fp8_weight = True

    def cast_to_float8(self, inpt_tensor):
        scale = amax_to_scale(
            tensor_to_amax(inpt_tensor), torch.float8_e4m3fn, inpt_tensor.dtype
        )
        return Float8Tensor.to_float8(inpt_tensor, scale, torch.float8_e4m3fn)

    def cast_to_float8e5m2_bw(self, gradY):
        return NoopFwToFloat8E5M2Bw.apply(gradY, self.emulate)

    @classmethod
    def from_float(cls, mod, emulate: bool = False):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            emulate (bool): whether to emulate fp8 matmul logic in float32
        """
        # TODO Follow up! This is a great idea but we need the mixin base to create real
        # Tensors and the Linear base to create empty params
        # with torch.device("meta"):
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        if mod.bias is not None:
            new_mod.bias_dtype = mod.bias.dtype
        # I think its okay to send all params and buffers to device
        new_mod.to(mod.weight.device)
        new_mod.add_weight_tag()
        return new_mod
