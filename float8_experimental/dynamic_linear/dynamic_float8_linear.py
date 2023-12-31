# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A wrapper around a `torch.nn.Linear` module which does fp8 compute.
"""

import torch

from float8_experimental.float8_tensor import Float8Tensor, calculate_amax_and_cast_to_float8
from float8_experimental.float8_utils import tensor_to_scale, to_fp8_saturated
import float8_experimental.config as config


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
        gradY_scale = tensor_to_scale(gradY, torch.float8_e5m2)
        gradY_scaled = gradY * gradY_scale
        bits_fp8 = to_fp8_saturated(gradY_scaled, torch.float8_e5m2)
        return (
            Float8Tensor(bits_fp8, gradY_scale, gradY.dtype, emulate=ctx.emulate),
            None,
        )


class Float8DynamicLinear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute. By on the fly
    conversion to fp8 of the input and weight tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config.allocate_float8_weight_cache_buffers:
            # this is a buffer to get `to(dtype)` for free
            # TODO(future): hide this from serialization
            # TODO(future): force this to stay in float8_e4m3fn
            self.register_buffer(
                "cached_fp8_weight",
                torch.empty(self.weight.shape, dtype=torch.float8_e4m3fn),
            )

        self.add_weight_tag()

    def forward(self, x):
        x_fp8 = self.cast_x_to_float8(x)
        if getattr(self, "_w_fp8", None) is not None:  # FSDP handled the cast
            w_fp8 = self._w_fp8
        else:
            w_fp8 = self.cast_w_to_float8(self.weight)

        y = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)

        # Cast gradY to float8_e5m2 during backward
        y = self.cast_to_float8e5m2_bw(y)

        return y

    def add_weight_tag(self):
        # We add a tag to the weight nn.Parameter in order to signal
        # To FSDP that this param is a weight
        self.weight._is_fp8_weight = True

    def cast_x_to_float8(self, inpt_tensor):
        scale = tensor_to_scale(inpt_tensor, torch.float8_e4m3fn)
        return Float8Tensor.to_float8(
            inpt_tensor, scale, torch.float8_e4m3fn, emulate=self.emulate
        )

    def cast_w_to_float8(self, w):
        with torch.no_grad():
            scale = tensor_to_scale(w, torch.float8_e4m3fn)
            if config.weight_cache_enabled:
                assert config.allocate_float8_weight_cache_buffers, (
                    "float8 weight cache buffer must be allocated using "
                    + "`allocate_float8_weight_cache_buffers` to use the weight cache"
                )
                w_bits_fp8 = self.cached_fp8_weight
            else:
                # manual calculation of fp8 bits:
                # 1. calculate the bits without Float8Tensor, without grad
                # 2. store the bits here
                # 3. create Float8Tensor from the bits calculated in 2
                # motivation: this will take care of saving the bits without
                # interacting with tensor subclasses, as w_fp8._data is not
                # currently traceable by dynamo
                w_bits_fp8 = calculate_amax_and_cast_to_float8(
                    w, scale, torch.float8_e4m3fn, amax_buffer=None 
                )
                if config.allocate_float8_weight_cache_buffers:
                    self.cached_fp8_weight.copy_(w_bits_fp8)
        w_fp8 = Float8Tensor.to_float8(
            w,
            scale,
            torch.float8_e4m3fn,
            emulate=self.emulate,
            cached_casted_weight=w_bits_fp8,
        )
        return w_fp8


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
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.emulate = emulate
        new_mod.add_weight_tag()
        if config.allocate_float8_weight_cache_buffers:
            new_mod.cached_fp8_weight = torch.empty(
                new_mod.cached_fp8_weight.shape, 
                dtype=new_mod.cached_fp8_weight.dtype,
                device=new_mod.weight.device)
        return new_mod
