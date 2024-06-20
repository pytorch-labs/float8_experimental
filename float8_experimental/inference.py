# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Defines an nn module designed to be used during inference
"""
from dataclasses import dataclass

from enum import auto, Enum
from typing import List, Optional

import torch
import torch.nn as nn
from float8_experimental.float8_linear_utils import swap_linear_layers

from float8_experimental.float8_tensor import (
    Float8Tensor,
    ScaledMMConfig,
    tensor_already_casted_to_fp8,
    to_fp8_no_autograd,
)
from float8_experimental.float8_utils import e4m3_dtype, tensor_to_scale


class ActivationCasting(Enum):
    """Types of quantization to perform on the activations

    WEIGHT_ONLY: Only quantize the weight, no activation casting, weight will be dequantized in the forward pass
    STATIC: Activation is quantized during model initialization with a static scale
    DYNAMIC: Activation is quantized during forward pass with a dynamic scale calculated from the input activation
    """

    WEIGHT_ONLY = auto()
    DYNAMIC = auto()
    STATIC = auto()


@dataclass(frozen=True)
class QuantConfig:
    """Defines the configuration for the quantization to fp8 of a linear module

    Args:
        activation_casting: The type of quantization to perform on the activations
        activation_scale: The scale of the input to this linear module, used for static quantization only
    """

    activation_casting: ActivationCasting
    activation_scale: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.activation_casting == ActivationCasting.STATIC:
            assert isinstance(
                self.activation_scale, torch.Tensor
            ), "When activation_casting is 'static', activation_scale must be a tensor."


class Float8LinearInference(torch.nn.Linear):
    """
    This is a wrapper around torch.nn.Linear that supports FP8 inference
    Supported forms of infernce:
        - FP8 inference with fp32 matmul - weight only
        - FP8 inference with fp8 matmul and dynamic weight casting
        - FP8 inference with fp8 matmul and static weight casting
    """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_casting == ActivationCasting.WEIGHT_ONLY:
            return torch.nn.functional.linear(
                input, self.weight.to_original_precision()
            )

        x_fp8 = cast_to_float8_e4m3fn(
            input, self.forward_config, activation_scale=self.activation_scale
        )
        return torch.nn.functional.linear(x_fp8, self.weight, self.bias)

    # Builder functions for Float8LinearInference
    def quantize_weight(self, dtype: torch.dtype = e4m3_dtype) -> None:
        """This functions converts the weight to a Float8Tensor and sets its requires_grad to False.

        Args:
            dtype: The dtype to quantize the weight to. Default is e4m3_dtype.

        Note:
            This function is typically called during inference to quantize the weight once since
            the weight is not updated during inference.

        """
        assert not isinstance(
            self.weight, Float8Tensor
        ), "Weight has already been quantized, cannot quantize again."
        scale = tensor_to_scale(self.weight, dtype)
        quantized_weight = to_fp8_no_autograd(
            self.weight,
            scale,
            dtype,
            self.forward_config,
        )
        self.weight = nn.Parameter(quantized_weight)
        self.weight.requires_grad = False

    @classmethod
    def create_meta_class(
        cls, in_features: int, out_features: int
    ) -> "Float8LinearInference":
        with torch.device("meta"):
            return cls(in_features=in_features, out_features=out_features, bias=False)

    def set_mm_config(self, use_fast_accum: bool = True) -> "Float8LinearInference":
        """TODO Hardcode for now but we could/should likely add this to the constructor"""
        self.forward_config: ScaledMMConfig = ScaledMMConfig(False, use_fast_accum)
        return self

    def set_weight_and_bias(
        self, weight: torch.nn.Parameter, bias: Optional[torch.nn.Parameter]
    ) -> "Float8LinearInference":
        self.weight = weight
        self.bias = bias
        return self

    def set_quantization_config(
        self,
        quant_config: QuantConfig,
    ) -> "Float8LinearInference":
        # We destructure the quant_config into the individual fields
        # If an activation config is passed in we want to register that as a buffer
        self.activation_casting: ActivationCasting = quant_config.activation_casting
        self.quantize_weight()

        if self.activation_casting == ActivationCasting.STATIC:
            self.register_buffer("activation_scale", quant_config.activation_scale)
        else:
            self.activation_scale = None
        return self

    @classmethod
    def from_float(
        cls,
        module: nn.Module,
        quant_config: QuantConfig,
    ) -> "Float8LinearInference":
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            quant_config (QuantConfig): Configuration for the weight and activation casting
        """
        return (
            cls.create_meta_class(module.in_features, module.out_features)
            .set_weight_and_bias(module.weight, module.bias)
            .set_mm_config()
            .set_quantization_config(quant_config)
        )


def cast_to_float8_e4m3fn(
    inpt_tensor: torch.Tensor,
    mm_config: ScaledMMConfig,
    reduce_amax: bool = False,
    activation_scale: Optional[torch.Tensor] = None,
) -> Float8Tensor:
    """Casts an input tensor to the Float8 (e4m3fn) format for efficient computation.

    Args:
        inpt_tensor: The input tensor to be cast.
        mm_config: Configuration settings for the matrix multiplication
        reduce_amax: Whether to reduce the amax (absolute maximum) among the local distributed group.
        activation_scale: Optional tensor specifying the scale for activation. Default is None.

    Returns:
        Float8Tensor: The input tensor cast to Float8 (e4m3fn) format.

    Note:
        If the input tensor is already in Float8 format, it is returned as is without re-casting.
    """
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = (
        activation_scale
        if activation_scale is not None
        else tensor_to_scale(inpt_tensor, e4m3_dtype, reduce_amax)
    )
    return Float8Tensor.to_float8(inpt_tensor, scale, e4m3_dtype, mm_config=mm_config)


def quantize_to_float8(
    module: nn.Module,
    quant_config: QuantConfig,
    *,
    skip_fqn_list: Optional[List[str]] = None,
) -> nn.Module:
    """
    Converts torch.nn.Linear layers in the given module to Float8LinearInference.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module (nn.Module): The module to modify.
        quant_config (QuantConfig): Quantization configuration for Float8 conversion.
        skip_fqn_list (List[str], optional): List of module FQNs to skip during conversion.

    Returns:
        nn.Module: The modified module with applicable Linear layers converted to Float8.

    Raises:
        AssertionError: If a root-level nn.Linear with children is encountered.
    """
    return swap_linear_layers(
        module,
        lambda m: Float8LinearInference.from_float(m, quant_config),
        skip_fqn_list=skip_fqn_list,
    )
