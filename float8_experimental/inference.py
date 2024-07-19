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
from typing import Callable, List, Optional

import float8_experimental.config as config

import torch
import torch.nn as nn
from float8_experimental.float8_linear_utils import swap_linear_layers

from float8_experimental.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
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

    # TODO: A better name would be NONE, we should unify this with torchao
    WEIGHT_ONLY = auto()
    DYNAMIC = auto()
    STATIC = auto()


@dataclass(frozen=True)
class QuantConfig:
    """Defines the configuration for the quantization to fp8 of a linear module

    Args:
        activation_casting: The type of quantization to perform on the activations
        static_quantization_scale: The scale of the input to this linear module, used for static quantization only
    """

    activation_casting: ActivationCasting
    static_quantization_scale: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.activation_casting == ActivationCasting.STATIC:
            assert isinstance(
                self.static_quantization_scale, torch.Tensor
            ), "When activation_casting is 'static', activation_scale must be a tensor."


class Float8InferenceLinear(torch.nn.Linear):
    """
    This is a wrapper around torch.nn.Linear that supports FP8 inference
    Supported forms of inference:
        - FP8 inference with high precision matmul - weight only
        - FP8 inference with fp8 matmul and dynamic weight casting
        - FP8 inference with fp8 matmul and static weight casting
    """

    def __init__(
        self,
        # FP8 specific arguments
        quant_config: QuantConfig,
        linear_mm_config: LinearMMConfig,
        # nn.Linear arguments
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # Construct the superclass this will create dummy weights and biases
        super().__init__(in_features, out_features, bias, device, dtype)
        self.linear_mm_config = linear_mm_config
        self.activation_casting = quant_config.activation_casting
        if self.activation_casting == ActivationCasting.STATIC:
            self.register_buffer(
                "static_quantization_scale", quant_config.static_quantization_scale
            )
        else:
            self.static_quantization_scale = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_casting == ActivationCasting.WEIGHT_ONLY:
            return torch.nn.functional.linear(
                input, self.weight.to_original_precision()
            )

        x_fp8 = cast_to_float8_e4m3_inference(
            input,
            self.linear_mm_config,
            static_quantization_scale=self.static_quantization_scale,
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
            self.linear_mm_config,
            gemm_input_role=GemmInputRole.W,
        )
        self.weight = nn.Parameter(quantized_weight)
        self.weight.requires_grad = False

    def set_weight_and_bias(
        self, weight: torch.nn.Parameter, bias: Optional[torch.nn.Parameter]
    ):
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_float(
        cls, module: nn.Module, quant_config: QuantConfig, use_fast_accum: bool
    ) -> "Float8InferenceLinear":
        """
        Create an nn.Linear with fp8 compute from another nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            quant_config (QuantConfig): Configuration for the weight and activation casting
        """
        forward_config = ScaledMMConfig(
            False, use_fast_accum, pad_inner_dim=config.pad_inner_dim
        )
        linear_mm_config = LinearMMConfig(
            forward_config, forward_config, forward_config
        )
        linear = cls(
            quant_config,
            linear_mm_config,
            module.in_features,
            module.out_features,
            False,
            device=torch.device("meta"),
        )
        linear.set_weight_and_bias(module.weight, module.bias)
        linear.quantize_weight()
        return linear


def cast_to_float8_e4m3_inference(
    inpt_tensor: torch.Tensor,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    static_quantization_scale: Optional[torch.Tensor] = None,
) -> Float8Tensor:
    """Casts an input tensor to the Float8 (e4m3fn*)

    Args:
        inpt_tensor: The input tensor to be cast.
        linear_mm_config: Configuration settings for the matrix multiplication
        reduce_amax: Whether to reduce the amax (absolute maximum) among the local distributed group.
        static_quantization_scale: Optional tensor specifying the scale for activation. Default is None.

    Returns:
        Float8Tensor: The input tensor cast to Float8 (e4m3fn) format.

    Note:
        If the input tensor is already in Float8 format, it is returned as is without re-casting.
    """
    if tensor_already_casted_to_fp8(inpt_tensor):
        return inpt_tensor
    scale = (
        static_quantization_scale
        if static_quantization_scale is not None
        else tensor_to_scale(inpt_tensor, e4m3_dtype, reduce_amax)
    )
    return Float8Tensor.to_float8(
        inpt_tensor,
        scale,
        e4m3_dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=GemmInputRole.X,
    )


def quantize_to_float8(
    module: nn.Module,
    quant_config: QuantConfig,
    *,
    layer_filter_fn: Optional[Callable[[str, nn.Module], bool]] = None,
    use_fast_accum: bool = True,
) -> Optional[nn.Module]:
    """
    Converts torch.nn.Linear layers in the given module to Float8InferenceLinear.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module (nn.Module): The module to modify.
        quant_config (QuantConfig): Quantization configuration for Float8 conversion.
        layer_filter_fn: If specified, only the modules that
            that pass the filter function will be swapped. The inputs to the
            filter function are the FQN and module instance.
        use_fast_accum : Whether to enable fast accumulation for the Float8InferenceLinear. Defaults to True.

    Returns:
        nn.Module: The modified module with applicable Linear layers converted to Float8.

    Raises:
        AssertionError: If a root-level nn.Linear with children is encountered.
    """
    return swap_linear_layers(
        module,
        lambda m: Float8InferenceLinear.from_float(m, quant_config, use_fast_accum),
        layer_filter_fn=layer_filter_fn,
    )
