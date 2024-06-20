# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import io
import random
import unittest

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear_utils import swap_linear_with_float8_linear
from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import compute_error
from float8_experimental.inference import (
    ActivationCasting,
    Float8LinearInference,
    QuantConfig,
    quantize_to_float8,
)


random.seed(0)
torch.manual_seed(0)

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4096, 14336, bias=False)
        self.w3 = nn.Linear(4096, 14336, bias=False)
        self.w2 = nn.Linear(14336, 4096, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class TestHPTrainToFP8LinearInference:
    def base_test_mlp_transform(self, base_mlp, quantized_mlp, input_tensor):
        with torch.no_grad():
            base_output = base_mlp(input_tensor)
            transformed_output = quantized_mlp(input_tensor)

        # Compute and check SQNR
        sqnr = compute_error(base_output, transformed_output)
        assert sqnr.item() > 20, f"SQNR is too low: {sqnr.item()} dB"

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_H100,
        "CUDA not available or on non H100 machine",
    )
    def test_dynamic_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        dynamic_fp8_mlp = copy.deepcopy(original_mlp)

        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(dynamic_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_dynamic_fp8_mlp = torch.compile(
            dynamic_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_dynamic_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_H100,
        "CUDA not available or on non H100 machine",
    )
    def test_static_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(
            ActivationCasting.STATIC,
            activation_scale=torch.tensor([1.0], device="cuda", dtype=torch.float32),
        )
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_H100,
        "CUDA not available or on non H100 machine",
    )
    def test_weight_only_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(ActivationCasting.WEIGHT_ONLY)
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )


class TestFP8TrainToFP8LinearInference:
    def train(self, model: nn.Module, dtype: torch.dtype):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        target_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
        for _ in range(10):
            input_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_H100,
        "CUDA not available or on non H100 machine",
    )
    def test_fp8_save_and_load(self, dtype: torch.dtype):
        # Initialize FP8 model
        fp8_mlp = FeedForward().to("cuda", dtype=torch.float32)
        fp8_mlp.reset_parameters()
        swap_linear_with_float8_linear(
            fp8_mlp,
            Float8DynamicLinear,
        )

        # Train the model
        self.train(fp8_mlp, dtype)

        # Generate input tensor and original out
        input_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
        og_out = fp8_mlp(input_tensor)

        # Save model state dict
        buffer = io.BytesIO()
        torch.save(fp8_mlp.state_dict(), buffer)

        # Reset buffer position to the beginning
        buffer.seek(0)

        # Later on you load the model, will be w/ Float8DynamicLinear on meta device
        with torch.device("meta"):
            new_fp8_mlp = FeedForward().to(dtype=dtype)
            swap_linear_with_float8_linear(
                new_fp8_mlp,
                Float8DynamicLinear,
            )

        # Load the actual data
        new_fp8_mlp.load_state_dict(
            torch.load(buffer, weights_only=True), strict=True, assign=True
        )

        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(new_fp8_mlp, quant_config)

        fp8_mod_count = 0
        for module in new_fp8_mlp.modules():
            if isinstance(module, Float8LinearInference):
                assert isinstance(module.weight, Float8Tensor)
                assert module.weight.requires_grad is False
                fp8_mod_count += 1
        assert fp8_mod_count == 3, "Expected 3 FP8 modules, got {}".format(
            fp8_mod_count
        )

        new_out = new_fp8_mlp(input_tensor)

        # Assert exact equality
        assert torch.all(og_out == new_out).item()


if __name__ == "__main__":
    pytest.main([__file__])
