# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Lets define a few top level things here
from float8_experimental.config import Float8LinearConfig
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import swap_linear_with_float8_linear
from float8_experimental.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)

# Needed to load Float8Tensor with weights_only = True
from torch.serialization import add_safe_globals

add_safe_globals([Float8Tensor, ScaledMMConfig, GemmInputRole, LinearMMConfig])

__all__ = [
    # configuration
    "Float8LinearConfig",
    # top level UX
    "swap_linear_with_float8_linear",
    # TODO(future): remove Float8Tensor and Float8Linear from public API
    "Float8Tensor",
    "Float8Linear",
]
