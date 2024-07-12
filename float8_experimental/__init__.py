# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Lets define a few top level things here
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_tensor import Float8Tensor, ScaledMMConfig, GemmInputRole

# Needed to load Float8Tensor with weights_only = True
from torch.serialization import add_safe_globals

add_safe_globals([Float8Tensor, ScaledMMConfig, GemmInputRole])

__all__ = ["Float8Tensor", "Float8Linear"]
