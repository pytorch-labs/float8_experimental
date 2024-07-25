# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import enum
from dataclasses import dataclass


class TensorScalingType(enum.Enum):
    DELAYED = "delayed"
    DYNAMIC = "dynamic"

    def short_str(self):
        if self is TensorScalingType.DELAYED:
            return "del"
        else:
            assert self is TensorScalingType.DYNAMIC
            return "dyn"


@dataclass(frozen=True)
class Float8TensorCastConfig:
    """
    Configuration for casting a single tensor to float8
    """

    scaling_type: TensorScalingType = TensorScalingType.DYNAMIC


@dataclass(frozen=True)
class Float8LinearConfig:
    """
    Configuration for converting a `torch.nn.Linear` module to float8
    for training.
    """

    #
    # Per-tensor configuration for `input`, `weight`, `grad_output`
    #
    cast_config_input: Float8TensorCastConfig = Float8TensorCastConfig()
    cast_config_weight: Float8TensorCastConfig = Float8TensorCastConfig()
    cast_config_grad_output: Float8TensorCastConfig = Float8TensorCastConfig()

    #
    # Per-linear configuration
    #

    # If True, on the first iteration of Float8Linear the amaxes will be
    # initialized with the incoming data. As of 2023-12-30, this doesn't work
    # with autocast + torch.compile + FSDP. Enabling this option is nice for
    # testing, but this is not necessary for real training jobs.
    enable_amax_init: bool = True

    # If True, pre-forward and post-forward functions are run. As of 2023-12-30,
    # this doesn't work with autocast + torch.compile + FSDP. Enabling this
    # option is useful for safety, but not strictly necessary.
    enable_pre_and_post_forward: bool = True

    # If True, then uses a tensor subclass for the float8 linear module's weight that
    # implements pre/post-all-gather methods to do float8 all-gather with FSDP2.
    enable_fsdp_float8_all_gather: bool = False

    # If True, then prior to performing the fp8 scaled mamtmul we will pad the
    # inner dimension of a (dim 1) and b (dim 2) with 0s. This is needed for matmuls
    # _scaled_mm since it has the strong constraint that for M,N,K  N, K must be a multiple of 16.
    # This can cause a memory spike however so we keep this off by default.
    pad_inner_dim: bool = False

    # If True, emulation is used instead of hardware accelerated gemm
    emulate: bool = False


# If True, use 'fnuz' float8 types for calculations.
# Currently, ROCm only supports fnuz variants.
# TODO(future PR): move this to Float8LinearConfig
use_fnuz_dtype = False
