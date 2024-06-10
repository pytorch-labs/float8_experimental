# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# If True, on the first iteration of Float8Linear the amaxes will be
# initialized with the incoming data. As of 2023-12-30, this doesn't work
# with autocast + torch.compile + FSDP. Enabling this option is nice for
# testing, but this is not necessary for real training jobs.
enable_amax_init = True

# If True, pre-forward and post-forward functions are run. As of 2023-12-30,
# this doesn't work with autocast + torch.compile + FSDP. Enabling this
# option is useful for safety, but not strictly necessary.
enable_pre_and_post_forward = True

# If True, then uses a tensor subclass for the fp8 linear module's weight that
# implements pre/post-all-gather methods to do fp8 all-gather with FSDP2.
# Only dynamic scaling is supported for now.
enable_fsdp_fp8_all_gather = False

# If True, use 'fnuz' float8 types for calculations.
# Currently, ROCm only supports fnuz variants.
use_fnuz_dtype = False
