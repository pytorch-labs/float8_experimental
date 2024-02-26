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

# If True, dynamic linear uses hooks for activation casting
# TODO(before land): add test coverage for both cases
# dynamic_use_activation_hooks = True
# dynamic_use_activation_hooks = False

use_fused_cast = True
