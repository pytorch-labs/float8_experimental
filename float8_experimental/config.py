# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#
# Weight caching.
#

# If True, allocates buffers for float8 weight cache
allocate_float8_weight_cache_buffers = False

# A global flag for controlling the weight cache, off by default. Intended
# usage is for users to modify this from their training loop directly
# according to their microbatching/pipeline parallel setup.
# Note: this is currently a global flag for simplicity and dynamo performance.
weight_cache_enabled = False

#
# Other
#

# If True, on the first iteration of Float8Linear the amaxes will be
# initialized with the incoming data. As of 2023-12-30, this doesn't work
# with autocast + torch.compile + FSDP. Enabling this option is nice for
# testing, but this is not necessary for real training jobs.
enable_amax_init = True

# If True, pre-forward and post-forward functions are run. As of 2023-12-30,
# this doesn't work with autocast + torch.compile + FSDP. Enabling this
# option is useful for safety, but not strictly necessary.
enable_pre_and_post_forward = True
