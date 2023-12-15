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
