# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import collections
import json


def profiler_output_to_time_by_kernel_name(prof):
    """
    Input: a profiler with captured events.
    Output: a deduplicated list of GPU time in nanoseconds grouped by CPU kernel name

    Note that if there are user_annotations in the captured events, `torch.profiler`
    will include their time in the total GPU time displayed at the bottom of
    `key_averages.table()`. The filter below excludes them to prevent double
    counting.
    """
    key_averages = prof.key_averages()
    thresh = 1e-10
    kernel_name_to_gpu_time_us = collections.defaultdict(float)
    for e in key_averages:
        # manually filter top-level CPU events with attributed CUDA time
        # example CPU event row:
        #                                               aten::addmm         0.83%      76.554us         0.98%      90.846us      90.846us       1.022ms        31.82%       1.022ms       1.022ms             1
        # and it maps to this CUDA event:
        #   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64...         0.00%       0.000us         0.00%       0.000us       0.000us       1.022ms        31.82%       1.022ms       1.022ms             1
        if not (e.self_cpu_time_total > thresh and e.self_device_time_total > thresh):
            continue
        kernel_name_to_gpu_time_us[e.key] = e.self_device_time_total
    return kernel_name_to_gpu_time_us


def profiler_output_to_gpu_time_for_key(prof, key):
    """
    Input: an event name
    Output: sum of GPU time of all events with that name in `prof`

    This is useful to get the total time of a user annotation
    """
    total = 0
    for e in prof.profiler.function_events:
        if e.key == key:
            total += e.device_time_total
    return total
