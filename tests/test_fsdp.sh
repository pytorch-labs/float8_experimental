#!/bin/bash

# terminate script on first error
set -e

launch() {
    echo "launching IS_FP8 $IS_FP8"

    # generate the test data
    python tests/test_fsdp.py --mode generate --is_fp8 $IS_FP8

    # generate single GPU model output and updated state dict
    python tests/test_fsdp.py --mode single_gpu --is_fp8 $IS_FP8

    # generate FSDP model output and updated state dict
    # the NCCL_DEBUG setting is to avoid log spew
    # the CUDA_VISIBLE_DEVICES setting is for easy debugging
    NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python tests/test_fsdp.py \
        --mode fsdp --is_fp8 $IS_FP8

    # compare the outputs and state dicts and verify equivalence
    python tests/test_fsdp.py --mode analyze --is_fp8 $IS_FP8

    echo "done"
}

for IS_FP8 in False True
do
    launch
done
