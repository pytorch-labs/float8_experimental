#!/bin/bash

# the NCCL_DEBUG setting is to avoid log spew
# the CUDA_VISIBLE_DEVICES setting is for easy debugging
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python test/test_fsdp_compile.py

echo "done!"
