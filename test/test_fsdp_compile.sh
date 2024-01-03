#!/bin/bash

# terminate script on first error
set -e

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python test/test_fsdp_compile.py
