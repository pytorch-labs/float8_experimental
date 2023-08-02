#!/bin/bash

# generate the test data
python float8_playground/test_fsdp.py --mode generate

# generate single GPU model output and updated state dict
python float8_playground/test_fsdp.py --mode single_gpu

# generate FSDP model output and updated state dict
# the NCCL_DEBUG setting is to avoid log spew
# the CUDA_VISIBLE_DEVICES setting is for easy debugging
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python float8_playground/test_fsdp.py --mode fsdp

# compare the outputs and state dicts and verify equivalence
python float8_playground/test_fsdp.py --mode analyze
