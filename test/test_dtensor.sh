#!/bin/bash

# terminate script on first error
set -e

NCCL_DEBUG=WARN torchrun --nproc_per_node 2 test/test_dtensor.py
