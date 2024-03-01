#!/bin/bash

# terminate script on first error
set -e

if python -c 'import torch;print(torch.cuda.is_available())' | grep -q "False"; then
    echo "Skipping test_tp.sh because no CUDA devices are available."
    exit
fi

NCCL_DEBUG=WARN torchrun --nproc_per_node 2 test/test_tp.py
