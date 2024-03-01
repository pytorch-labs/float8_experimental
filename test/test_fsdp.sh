#!/bin/bash

# terminate script on first error
set -e

launch() {
    echo "launching IS_FP8 $IS_FP8, compile_fsdp $COMPILE, fullgraph $FULLGRAPH"

    # generate the test data
    python test/test_fsdp.py --mode generate --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH
    echo "Success: ✅"

    # generate single GPU model output and updated state dict
    python test/test_fsdp.py --mode single_gpu --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH
    echo "Success: ✅"

    # generate FSDP model output and updated state dict
    # the NCCL_DEBUG setting is to avoid log spew
    # the CUDA_VISIBLE_DEVICES setting is for easy debugging
    # the NCCL_NET setting is to work around transient issues on a
    #   specific host (`devgpu001.nha2`)
    NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 NCCL_NET=SOCKET python test/test_fsdp.py \
        --mode fsdp --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH

    # compare the outputs and state dicts and verify equivalence
    python test/test_fsdp.py --mode analyze --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH
    echo "Success: ✅"

    echo "✅ All Tests Passed ✅"
}

if python -c 'import torch;print(torch.cuda.is_available())' | grep -q "False"; then
    echo "Skipping test_fsdp.sh because no CUDA devices are available."
    return
fi

# IS_FP8, COMPILE, FULLGRAPH
for i in False,False,False True,False,False True,True,False
do
    IFS=","; set -- $i;
    IS_FP8=$1; COMPILE=$2; FULLGRAPH=$3
    launch
done
