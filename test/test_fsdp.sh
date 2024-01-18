#!/bin/bash

# terminate script on first error
set -e

launch() {
    echo "Launching test with the following configuration:"
    echo "IS_FP8:                   $IS_FP8"
    echo "compile_fsdp:             $COMPILE"
    echo "fullgraph:                $FULLGRAPH"
    echo "recompute_weight_cast:    $RECOMPUTE"

    # generate the test data
    python test/test_fsdp.py --mode generate --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH --recompute_weight_cast $RECOMPUTE
    echo "Success: ✅"

    # generate single GPU model output and updated state dict
    python test/test_fsdp.py --mode single_gpu --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH --recompute_weight_cast $RECOMPUTE
    echo "Success: ✅"

    # generate FSDP model output and updated state dict
    # the NCCL_DEBUG setting is to avoid log spew
    # the CUDA_VISIBLE_DEVICES setting is for easy debugging
    # the NCCL_NET setting is to work around transient issues on a
    #   specific host (`devgpu001.nha2`)
    NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 NCCL_NET=SOCKET python test/test_fsdp.py \
        --mode fsdp --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH --recompute_weight_cast $RECOMPUTE

    # compare the outputs and state dicts and verify equivalence
    python test/test_fsdp.py --mode analyze --is_fp8 $IS_FP8 --compile_fsdp $COMPILE --fullgraph $FULLGRAPH --recompute_weight_cast $RECOMPUTE
    echo "Success: ✅"

    echo "✅ All Tests Passed ✅"
}

# Loop over different combinations of settings
for i in False,False,False,False \
         True,False,False,False  \
         True,True,False,False   \
         True,False,False,True   \
         True,True,False,True
do
    # Split the string into variables
    IFS=","
    set -- $i

    # Assign each variable to a more descriptive name
    IS_FP8=$1
    COMPILE=$2
    FULLGRAPH=$3
    RECOMPUTE=$4

    # Launch the test with the current settings
    launch
done
