#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)


# Set USE_FNUZ_DTYPE environment variable if IS_ROCM is not empty
if [ -n "$IS_ROCM" ]; then
    export USE_FNUZ_DTYPE=true
    echo "ROCm detected. Set USE_FNUZ_DTYPE=true"
else
    export USE_FNUZ_DTYPE=false
    echo "ROCm not detected. Set USE_FNUZ_DTYPE=false"
fi

pytest test/test_base.py
pytest test/test_sam.py
pytest test/test_compile.py

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
./test/test_fsdp.sh
./test/test_fsdp_compile.sh
./test/test_dtensor.sh
pytest test/test_fsdp2/test_fsdp2_eager.py
fi

echo "all tests successful"
