#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)

pytest test/test_base.py
pytest test/test_compile.py
pytest test/test_inference_flows.py
pytest test/test_numerics_integration.py

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
./test/test_fsdp.sh
./test/test_fsdp_compile.sh
./test/test_dtensor.sh
pytest test/test_fsdp2/test_fsdp2_eager.py
fi

echo "all tests successful"
