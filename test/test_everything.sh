#!/bin/bash

# terminate script on first error
set -e

pytest test/test_base.py
pytest test/test_sam.py
pytest test/test_compile.py
./test/test_fsdp.sh
./test/test_fsdp_compile.sh
./test/test_dtensor.sh
pytest test/test_fsdp2/test_fsdp2_eager.py

echo "all tests successful"
