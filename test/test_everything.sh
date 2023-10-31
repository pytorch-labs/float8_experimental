#!/bin/bash

# terminate script on first error
set -e

pytest test/test_base.py
pytest test/test_sam.py
pytest test/test_compile.py
./test/test_fsdp.sh
./test/test_tp.sh

echo "all tests successful"
