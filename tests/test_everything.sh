#!/bin/bash

# terminate script on first error
set -e

python tests/test.py
python tests/test_sam.py
./tests/test_fsdp.sh

echo "all tests successful"
