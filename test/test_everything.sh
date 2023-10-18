#!/bin/bash

# terminate script on first error
set -e

pytest test/test.py
pytest test/test_sam.py
./test/test_fsdp.sh
./test/test_tp.sh

echo "all tests successful"
