#!/bin/bash

# terminate script on first error
set -e

pytest tests/test.py
pytest tests/test_sam.py
./tests/test_fsdp.sh

echo "all tests successful"
