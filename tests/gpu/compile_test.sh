#!/bin/bash
cd "$(dirname "$0")"
nvcc -o cuda_test cuda_test.cu
if [ $? -eq 0 ]; then
    echo "CUDA test compiled successfully"
    echo "Run with: ./cuda_test"
else
    echo "CUDA compilation failed"
    exit 1
fi