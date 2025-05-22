#!/bin/bash

# Test script for GPU Memory Manager
set -e

echo "🔨 Building GPU Memory Manager Test..."

# Create build directory if it doesn't exist
mkdir -p ../build/tests

# Compile the test
nvcc -o ../build/tests/gpu_memory_manager_test \
    gpu_memory_manager_test.cpp \
    ../src/core/memory_manager.cu \
    -I../src \
    -std=c++17 \
    -O2 \
    -g

echo "✅ Build successful!"
echo ""
echo "🚀 Running GPU Memory Manager Test..."
echo ""

# Run the test
cd ../build/tests
./gpu_memory_manager_test

echo ""
echo "🎯 Test execution completed!"