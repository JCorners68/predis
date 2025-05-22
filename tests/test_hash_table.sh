#!/bin/bash

# Test script for GPU Hash Table
set -e

echo "🔨 Building GPU Hash Table Test..."

# Create build directory if it doesn't exist
mkdir -p ../build/tests

# Compile the test
nvcc -o ../build/tests/gpu_hash_table_test \
    gpu_hash_table_test.cpp \
    ../src/core/data_structures/gpu_hash_table.cu \
    ../src/core/memory_manager.cu \
    -I../src \
    -std=c++17 \
    -O2 \
    -g \
    -lcuda \
    -lcudart

echo "✅ Build successful!"
echo ""
echo "🚀 Running GPU Hash Table Test..."
echo ""

# Run the test
cd ../build/tests
./gpu_hash_table_test

echo ""
echo "🎯 Test execution completed!"