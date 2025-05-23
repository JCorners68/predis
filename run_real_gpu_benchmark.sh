#!/bin/bash

# Script to compile and run real GPU benchmark
# This executes actual GPU operations and measures real performance

echo "=== Compiling Real GPU Benchmark ==="
echo "This benchmark executes actual GPU operations, not simulated results"
echo ""

# Create benchmark_results directory if it doesn't exist
mkdir -p benchmark_results

# Compile the real GPU benchmark
nvcc -o real_gpu_benchmark src/benchmarks/real_gpu_benchmark.cu -O3 -std=c++14

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run the benchmark
echo "=== Running Real GPU Benchmark ==="
./real_gpu_benchmark

if [ $? -eq 0 ]; then
    # Move results to benchmark_results directory with timestamp
    timestamp=$(date +%s)
    mv real_benchmark_results.json benchmark_results/real_gpu_results_${timestamp}.json
    echo ""
    echo "Benchmark completed successfully!"
    echo "Results saved to: benchmark_results/real_gpu_results_${timestamp}.json"
else
    echo "Benchmark execution failed!"
    exit 1
fi