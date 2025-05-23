#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

// Simple GPU write kernel
__global__ void simple_write_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate write with some computation
        data[idx] = idx * idx + 42;
    }
}

int main() {
    std::cout << "=== GPU Write Performance Test ===" << std::endl;
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    
    // Test parameters
    const int N = 10000000; // 10M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Warm up
    simple_write_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    const int NUM_RUNS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_RUNS; i++) {
        simple_write_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    double ops_per_run = N;
    double total_ops = ops_per_run * NUM_RUNS;
    double ops_per_sec = (total_ops / duration) * 1000.0;
    
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Throughput: " << ops_per_sec / 1e9 << " billion ops/sec" << std::endl;
    std::cout << "Bandwidth: " << (total_ops * sizeof(int)) / (duration / 1000.0) / 1e9 << " GB/s" << std::endl;
    
    // Cleanup
    cudaFree(d_data);
    
    std::cout << "\nThis shows REAL GPU performance, not simulated!" << std::endl;
    
    return 0;
}