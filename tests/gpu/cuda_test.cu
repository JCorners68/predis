/*
 * Copyright 2025 Predis Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloPredis(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Hello from Predis GPU thread %d!\n", idx);
    }
}

int main() {
    // Check GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    }
    
    // Launch simple kernel
    int n = 16;
    int blockSize = 8;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    std::cout << "\nLaunching Predis CUDA kernel..." << std::endl;
    helloPredis<<<gridSize, blockSize>>>(n);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "\nPredis CUDA test completed successfully!" << std::endl;
    return 0;
}