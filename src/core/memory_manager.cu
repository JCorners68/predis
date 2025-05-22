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

#include "memory_manager.h"
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>

namespace predis {
namespace core {

struct MemoryManager::Impl {
    bool initialized = false;
    size_t max_memory_bytes = 0;
    size_t allocated_bytes = 0;
    size_t allocation_count = 0;
    void* gpu_memory_pool = nullptr;
    std::unordered_map<void*, size_t> allocations;
    std::mutex allocation_mutex;
    
    // Simple pool allocator
    size_t pool_block_size = 0;
    size_t pool_num_blocks = 0;
    void* pool_start = nullptr;
    std::vector<bool> pool_block_used;
};

MemoryManager::MemoryManager() : pImpl(std::make_unique<Impl>()) {
}

MemoryManager::~MemoryManager() {
    if (pImpl->initialized) {
        shutdown();
    }
}

bool MemoryManager::initialize(size_t max_memory_bytes) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    // Get GPU device info
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Set device 0
    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get memory info
    size_t free_bytes, total_bytes;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get GPU memory info: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Determine max memory to use (80% of available if not specified)
    if (max_memory_bytes == 0) {
        pImpl->max_memory_bytes = static_cast<size_t>(free_bytes * 0.8);
    } else {
        pImpl->max_memory_bytes = std::min(max_memory_bytes, free_bytes);
    }
    
    std::cout << "GPU Memory Info:" << std::endl;
    std::cout << "  Total: " << (total_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Free:  " << (free_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Using: " << (pImpl->max_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    
    pImpl->initialized = true;
    return true;
}

void MemoryManager::shutdown() {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    // Free all allocations
    for (auto& [ptr, size] : pImpl->allocations) {
        cudaFree(ptr);
    }
    pImpl->allocations.clear();
    
    // Free memory pool if exists
    if (pImpl->pool_start) {
        cudaFree(pImpl->pool_start);
        pImpl->pool_start = nullptr;
    }
    
    pImpl->allocated_bytes = 0;
    pImpl->allocation_count = 0;
    pImpl->initialized = false;
    
    std::cout << "GPU Memory Manager shutdown complete" << std::endl;
}

void* MemoryManager::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!pImpl->initialized) {
        std::cerr << "MemoryManager not initialized" << std::endl;
        return nullptr;
    }
    
    if (pImpl->allocated_bytes + size > pImpl->max_memory_bytes) {
        std::cerr << "Out of GPU memory: requested " << size << " bytes, " 
                  << (pImpl->max_memory_bytes - pImpl->allocated_bytes) << " available" << std::endl;
        return nullptr;
    }
    
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    
    pImpl->allocations[ptr] = size;
    pImpl->allocated_bytes += size;
    pImpl->allocation_count++;
    
    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!ptr) return;
    
    auto it = pImpl->allocations.find(ptr);
    if (it != pImpl->allocations.end()) {
        size_t size = it->second;
        pImpl->allocated_bytes -= size;
        pImpl->allocation_count--;
        pImpl->allocations.erase(it);
        
        cudaFree(ptr);
    }
}

bool MemoryManager::create_pool(size_t block_size, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (pImpl->pool_start) {
        std::cerr << "Memory pool already exists" << std::endl;
        return false;
    }
    
    size_t pool_size = block_size * num_blocks;
    cudaError_t error = cudaMalloc(&pImpl->pool_start, pool_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create memory pool: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    pImpl->pool_block_size = block_size;
    pImpl->pool_num_blocks = num_blocks;
    pImpl->pool_block_used.resize(num_blocks, false);
    
    std::cout << "Created GPU memory pool: " << num_blocks << " blocks of " 
              << block_size << " bytes each" << std::endl;
    
    return true;
}

void* MemoryManager::allocate_from_pool(size_t size) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!pImpl->pool_start || size > pImpl->pool_block_size) {
        return nullptr;
    }
    
    // Find free block
    for (size_t i = 0; i < pImpl->pool_num_blocks; ++i) {
        if (!pImpl->pool_block_used[i]) {
            pImpl->pool_block_used[i] = true;
            return static_cast<char*>(pImpl->pool_start) + (i * pImpl->pool_block_size);
        }
    }
    
    return nullptr;  // Pool exhausted
}

void MemoryManager::deallocate_to_pool(void* ptr) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!ptr || !pImpl->pool_start) return;
    
    // Calculate block index
    char* block_ptr = static_cast<char*>(ptr);
    char* pool_ptr = static_cast<char*>(pImpl->pool_start);
    
    if (block_ptr < pool_ptr) return;
    
    size_t offset = block_ptr - pool_ptr;
    size_t block_index = offset / pImpl->pool_block_size;
    
    if (block_index < pImpl->pool_num_blocks) {
        pImpl->pool_block_used[block_index] = false;
    }
}

MemoryManager::MemoryStats MemoryManager::get_stats() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    MemoryStats stats;
    stats.total_bytes = pImpl->max_memory_bytes;
    stats.allocated_bytes = pImpl->allocated_bytes;
    stats.free_bytes = pImpl->max_memory_bytes - pImpl->allocated_bytes;
    stats.allocation_count = pImpl->allocation_count;
    
    if (pImpl->max_memory_bytes > 0) {
        stats.fragmentation_ratio = static_cast<double>(pImpl->allocated_bytes) / pImpl->max_memory_bytes;
    }
    
    return stats;
}

bool MemoryManager::is_out_of_memory() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    return pImpl->allocated_bytes >= pImpl->max_memory_bytes * 0.95;  // 95% threshold
}

void MemoryManager::defragment() {
    std::cout << "GPU memory defragmentation not implemented yet" << std::endl;
}

} // namespace core
} // namespace predis