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
#include <iomanip>
#include <vector>

namespace predis {
namespace core {

struct MemoryManager::Impl {
    bool initialized = false;
    size_t max_memory_bytes = 0;
    size_t allocated_bytes = 0;
    size_t allocation_count = 0;
    void* gpu_memory_pool = nullptr;
    std::unordered_map<void*, size_t> allocations;
    mutable std::mutex allocation_mutex;
    
    // Enhanced pool allocator with multiple sizes
    struct MemoryPool {
        size_t block_size;
        size_t num_blocks;
        void* pool_start;
        std::vector<bool> block_used;
        size_t allocated_blocks;
        
        MemoryPool() : block_size(0), num_blocks(0), pool_start(nullptr), allocated_blocks(0) {}
    };
    
    std::vector<MemoryPool> pools;
    
    // Fragmentation tracking
    size_t peak_allocated_bytes = 0;
    size_t total_allocations_ever = 0;
    size_t total_deallocations_ever = 0;
    
    // Variable-size allocation tracking
    std::unordered_map<size_t, size_t> size_histogram;  // size -> count
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
    
    // Free all memory pools
    for (auto& pool : pImpl->pools) {
        if (pool.pool_start) {
            cudaFree(pool.pool_start);
        }
    }
    pImpl->pools.clear();
    
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
    pImpl->total_allocations_ever++;
    
    // Track peak memory usage
    if (pImpl->allocated_bytes > pImpl->peak_allocated_bytes) {
        pImpl->peak_allocated_bytes = pImpl->allocated_bytes;
    }
    
    // Update size histogram for variable-size tracking
    pImpl->size_histogram[size]++;
    
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
        pImpl->total_deallocations_ever++;
        pImpl->allocations.erase(it);
        
        // Update size histogram
        auto hist_it = pImpl->size_histogram.find(size);
        if (hist_it != pImpl->size_histogram.end()) {
            hist_it->second--;
            if (hist_it->second == 0) {
                pImpl->size_histogram.erase(hist_it);
            }
        }
        
        cudaFree(ptr);
    }
}

bool MemoryManager::create_pool(size_t block_size, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!pImpl->initialized) {
        std::cerr << "MemoryManager not initialized" << std::endl;
        return false;
    }
    
    // Check if pool with this block size already exists
    for (const auto& pool : pImpl->pools) {
        if (pool.block_size == block_size) {
            std::cerr << "Memory pool with block size " << block_size << " already exists" << std::endl;
            return false;
        }
    }
    
    size_t pool_size = block_size * num_blocks;
    if (pImpl->allocated_bytes + pool_size > pImpl->max_memory_bytes) {
        std::cerr << "Cannot create pool: would exceed memory limit" << std::endl;
        return false;
    }
    
    Impl::MemoryPool new_pool;
    cudaError_t error = cudaMalloc(&new_pool.pool_start, pool_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create memory pool: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    new_pool.block_size = block_size;
    new_pool.num_blocks = num_blocks;
    new_pool.block_used.resize(num_blocks, false);
    new_pool.allocated_blocks = 0;
    
    pImpl->pools.push_back(std::move(new_pool));
    pImpl->allocated_bytes += pool_size;
    
    std::cout << "Created GPU memory pool: " << num_blocks << " blocks of " 
              << block_size << " bytes each (" << (pool_size / 1024 / 1024) << " MB)" << std::endl;
    
    return true;
}

void* MemoryManager::allocate_from_pool(size_t size) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!pImpl->initialized) {
        return nullptr;
    }
    
    // Find best-fit pool (smallest block size that can accommodate the request)
    Impl::MemoryPool* best_pool = nullptr;
    for (auto& pool : pImpl->pools) {
        if (pool.block_size >= size) {
            if (!best_pool || pool.block_size < best_pool->block_size) {
                best_pool = &pool;
            }
        }
    }
    
    if (!best_pool) {
        return nullptr;  // No suitable pool found
    }
    
    // Find free block in the selected pool
    for (size_t i = 0; i < best_pool->num_blocks; ++i) {
        if (!best_pool->block_used[i]) {
            best_pool->block_used[i] = true;
            best_pool->allocated_blocks++;
            return static_cast<char*>(best_pool->pool_start) + (i * best_pool->block_size);
        }
    }
    
    return nullptr;  // Pool exhausted
}

void MemoryManager::deallocate_to_pool(void* ptr) {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!ptr) return;
    
    // Find the pool containing this pointer and deallocate
    for (auto& pool : pImpl->pools) {
        if (ptr >= pool.pool_start && 
            ptr < static_cast<char*>(pool.pool_start) + (pool.num_blocks * pool.block_size)) {
            
            char* block_ptr = static_cast<char*>(ptr);
            char* pool_ptr = static_cast<char*>(pool.pool_start);
            size_t offset = block_ptr - pool_ptr;
            size_t block_index = offset / pool.block_size;
            
            if (block_index < pool.num_blocks && pool.block_used[block_index]) {
                pool.block_used[block_index] = false;
                pool.allocated_blocks--;
            }
            break;
        }
    }
}

MemoryManager::MemoryStats MemoryManager::get_stats() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    MemoryStats stats;
    stats.total_bytes = pImpl->max_memory_bytes;
    stats.allocated_bytes = pImpl->allocated_bytes;
    stats.free_bytes = pImpl->max_memory_bytes - pImpl->allocated_bytes;
    stats.allocation_count = pImpl->allocation_count;
    stats.peak_allocated_bytes = pImpl->peak_allocated_bytes;
    stats.total_allocations_ever = pImpl->total_allocations_ever;
    stats.total_deallocations_ever = pImpl->total_deallocations_ever;
    stats.active_pools = pImpl->pools.size();
    
    // Add pool statistics to total
    for (const auto& pool : pImpl->pools) {
        // Pool memory is already counted in allocated_bytes during creation
        // but we need to subtract unused pool space from effective allocation
        size_t unused_pool_bytes = (pool.num_blocks - pool.allocated_blocks) * pool.block_size;
        stats.free_bytes += unused_pool_bytes;
    }
    
    // Calculate actual fragmentation ratio
    if (pImpl->max_memory_bytes > 0) {
        // Count fragmented blocks in pools
        size_t total_pool_blocks = 0;
        size_t used_pool_blocks = 0;
        size_t fragmented_blocks = 0;
        
        for (const auto& pool : pImpl->pools) {
            total_pool_blocks += pool.num_blocks;
            used_pool_blocks += pool.allocated_blocks;
            
            // Count fragmentation (free blocks between used blocks)
            bool in_used_sequence = false;
            for (size_t i = 0; i < pool.block_used.size(); ++i) {
                if (pool.block_used[i]) {
                    in_used_sequence = true;
                } else if (in_used_sequence) {
                    // Free block after used blocks = fragmentation
                    fragmented_blocks++;
                }
            }
        }
        
        if (total_pool_blocks > 0) {
            stats.fragmentation_ratio = static_cast<double>(fragmented_blocks) / total_pool_blocks;
        } else {
            stats.fragmentation_ratio = static_cast<double>(pImpl->allocated_bytes) / pImpl->max_memory_bytes;
        }
    }
    
    return stats;
}

bool MemoryManager::is_out_of_memory() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    if (!pImpl->initialized || pImpl->max_memory_bytes == 0) {
        return false;  // Not initialized yet, so not out of memory
    }
    
    return pImpl->allocated_bytes >= pImpl->max_memory_bytes * 0.95;  // 95% threshold
}

void MemoryManager::defragment() {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    std::cout << "Starting GPU memory defragmentation..." << std::endl;
    
    // Defragment each pool by compacting used blocks
    for (auto& pool : pImpl->pools) {
        if (pool.allocated_blocks == 0) continue;
        
        std::vector<bool> new_block_used(pool.num_blocks, false);
        size_t write_index = 0;
        
        // Compact used blocks to the beginning
        for (size_t read_index = 0; read_index < pool.num_blocks; ++read_index) {
            if (pool.block_used[read_index]) {
                if (read_index != write_index) {
                    // Move data from read_index to write_index
                    char* src = static_cast<char*>(pool.pool_start) + (read_index * pool.block_size);
                    char* dst = static_cast<char*>(pool.pool_start) + (write_index * pool.block_size);
                    
                    cudaError_t error = cudaMemcpy(dst, src, pool.block_size, cudaMemcpyDeviceToDevice);
                    if (error != cudaSuccess) {
                        std::cerr << "Defragmentation memcpy failed: " << cudaGetErrorString(error) << std::endl;
                        return;
                    }
                }
                new_block_used[write_index] = true;
                write_index++;
            }
        }
        
        pool.block_used = std::move(new_block_used);
    }
    
    std::cout << "GPU memory defragmentation complete" << std::endl;
}

bool MemoryManager::create_common_pools() {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    // Create standard pool sizes optimized for cache workloads
    struct PoolConfig {
        size_t block_size;
        size_t num_blocks;
        const char* description;
    };
    
    std::vector<PoolConfig> configs = {
        {64,   10000, "small values (64B)"},      // 640KB total
        {256,  5000,  "medium values (256B)"},    // 1.28MB total  
        {1024, 2000,  "large values (1KB)"},      // 2MB total
        {4096, 500,   "extra large values (4KB)"} // 2MB total
    };
    
    bool all_success = true;
    for (const auto& config : configs) {
        if (!create_pool(config.block_size, config.num_blocks)) {
            std::cerr << "Failed to create pool for " << config.description << std::endl;
            all_success = false;
        } else {
            std::cout << "Created pool for " << config.description << std::endl;
        }
    }
    
    return all_success;
}

size_t MemoryManager::get_pool_count() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    return pImpl->pools.size();
}

void MemoryManager::print_pool_stats() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    std::cout << "\n=== GPU Memory Pool Statistics ===" << std::endl;
    std::cout << "Active pools: " << pImpl->pools.size() << std::endl;
    
    for (size_t i = 0; i < pImpl->pools.size(); ++i) {
        const auto& pool = pImpl->pools[i];
        double utilization = static_cast<double>(pool.allocated_blocks) / pool.num_blocks * 100.0;
        
        std::cout << "Pool " << i << ": " 
                  << pool.block_size << "B blocks, "
                  << pool.allocated_blocks << "/" << pool.num_blocks << " used ("
                  << std::fixed << std::setprecision(1) << utilization << "%)"
                  << std::endl;
    }
    
    std::cout << "=================================" << std::endl;
}

bool MemoryManager::has_memory_leaks() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    // Check for unmatched allocations
    size_t unmatched_allocs = pImpl->total_allocations_ever - pImpl->total_deallocations_ever;
    return (unmatched_allocs != pImpl->allocation_count) || !pImpl->allocations.empty();
}

void MemoryManager::print_allocation_report() const {
    std::lock_guard<std::mutex> lock(pImpl->allocation_mutex);
    
    std::cout << "\n=== GPU Memory Allocation Report ===" << std::endl;
    std::cout << "Total allocations ever: " << pImpl->total_allocations_ever << std::endl;
    std::cout << "Total deallocations ever: " << pImpl->total_deallocations_ever << std::endl;
    std::cout << "Current active allocations: " << pImpl->allocation_count << std::endl;
    std::cout << "Peak memory usage: " << (pImpl->peak_allocated_bytes / 1024 / 1024) << " MB" << std::endl;
    
    if (has_memory_leaks()) {
        std::cout << "⚠️  MEMORY LEAKS DETECTED!" << std::endl;
        std::cout << "Active allocation map size: " << pImpl->allocations.size() << std::endl;
    } else {
        std::cout << "✅ No memory leaks detected" << std::endl;
    }
    
    // Size distribution
    if (!pImpl->size_histogram.empty()) {
        std::cout << "\nAllocation size distribution:" << std::endl;
        for (const auto& [size, count] : pImpl->size_histogram) {
            std::cout << "  " << size << "B: " << count << " allocations" << std::endl;
        }
    }
    
    std::cout << "====================================" << std::endl;
}

} // namespace core
} // namespace predis