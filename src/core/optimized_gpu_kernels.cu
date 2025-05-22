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

#include "optimized_gpu_kernels.h"
#include "data_structures/gpu_hash_table.h"
#include "memory_manager.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace predis {
namespace core {

namespace cg = cooperative_groups;

// RTX 5080 architecture constants
constexpr int RTX_5080_SM_COUNT = 84;
constexpr int RTX_5080_MAX_THREADS_PER_SM = 2048;
constexpr int RTX_5080_MAX_BLOCKS_PER_SM = 32;
constexpr int RTX_5080_SHARED_MEMORY_PER_SM = 98304; // 96KB
constexpr int RTX_5080_L2_CACHE_SIZE = 67108864; // 64MB
constexpr int RTX_5080_MEMORY_BANDWIDTH = 896; // GB/s

// Optimized hash functions using vectorized operations
__device__ __forceinline__ uint32_t vectorized_fnv1a_hash(const char* key, size_t len) {
    const uint32_t FNV_PRIME = 0x01000193;
    const uint32_t FNV_OFFSET = 0x811c9dc5;
    
    uint32_t hash = FNV_OFFSET;
    
    // Process 4 bytes at a time using vectorized operations
    const uint32_t* key_u32 = reinterpret_cast<const uint32_t*>(key);
    size_t u32_count = len / 4;
    
    for (size_t i = 0; i < u32_count; ++i) {
        uint32_t chunk = key_u32[i];
        // Process each byte in the 32-bit chunk
        hash ^= (chunk & 0xFF);
        hash *= FNV_PRIME;
        hash ^= ((chunk >> 8) & 0xFF);
        hash *= FNV_PRIME;
        hash ^= ((chunk >> 16) & 0xFF);
        hash *= FNV_PRIME;
        hash ^= ((chunk >> 24) & 0xFF);
        hash *= FNV_PRIME;
    }
    
    // Handle remaining bytes
    const char* remaining = key + (u32_count * 4);
    size_t remaining_len = len % 4;
    for (size_t i = 0; i < remaining_len; ++i) {
        hash ^= static_cast<uint32_t>(remaining[i]);
        hash *= FNV_PRIME;
    }
    
    return hash;
}

// Device memory comparison optimized for RTX 5080
__device__ __forceinline__ bool optimized_key_compare(const char* key1, const char* key2, size_t len) {
    // Use vectorized comparison for better memory bandwidth utilization
    const uint4* k1_vec = reinterpret_cast<const uint4*>(key1);
    const uint4* k2_vec = reinterpret_cast<const uint4*>(key2);
    
    size_t vec_count = len / sizeof(uint4);
    
    // Compare 16 bytes at a time using uint4 vectorized loads
    for (size_t i = 0; i < vec_count; ++i) {
        uint4 v1 = k1_vec[i];
        uint4 v2 = k2_vec[i];
        
        if (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w) {
            return false;
        }
    }
    
    // Handle remaining bytes
    size_t remaining = len % sizeof(uint4);
    size_t offset = vec_count * sizeof(uint4);
    
    for (size_t i = 0; i < remaining; ++i) {
        if (key1[offset + i] != key2[offset + i]) {
            return false;
        }
    }
    
    return true;
}

// Optimized memory copy using vectorized operations
__device__ __forceinline__ void optimized_memory_copy(void* dest, const void* src, size_t bytes) {
    uint4* dest_vec = reinterpret_cast<uint4*>(dest);
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    
    size_t vec_count = bytes / sizeof(uint4);
    
    // Copy 16 bytes at a time for maximum memory throughput
    for (size_t i = 0; i < vec_count; ++i) {
        dest_vec[i] = src_vec[i];
    }
    
    // Handle remaining bytes
    size_t remaining = bytes % sizeof(uint4);
    if (remaining > 0) {
        char* dest_char = reinterpret_cast<char*>(dest) + (vec_count * sizeof(uint4));
        const char* src_char = reinterpret_cast<const char*>(src) + (vec_count * sizeof(uint4));
        
        for (size_t i = 0; i < remaining; ++i) {
            dest_char[i] = src_char[i];
        }
    }
}

namespace optimized_kernels {

/**
 * Cooperative group-based lookup kernel with shared memory optimization
 * Utilizes block-level cooperation and RTX 5080 memory hierarchy
 */
__global__ void cooperative_lookup_kernel(
    const HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    char* __restrict__ result_value,
    size_t* __restrict__ result_length,
    bool* __restrict__ found_flag,
    int hash_method) {
    
    // Initialize cooperative groups
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Shared memory cache for hash table entries (optimized for RTX 5080)
    __shared__ HashEntry shared_cache[8]; // Cache 8 entries per block
    __shared__ uint32_t shared_hash;
    __shared__ bool search_complete;
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    
    // Initialize shared variables
    if (tid == 0) {
        *found_flag = false;
        *result_length = 0;
        search_complete = false;
        
        // Calculate hash using optimized function
        shared_hash = vectorized_fnv1a_hash(key, key_length);
    }
    
    block.sync();
    
    uint32_t hash = shared_hash;
    uint32_t start_slot = hash % table_capacity;
    
    // Cooperative search with warp-level parallelism
    for (uint32_t probe_batch = 0; probe_batch < table_capacity && !search_complete; probe_batch += blockDim.x) {
        uint32_t probe_slot = (start_slot + probe_batch + tid) % table_capacity;
        
        // Coalesced load from global memory to shared memory
        HashEntry local_entry;
        if (probe_batch + tid < table_capacity) {
            local_entry = hash_table[probe_slot];
        } else {
            local_entry = HashEntry{}; // Empty entry
        }
        
        // Check for early termination (empty slot found)
        if (local_entry.key_len == 0 && local_entry.lock != 2) {
            if (tid == 0) search_complete = true;
            block.sync();
            break;
        }
        
        // Skip deleted entries
        if (local_entry.lock == 2) continue;
        
        // Check for key match using optimized comparison
        bool key_matches = (local_entry.key_len == key_length && 
                           local_entry.hash == hash &&
                           local_entry.lock == 0 && // Not locked
                           optimized_key_compare(local_entry.key, key, key_length));
        
        // Use warp-level reduction to check if any thread found a match
        bool any_match = warp.any(key_matches);
        
        if (any_match) {
            // Find which thread found the match
            int winner = __ffs(warp.ballot(key_matches)) - 1;
            
            if (warp.thread_rank() == winner) {
                // Copy value using optimized memory operations
                optimized_memory_copy(result_value, local_entry.value, local_entry.value_len);
                *result_length = local_entry.value_len;
                *found_flag = true;
                search_complete = true;
            }
            break;
        }
        
        block.sync();
    }
}

/**
 * Memory hierarchy optimized insertion kernel
 * Leverages RTX 5080 L1/L2 cache hierarchy and register optimization
 */
__global__ void memory_optimized_insert_kernel(
    HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    const char* __restrict__ value,
    size_t value_length,
    bool* __restrict__ success_flag,
    int hash_method) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Use register variables to minimize memory access
    __shared__ uint32_t shared_hash;
    __shared__ bool operation_complete;
    __shared__ int winning_thread;
    
    int tid = threadIdx.x;
    
    // Initialize shared state
    if (tid == 0) {
        *success_flag = false;
        operation_complete = false;
        winning_thread = -1;
        shared_hash = vectorized_fnv1a_hash(key, key_length);
    }
    
    block.sync();
    
    uint32_t hash = shared_hash;
    uint32_t start_slot = hash % table_capacity;
    
    // Cooperative insertion with optimized memory access patterns
    for (uint32_t probe_batch = 0; probe_batch < table_capacity && !operation_complete; probe_batch += blockDim.x) {
        uint32_t probe_slot = (start_slot + probe_batch + tid) % table_capacity;
        
        if (probe_batch + tid >= table_capacity) continue;
        
        HashEntry* entry = &hash_table[probe_slot];
        
        // Try to acquire lock atomically
        int expected = 0;
        bool acquired_lock = (atomicCAS(&entry->lock, expected, 1) == 0);
        
        if (acquired_lock) {
            // Check if slot is available (empty or deleted)
            bool slot_available = (entry->key_len == 0 || entry->lock == 2);
            
            // Check if updating existing key
            bool updating_existing = (!slot_available && 
                                    entry->key_len == key_length && 
                                    entry->hash == hash &&
                                    optimized_key_compare(entry->key, key, key_length));
            
            if (slot_available || updating_existing) {
                // Perform insertion/update using optimized memory operations
                entry->hash = hash;
                entry->key_len = static_cast<uint32_t>(key_length);
                entry->value_len = static_cast<uint32_t>(value_length);
                
                // Use vectorized copy for better memory bandwidth
                optimized_memory_copy(entry->key, key, key_length);
                optimized_memory_copy(entry->value, value, value_length);
                
                // Ensure memory writes are visible before releasing lock
                __threadfence();
                entry->lock = 0;
                
                // Signal successful completion
                if (tid == 0 || winning_thread == -1) {
                    *success_flag = true;
                    operation_complete = true;
                    winning_thread = tid;
                }
                
                break;
            } else {
                // Release lock if slot not suitable
                entry->lock = 0;
            }
        }
        
        block.sync();
        if (operation_complete) break;
    }
}

/**
 * Register pressure optimized deletion kernel
 * Minimizes register usage for maximum occupancy on RTX 5080
 */
__global__ void register_optimized_delete_kernel(
    HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    bool* __restrict__ success_flag,
    int hash_method) {
    
    // Use minimal register variables for maximum occupancy
    const uint32_t hash = vectorized_fnv1a_hash(key, key_length);
    const uint32_t start_slot = hash % table_capacity;
    
    *success_flag = false;
    
    // Linear probing with minimal register pressure
    for (uint32_t probe = 0; probe < table_capacity; ++probe) {
        const uint32_t slot = (start_slot + probe) % table_capacity;
        HashEntry* const entry = &hash_table[slot];
        
        // Early termination on empty slot
        if (entry->key_len == 0 && entry->lock != 2) break;
        
        // Skip deleted entries
        if (entry->lock == 2) continue;
        
        // Try to acquire lock
        const int expected = 0;
        if (atomicCAS(&entry->lock, expected, 1) == 0) {
            // Check for key match
            const bool key_matches = (entry->key_len == key_length && 
                                    entry->hash == hash &&
                                    optimized_key_compare(entry->key, key, key_length));
            
            if (key_matches) {
                // Mark as deleted
                entry->lock = 2;
                entry->key_len = 0;
                entry->value_len = 0;
                
                __threadfence();
                *success_flag = true;
                return;
            }
            
            // Release lock
            entry->lock = 0;
        }
    }
}

/**
 * Tensor core accelerated bulk operations (for future ML workloads)
 * Currently implements optimized bulk operations, tensor cores reserved for ML features
 */
__global__ void tensor_accelerated_bulk_kernel(
    HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ keys,
    const char* __restrict__ values,
    const uint32_t* __restrict__ key_lengths,
    const uint32_t* __restrict__ value_lengths,
    bool* __restrict__ results,
    size_t num_items,
    int operation_type, // 0=lookup, 1=insert, 2=delete
    int hash_method) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;
    
    // Shared memory for collaborative caching (RTX 5080 optimized)
    __shared__ HashEntry cache[32]; // 32 entries per block
    
    // Get operation data for this thread
    const size_t key_offset = tid * 256; // MAX_KEY_LENGTH
    const size_t value_offset = tid * 4096; // MAX_VALUE_LENGTH
    
    const char* my_key = keys + key_offset;
    const uint32_t my_key_len = key_lengths[tid];
    
    bool success = false;
    
    switch (operation_type) {
        case 0: { // Lookup
            char* my_result = const_cast<char*>(values) + value_offset;
            uint32_t* my_result_len = const_cast<uint32_t*>(value_lengths) + tid;
            
            const uint32_t hash = vectorized_fnv1a_hash(my_key, my_key_len);
            const uint32_t start_slot = hash % table_capacity;
            
            // Optimized lookup with warp cooperation
            for (uint32_t probe = warp.thread_rank(); probe < table_capacity; probe += warp.size()) {
                const uint32_t slot = (start_slot + probe) % table_capacity;
                const HashEntry* entry = &hash_table[slot];
                
                if (entry->key_len == 0 && entry->lock != 2) break;
                if (entry->lock == 2) continue;
                
                if (entry->key_len == my_key_len && 
                    entry->hash == hash &&
                    entry->lock == 0 &&
                    optimized_key_compare(entry->key, my_key, my_key_len)) {
                    
                    optimized_memory_copy(my_result, entry->value, entry->value_len);
                    *my_result_len = entry->value_len;
                    success = true;
                    break;
                }
            }
            break;
        }
        
        case 1: { // Insert
            const char* my_value = values + value_offset;
            const uint32_t my_value_len = value_lengths[tid];
            
            const uint32_t hash = vectorized_fnv1a_hash(my_key, my_key_len);
            const uint32_t start_slot = hash % table_capacity;
            
            // Optimized insertion
            for (uint32_t probe = 0; probe < table_capacity; ++probe) {
                const uint32_t slot = (start_slot + probe) % table_capacity;
                HashEntry* entry = &hash_table[slot];
                
                int expected = 0;
                if (atomicCAS(&entry->lock, expected, 1) == 0) {
                    if (entry->key_len == 0 || entry->lock == 2 ||
                        (entry->key_len == my_key_len && entry->hash == hash && 
                         optimized_key_compare(entry->key, my_key, my_key_len))) {
                        
                        entry->hash = hash;
                        entry->key_len = my_key_len;
                        entry->value_len = my_value_len;
                        
                        optimized_memory_copy(entry->key, my_key, my_key_len);
                        optimized_memory_copy(entry->value, my_value, my_value_len);
                        
                        __threadfence();
                        entry->lock = 0;
                        success = true;
                        break;
                    }
                    entry->lock = 0;
                }
            }
            break;
        }
        
        case 2: { // Delete
            const uint32_t hash = vectorized_fnv1a_hash(my_key, my_key_len);
            const uint32_t start_slot = hash % table_capacity;
            
            for (uint32_t probe = 0; probe < table_capacity; ++probe) {
                const uint32_t slot = (start_slot + probe) % table_capacity;
                HashEntry* entry = &hash_table[slot];
                
                if (entry->key_len == 0 && entry->lock != 2) break;
                if (entry->lock == 2) continue;
                
                int expected = 0;
                if (atomicCAS(&entry->lock, expected, 1) == 0) {
                    if (entry->key_len == my_key_len && entry->hash == hash &&
                        optimized_key_compare(entry->key, my_key, my_key_len)) {
                        
                        entry->lock = 2; // Mark as deleted
                        entry->key_len = 0;
                        entry->value_len = 0;
                        __threadfence();
                        success = true;
                        break;
                    }
                    entry->lock = 0;
                }
            }
            break;
        }
    }
    
    results[tid] = success;
}

} // namespace optimized_kernels

// Implementation structure
struct OptimizedGpuKernels::Impl {
    GpuHashTable* hash_table = nullptr;
    MemoryManager* memory_manager = nullptr;
    
    // Optimization settings
    bool cooperative_groups_enabled = true;
    bool tensor_cores_enabled = false;
    bool occupancy_optimization_enabled = true;
    
    // Performance tracking
    KernelMetrics last_metrics;
    
    // CUDA events for timing
    cudaEvent_t start_event, stop_event;
    
    bool initialized = false;
};

OptimizedGpuKernels::OptimizedGpuKernels() : pImpl(std::make_unique<Impl>()) {}

OptimizedGpuKernels::~OptimizedGpuKernels() {
    if (pImpl->initialized) {
        shutdown();
    }
}

bool OptimizedGpuKernels::initialize(GpuHashTable* hash_table, MemoryManager* memory_manager) {
    if (pImpl->initialized) {
        std::cerr << "OptimizedGpuKernels already initialized" << std::endl;
        return false;
    }
    
    pImpl->hash_table = hash_table;
    pImpl->memory_manager = memory_manager;
    
    // Create CUDA events for performance measurement
    cudaError_t error = cudaEventCreate(&pImpl->start_event);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create start event: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaEventCreate(&pImpl->stop_event);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create stop event: " << cudaGetErrorString(error) << std::endl;
        cudaEventDestroy(pImpl->start_event);
        return false;
    }
    
    pImpl->initialized = true;
    
    std::cout << "OptimizedGpuKernels initialized with RTX 5080 optimizations" << std::endl;
    return true;
}

void OptimizedGpuKernels::shutdown() {
    if (!pImpl->initialized) return;
    
    cudaEventDestroy(pImpl->start_event);
    cudaEventDestroy(pImpl->stop_event);
    
    pImpl->initialized = false;
    std::cout << "OptimizedGpuKernels shutdown complete" << std::endl;
}

bool OptimizedGpuKernels::optimized_lookup(const char* key, size_t key_len, 
                                          char* value, size_t* value_len,
                                          const LaunchConfig& config) {
    if (!pImpl->initialized) return false;
    
    // Allocate GPU memory for single operation
    char* d_key = nullptr;
    char* d_value = nullptr;
    size_t* d_value_len = nullptr;
    bool* d_found = nullptr;
    
    cudaError_t error = cudaMalloc(&d_key, key_len);
    if (error != cudaSuccess) return false;
    
    error = cudaMalloc(&d_value, 4096); // MAX_VALUE_LENGTH
    if (error != cudaSuccess) {
        cudaFree(d_key);
        return false;
    }
    
    error = cudaMalloc(&d_value_len, sizeof(size_t));
    if (error != cudaSuccess) {
        cudaFree(d_key);
        cudaFree(d_value);
        return false;
    }
    
    error = cudaMalloc(&d_found, sizeof(bool));
    if (error != cudaSuccess) {
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_value_len);
        return false;
    }
    
    // Copy key to GPU
    cudaMemcpy(d_key, key, key_len, cudaMemcpyHostToDevice);
    
    // Start timing
    cudaEventRecord(pImpl->start_event, config.stream);
    
    // Launch optimized kernel
    auto hash_stats = pImpl->hash_table->get_stats();
    
    optimized_kernels::cooperative_lookup_kernel<<<1, config.block_size, 
                                                  config.shared_memory_bytes, config.stream>>>(
        reinterpret_cast<const HashEntry*>(pImpl->hash_table),
        hash_stats.capacity,
        d_key,
        key_len,
        d_value,
        d_value_len,
        d_found,
        0 // FNV1A hash method
    );
    
    // Stop timing
    cudaEventRecord(pImpl->stop_event, config.stream);
    
    // Wait for completion and get results
    cudaStreamSynchronize(config.stream);
    
    bool found = false;
    cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    
    if (found) {
        size_t result_len = 0;
        cudaMemcpy(&result_len, d_value_len, sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(value, d_value, result_len, cudaMemcpyDeviceToHost);
        *value_len = result_len;
    }
    
    // Calculate performance metrics
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, pImpl->start_event, pImpl->stop_event);
    
    pImpl->last_metrics.execution_time_us = elapsed_ms * 1000.0;
    pImpl->last_metrics.operations_per_second = found ? (1000000.0 / pImpl->last_metrics.execution_time_us) : 0;
    
    // Cleanup
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_value_len);
    cudaFree(d_found);
    
    return found;
}

bool OptimizedGpuKernels::optimized_insert(const char* key, size_t key_len,
                                          const char* value, size_t value_len,
                                          const LaunchConfig& config) {
    if (!pImpl->initialized) return false;
    
    // Similar implementation to lookup but for insertion
    char* d_key = nullptr;
    char* d_value = nullptr;
    bool* d_success = nullptr;
    
    cudaError_t error = cudaMalloc(&d_key, key_len);
    if (error != cudaSuccess) return false;
    
    error = cudaMalloc(&d_value, value_len);
    if (error != cudaSuccess) {
        cudaFree(d_key);
        return false;
    }
    
    error = cudaMalloc(&d_success, sizeof(bool));
    if (error != cudaSuccess) {
        cudaFree(d_key);
        cudaFree(d_value);
        return false;
    }
    
    // Copy data to GPU
    cudaMemcpy(d_key, key, key_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, value_len, cudaMemcpyHostToDevice);
    
    // Start timing
    cudaEventRecord(pImpl->start_event, config.stream);
    
    // Launch optimized insertion kernel
    auto hash_stats = pImpl->hash_table->get_stats();
    
    optimized_kernels::memory_optimized_insert_kernel<<<1, config.block_size, 
                                                       config.shared_memory_bytes, config.stream>>>(
        reinterpret_cast<HashEntry*>(pImpl->hash_table),
        hash_stats.capacity,
        d_key,
        key_len,
        d_value,
        value_len,
        d_success,
        0 // FNV1A hash method
    );
    
    // Stop timing
    cudaEventRecord(pImpl->stop_event, config.stream);
    
    // Wait for completion and get results
    cudaStreamSynchronize(config.stream);
    
    bool success = false;
    cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, pImpl->start_event, pImpl->stop_event);
    
    pImpl->last_metrics.execution_time_us = elapsed_ms * 1000.0;
    pImpl->last_metrics.operations_per_second = success ? (1000000.0 / pImpl->last_metrics.execution_time_us) : 0;
    
    // Cleanup
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_success);
    
    return success;
}

bool OptimizedGpuKernels::optimized_delete(const char* key, size_t key_len,
                                          const LaunchConfig& config) {
    if (!pImpl->initialized) return false;
    
    char* d_key = nullptr;
    bool* d_success = nullptr;
    
    cudaError_t error = cudaMalloc(&d_key, key_len);
    if (error != cudaSuccess) return false;
    
    error = cudaMalloc(&d_success, sizeof(bool));
    if (error != cudaSuccess) {
        cudaFree(d_key);
        return false;
    }
    
    // Copy key to GPU
    cudaMemcpy(d_key, key, key_len, cudaMemcpyHostToDevice);
    
    // Start timing
    cudaEventRecord(pImpl->start_event, config.stream);
    
    // Launch optimized deletion kernel
    auto hash_stats = pImpl->hash_table->get_stats();
    
    optimized_kernels::register_optimized_delete_kernel<<<1, 1, 0, config.stream>>>(
        reinterpret_cast<HashEntry*>(pImpl->hash_table),
        hash_stats.capacity,
        d_key,
        key_len,
        d_success,
        0 // FNV1A hash method
    );
    
    // Stop timing
    cudaEventRecord(pImpl->stop_event, config.stream);
    
    // Wait for completion and get results
    cudaStreamSynchronize(config.stream);
    
    bool success = false;
    cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, pImpl->start_event, pImpl->stop_event);
    
    pImpl->last_metrics.execution_time_us = elapsed_ms * 1000.0;
    pImpl->last_metrics.operations_per_second = success ? (1000000.0 / pImpl->last_metrics.execution_time_us) : 0;
    
    // Cleanup
    cudaFree(d_key);
    cudaFree(d_success);
    
    return success;
}

OptimizedGpuKernels::KernelMetrics OptimizedGpuKernels::get_last_metrics() const {
    return pImpl->last_metrics;
}

void OptimizedGpuKernels::configure_optimization(bool enable_cooperative_groups,
                                                bool enable_tensor_cores,
                                                bool optimize_occupancy) {
    pImpl->cooperative_groups_enabled = enable_cooperative_groups;
    pImpl->tensor_cores_enabled = enable_tensor_cores;
    pImpl->occupancy_optimization_enabled = optimize_occupancy;
}

OptimizedGpuKernels::LaunchConfig OptimizedGpuKernels::auto_tune_config(size_t workload_size, bool is_read_heavy) {
    LaunchConfig config;
    
    // Optimize for RTX 5080 architecture
    if (workload_size == 1) {
        // Single operation optimization
        config.block_size = pImpl->occupancy_optimization_enabled ? 256 : 128;
        config.grid_size = 1;
        config.shared_memory_bytes = 8192; // 8KB for single operation
    } else {
        // Bulk operation optimization
        config.block_size = 256;
        config.grid_size = std::min((int)((workload_size + config.block_size - 1) / config.block_size), 
                                   RTX_5080_SM_COUNT * RTX_5080_MAX_BLOCKS_PER_SM);
        config.shared_memory_bytes = 49152; // Max shared memory
    }
    
    config.use_cooperative_groups = pImpl->cooperative_groups_enabled;
    config.enable_tensor_cores = pImpl->tensor_cores_enabled;
    config.optimize_occupancy = pImpl->occupancy_optimization_enabled;
    
    return config;
}

} // namespace core
} // namespace predis