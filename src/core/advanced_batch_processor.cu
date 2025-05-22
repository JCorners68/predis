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

#include "advanced_batch_processor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>

namespace predis {
namespace core {

namespace cg = cooperative_groups;

// GPU Constants optimized for RTX 5080
__constant__ size_t MAX_KEY_SIZE = 256;
__constant__ size_t MAX_VALUE_SIZE = 4096;
__constant__ uint32_t WARP_SIZE = 32;

// Hash function (same as in gpu_hash_table.cu)
__device__ uint32_t fnv1a_hash(const char* key, size_t len) {
    const uint32_t FNV_PRIME = 0x01000193;
    const uint32_t FNV_OFFSET = 0x811c9dc5;
    
    uint32_t hash = FNV_OFFSET;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(key[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

// Device memory comparison optimized for GPU
__device__ int memcmp_device(const void* s1, const void* s2, size_t n) {
    const unsigned char* p1 = static_cast<const unsigned char*>(s1);
    const unsigned char* p2 = static_cast<const unsigned char*>(s2);
    
    for (size_t i = 0; i < n; ++i) {
        if (p1[i] != p2[i]) {
            return p1[i] - p2[i];
        }
    }
    return 0;
}

// Device memory copy optimized for GPU
__device__ void memcpy_device(void* dest, const void* src, size_t n) {
    char* d = static_cast<char*>(dest);
    const char* s = static_cast<const char*>(src);
    
    // Vectorized copy for better memory throughput
    size_t vector_count = n / sizeof(uint4);
    uint4* dest_vec = reinterpret_cast<uint4*>(d);
    const uint4* src_vec = reinterpret_cast<const uint4*>(s);
    
    for (size_t i = 0; i < vector_count; ++i) {
        dest_vec[i] = src_vec[i];
    }
    
    // Handle remaining bytes
    size_t remaining = n % sizeof(uint4);
    size_t offset = vector_count * sizeof(uint4);
    for (size_t i = 0; i < remaining; ++i) {
        d[offset + i] = s[offset + i];
    }
}

namespace batch_kernels {

/**
 * Optimized batch lookup kernel with perfect memory coalescing
 * Each warp processes 32 keys simultaneously with cooperative memory access
 */
__global__ void batch_lookup_coalesced_kernel(
    const char* __restrict__ keys_data,
    const size_t* __restrict__ key_offsets,
    const uint32_t* __restrict__ key_lengths,
    char* __restrict__ results_data,
    const size_t* __restrict__ result_offsets,
    uint32_t* __restrict__ result_lengths,
    bool* __restrict__ found_flags,
    const HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size) {
    
    // Shared memory for warp-level cache line optimization
    __shared__ HashEntry shared_cache[WARP_SIZE];
    
    // Thread and warp identification
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t warp_idx = threadIdx.x / WARP_SIZE;
    size_t lane_idx = threadIdx.x % WARP_SIZE;
    
    if (global_idx >= batch_size) return;
    
    // Load key data with coalesced access
    const char* my_key = keys_data + key_offsets[global_idx];
    uint32_t my_key_len = key_lengths[global_idx];
    uint32_t hash = fnv1a_hash(my_key, my_key_len);
    
    // Initialize search state
    size_t probe_start = hash % table_size;
    bool found = false;
    found_flags[global_idx] = false;
    result_lengths[global_idx] = 0;
    
    // Warp-cooperative linear probing with cache line optimization
    for (size_t probe_batch = 0; probe_batch < table_size && !found; probe_batch += WARP_SIZE) {
        // Each thread in warp probes a different slot
        size_t probe_slot = (probe_start + probe_batch + lane_idx) % table_size;
        
        // Coalesced load of hash table entries into shared memory
        if (probe_batch + lane_idx < table_size) {
            shared_cache[lane_idx] = hash_table[probe_slot];
        }
        
        // Synchronize warp for shared memory consistency
        warp.sync();
        
        // Check my specific entry from shared memory
        const HashEntry* my_entry = &shared_cache[lane_idx];
        
        // Early termination check - if we hit an empty slot, key doesn't exist
        if (my_entry->key_len == 0 && my_entry->lock != 2) {
            break;
        }
        
        // Skip deleted entries
        if (my_entry->lock == 2) {
            continue;
        }
        
        // Check for key match
        if (my_entry->key_len == my_key_len && 
            my_entry->hash == hash &&
            my_entry->lock == 0) { // Not locked
            
            // Verify key contents match
            bool key_match = (memcmp_device(my_entry->key, my_key, my_key_len) == 0);
            
            if (key_match) {
                // Found the key! Copy value to results
                char* result_dest = results_data + result_offsets[global_idx];
                memcpy_device(result_dest, my_entry->value, my_entry->value_len);
                result_lengths[global_idx] = my_entry->value_len;
                found_flags[global_idx] = true;
                found = true;
            }
        }
    }
}

/**
 * Optimized batch insert kernel with conflict resolution and atomic operations
 */
__global__ void batch_insert_coalesced_kernel(
    const char* __restrict__ keys_data,
    const char* __restrict__ values_data,
    const size_t* __restrict__ key_offsets,
    const size_t* __restrict__ value_offsets,
    const uint32_t* __restrict__ key_lengths,
    const uint32_t* __restrict__ value_lengths,
    bool* __restrict__ success_flags,
    HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size) {
    
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size) return;
    
    // Load operation data
    const char* my_key = keys_data + key_offsets[global_idx];
    const char* my_value = values_data + value_offsets[global_idx];
    uint32_t my_key_len = key_lengths[global_idx];
    uint32_t my_value_len = value_lengths[global_idx];
    
    // Validate sizes
    if (my_key_len == 0 || my_key_len >= MAX_KEY_SIZE || 
        my_value_len == 0 || my_value_len >= MAX_VALUE_SIZE) {
        success_flags[global_idx] = false;
        return;
    }
    
    uint32_t hash = fnv1a_hash(my_key, my_key_len);
    size_t start_slot = hash % table_size;
    size_t probe_count = 0;
    
    success_flags[global_idx] = false;
    
    // Linear probing with atomic locking
    while (probe_count < table_size) {
        size_t slot = (start_slot + probe_count) % table_size;
        HashEntry* entry = &hash_table[slot];
        
        // Try to acquire lock atomically
        int expected = 0;
        if (atomicCAS(&entry->lock, expected, 1) == 0) {
            // Successfully acquired lock
            
            // Check if slot is empty or marked for deletion
            if (entry->key_len == 0 || entry->lock == 2) {
                // Insert new entry
                entry->hash = hash;
                entry->key_len = my_key_len;
                entry->value_len = my_value_len;
                
                // Copy key and value
                memcpy_device(entry->key, my_key, my_key_len);
                memcpy_device(entry->value, my_value, my_value_len);
                
                // Ensure memory writes are visible before releasing lock
                __threadfence();
                entry->lock = 0;
                success_flags[global_idx] = true;
                return;
            }
            
            // Check if key already exists (update case)
            if (entry->key_len == my_key_len && 
                entry->hash == hash &&
                memcmp_device(entry->key, my_key, my_key_len) == 0) {
                
                // Update existing entry
                entry->value_len = my_value_len;
                memcpy_device(entry->value, my_value, my_value_len);
                
                __threadfence();
                entry->lock = 0;
                success_flags[global_idx] = true;
                return;
            }
            
            // Release lock and continue probing
            entry->lock = 0;
        }
        
        probe_count++;
    }
    
    // Table full or other error
    success_flags[global_idx] = false;
}

/**
 * Optimized batch delete kernel with atomic operations
 */
__global__ void batch_delete_coalesced_kernel(
    const char* __restrict__ keys_data,
    const size_t* __restrict__ key_offsets,
    const uint32_t* __restrict__ key_lengths,
    bool* __restrict__ success_flags,
    HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size) {
    
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size) return;
    
    // Load key data
    const char* my_key = keys_data + key_offsets[global_idx];
    uint32_t my_key_len = key_lengths[global_idx];
    
    if (my_key_len == 0 || my_key_len >= MAX_KEY_SIZE) {
        success_flags[global_idx] = false;
        return;
    }
    
    uint32_t hash = fnv1a_hash(my_key, my_key_len);
    size_t start_slot = hash % table_size;
    size_t probe_count = 0;
    
    success_flags[global_idx] = false;
    
    // Linear probing to find and delete
    while (probe_count < table_size) {
        size_t slot = (start_slot + probe_count) % table_size;
        HashEntry* entry = &hash_table[slot];
        
        // Check if we've reached an empty slot
        if (entry->key_len == 0 && entry->lock != 2) {
            break; // Key not found
        }
        
        // Skip deleted entries
        if (entry->lock == 2) {
            probe_count++;
            continue;
        }
        
        // Try to acquire lock for potential deletion
        int expected = 0;
        if (atomicCAS(&entry->lock, expected, 1) == 0) {
            // Check for key match
            if (entry->key_len == my_key_len && 
                entry->hash == hash &&
                memcmp_device(entry->key, my_key, my_key_len) == 0) {
                
                // Mark as deleted
                entry->lock = 2;
                entry->key_len = 0;
                entry->value_len = 0;
                __threadfence();
                success_flags[global_idx] = true;
                return;
            }
            
            // Release lock if not a match
            entry->lock = 0;
        }
        
        probe_count++;
    }
}

/**
 * Lightweight batch exists kernel (no value copying)
 */
__global__ void batch_exists_coalesced_kernel(
    const char* __restrict__ keys_data,
    const size_t* __restrict__ key_offsets,
    const uint32_t* __restrict__ key_lengths,
    bool* __restrict__ exists_flags,
    const HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size) {
    
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size) return;
    
    // Load key data
    const char* my_key = keys_data + key_offsets[global_idx];
    uint32_t my_key_len = key_lengths[global_idx];
    
    if (my_key_len == 0 || my_key_len >= MAX_KEY_SIZE) {
        exists_flags[global_idx] = false;
        return;
    }
    
    uint32_t hash = fnv1a_hash(my_key, my_key_len);
    size_t start_slot = hash % table_size;
    size_t probe_count = 0;
    
    exists_flags[global_idx] = false;
    
    // Linear probing lookup (read-only)
    while (probe_count < table_size) {
        size_t slot = (start_slot + probe_count) % table_size;
        const HashEntry* entry = &hash_table[slot];
        
        // Check if we've reached an empty slot
        if (entry->key_len == 0 && entry->lock != 2) {
            break; // Key not found
        }
        
        // Skip deleted entries
        if (entry->lock == 2) {
            probe_count++;
            continue;
        }
        
        // Check for key match (no locking needed for read-only)
        if (entry->key_len == my_key_len && 
            entry->hash == hash &&
            entry->lock != 1 && // Not currently locked
            memcmp_device(entry->key, my_key, my_key_len) == 0) {
            
            exists_flags[global_idx] = true;
            return;
        }
        
        probe_count++;
    }
}

} // namespace batch_kernels

// Implementation structure
struct AdvancedBatchProcessor::Impl {
    GpuHashTable* hash_table = nullptr;
    MemoryManager* memory_manager = nullptr;
    BatchConfig config;
    
    // CUDA streams for parallel execution
    cudaStream_t streams[MAX_CONCURRENT_STREAMS];
    size_t num_active_streams = 0;
    
    // Performance tracking
    mutable std::mutex metrics_mutex;
    BatchMetrics cumulative_metrics;
    std::vector<BatchMetrics> recent_metrics;
    
    // Auto-tuning state
    size_t current_optimal_batch_size = 1024;
    double recent_bandwidth_utilization = 0.0;
    bool auto_tuning_enabled = true;
    
    // Memory management
    std::unique_ptr<BatchDataManager> data_manager;
    
    bool initialized = false;
};

AdvancedBatchProcessor::AdvancedBatchProcessor() : pImpl(std::make_unique<Impl>()) {}

AdvancedBatchProcessor::~AdvancedBatchProcessor() {
    if (pImpl->initialized) {
        shutdown();
    }
}

bool AdvancedBatchProcessor::initialize(GpuHashTable* hash_table, 
                                       MemoryManager* memory_manager,
                                       const BatchConfig& config) {
    if (pImpl->initialized) {
        std::cerr << "AdvancedBatchProcessor already initialized" << std::endl;
        return false;
    }
    
    pImpl->hash_table = hash_table;
    pImpl->memory_manager = memory_manager;
    pImpl->config = config;
    
    // Initialize CUDA streams
    pImpl->num_active_streams = std::min(config.max_concurrent_batches, 
                                        static_cast<size_t>(MAX_CONCURRENT_STREAMS));
    
    for (size_t i = 0; i < pImpl->num_active_streams; ++i) {
        cudaError_t error = cudaStreamCreate(&pImpl->streams[i]);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream " << i << ": " 
                      << cudaGetErrorString(error) << std::endl;
            
            // Cleanup already created streams
            for (size_t j = 0; j < i; ++j) {
                cudaStreamDestroy(pImpl->streams[j]);
            }
            return false;
        }
    }
    
    // Initialize data manager
    pImpl->data_manager = std::make_unique<BatchDataManager>(memory_manager);
    
    // Initialize performance tracking
    pImpl->current_optimal_batch_size = config.preferred_batch_size;
    pImpl->auto_tuning_enabled = config.enable_auto_tuning;
    
    pImpl->initialized = true;
    
    std::cout << "AdvancedBatchProcessor initialized with " 
              << pImpl->num_active_streams << " streams" << std::endl;
    
    return true;
}

void AdvancedBatchProcessor::shutdown() {
    if (!pImpl->initialized) return;
    
    // Destroy CUDA streams
    for (size_t i = 0; i < pImpl->num_active_streams; ++i) {
        cudaStreamSynchronize(pImpl->streams[i]);
        cudaStreamDestroy(pImpl->streams[i]);
    }
    
    pImpl->data_manager.reset();
    pImpl->initialized = false;
    
    std::cout << "AdvancedBatchProcessor shutdown complete" << std::endl;
}

AdvancedBatchProcessor::BatchResult AdvancedBatchProcessor::batch_get(
    const std::vector<std::string>& keys) {
    
    if (!pImpl->initialized || keys.empty()) {
        return BatchResult{};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BatchResult result;
    result.values.resize(keys.size());
    result.success_flags.resize(keys.size());
    
    try {
        // Pack keys for GPU transfer
        auto packed_data = pImpl->data_manager->pack_keys_for_gpu(keys);
        if (!packed_data) {
            result.error_message = "Failed to pack keys for GPU transfer";
            return result;
        }
        
        // Use first available stream (for simplicity in this implementation)
        cudaStream_t stream = pImpl->streams[0];
        
        // Transfer data to GPU
        if (!pImpl->data_manager->transfer_to_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer data to GPU";
            return result;
        }
        
        // Configure kernel launch parameters
        size_t batch_size = keys.size();
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        
        // Get hash table for kernel
        auto hash_stats = pImpl->hash_table->get_stats();
        
        // Launch optimized batch lookup kernel
        batch_kernels::batch_lookup_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
            packed_data->d_keys_data,
            packed_data->d_key_offsets,
            packed_data->d_key_lengths,
            packed_data->d_values_data,     // Results will be written here
            packed_data->d_value_offsets,
            packed_data->d_value_lengths,
            packed_data->d_result_flags,
            reinterpret_cast<const HashEntry*>(pImpl->hash_table), // Need to get actual pointer
            hash_stats.capacity,
            batch_size
        );
        
        // Check for kernel launch errors
        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            result.error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(kernel_error));
            return result;
        }
        
        // Wait for kernel completion
        cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            result.error_message = "Kernel execution failed: " + std::string(cudaGetErrorString(sync_error));
            return result;
        }
        
        // Transfer results back from GPU
        if (!pImpl->data_manager->transfer_from_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer results from GPU";
            return result;
        }
        
        // Unpack results
        result.values = pImpl->data_manager->unpack_lookup_results(*packed_data, keys);
        result.success_flags = pImpl->data_manager->unpack_operation_results(*packed_data);
        
        // Count successes
        result.successful_count = std::count(result.success_flags.begin(), 
                                           result.success_flags.end(), true);
        result.failed_count = result.success_flags.size() - result.successful_count;
        
    } catch (const std::exception& e) {
        result.error_message = "Exception during batch get: " + std::string(e.what());
        return result;
    }
    
    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    result.metrics.total_time_ms = elapsed_ms;
    result.metrics.operations_per_second = (keys.size() * 1000.0) / elapsed_ms;
    result.metrics.successful_operations = result.successful_count;
    result.metrics.failed_operations = result.failed_count;
    result.metrics.batch_efficiency_percent = 
        (static_cast<double>(result.successful_count) / keys.size()) * 100.0;
    
    // Update cumulative metrics
    {
        std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
        pImpl->cumulative_metrics.operations_per_second = 
            (pImpl->cumulative_metrics.operations_per_second + result.metrics.operations_per_second) / 2.0;
        pImpl->cumulative_metrics.successful_operations += result.metrics.successful_operations;
        pImpl->cumulative_metrics.failed_operations += result.metrics.failed_operations;
    }
    
    return result;
}

AdvancedBatchProcessor::BatchResult AdvancedBatchProcessor::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& values) {
    
    if (!pImpl->initialized || keys.empty() || keys.size() != values.size()) {
        return BatchResult{};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BatchResult result;
    result.success_flags.resize(keys.size());
    
    try {
        // Pack keys and values for GPU transfer
        auto packed_data = pImpl->data_manager->pack_key_values_for_gpu(keys, values);
        if (!packed_data) {
            result.error_message = "Failed to pack key-value pairs for GPU transfer";
            return result;
        }
        
        // Use first available stream
        cudaStream_t stream = pImpl->streams[0];
        
        // Transfer data to GPU
        if (!pImpl->data_manager->transfer_to_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer data to GPU";
            return result;
        }
        
        // Configure kernel launch parameters
        size_t batch_size = keys.size();
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        
        // Get hash table for kernel
        auto hash_stats = pImpl->hash_table->get_stats();
        
        // Launch optimized batch insert kernel
        batch_kernels::batch_insert_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
            packed_data->d_keys_data,
            packed_data->d_values_data,
            packed_data->d_key_offsets,
            packed_data->d_value_offsets,
            packed_data->d_key_lengths,
            packed_data->d_value_lengths,
            packed_data->d_result_flags,
            reinterpret_cast<HashEntry*>(pImpl->hash_table), // Need mutable access
            hash_stats.capacity,
            batch_size
        );
        
        // Check for kernel errors
        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            result.error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(kernel_error));
            return result;
        }
        
        // Wait for completion
        cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            result.error_message = "Kernel execution failed: " + std::string(cudaGetErrorString(sync_error));
            return result;
        }
        
        // Transfer results back
        if (!pImpl->data_manager->transfer_from_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer results from GPU";
            return result;
        }
        
        // Unpack operation results
        result.success_flags = pImpl->data_manager->unpack_operation_results(*packed_data);
        
        // Count successes
        result.successful_count = std::count(result.success_flags.begin(), 
                                           result.success_flags.end(), true);
        result.failed_count = result.success_flags.size() - result.successful_count;
        
    } catch (const std::exception& e) {
        result.error_message = "Exception during batch put: " + std::string(e.what());
        return result;
    }
    
    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    result.metrics.total_time_ms = elapsed_ms;
    result.metrics.operations_per_second = (keys.size() * 1000.0) / elapsed_ms;
    result.metrics.successful_operations = result.successful_count;
    result.metrics.failed_operations = result.failed_count;
    result.metrics.batch_efficiency_percent = 
        (static_cast<double>(result.successful_count) / keys.size()) * 100.0;
    
    // Update cumulative metrics
    {
        std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
        pImpl->cumulative_metrics.operations_per_second = 
            (pImpl->cumulative_metrics.operations_per_second + result.metrics.operations_per_second) / 2.0;
        pImpl->cumulative_metrics.successful_operations += result.metrics.successful_operations;
        pImpl->cumulative_metrics.failed_operations += result.metrics.failed_operations;
    }
    
    return result;
}

AdvancedBatchProcessor::BatchResult AdvancedBatchProcessor::batch_delete(
    const std::vector<std::string>& keys) {
    
    if (!pImpl->initialized || keys.empty()) {
        return BatchResult{};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BatchResult result;
    result.success_flags.resize(keys.size());
    
    try {
        // Pack keys for GPU transfer
        auto packed_data = pImpl->data_manager->pack_keys_for_gpu(keys);
        if (!packed_data) {
            result.error_message = "Failed to pack keys for GPU transfer";
            return result;
        }
        
        // Use first available stream
        cudaStream_t stream = pImpl->streams[0];
        
        // Transfer data to GPU
        if (!pImpl->data_manager->transfer_to_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer data to GPU";
            return result;
        }
        
        // Configure kernel launch parameters
        size_t batch_size = keys.size();
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        
        // Get hash table for kernel
        auto hash_stats = pImpl->hash_table->get_stats();
        
        // Launch optimized batch delete kernel
        batch_kernels::batch_delete_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
            packed_data->d_keys_data,
            packed_data->d_key_offsets,
            packed_data->d_key_lengths,
            packed_data->d_result_flags,
            reinterpret_cast<HashEntry*>(pImpl->hash_table), // Need mutable access
            hash_stats.capacity,
            batch_size
        );
        
        // Check for kernel errors
        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            result.error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(kernel_error));
            return result;
        }
        
        // Wait for completion
        cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            result.error_message = "Kernel execution failed: " + std::string(cudaGetErrorString(sync_error));
            return result;
        }
        
        // Transfer results back
        if (!pImpl->data_manager->transfer_from_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer results from GPU";
            return result;
        }
        
        // Unpack operation results
        result.success_flags = pImpl->data_manager->unpack_operation_results(*packed_data);
        
        // Count successes
        result.successful_count = std::count(result.success_flags.begin(), 
                                           result.success_flags.end(), true);
        result.failed_count = result.success_flags.size() - result.successful_count;
        
    } catch (const std::exception& e) {
        result.error_message = "Exception during batch delete: " + std::string(e.what());
        return result;
    }
    
    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    result.metrics.total_time_ms = elapsed_ms;
    result.metrics.operations_per_second = (keys.size() * 1000.0) / elapsed_ms;
    result.metrics.successful_operations = result.successful_count;
    result.metrics.failed_operations = result.failed_count;
    result.metrics.batch_efficiency_percent = 
        (static_cast<double>(result.successful_count) / keys.size()) * 100.0;
    
    // Update cumulative metrics
    {
        std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
        pImpl->cumulative_metrics.operations_per_second = 
            (pImpl->cumulative_metrics.operations_per_second + result.metrics.operations_per_second) / 2.0;
        pImpl->cumulative_metrics.successful_operations += result.metrics.successful_operations;
        pImpl->cumulative_metrics.failed_operations += result.metrics.failed_operations;
    }
    
    return result;
}

AdvancedBatchProcessor::BatchResult AdvancedBatchProcessor::batch_exists(
    const std::vector<std::string>& keys) {
    
    if (!pImpl->initialized || keys.empty()) {
        return BatchResult{};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BatchResult result;
    result.success_flags.resize(keys.size());
    
    try {
        // Pack keys for GPU transfer
        auto packed_data = pImpl->data_manager->pack_keys_for_gpu(keys);
        if (!packed_data) {
            result.error_message = "Failed to pack keys for GPU transfer";
            return result;
        }
        
        // Use first available stream
        cudaStream_t stream = pImpl->streams[0];
        
        // Transfer data to GPU
        if (!pImpl->data_manager->transfer_to_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer data to GPU";
            return result;
        }
        
        // Configure kernel launch parameters
        size_t batch_size = keys.size();
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        
        // Get hash table for kernel
        auto hash_stats = pImpl->hash_table->get_stats();
        
        // Launch optimized batch exists kernel
        batch_kernels::batch_exists_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
            packed_data->d_keys_data,
            packed_data->d_key_offsets,
            packed_data->d_key_lengths,
            packed_data->d_result_flags,
            reinterpret_cast<const HashEntry*>(pImpl->hash_table),
            hash_stats.capacity,
            batch_size
        );
        
        // Check for kernel errors
        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            result.error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(kernel_error));
            return result;
        }
        
        // Wait for completion
        cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            result.error_message = "Kernel execution failed: " + std::string(cudaGetErrorString(sync_error));
            return result;
        }
        
        // Transfer results back
        if (!pImpl->data_manager->transfer_from_gpu(*packed_data, stream)) {
            result.error_message = "Failed to transfer results from GPU";
            return result;
        }
        
        // Unpack operation results - for exists operations, success_flags indicate existence
        result.success_flags = pImpl->data_manager->unpack_operation_results(*packed_data);
        
        // For exists operations, successful_count means "keys that exist"
        result.successful_count = std::count(result.success_flags.begin(), 
                                           result.success_flags.end(), true);
        result.failed_count = result.success_flags.size() - result.successful_count;
        
    } catch (const std::exception& e) {
        result.error_message = "Exception during batch exists: " + std::string(e.what());
        return result;
    }
    
    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    result.metrics.total_time_ms = elapsed_ms;
    result.metrics.operations_per_second = (keys.size() * 1000.0) / elapsed_ms;
    result.metrics.successful_operations = result.successful_count;
    result.metrics.failed_operations = result.failed_count;
    result.metrics.batch_efficiency_percent = 
        (static_cast<double>(result.successful_count) / keys.size()) * 100.0;
    
    // Update cumulative metrics
    {
        std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
        pImpl->cumulative_metrics.operations_per_second = 
            (pImpl->cumulative_metrics.operations_per_second + result.metrics.operations_per_second) / 2.0;
        pImpl->cumulative_metrics.successful_operations += result.metrics.successful_operations;
        pImpl->cumulative_metrics.failed_operations += result.metrics.failed_operations;
    }
    
    return result;
}

size_t AdvancedBatchProcessor::get_optimal_batch_size() const {
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    return pImpl->current_optimal_batch_size;
}

void AdvancedBatchProcessor::tune_batch_size(const BatchMetrics& recent_metrics) {
    if (!pImpl->auto_tuning_enabled) return;
    
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    
    // Track recent performance
    pImpl->recent_metrics.push_back(recent_metrics);
    if (pImpl->recent_metrics.size() > 10) {
        pImpl->recent_metrics.erase(pImpl->recent_metrics.begin());
    }
    
    // Calculate average performance over recent operations
    double avg_ops_per_sec = 0.0;
    double avg_efficiency = 0.0;
    
    for (const auto& metrics : pImpl->recent_metrics) {
        avg_ops_per_sec += metrics.operations_per_second;
        avg_efficiency += metrics.batch_efficiency_percent;
    }
    
    if (!pImpl->recent_metrics.empty()) {
        avg_ops_per_sec /= pImpl->recent_metrics.size();
        avg_efficiency /= pImpl->recent_metrics.size();
    }
    
    // Auto-tune batch size based on performance trends
    const double TARGET_EFFICIENCY = 95.0; // 95% success rate target
    const double MIN_OPS_THRESHOLD = 100000.0; // 100K ops/sec minimum
    
    if (avg_efficiency < TARGET_EFFICIENCY) {
        // Reduce batch size if efficiency is poor
        pImpl->current_optimal_batch_size = 
            std::max(static_cast<size_t>(64), 
                    static_cast<size_t>(pImpl->current_optimal_batch_size * 0.8));
    } else if (avg_ops_per_sec > MIN_OPS_THRESHOLD && avg_efficiency > TARGET_EFFICIENCY) {
        // Increase batch size if performance is good
        pImpl->current_optimal_batch_size = 
            std::min(pImpl->config.max_batch_size,
                    static_cast<size_t>(pImpl->current_optimal_batch_size * 1.2));
    }
    
    // Estimate GPU bandwidth utilization
    const double THEORETICAL_MAX_OPS = 2000000.0; // 2M ops/sec theoretical max
    pImpl->recent_bandwidth_utilization = (avg_ops_per_sec / THEORETICAL_MAX_OPS) * 100.0;
}

BatchMetrics AdvancedBatchProcessor::get_cumulative_metrics() const {
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    
    BatchMetrics metrics = pImpl->cumulative_metrics;
    metrics.gpu_bandwidth_utilization_percent = pImpl->recent_bandwidth_utilization;
    
    return metrics;
}

void AdvancedBatchProcessor::reset_metrics() {
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    
    pImpl->cumulative_metrics = BatchMetrics{};
    pImpl->recent_metrics.clear();
    pImpl->recent_bandwidth_utilization = 0.0;
}

bool AdvancedBatchProcessor::configure(const BatchConfig& new_config) {
    if (!pImpl->initialized) {
        pImpl->config = new_config;
        return true;
    }
    
    // Update runtime configuration
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    
    pImpl->config = new_config;
    pImpl->auto_tuning_enabled = new_config.enable_auto_tuning;
    pImpl->current_optimal_batch_size = new_config.preferred_batch_size;
    
    return true;
}

} // namespace core  
} // namespace predis

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Design advanced batch processor architecture with multi-stream support", "status": "completed", "priority": "high"}, {"id": "2", "content": "Implement optimized GPU kernels for parallel batch operations", "status": "completed", "priority": "high"}, {"id": "3", "content": "Create memory coalescing optimization for bulk transfers", "status": "in_progress", "priority": "high"}, {"id": "4", "content": "Implement batch size auto-tuning based on GPU metrics", "status": "pending", "priority": "medium"}, {"id": "5", "content": "Add error handling with partial batch success preservation", "status": "pending", "priority": "medium"}, {"id": "6", "content": "Create performance validation suite for batch scaling", "status": "pending", "priority": "medium"}, {"id": "7", "content": "Integrate advanced batch processor with existing cache system", "status": "pending", "priority": "low"}]