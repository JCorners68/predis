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

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <memory>

namespace predis {
namespace core {

// Forward declarations
struct HashEntry;
class MemoryManager;
class GpuHashTable;

/**
 * Advanced GPU kernel optimizations targeting RTX 5080 architecture
 * Focuses on maximum single-operation performance through:
 * - CUDA cooperative groups for block-level parallelism
 * - Advanced memory hierarchy optimization (L1/L2 cache, shared memory)  
 * - GPU occupancy optimization and register pressure management
 * - Tensor core utilization for applicable workloads
 */
class OptimizedGpuKernels {
public:
    /**
     * Kernel launch configuration optimized for RTX 5080
     */
    struct LaunchConfig {
        int block_size = 256;              // Optimized for SM occupancy
        int grid_size = 1024;              // Utilizing all 84 SMs
        int shared_memory_bytes = 49152;   // Max shared memory per block
        cudaStream_t stream = nullptr;
        
        // RTX 5080 specific optimizations
        bool use_cooperative_groups = true;
        bool enable_tensor_cores = false;  // For future ML workloads
        bool optimize_occupancy = true;
        
        LaunchConfig() = default;
        LaunchConfig(int blocks, int threads, cudaStream_t s = nullptr) 
            : block_size(threads), grid_size(blocks), stream(s) {}
    };

    /**
     * Performance metrics for kernel optimization tracking
     */
    struct KernelMetrics {
        double execution_time_us = 0.0;
        double memory_bandwidth_gbps = 0.0;
        double gpu_occupancy_percent = 0.0;
        size_t shared_memory_usage_bytes = 0;
        size_t register_usage_per_thread = 0;
        size_t operations_per_second = 0;
        
        // RTX 5080 specific metrics
        double sm_utilization_percent = 0.0;
        double cache_hit_rate_percent = 0.0;
        double memory_coalescing_efficiency = 0.0;
    };

    OptimizedGpuKernels() = default;
    ~OptimizedGpuKernels() = default;

    /**
     * Initialize optimized kernels with target hardware configuration
     */
    bool initialize(GpuHashTable* hash_table, MemoryManager* memory_manager);

    /**
     * Shutdown and cleanup resources
     */
    void shutdown();

    /**
     * Optimized single-key lookup with cooperative groups
     * Target: 10x+ improvement over baseline implementation
     */
    bool optimized_lookup(const char* key, size_t key_len, 
                         char* value, size_t* value_len,
                         const LaunchConfig& config = LaunchConfig{});

    /**
     * Optimized single-key insertion with memory hierarchy optimization
     * Target: 10x+ improvement over baseline implementation  
     */
    bool optimized_insert(const char* key, size_t key_len,
                         const char* value, size_t value_len,
                         const LaunchConfig& config = LaunchConfig{});

    /**
     * Optimized single-key deletion with register pressure management
     */
    bool optimized_delete(const char* key, size_t key_len,
                         const LaunchConfig& config = LaunchConfig{});

    /**
     * Optimized bulk operations leveraging tensor cores where applicable
     */
    bool optimized_bulk_insert(const char* keys, const char* values,
                              const uint32_t* key_lengths, const uint32_t* value_lengths,
                              bool* results, size_t num_items,
                              const LaunchConfig& config = LaunchConfig{});

    bool optimized_bulk_lookup(const char* keys, char* values,
                              const uint32_t* key_lengths, uint32_t* value_lengths,
                              bool* results, size_t num_items,
                              const LaunchConfig& config = LaunchConfig{});

    /**
     * Get performance metrics for the last kernel execution
     */
    KernelMetrics get_last_metrics() const;

    /**
     * Configure kernel optimization settings
     */
    void configure_optimization(bool enable_cooperative_groups,
                               bool enable_tensor_cores,
                               bool optimize_occupancy);

    /**
     * Auto-tune kernel parameters based on workload characteristics
     */
    LaunchConfig auto_tune_config(size_t workload_size, bool is_read_heavy = true);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

// Namespace for optimized GPU kernel implementations
namespace optimized_kernels {

/**
 * Cooperative group-based lookup kernel with shared memory optimization
 */
__global__ void cooperative_lookup_kernel(
    const HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    char* __restrict__ result_value,
    size_t* __restrict__ result_length,
    bool* __restrict__ found_flag,
    int hash_method
);

/**
 * Memory hierarchy optimized insertion kernel
 */
__global__ void memory_optimized_insert_kernel(
    HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    const char* __restrict__ value,
    size_t value_length,
    bool* __restrict__ success_flag,
    int hash_method
);

/**
 * Register pressure optimized deletion kernel
 */
__global__ void register_optimized_delete_kernel(
    HashEntry* __restrict__ hash_table,
    size_t table_capacity,
    const char* __restrict__ key,
    size_t key_length,
    bool* __restrict__ success_flag,
    int hash_method
);

/**
 * Bulk operations with tensor core acceleration (future ML workloads)
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
    int hash_method
);

} // namespace optimized_kernels

} // namespace core
} // namespace predis