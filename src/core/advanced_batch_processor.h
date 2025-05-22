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

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <optional>
#include <cuda_runtime.h>
#include "gpu_hash_table.h"
#include "memory_manager.h"

namespace predis {
namespace core {

/**
 * @brief Advanced batch processor leveraging GPU parallelism for maximum throughput
 * 
 * Implements optimized batch operations with:
 * - Multi-stream parallel execution
 * - Memory coalescing optimization
 * - Dynamic batch size tuning
 * - Partial failure handling
 * - Performance scaling validation
 */
class AdvancedBatchProcessor {
public:
    enum class OperationType {
        GET,
        PUT,
        DELETE,
        EXISTS
    };

    struct BatchMetrics {
        double operations_per_second = 0.0;
        double memory_bandwidth_gbps = 0.0;
        double gpu_utilization_percent = 0.0;
        double batch_efficiency_percent = 0.0;
        size_t successful_operations = 0;
        size_t failed_operations = 0;
        double total_time_ms = 0.0;
        size_t optimal_batch_size = 0;
    };

    struct BatchConfig {
        size_t min_batch_size = 32;          // Minimum for GPU efficiency
        size_t max_batch_size = 8192;        // Maximum before memory constraints
        size_t preferred_batch_size = 1024;  // Optimal for RTX 5080
        size_t max_concurrent_batches = 8;   // Number of CUDA streams
        double target_memory_bandwidth = 0.8; // 80% of theoretical maximum
        bool enable_auto_tuning = true;      // Dynamic batch size optimization
        bool enable_error_recovery = true;   // Partial batch failure handling
    };

    struct BatchResult {
        std::vector<std::optional<std::string>> values;
        std::vector<bool> success_flags;
        BatchMetrics metrics;
        size_t successful_count = 0;
        size_t failed_count = 0;
        std::string error_message;
    };

private:
    static constexpr size_t MAX_KEY_LENGTH = 256;
    static constexpr size_t MAX_VALUE_LENGTH = 4096;
    static constexpr size_t MAX_CONCURRENT_STREAMS = 16;
    static constexpr double RTX5080_MEMORY_BANDWIDTH_GBPS = 800.0;

    struct BatchOperation {
        OperationType type;
        std::vector<std::string> keys;
        std::vector<std::string> values; // For PUT operations
        size_t start_index;
        size_t count;
        cudaStream_t stream;
        cudaEvent_t completion_event;
        
        // GPU memory pointers
        char* d_keys_data = nullptr;
        char* d_values_data = nullptr;
        size_t* d_key_offsets = nullptr;
        size_t* d_value_offsets = nullptr;
        uint32_t* d_key_lengths = nullptr;
        uint32_t* d_value_lengths = nullptr;
        bool* d_success_flags = nullptr;
        
        BatchOperation() {
            cudaEventCreate(&completion_event);
        }
        
        ~BatchOperation() {
            cudaEventDestroy(completion_event);
        }
    };

public:
    AdvancedBatchProcessor();
    ~AdvancedBatchProcessor();

    // Initialization and configuration
    bool initialize(GpuHashTable* hash_table, MemoryManager* memory_manager, 
                   const BatchConfig& config = BatchConfig{});
    void shutdown();
    
    // Configuration management
    void update_config(const BatchConfig& config);
    BatchConfig get_config() const;
    
    // Advanced batch operations
    BatchResult batch_get(const std::vector<std::string>& keys);
    BatchResult batch_put(const std::vector<std::pair<std::string, std::string>>& key_value_pairs);
    BatchResult batch_delete(const std::vector<std::string>& keys);
    BatchResult batch_exists(const std::vector<std::string>& keys);
    
    // Performance optimization
    size_t calculate_optimal_batch_size(size_t total_operations, OperationType type);
    BatchMetrics get_performance_metrics() const;
    void reset_performance_metrics();
    
    // Auto-tuning and adaptation
    void enable_auto_tuning(bool enabled);
    void tune_batch_parameters();
    
    // Performance validation and testing
    bool validate_batch_scaling(size_t min_size, size_t max_size, size_t step_size);
    std::vector<BatchMetrics> benchmark_batch_sizes(const std::vector<size_t>& batch_sizes);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief GPU kernels for optimized batch operations
 */
namespace batch_kernels {

// Optimized batch lookup kernel with memory coalescing
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
    size_t batch_size
);

// Optimized batch insert kernel with conflict resolution
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
    size_t batch_size
);

// Optimized batch delete kernel with atomic operations
__global__ void batch_delete_coalesced_kernel(
    const char* __restrict__ keys_data,
    const size_t* __restrict__ key_offsets,
    const uint32_t* __restrict__ key_lengths,
    bool* __restrict__ success_flags,
    HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size
);

// Batch exists kernel (lightweight lookup without value copy)
__global__ void batch_exists_coalesced_kernel(
    const char* __restrict__ keys_data,
    const size_t* __restrict__ key_offsets,
    const uint32_t* __restrict__ key_lengths,
    bool* __restrict__ exists_flags,
    const HashEntry* __restrict__ hash_table,
    size_t table_size,
    size_t batch_size
);

} // namespace batch_kernels

/**
 * @brief Utilities for batch data preparation and memory management
 */
class BatchDataManager {
public:
    struct PackedBatchData {
        // Host memory (pinned for fast transfer)
        char* h_keys_data = nullptr;
        char* h_values_data = nullptr;
        size_t* h_key_offsets = nullptr;
        size_t* h_value_offsets = nullptr;
        uint32_t* h_key_lengths = nullptr;
        uint32_t* h_value_lengths = nullptr;
        
        // GPU memory
        char* d_keys_data = nullptr;
        char* d_values_data = nullptr;
        size_t* d_key_offsets = nullptr;
        size_t* d_value_offsets = nullptr;
        uint32_t* d_key_lengths = nullptr;
        uint32_t* d_value_lengths = nullptr;
        bool* d_result_flags = nullptr;
        
        size_t keys_data_size = 0;
        size_t values_data_size = 0;
        size_t batch_size = 0;
        
        PackedBatchData() = default;
        ~PackedBatchData();
        
        // Non-copyable, movable
        PackedBatchData(const PackedBatchData&) = delete;
        PackedBatchData& operator=(const PackedBatchData&) = delete;
        PackedBatchData(PackedBatchData&&) = default;
        PackedBatchData& operator=(PackedBatchData&&) = default;
    };

public:
    BatchDataManager(MemoryManager* memory_manager);
    ~BatchDataManager();

    // Pack string data into GPU-optimized layout
    std::unique_ptr<PackedBatchData> pack_keys_for_gpu(const std::vector<std::string>& keys);
    std::unique_ptr<PackedBatchData> pack_key_values_for_gpu(
        const std::vector<std::pair<std::string, std::string>>& key_value_pairs);
    
    // Transfer data to/from GPU with optimal memory patterns
    bool transfer_to_gpu(PackedBatchData& data, cudaStream_t stream);
    bool transfer_from_gpu(PackedBatchData& data, cudaStream_t stream);
    
    // Unpack results back to application format
    std::vector<std::optional<std::string>> unpack_lookup_results(
        const PackedBatchData& data, const std::vector<std::string>& original_keys);
    std::vector<bool> unpack_operation_results(const PackedBatchData& data);

private:
    MemoryManager* memory_manager_;
    
    // Memory pools for different operation types
    struct MemoryPool {
        std::vector<void*> available_buffers;
        size_t buffer_size;
        size_t max_buffers;
    };
    
    std::map<size_t, MemoryPool> host_memory_pools_;
    std::map<size_t, MemoryPool> device_memory_pools_;
    
    void* allocate_from_pool(std::map<size_t, MemoryPool>& pools, size_t size, bool is_device);
    void return_to_pool(std::map<size_t, MemoryPool>& pools, void* ptr, size_t size);
};

} // namespace core
} // namespace predis