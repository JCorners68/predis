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
#include <memory>
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <atomic>
#include <chrono>

namespace predis {
namespace core {

// Forward declarations
class MemoryManager;
class GpuHashTable;

/**
 * Memory Pipeline Optimizer for sustained high-throughput operations
 * 
 * Implements sophisticated memory pipeline optimization through:
 * - Asynchronous memory transfers with compute overlap
 * - Multi-buffer ping-pong patterns for continuous GPU utilization
 * - NUMA-aware CPU memory allocation for optimal bandwidth
 * - GPU memory pool management with fragmentation prevention
 * - Stream synchronization and pipeline coordination
 * 
 * Target: Sustained >2M ops/sec throughput with <1ms latency
 */
class MemoryPipelineOptimizer {
public:
    /**
     * Pipeline configuration optimized for RTX 5080 and high-performance CPUs
     */
    struct PipelineConfig {
        // Buffer management
        size_t num_pipeline_stages = 4;        // Number of pipeline stages
        size_t buffer_size_mb = 64;            // Size of each buffer in MB
        size_t max_concurrent_transfers = 8;   // Max concurrent H2D/D2H transfers
        
        // Memory allocation strategy
        bool enable_numa_optimization = true;  // NUMA-aware allocation
        bool use_pinned_memory = true;         // Use pinned memory for transfers
        bool enable_memory_pooling = true;     // Enable memory pool management
        
        // Pipeline optimization
        bool enable_compute_overlap = true;    // Overlap compute with transfers
        bool use_async_transfers = true;       // Use asynchronous memory transfers
        double target_bandwidth_utilization = 0.85; // Target 85% bandwidth utilization
        
        // Performance tuning
        size_t prefetch_queue_depth = 16;      // Prefetch queue depth
        size_t batch_coalescing_threshold = 256; // Coalesce batches below this size
        std::chrono::microseconds pipeline_timeout{1000}; // Pipeline stage timeout
        
        PipelineConfig() = default;
    };

    /**
     * Pipeline performance metrics
     */
    struct PipelineMetrics {
        // Throughput metrics
        double sustained_ops_per_second = 0.0;
        double peak_ops_per_second = 0.0;
        double memory_bandwidth_gbps = 0.0;
        double gpu_utilization_percent = 0.0;
        
        // Latency metrics
        double average_latency_ms = 0.0;
        double p99_latency_ms = 0.0;
        double pipeline_efficiency_percent = 0.0;
        
        // Pipeline stage metrics
        std::vector<double> stage_utilization_percent;
        std::vector<double> stage_latency_ms;
        size_t active_pipeline_stages = 0;
        
        // Memory metrics
        size_t total_memory_allocated_mb = 0;
        size_t memory_pool_fragmentation_percent = 0;
        double numa_allocation_efficiency = 0.0;
        
        // Error tracking
        size_t pipeline_stalls = 0;
        size_t memory_allocation_failures = 0;
        size_t transfer_timeouts = 0;
    };

    /**
     * Asynchronous operation handle for pipeline operations
     */
    class AsyncOperation {
    public:
        enum class Status { PENDING, IN_PROGRESS, COMPLETED, FAILED };
        enum class Type { LOOKUP, INSERT, DELETE, BATCH_LOOKUP, BATCH_INSERT };
        
        AsyncOperation(Type type, size_t operation_id);
        ~AsyncOperation() = default;
        
        Status get_status() const;
        bool is_ready() const;
        void wait() const;
        bool wait_for(std::chrono::milliseconds timeout) const;
        
        // Result retrieval (blocking)
        template<typename T>
        T get_result();
        
        // Cancel operation (best effort)
        void cancel();
        
    private:
        friend class MemoryPipelineOptimizer;
        
        Type type_;
        size_t operation_id_;
        std::atomic<Status> status_{Status::PENDING};
        std::shared_future<void> completion_future_;
        std::any result_data_;
        std::string error_message_;
    };

    MemoryPipelineOptimizer() = default;
    ~MemoryPipelineOptimizer() = default;

    /**
     * Initialize the memory pipeline optimizer
     */
    bool initialize(MemoryManager* memory_manager, 
                   GpuHashTable* hash_table,
                   const PipelineConfig& config = PipelineConfig{});

    /**
     * Shutdown and cleanup all pipeline resources
     */
    void shutdown();

    /**
     * Configure pipeline optimization settings
     */
    bool configure(const PipelineConfig& config);

    /**
     * Asynchronous cache operations with pipeline optimization
     */
    std::unique_ptr<AsyncOperation> async_lookup(const char* key, size_t key_len);
    std::unique_ptr<AsyncOperation> async_insert(const char* key, size_t key_len,
                                                 const char* value, size_t value_len);
    std::unique_ptr<AsyncOperation> async_delete(const char* key, size_t key_len);

    /**
     * Asynchronous batch operations with advanced pipeline optimization
     */
    std::unique_ptr<AsyncOperation> async_batch_lookup(const std::vector<std::string>& keys);
    std::unique_ptr<AsyncOperation> async_batch_insert(const std::vector<std::string>& keys,
                                                       const std::vector<std::string>& values);
    std::unique_ptr<AsyncOperation> async_batch_delete(const std::vector<std::string>& keys);

    /**
     * Pipeline control and monitoring
     */
    PipelineMetrics get_pipeline_metrics() const;
    void reset_metrics();
    
    /**
     * Pipeline health and optimization
     */
    bool is_pipeline_healthy() const;
    void optimize_pipeline_configuration();
    void flush_pipeline();  // Complete all pending operations
    
    /**
     * Memory pool management
     */
    size_t get_available_memory_mb() const;
    void defragment_memory_pools();
    void prefetch_memory_for_operations(size_t estimated_operations);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * NUMA-aware memory allocator for optimal CPU-GPU memory bandwidth
 */
class NumaMemoryAllocator {
public:
    struct AllocationInfo {
        void* ptr = nullptr;
        size_t size = 0;
        int numa_node = -1;
        bool is_pinned = false;
        std::chrono::steady_clock::time_point allocation_time;
    };

    NumaMemoryAllocator() = default;
    ~NumaMemoryAllocator() = default;

    bool initialize();
    void shutdown();

    /**
     * NUMA-optimized memory allocation
     */
    AllocationInfo allocate_pinned_memory(size_t size, int preferred_numa_node = -1);
    void deallocate_memory(const AllocationInfo& info);

    /**
     * Get optimal NUMA node for GPU operations
     */
    int get_optimal_numa_node_for_gpu(int gpu_device_id = 0) const;
    
    /**
     * Memory bandwidth testing and optimization
     */
    double measure_memory_bandwidth(int numa_node, size_t test_size_mb = 64) const;
    std::vector<int> get_numa_topology() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Multi-buffer ping-pong manager for continuous GPU utilization
 */
class PingPongBufferManager {
public:
    struct BufferSet {
        void* host_buffer = nullptr;
        void* device_buffer = nullptr;
        size_t buffer_size = 0;
        cudaStream_t transfer_stream = nullptr;
        cudaEvent_t transfer_complete_event = nullptr;
        std::atomic<bool> in_use{false};
        std::atomic<bool> transfer_in_progress{false};
    };

    PingPongBufferManager() = default;
    ~PingPongBufferManager() = default;

    bool initialize(size_t buffer_size, size_t num_buffer_sets, 
                   NumaMemoryAllocator* numa_allocator = nullptr);
    void shutdown();

    /**
     * Get next available buffer set for operations
     */
    BufferSet* acquire_buffer_set(std::chrono::milliseconds timeout = std::chrono::milliseconds{100});
    void release_buffer_set(BufferSet* buffer_set);

    /**
     * Asynchronous memory transfer operations
     */
    bool start_host_to_device_transfer(BufferSet* buffer_set, const void* src_data, size_t size);
    bool start_device_to_host_transfer(BufferSet* buffer_set, void* dst_data, size_t size);
    bool wait_for_transfer_completion(BufferSet* buffer_set, 
                                    std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});

    /**
     * Buffer management and monitoring
     */
    size_t get_available_buffer_count() const;
    double get_buffer_utilization_percent() const;
    void prefetch_buffers(size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis