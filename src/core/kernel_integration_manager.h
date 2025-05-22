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

#include "optimized_gpu_kernels.h"
#include "advanced_batch_processor.h"
#include "data_structures/gpu_hash_table.h"
#include "memory_manager.h"
#include <memory>
#include <string>
#include <vector>

namespace predis {
namespace core {

/**
 * Integration manager that seamlessly connects optimized GPU kernels
 * with the existing cache system. Provides automatic kernel selection
 * based on workload characteristics and performance optimization.
 */
class KernelIntegrationManager {
public:
    /**
     * Kernel selection strategy for different workload patterns
     */
    enum class KernelStrategy {
        AUTO_SELECT,        // Automatically choose best kernel based on workload
        ALWAYS_OPTIMIZED,   // Always use optimized kernels
        ALWAYS_BASELINE,    // Always use baseline kernels (for comparison)
        HYBRID_ADAPTIVE     // Adaptive switching based on performance monitoring
    };

    /**
     * Performance monitoring configuration
     */
    struct MonitoringConfig {
        bool enable_performance_tracking = true;
        size_t performance_window_size = 100;  // Operations to track for performance
        double optimization_threshold = 2.0;   // Switch to optimized if >2x improvement
        bool enable_auto_tuning = true;
        double measurement_interval_seconds = 1.0;
    };

    /**
     * Integrated operation statistics
     */
    struct OperationStats {
        size_t total_operations = 0;
        size_t optimized_operations = 0;
        size_t baseline_operations = 0;
        double average_ops_per_second = 0.0;
        double optimization_improvement_factor = 1.0;
        double gpu_utilization_percent = 0.0;
        std::string current_strategy = "AUTO_SELECT";
    };

    KernelIntegrationManager() = default;
    ~KernelIntegrationManager() = default;

    /**
     * Initialize the integration manager with existing cache components
     */
    bool initialize(GpuHashTable* hash_table, 
                   MemoryManager* memory_manager,
                   AdvancedBatchProcessor* batch_processor = nullptr);

    /**
     * Shutdown and cleanup all resources
     */
    void shutdown();

    /**
     * Configure kernel selection strategy and monitoring
     */
    void configure(KernelStrategy strategy, const MonitoringConfig& config = MonitoringConfig{});

    /**
     * Integrated cache operations that automatically select optimal kernels
     */
    bool lookup(const char* key, size_t key_len, char* value, size_t* value_len);
    bool insert(const char* key, size_t key_len, const char* value, size_t value_len);
    bool remove(const char* key, size_t key_len);
    bool exists(const char* key, size_t key_len);

    /**
     * Batch operations with automatic kernel optimization
     */
    struct BatchResult {
        std::vector<bool> success_flags;
        std::vector<std::string> values; // For lookup operations
        size_t successful_operations = 0;
        size_t failed_operations = 0;
        double total_time_ms = 0.0;
        double operations_per_second = 0.0;
        std::string kernel_used = "AUTO_SELECTED";
        bool used_optimized_kernel = false;
    };

    BatchResult batch_lookup(const std::vector<std::string>& keys);
    BatchResult batch_insert(const std::vector<std::string>& keys, 
                            const std::vector<std::string>& values);
    BatchResult batch_remove(const std::vector<std::string>& keys);
    BatchResult batch_exists(const std::vector<std::string>& keys);

    /**
     * Performance monitoring and statistics
     */
    OperationStats get_operation_stats() const;
    void reset_statistics();

    /**
     * Manual kernel optimization control
     */
    void force_kernel_selection(bool use_optimized);
    bool is_optimized_kernel_available() const;

    /**
     * Get performance recommendations based on current workload
     */
    std::string get_performance_recommendations() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis