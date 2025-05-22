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

#include "kernel_integration_manager.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <numeric>

namespace predis {
namespace core {

struct KernelIntegrationManager::Impl {
    // Core components
    GpuHashTable* hash_table = nullptr;
    MemoryManager* memory_manager = nullptr;
    AdvancedBatchProcessor* batch_processor = nullptr;
    
    // Optimized kernels
    std::unique_ptr<OptimizedGpuKernels> optimized_kernels;
    
    // Configuration
    KernelStrategy strategy = KernelStrategy::AUTO_SELECT;
    MonitoringConfig monitoring_config;
    
    // Performance tracking
    mutable std::mutex stats_mutex;
    OperationStats operation_stats;
    std::vector<double> recent_performance_samples;
    std::vector<bool> recent_kernel_choices; // true = optimized, false = baseline
    
    // Timing
    std::chrono::steady_clock::time_point last_measurement_time;
    
    // State
    bool initialized = false;
    bool force_optimized = false;
    bool force_baseline = false;
    
    // Performance thresholds
    static constexpr double MIN_OPS_FOR_OPTIMIZATION = 1000.0; // Switch to optimized if >1K ops/sec
    static constexpr size_t MIN_SAMPLES_FOR_DECISION = 10;
};

KernelIntegrationManager::KernelIntegrationManager() : pImpl(std::make_unique<Impl>()) {}

KernelIntegrationManager::~KernelIntegrationManager() {
    if (pImpl->initialized) {
        shutdown();
    }
}

bool KernelIntegrationManager::initialize(GpuHashTable* hash_table, 
                                         MemoryManager* memory_manager,
                                         AdvancedBatchProcessor* batch_processor) {
    if (pImpl->initialized) {
        std::cerr << "KernelIntegrationManager already initialized" << std::endl;
        return false;
    }
    
    if (!hash_table || !memory_manager) {
        std::cerr << "Invalid hash table or memory manager provided" << std::endl;
        return false;
    }
    
    pImpl->hash_table = hash_table;
    pImpl->memory_manager = memory_manager;
    pImpl->batch_processor = batch_processor;
    
    // Initialize optimized kernels
    pImpl->optimized_kernels = std::make_unique<OptimizedGpuKernels>();
    if (!pImpl->optimized_kernels->initialize(hash_table, memory_manager)) {
        std::cerr << "Failed to initialize optimized GPU kernels" << std::endl;
        return false;
    }
    
    // Configure optimized kernels for maximum performance
    pImpl->optimized_kernels->configure_optimization(true, false, true); // Enable cooperative groups and occupancy optimization
    
    // Initialize timing
    pImpl->last_measurement_time = std::chrono::steady_clock::now();
    
    pImpl->initialized = true;
    
    std::cout << "KernelIntegrationManager initialized with optimized GPU kernels" << std::endl;
    return true;
}

void KernelIntegrationManager::shutdown() {
    if (!pImpl->initialized) return;
    
    if (pImpl->optimized_kernels) {
        pImpl->optimized_kernels->shutdown();
    }
    
    pImpl->initialized = false;
    std::cout << "KernelIntegrationManager shutdown complete" << std::endl;
}

void KernelIntegrationManager::configure(KernelStrategy strategy, const MonitoringConfig& config) {
    pImpl->strategy = strategy;
    pImpl->monitoring_config = config;
    
    std::cout << "KernelIntegrationManager configured with strategy: ";
    switch (strategy) {
        case KernelStrategy::AUTO_SELECT:
            std::cout << "AUTO_SELECT";
            break;
        case KernelStrategy::ALWAYS_OPTIMIZED:
            std::cout << "ALWAYS_OPTIMIZED";
            pImpl->force_optimized = true;
            break;
        case KernelStrategy::ALWAYS_BASELINE:
            std::cout << "ALWAYS_BASELINE";
            pImpl->force_baseline = true;
            break;
        case KernelStrategy::HYBRID_ADAPTIVE:
            std::cout << "HYBRID_ADAPTIVE";
            break;
    }
    std::cout << std::endl;
}

bool KernelIntegrationManager::lookup(const char* key, size_t key_len, char* value, size_t* value_len) {
    if (!pImpl->initialized) return false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = false;
    bool used_optimized = false;
    
    // Determine which kernel to use
    bool use_optimized_kernel = false;
    
    if (pImpl->force_optimized || pImpl->strategy == KernelStrategy::ALWAYS_OPTIMIZED) {
        use_optimized_kernel = true;
    } else if (pImpl->force_baseline || pImpl->strategy == KernelStrategy::ALWAYS_BASELINE) {
        use_optimized_kernel = false;
    } else {
        // Auto-select based on performance history and strategy
        use_optimized_kernel = should_use_optimized_kernel();
    }
    
    // Execute operation using selected kernel
    if (use_optimized_kernel && pImpl->optimized_kernels) {
        auto config = pImpl->optimized_kernels->auto_tune_config(1, true); // Single lookup, read-heavy
        success = pImpl->optimized_kernels->optimized_lookup(key, key_len, value, value_len, config);
        used_optimized = true;
    } else {
        // Use baseline kernel from GpuHashTable
        uint32_t result_len = 0;
        success = pImpl->hash_table->lookup(key, key_len, value, &result_len);
        if (success && value_len) {
            *value_len = result_len;
        }
        used_optimized = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    // Update performance statistics
    update_performance_stats(elapsed_us, used_optimized, success);
    
    return success;
}

bool KernelIntegrationManager::insert(const char* key, size_t key_len, const char* value, size_t value_len) {
    if (!pImpl->initialized) return false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = false;
    bool used_optimized = false;
    
    // Determine which kernel to use
    bool use_optimized_kernel = false;
    
    if (pImpl->force_optimized || pImpl->strategy == KernelStrategy::ALWAYS_OPTIMIZED) {
        use_optimized_kernel = true;
    } else if (pImpl->force_baseline || pImpl->strategy == KernelStrategy::ALWAYS_BASELINE) {
        use_optimized_kernel = false;
    } else {
        use_optimized_kernel = should_use_optimized_kernel();
    }
    
    // Execute operation using selected kernel
    if (use_optimized_kernel && pImpl->optimized_kernels) {
        auto config = pImpl->optimized_kernels->auto_tune_config(1, false); // Single insert, write operation
        success = pImpl->optimized_kernels->optimized_insert(key, key_len, value, value_len, config);
        used_optimized = true;
    } else {
        // Use baseline kernel from GpuHashTable
        success = pImpl->hash_table->insert(key, key_len, value, value_len);
        used_optimized = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    // Update performance statistics
    update_performance_stats(elapsed_us, used_optimized, success);
    
    return success;
}

bool KernelIntegrationManager::remove(const char* key, size_t key_len) {
    if (!pImpl->initialized) return false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = false;
    bool used_optimized = false;
    
    // Determine which kernel to use
    bool use_optimized_kernel = should_use_optimized_kernel();
    
    if (pImpl->force_optimized || pImpl->strategy == KernelStrategy::ALWAYS_OPTIMIZED) {
        use_optimized_kernel = true;
    } else if (pImpl->force_baseline || pImpl->strategy == KernelStrategy::ALWAYS_BASELINE) {
        use_optimized_kernel = false;
    }
    
    // Execute operation using selected kernel
    if (use_optimized_kernel && pImpl->optimized_kernels) {
        auto config = pImpl->optimized_kernels->auto_tune_config(1, false); // Single delete
        success = pImpl->optimized_kernels->optimized_delete(key, key_len, config);
        used_optimized = true;
    } else {
        // Use baseline kernel from GpuHashTable
        success = pImpl->hash_table->remove(key, key_len);
        used_optimized = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    // Update performance statistics
    update_performance_stats(elapsed_us, used_optimized, success);
    
    return success;
}

bool KernelIntegrationManager::exists(const char* key, size_t key_len) {
    // For exists, we can use lookup and just check success without returning value
    char dummy_value[1];
    size_t dummy_len = 0;
    return lookup(key, key_len, dummy_value, &dummy_len);
}

KernelIntegrationManager::BatchResult KernelIntegrationManager::batch_lookup(const std::vector<std::string>& keys) {
    if (!pImpl->initialized) return BatchResult{};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    BatchResult result;
    
    // For batch operations, prefer the advanced batch processor if available
    if (pImpl->batch_processor) {
        auto batch_result = pImpl->batch_processor->batch_get(keys);
        
        result.success_flags = batch_result.success_flags;
        result.values = batch_result.values;
        result.successful_operations = batch_result.successful_count;
        result.failed_operations = batch_result.failed_count;
        result.operations_per_second = batch_result.metrics.operations_per_second;
        result.kernel_used = "ADVANCED_BATCH_PROCESSOR";
        result.used_optimized_kernel = true;
    } else {
        // Fallback to individual operations
        result.success_flags.resize(keys.size());
        result.values.resize(keys.size());
        result.successful_operations = 0;
        
        for (size_t i = 0; i < keys.size(); ++i) {
            char value_buffer[4096];
            size_t value_len = 0;
            
            bool success = lookup(keys[i].c_str(), keys[i].length(), value_buffer, &value_len);
            result.success_flags[i] = success;
            
            if (success) {
                result.values[i] = std::string(value_buffer, value_len);
                result.successful_operations++;
            }
        }
        
        result.failed_operations = keys.size() - result.successful_operations;
        result.kernel_used = "INDIVIDUAL_OPTIMIZED";
        result.used_optimized_kernel = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (result.total_time_ms > 0) {
        result.operations_per_second = (keys.size() * 1000.0) / result.total_time_ms;
    }
    
    return result;
}

KernelIntegrationManager::BatchResult KernelIntegrationManager::batch_insert(
    const std::vector<std::string>& keys, 
    const std::vector<std::string>& values) {
    
    if (!pImpl->initialized || keys.size() != values.size()) return BatchResult{};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    BatchResult result;
    
    // Use advanced batch processor if available
    if (pImpl->batch_processor) {
        auto batch_result = pImpl->batch_processor->batch_put(keys, values);
        
        result.success_flags = batch_result.success_flags;
        result.successful_operations = batch_result.successful_count;
        result.failed_operations = batch_result.failed_count;
        result.operations_per_second = batch_result.metrics.operations_per_second;
        result.kernel_used = "ADVANCED_BATCH_PROCESSOR";
        result.used_optimized_kernel = true;
    } else {
        // Fallback to individual operations
        result.success_flags.resize(keys.size());
        result.successful_operations = 0;
        
        for (size_t i = 0; i < keys.size(); ++i) {
            bool success = insert(keys[i].c_str(), keys[i].length(), 
                                 values[i].c_str(), values[i].length());
            result.success_flags[i] = success;
            
            if (success) {
                result.successful_operations++;
            }
        }
        
        result.failed_operations = keys.size() - result.successful_operations;
        result.kernel_used = "INDIVIDUAL_OPTIMIZED";
        result.used_optimized_kernel = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (result.total_time_ms > 0) {
        result.operations_per_second = (keys.size() * 1000.0) / result.total_time_ms;
    }
    
    return result;
}

KernelIntegrationManager::BatchResult KernelIntegrationManager::batch_remove(const std::vector<std::string>& keys) {
    if (!pImpl->initialized) return BatchResult{};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    BatchResult result;
    
    // Use advanced batch processor if available
    if (pImpl->batch_processor) {
        auto batch_result = pImpl->batch_processor->batch_delete(keys);
        
        result.success_flags = batch_result.success_flags;
        result.successful_operations = batch_result.successful_count;
        result.failed_operations = batch_result.failed_count;
        result.operations_per_second = batch_result.metrics.operations_per_second;
        result.kernel_used = "ADVANCED_BATCH_PROCESSOR";
        result.used_optimized_kernel = true;
    } else {
        // Fallback to individual operations
        result.success_flags.resize(keys.size());
        result.successful_operations = 0;
        
        for (size_t i = 0; i < keys.size(); ++i) {
            bool success = remove(keys[i].c_str(), keys[i].length());
            result.success_flags[i] = success;
            
            if (success) {
                result.successful_operations++;
            }
        }
        
        result.failed_operations = keys.size() - result.successful_operations;
        result.kernel_used = "INDIVIDUAL_OPTIMIZED";
        result.used_optimized_kernel = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (result.total_time_ms > 0) {
        result.operations_per_second = (keys.size() * 1000.0) / result.total_time_ms;
    }
    
    return result;
}

KernelIntegrationManager::BatchResult KernelIntegrationManager::batch_exists(const std::vector<std::string>& keys) {
    if (!pImpl->initialized) return BatchResult{};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    BatchResult result;
    
    // Use advanced batch processor if available
    if (pImpl->batch_processor) {
        auto batch_result = pImpl->batch_processor->batch_exists(keys);
        
        result.success_flags = batch_result.success_flags;
        result.successful_operations = batch_result.successful_count;
        result.failed_operations = batch_result.failed_count;
        result.operations_per_second = batch_result.metrics.operations_per_second;
        result.kernel_used = "ADVANCED_BATCH_PROCESSOR";
        result.used_optimized_kernel = true;
    } else {
        // Fallback to individual operations
        result.success_flags.resize(keys.size());
        result.successful_operations = 0;
        
        for (size_t i = 0; i < keys.size(); ++i) {
            bool success = exists(keys[i].c_str(), keys[i].length());
            result.success_flags[i] = success;
            
            if (success) {
                result.successful_operations++;
            }
        }
        
        result.failed_operations = keys.size() - result.successful_operations;
        result.kernel_used = "INDIVIDUAL_OPTIMIZED";
        result.used_optimized_kernel = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (result.total_time_ms > 0) {
        result.operations_per_second = (keys.size() * 1000.0) / result.total_time_ms;
    }
    
    return result;
}

KernelIntegrationManager::OperationStats KernelIntegrationManager::get_operation_stats() const {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    return pImpl->operation_stats;
}

void KernelIntegrationManager::reset_statistics() {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    pImpl->operation_stats = OperationStats{};
    pImpl->recent_performance_samples.clear();
    pImpl->recent_kernel_choices.clear();
}

void KernelIntegrationManager::force_kernel_selection(bool use_optimized) {
    pImpl->force_optimized = use_optimized;
    pImpl->force_baseline = !use_optimized;
    
    std::cout << "Forced kernel selection: " << (use_optimized ? "OPTIMIZED" : "BASELINE") << std::endl;
}

bool KernelIntegrationManager::is_optimized_kernel_available() const {
    return pImpl->optimized_kernels && pImpl->initialized;
}

std::string KernelIntegrationManager::get_performance_recommendations() const {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    
    std::string recommendations = "Performance Recommendations:\n";
    
    if (pImpl->operation_stats.total_operations < 100) {
        recommendations += "- Insufficient data for meaningful recommendations (need >100 operations)\n";
        return recommendations;
    }
    
    double optimized_ratio = static_cast<double>(pImpl->operation_stats.optimized_operations) / 
                            pImpl->operation_stats.total_operations;
    
    if (pImpl->operation_stats.optimization_improvement_factor > 5.0) {
        recommendations += "- Excellent optimization performance (>5x improvement)\n";
        recommendations += "- Consider using ALWAYS_OPTIMIZED strategy for maximum performance\n";
    } else if (pImpl->operation_stats.optimization_improvement_factor > 2.0) {
        recommendations += "- Good optimization performance (>2x improvement)\n";
        recommendations += "- Continue with AUTO_SELECT or HYBRID_ADAPTIVE strategy\n";
    } else {
        recommendations += "- Limited optimization benefit (<2x improvement)\n";
        recommendations += "- Consider workload analysis or ALWAYS_BASELINE for consistency\n";
    }
    
    if (pImpl->operation_stats.gpu_utilization_percent < 50.0) {
        recommendations += "- Low GPU utilization detected\n";
        recommendations += "- Consider increasing batch sizes or concurrent operations\n";
    }
    
    if (optimized_ratio > 0.8) {
        recommendations += "- High optimized kernel usage (>80%)\n";
        recommendations += "- System is well-optimized for current workload\n";
    }
    
    return recommendations;
}

// Private helper methods
bool KernelIntegrationManager::should_use_optimized_kernel() const {
    if (pImpl->strategy == KernelStrategy::ALWAYS_OPTIMIZED) return true;
    if (pImpl->strategy == KernelStrategy::ALWAYS_BASELINE) return false;
    
    // For AUTO_SELECT and HYBRID_ADAPTIVE, use performance-based decision
    if (pImpl->recent_performance_samples.size() < Impl::MIN_SAMPLES_FOR_DECISION) {
        // Not enough data, default to optimized
        return true;
    }
    
    // Calculate recent average performance
    double avg_performance = std::accumulate(pImpl->recent_performance_samples.begin(), 
                                           pImpl->recent_performance_samples.end(), 0.0) /
                            pImpl->recent_performance_samples.size();
    
    // Use optimized if recent performance suggests benefit
    return avg_performance > Impl::MIN_OPS_FOR_OPTIMIZATION;
}

void KernelIntegrationManager::update_performance_stats(double elapsed_us, bool used_optimized, bool success) {
    if (!success) return; // Only track successful operations
    
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    
    pImpl->operation_stats.total_operations++;
    if (used_optimized) {
        pImpl->operation_stats.optimized_operations++;
    } else {
        pImpl->operation_stats.baseline_operations++;
    }
    
    // Calculate operations per second for this operation
    double ops_per_second = 1000000.0 / elapsed_us; // Convert microseconds to ops/sec
    
    // Update running average
    double total_ops = static_cast<double>(pImpl->operation_stats.total_operations);
    pImpl->operation_stats.average_ops_per_second = 
        (pImpl->operation_stats.average_ops_per_second * (total_ops - 1.0) + ops_per_second) / total_ops;
    
    // Track recent performance samples
    pImpl->recent_performance_samples.push_back(ops_per_second);
    pImpl->recent_kernel_choices.push_back(used_optimized);
    
    // Maintain sliding window
    if (pImpl->recent_performance_samples.size() > pImpl->monitoring_config.performance_window_size) {
        pImpl->recent_performance_samples.erase(pImpl->recent_performance_samples.begin());
        pImpl->recent_kernel_choices.erase(pImpl->recent_kernel_choices.begin());
    }
    
    // Calculate optimization improvement factor
    if (pImpl->operation_stats.baseline_operations > 0 && pImpl->operation_stats.optimized_operations > 0) {
        // This is a simplified calculation - in a real implementation, you'd track separate averages
        pImpl->operation_stats.optimization_improvement_factor = 
            pImpl->operation_stats.average_ops_per_second / Impl::MIN_OPS_FOR_OPTIMIZATION;
    }
    
    // Update strategy string
    switch (pImpl->strategy) {
        case KernelStrategy::AUTO_SELECT:
            pImpl->operation_stats.current_strategy = "AUTO_SELECT";
            break;
        case KernelStrategy::ALWAYS_OPTIMIZED:
            pImpl->operation_stats.current_strategy = "ALWAYS_OPTIMIZED";
            break;
        case KernelStrategy::ALWAYS_BASELINE:
            pImpl->operation_stats.current_strategy = "ALWAYS_BASELINE";
            break;
        case KernelStrategy::HYBRID_ADAPTIVE:
            pImpl->operation_stats.current_strategy = "HYBRID_ADAPTIVE";
            break;
    }
}

} // namespace core
} // namespace predis