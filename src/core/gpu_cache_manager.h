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
#include <memory>
#include <vector>
#include <chrono>

namespace predis {
namespace core {

/**
 * @brief GPU-accelerated cache manager with real/mock mode switching
 * 
 * Provides high-performance key-value storage using GPU memory and hash tables
 * with fallback to mock implementation for development and testing
 */
class GpuCacheManager {
public:
    enum class Mode {
        MOCK,    // Use mock implementation for development/testing
        REAL_GPU // Use actual GPU memory and hash table
    };

    enum class HashMethod {
        FNV1A,
        MURMUR3
    };

    struct Config {
        Mode mode = Mode::REAL_GPU;
        HashMethod hash_method = HashMethod::FNV1A;
        size_t initial_capacity = 1024 * 1024;  // 1M entries
        size_t max_memory_mb = 12800;           // 80% of 16GB
        bool enable_statistics = true;
        bool enable_prefetching = false;
    };

    struct CacheStats {
        size_t total_entries = 0;
        size_t memory_usage_mb = 0;
        size_t memory_capacity_mb = 0;
        double load_factor = 0.0;
        size_t total_operations = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        double hit_ratio = 0.0;
        double avg_operation_time_ms = 0.0;
        Mode current_mode = Mode::MOCK;
        
        // Performance metrics
        double operations_per_second = 0.0;
        double peak_operations_per_second = 0.0;
        size_t peak_memory_usage_mb = 0;
    };

    struct BatchResult {
        size_t successful_operations = 0;
        size_t failed_operations = 0;
        double total_time_ms = 0.0;
        double operations_per_second = 0.0;
    };

public:
    GpuCacheManager();
    ~GpuCacheManager();

    // Initialization and configuration
    bool initialize();
    bool initialize(const Config& config);
    void shutdown();
    bool switch_mode(Mode new_mode);
    Mode get_current_mode() const;

    // Basic cache operations
    bool get(const std::string& key, std::string& value);
    bool put(const std::string& key, const std::string& value, int ttl_seconds = 0);
    bool remove(const std::string& key);
    bool exists(const std::string& key);
    void clear();

    // Batch operations for GPU efficiency
    BatchResult batch_get(const std::vector<std::string>& keys, std::vector<std::string>& values);
    BatchResult batch_put(const std::vector<std::pair<std::string, std::string>>& key_value_pairs);
    BatchResult batch_remove(const std::vector<std::string>& keys);

    // Statistics and monitoring
    CacheStats get_stats() const;
    void reset_stats();
    void print_performance_report() const;

    // Performance testing and comparison
    bool run_performance_comparison(size_t num_operations = 10000);
    bool validate_mock_vs_real_consistency(size_t num_tests = 1000);

    // Advanced features
    void configure_prefetching(bool enabled, double confidence_threshold = 0.7);
    void hint_related_keys(const std::vector<std::string>& keys);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis