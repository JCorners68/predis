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
#include <unordered_map>
#include <memory>

namespace predis {
namespace api {

/**
 * @brief Main client interface for Predis GPU cache with mock/real GPU switching
 * 
 * Provides Redis-compatible API with ML extensions and seamless switching
 * between mock and real GPU implementations for development and testing.
 */
class PredisClient {
public:
    enum class Mode {
        AUTO_DETECT,    // Automatically detect and choose best available mode
        MOCK_ONLY,      // Force mock implementation for testing/development
        REAL_GPU_ONLY,  // Force real GPU implementation
        HYBRID          // Use both for performance comparison
    };

    enum class ConsistencyLevel {
        STRONG,
        RELAXED
    };

    struct PrefetchConfig {
        bool enabled = true;
        double confidence_threshold = 0.7;
        int max_prefetch_keys = 200;
        int max_prefetch_size_mb = 100;
        int prefetch_ttl = 30;
    };

    struct Stats {
        size_t total_keys = 0;
        double hit_ratio = 0.0;
        double memory_usage_mb = 0.0;
        double prefetch_hit_ratio = 0.0;
        double operations_per_second = 0.0;
        double avg_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        
        // Implementation mode tracking
        bool using_real_gpu = false;
        std::string implementation_mode;
        
        // Performance comparison data
        double mock_ops_per_second = 0.0;
        double real_gpu_ops_per_second = 0.0;
        double performance_improvement_ratio = 0.0;
    };

public:
    PredisClient();
    ~PredisClient();

    // Connection and mode management
    bool connect(const std::string& host = "localhost", int port = 6379, Mode mode = Mode::AUTO_DETECT);
    void disconnect();
    bool is_connected() const;
    
    // Mode switching and configuration
    bool switch_mode(Mode new_mode);
    Mode get_current_mode() const;
    bool is_using_real_gpu() const;

    // Basic cache operations
    bool get(const std::string& key, std::string& value);
    bool put(const std::string& key, const std::string& value, int ttl = 0);
    bool remove(const std::string& key);
    
    // Batch operations
    std::vector<std::string> mget(const std::vector<std::string>& keys);
    bool mput(const std::unordered_map<std::string, std::string>& key_values);
    bool mdelete(const std::vector<std::string>& keys);

    // Consistency control
    void set_consistency_level(ConsistencyLevel level);
    ConsistencyLevel get_consistency_level() const;

    // Prefetching configuration
    void configure_prefetching(const PrefetchConfig& config);
    PrefetchConfig get_prefetch_config() const;

    // ML hints
    void hint_related_keys(const std::vector<std::string>& keys);
    void hint_sequence(const std::vector<std::string>& keys);

    // Statistics and monitoring
    Stats get_stats() const;
    void reset_stats();
    void flush_all();
    
    // Performance testing and comparison
    bool run_performance_comparison(size_t num_operations = 10000);
    bool validate_consistency(size_t num_tests = 1000);
    void print_performance_report() const;
    
    // Advanced GPU features (real GPU mode only)
    bool configure_gpu_memory(size_t max_memory_mb = 0);  // 0 = auto-detect
    bool defragment_gpu_memory();
    void print_gpu_memory_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace api
} // namespace predis