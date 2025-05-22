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
 * @brief Main client interface for Predis GPU cache
 * 
 * Provides Redis-compatible API with ML extensions
 */
class PredisClient {
public:
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
        size_t memory_usage_mb = 0;
        double prefetch_hit_ratio = 0.0;
        size_t operations_per_second = 0;
    };

public:
    PredisClient();
    ~PredisClient();

    // Connection management
    bool connect(const std::string& host = "localhost", int port = 6379);
    void disconnect();
    bool is_connected() const;

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
    void flush_all();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace api
} // namespace predis