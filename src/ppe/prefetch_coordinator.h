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

namespace predis {
namespace ppe {

/**
 * @brief Main coordinator for predictive prefetching engine
 * 
 * Orchestrates ML model training, inference, and prefetching decisions
 */
class PrefetchCoordinator {
public:
    enum class ModelType {
        NGBOOST,
        QUANTILE_LSTM,
        LIGHTWEIGHT_MLP
    };

    struct PrefetchStats {
        size_t predictions_made = 0;
        size_t prefetches_executed = 0;
        size_t prefetch_hits = 0;
        double accuracy = 0.0;
        double hit_rate_improvement = 0.0;
    };

    struct AccessPattern {
        std::string key;
        uint64_t timestamp;
        bool was_hit;
        size_t value_size;
    };

public:
    PrefetchCoordinator();
    ~PrefetchCoordinator();

    // Initialization
    bool initialize(ModelType model_type = ModelType::NGBOOST);
    void shutdown();

    // Access pattern logging
    void log_access(const AccessPattern& pattern);
    void process_access_logs();

    // ML model management
    bool train_model();
    bool load_model(const std::string& model_path);
    bool save_model(const std::string& model_path);

    // Prediction and prefetching
    std::vector<std::string> predict_next_keys(double confidence_threshold = 0.7);
    void execute_prefetching(const std::vector<std::string>& keys);

    // Statistics and monitoring
    PrefetchStats get_stats() const;
    void reset_stats();

    // Configuration
    void set_confidence_threshold(double threshold);
    void set_max_prefetch_keys(size_t max_keys);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ppe
} // namespace predis