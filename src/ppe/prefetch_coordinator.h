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
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>

namespace predis {

// Forward declarations
namespace ml {
    class InferenceEngine;
    class BaseModel;
    class FeatureEngineering;
}

namespace logger {
    class OptimizedAccessLogger;
}

namespace core {
    class SimpleCacheManager;
}

namespace ppe {

/**
 * @brief Enhanced prefetch coordinator with ML integration
 * 
 * Integrates with Story 3.4 ML models for intelligent prefetching
 */
class PrefetchCoordinator {
public:
    enum class ModelType {
        LSTM,           // From Story 3.4
        XGBOOST,        // From Story 3.4
        ENSEMBLE,       // From Story 3.4
        ADAPTIVE        // Dynamic model selection
    };

    struct PrefetchConfig {
        ModelType model_type = ModelType::ENSEMBLE;
        double confidence_threshold = 0.7;
        size_t max_prefetch_keys = 100;
        size_t prefetch_queue_size = 1000;
        size_t batch_size = 64;
        float batch_timeout_ms = 10.0f;
        bool enable_background_prefetch = true;
        bool enable_adaptive_threshold = true;
        bool use_gpu = true;
        int prefetch_threads = 2;
    };

    struct PrefetchStats {
        std::atomic<uint64_t> predictions_made{0};
        std::atomic<uint64_t> prefetches_executed{0};
        std::atomic<uint64_t> prefetch_hits{0};
        std::atomic<uint64_t> prefetch_misses{0};
        std::atomic<uint64_t> false_positives{0};
        std::atomic<uint64_t> cache_evictions_prevented{0};
        std::atomic<float> avg_confidence{0.0f};
        std::atomic<float> avg_latency_ms{0.0f};
        std::chrono::time_point<std::chrono::system_clock> start_time;
        
        float getHitRate() const {
            uint64_t total = prefetch_hits + prefetch_misses;
            return total > 0 ? static_cast<float>(prefetch_hits) / total : 0.0f;
        }
        
        float getPrecision() const {
            uint64_t total = prefetch_hits + false_positives;
            return total > 0 ? static_cast<float>(prefetch_hits) / total : 0.0f;
        }
        
        float getHitRateImprovement(float baseline_hit_rate) const {
            return getHitRate() - baseline_hit_rate;
        }
    };

    struct PrefetchRequest {
        std::string key;
        float confidence;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
        int priority;  // Higher = more important
        
        bool operator<(const PrefetchRequest& other) const {
            // Priority queue: higher priority first, then higher confidence
            if (priority != other.priority) return priority < other.priority;
            return confidence < other.confidence;
        }
    };

    struct AccessEvent {
        std::string key;
        uint64_t timestamp;
        bool was_hit;
        size_t value_size;
        std::vector<std::string> related_keys;  // For relationship tracking
    };

    // Callback types for integration
    using PrefetchCallback = std::function<void(const std::string& key, bool success)>;
    using EvictionCallback = std::function<bool(const std::string& key)>;

public:
    explicit PrefetchCoordinator(const PrefetchConfig& config = PrefetchConfig());
    ~PrefetchCoordinator();

    // Initialization and lifecycle
    bool initialize();
    void shutdown();
    bool isRunning() const { return running_; }

    // ML model management
    void setInferenceEngine(std::shared_ptr<ml::InferenceEngine> engine);
    void setFeatureEngineering(std::shared_ptr<ml::FeatureEngineering> feature_eng);
    bool loadModel(const std::string& model_path);
    bool saveModel(const std::string& model_path);
    void updateModel(std::shared_ptr<ml::BaseModel> new_model);

    // Access pattern integration
    void setAccessLogger(std::shared_ptr<logger::OptimizedAccessLogger> logger);
    void logAccess(const AccessEvent& event);
    void processAccessBatch(const std::vector<AccessEvent>& events);

    // Prefetch operations
    std::vector<std::string> predictNextKeys(const std::string& current_key, 
                                           size_t max_keys = 10);
    void schedulePrefetch(const std::string& key, float confidence, int priority = 0);
    void executePrefetch(const PrefetchRequest& request);
    
    // Cache integration
    void setCacheManager(std::shared_ptr<core::SimpleCacheManager> cache);
    void registerPrefetchCallback(PrefetchCallback callback);
    void registerEvictionCallback(EvictionCallback callback);

    // Monitoring and statistics
    PrefetchStats getStats() const { return stats_; }
    void resetStats();
    float getCurrentHitRate() const;
    float getBaselineHitRate() const { return baseline_hit_rate_; }
    void updateBaselineHitRate(float rate) { baseline_hit_rate_ = rate; }

    // Configuration
    void updateConfig(const PrefetchConfig& config);
    PrefetchConfig getConfig() const { return config_; }
    
    // Adaptive threshold management
    void enableAdaptiveThreshold(bool enable);
    float getCurrentThreshold() const { return current_threshold_; }
    void adjustThreshold(float delta);

    // A/B testing support
    void enableABTesting(bool enable, float test_percentage = 0.1f);
    std::pair<float, float> getABTestResults() const;

private:
    PrefetchConfig config_;
    std::atomic<bool> running_{false};
    
    // ML components
    std::shared_ptr<ml::InferenceEngine> inference_engine_;
    std::shared_ptr<ml::FeatureEngineering> feature_engineering_;
    std::shared_ptr<logger::OptimizedAccessLogger> access_logger_;
    std::shared_ptr<core::SimpleCacheManager> cache_manager_;
    
    // Prefetch queue and threading
    std::priority_queue<PrefetchRequest> prefetch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> prefetch_threads_;
    
    // Access pattern tracking
    std::unordered_map<std::string, std::vector<AccessEvent>> access_history_;
    std::mutex history_mutex_;
    size_t max_history_per_key_ = 1000;
    
    // Performance tracking
    PrefetchStats stats_;
    float baseline_hit_rate_ = 0.0f;
    std::atomic<float> current_threshold_;
    
    // Callbacks
    std::vector<PrefetchCallback> prefetch_callbacks_;
    std::vector<EvictionCallback> eviction_callbacks_;
    
    // A/B testing
    bool ab_testing_enabled_ = false;
    float ab_test_percentage_ = 0.1f;
    std::atomic<uint64_t> ab_control_hits_{0};
    std::atomic<uint64_t> ab_control_total_{0};
    std::atomic<uint64_t> ab_test_hits_{0};
    std::atomic<uint64_t> ab_test_total_{0};
    
    // Internal methods
    void prefetchWorkerLoop();
    std::vector<PrefetchRequest> collectBatch();
    void processPrefetchBatch(const std::vector<PrefetchRequest>& batch);
    
    std::vector<float> extractFeaturesForKey(const std::string& key);
    std::vector<std::vector<float>> prepareFeatureBatch(
        const std::vector<std::string>& keys);
    
    void updateStats(const PrefetchRequest& request, bool success);
    void adaptThreshold();
    bool shouldPrefetch(float confidence) const;
    
    // Cache coordination
    bool checkEvictionPolicy(const std::string& key);
    void notifyPrefetchComplete(const std::string& key, bool success);
};

// Factory for creating prefetch coordinators with optimized configurations
class PrefetchCoordinatorFactory {
public:
    static std::unique_ptr<PrefetchCoordinator> createDefault();
    static std::unique_ptr<PrefetchCoordinator> createHighAccuracy();
    static std::unique_ptr<PrefetchCoordinator> createLowLatency();
    static std::unique_ptr<PrefetchCoordinator> createAdaptive();
};

} // namespace ppe
} // namespace predis