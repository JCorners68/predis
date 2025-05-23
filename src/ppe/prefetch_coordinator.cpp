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

#include "prefetch_coordinator.h"
#include "../ml/inference_engine.h"
#include "../ml/feature_engineering.h"
#include "../ml/models/model_interfaces.h"
#include "../ml/models/ensemble_model.h"
#include "../logger/optimized_access_logger.h"
#include "../core/simple_cache_manager.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>

namespace predis {
namespace ppe {

PrefetchCoordinator::PrefetchCoordinator(const PrefetchConfig& config)
    : config_(config), current_threshold_(config.confidence_threshold) {
    stats_.start_time = std::chrono::system_clock::now();
}

PrefetchCoordinator::~PrefetchCoordinator() {
    shutdown();
}

bool PrefetchCoordinator::initialize() {
    if (running_) {
        return false;
    }
    
    // Initialize ML components if not already set
    if (!inference_engine_) {
        // Create inference engine based on configuration
        ml::InferenceEngineConfig engine_config;
        engine_config.batch_size = config_.batch_size;
        engine_config.batch_timeout_ms = config_.batch_timeout_ms;
        engine_config.use_gpu = config_.use_gpu;
        engine_config.num_worker_threads = 2;
        
        inference_engine_ = std::make_shared<ml::InferenceEngine>(engine_config);
        
        // Create and set the model
        ml::ModelConfig model_config;
        model_config.use_gpu = config_.use_gpu;
        
        std::unique_ptr<ml::BaseModel> model;
        switch (config_.model_type) {
            case ModelType::LSTM:
                model_config.model_type = "lstm";
                model_config.hidden_units = 128;
                model_config.num_layers = 2;
                model = ml::ModelFactory::createModel(model_config);
                break;
                
            case ModelType::XGBOOST:
                model_config.model_type = "xgboost";
                model_config.n_estimators = 100;
                model_config.max_depth = 6;
                model = ml::ModelFactory::createModel(model_config);
                break;
                
            case ModelType::ENSEMBLE:
            case ModelType::ADAPTIVE:
                model = ml::EnsembleModelFactory::createDefaultEnsemble();
                break;
        }
        
        inference_engine_->setModel(std::move(model));
        inference_engine_->start();
    }
    
    if (!feature_engineering_) {
        ml::FeatureConfig feature_config;
        feature_config.window_size = 100;
        feature_config.use_temporal_features = true;
        feature_config.use_frequency_features = true;
        feature_config.use_sequence_features = true;
        feature_config.use_relationship_features = true;
        
        feature_engineering_ = std::make_shared<ml::FeatureEngineering>(feature_config);
    }
    
    running_ = true;
    
    // Start prefetch worker threads
    for (int i = 0; i < config_.prefetch_threads; ++i) {
        prefetch_threads_.emplace_back(&PrefetchCoordinator::prefetchWorkerLoop, this);
    }
    
    std::cout << "PrefetchCoordinator initialized with " << config_.prefetch_threads 
              << " worker threads\n";
    
    return true;
}

void PrefetchCoordinator::shutdown() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    queue_cv_.notify_all();
    
    // Wait for worker threads
    for (auto& thread : prefetch_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    prefetch_threads_.clear();
    
    // Stop inference engine
    if (inference_engine_) {
        inference_engine_->stop();
    }
    
    std::cout << "PrefetchCoordinator shutdown complete\n";
}

void PrefetchCoordinator::setInferenceEngine(std::shared_ptr<ml::InferenceEngine> engine) {
    inference_engine_ = engine;
}

void PrefetchCoordinator::setFeatureEngineering(std::shared_ptr<ml::FeatureEngineering> feature_eng) {
    feature_engineering_ = feature_eng;
}

void PrefetchCoordinator::setAccessLogger(std::shared_ptr<logger::OptimizedAccessLogger> logger) {
    access_logger_ = logger;
}

void PrefetchCoordinator::setCacheManager(std::shared_ptr<core::SimpleCacheManager> cache) {
    cache_manager_ = cache;
}

void PrefetchCoordinator::logAccess(const AccessEvent& event) {
    // Update access history
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        auto& history = access_history_[event.key];
        history.push_back(event);
        
        // Limit history size
        if (history.size() > max_history_per_key_) {
            history.erase(history.begin(), history.begin() + (history.size() - max_history_per_key_));
        }
    }
    
    // Log to access logger if available
    if (access_logger_) {
        access_logger_->logAccess(event.key, event.timestamp, 
                                 event.was_hit ? logger::AccessType::HIT : logger::AccessType::MISS);
    }
    
    // Trigger prediction for next keys
    if (config_.enable_background_prefetch) {
        auto predicted_keys = predictNextKeys(event.key, config_.max_prefetch_keys);
        
        for (const auto& next_key : predicted_keys) {
            // Extract confidence from prediction (simplified)
            float confidence = 0.8f; // Would come from ML model
            schedulePrefetch(next_key, confidence);
        }
    }
}

void PrefetchCoordinator::processAccessBatch(const std::vector<AccessEvent>& events) {
    for (const auto& event : events) {
        logAccess(event);
    }
}

std::vector<std::string> PrefetchCoordinator::predictNextKeys(const std::string& current_key, 
                                                             size_t max_keys) {
    if (!inference_engine_ || !feature_engineering_) {
        return {};
    }
    
    // Extract features for the current key
    auto features = extractFeaturesForKey(current_key);
    if (features.empty()) {
        return {};
    }
    
    // Get predictions from ML model
    std::vector<std::vector<float>> feature_batch = {features};
    auto predictions = inference_engine_->predict(feature_batch);
    
    if (predictions.predictions.empty()) {
        return {};
    }
    
    // For demonstration, generate candidate keys based on patterns
    std::vector<std::string> predicted_keys;
    
    // Simple pattern: if key has numeric suffix, predict next numbers
    size_t pos = current_key.find_last_not_of("0123456789");
    if (pos != std::string::npos && pos < current_key.length() - 1) {
        std::string prefix = current_key.substr(0, pos + 1);
        std::string number_str = current_key.substr(pos + 1);
        
        try {
            int number = std::stoi(number_str);
            
            // Predict next sequential keys
            for (size_t i = 1; i <= max_keys && predicted_keys.size() < max_keys; ++i) {
                predicted_keys.push_back(prefix + std::to_string(number + i));
            }
        } catch (...) {
            // Not a valid number
        }
    }
    
    // Add related keys from access history
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        auto it = access_history_.find(current_key);
        if (it != access_history_.end() && !it->second.empty()) {
            for (const auto& event : it->second) {
                for (const auto& related_key : event.related_keys) {
                    if (predicted_keys.size() < max_keys && 
                        std::find(predicted_keys.begin(), predicted_keys.end(), related_key) == predicted_keys.end()) {
                        predicted_keys.push_back(related_key);
                    }
                }
            }
        }
    }
    
    stats_.predictions_made += predicted_keys.size();
    
    return predicted_keys;
}

void PrefetchCoordinator::schedulePrefetch(const std::string& key, float confidence, int priority) {
    if (!shouldPrefetch(confidence)) {
        return;
    }
    
    PrefetchRequest request;
    request.key = key;
    request.confidence = confidence;
    request.timestamp = std::chrono::high_resolution_clock::now();
    request.priority = priority;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (prefetch_queue_.size() >= config_.prefetch_queue_size) {
            // Queue is full, drop lowest priority request
            // Note: std::priority_queue doesn't support this directly
            return;
        }
        
        prefetch_queue_.push(request);
    }
    
    queue_cv_.notify_one();
}

void PrefetchCoordinator::executePrefetch(const PrefetchRequest& request) {
    if (!cache_manager_) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check if key is already in cache
    std::string value;
    if (cache_manager_->get(request.key, value)) {
        // Already in cache, no need to prefetch
        stats_.prefetch_hits++;
        updateStats(request, true);
        return;
    }
    
    // Check eviction policy
    if (!checkEvictionPolicy(request.key)) {
        return;
    }
    
    // Simulate prefetch from storage (in real implementation, would fetch from backend)
    // For now, we'll just mark it as executed
    bool success = true;
    
    if (success) {
        // In real implementation, would add to cache
        // cache_manager_->put(request.key, fetched_value);
        stats_.prefetches_executed++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float latency = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0f;
    
    // Update average latency
    float current_avg = stats_.avg_latency_ms.load();
    uint64_t count = stats_.prefetches_executed.load();
    stats_.avg_latency_ms = (current_avg * (count - 1) + latency) / count;
    
    updateStats(request, success);
    notifyPrefetchComplete(request.key, success);
}

void PrefetchCoordinator::registerPrefetchCallback(PrefetchCallback callback) {
    prefetch_callbacks_.push_back(callback);
}

void PrefetchCoordinator::registerEvictionCallback(EvictionCallback callback) {
    eviction_callbacks_.push_back(callback);
}

void PrefetchCoordinator::resetStats() {
    stats_.predictions_made = 0;
    stats_.prefetches_executed = 0;
    stats_.prefetch_hits = 0;
    stats_.prefetch_misses = 0;
    stats_.false_positives = 0;
    stats_.cache_evictions_prevented = 0;
    stats_.avg_confidence = 0.0f;
    stats_.avg_latency_ms = 0.0f;
    stats_.start_time = std::chrono::system_clock::now();
    
    // Reset A/B testing counters
    ab_control_hits_ = 0;
    ab_control_total_ = 0;
    ab_test_hits_ = 0;
    ab_test_total_ = 0;
}

float PrefetchCoordinator::getCurrentHitRate() const {
    return stats_.getHitRate();
}

void PrefetchCoordinator::updateConfig(const PrefetchConfig& config) {
    config_ = config;
    current_threshold_ = config.confidence_threshold;
    
    // Update inference engine config if needed
    if (inference_engine_) {
        ml::InferenceEngineConfig engine_config = inference_engine_->getConfig();
        engine_config.batch_size = config.batch_size;
        engine_config.batch_timeout_ms = config.batch_timeout_ms;
        engine_config.use_gpu = config.use_gpu;
        inference_engine_->updateConfig(engine_config);
    }
}

void PrefetchCoordinator::enableAdaptiveThreshold(bool enable) {
    config_.enable_adaptive_threshold = enable;
}

void PrefetchCoordinator::adjustThreshold(float delta) {
    float new_threshold = current_threshold_ + delta;
    new_threshold = std::max(0.0f, std::min(1.0f, new_threshold));
    current_threshold_ = new_threshold;
}

void PrefetchCoordinator::enableABTesting(bool enable, float test_percentage) {
    ab_testing_enabled_ = enable;
    ab_test_percentage_ = test_percentage;
    
    if (enable) {
        std::cout << "A/B testing enabled with " << (test_percentage * 100) 
                  << "% test traffic\n";
    }
}

std::pair<float, float> PrefetchCoordinator::getABTestResults() const {
    float control_hit_rate = ab_control_total_ > 0 ? 
        static_cast<float>(ab_control_hits_) / ab_control_total_ : 0.0f;
    float test_hit_rate = ab_test_total_ > 0 ? 
        static_cast<float>(ab_test_hits_) / ab_test_total_ : 0.0f;
    
    return {control_hit_rate, test_hit_rate};
}

bool PrefetchCoordinator::loadModel(const std::string& model_path) {
    if (!inference_engine_) {
        return false;
    }
    
    auto model = inference_engine_->getModel();
    if (model) {
        return model->loadModel(model_path);
    }
    
    return false;
}

bool PrefetchCoordinator::saveModel(const std::string& model_path) {
    if (!inference_engine_) {
        return false;
    }
    
    auto model = inference_engine_->getModel();
    if (model) {
        return model->saveModel(model_path);
    }
    
    return false;
}

void PrefetchCoordinator::updateModel(std::shared_ptr<ml::BaseModel> new_model) {
    if (inference_engine_) {
        inference_engine_->updateModel(std::move(new_model));
    }
}

// Private methods

void PrefetchCoordinator::prefetchWorkerLoop() {
    while (running_) {
        auto batch = collectBatch();
        
        if (!batch.empty()) {
            processPrefetchBatch(batch);
        }
    }
}

std::vector<PrefetchCoordinator::PrefetchRequest> PrefetchCoordinator::collectBatch() {
    std::vector<PrefetchRequest> batch;
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for requests or timeout
    auto timeout = std::chrono::milliseconds(static_cast<int>(config_.batch_timeout_ms));
    queue_cv_.wait_for(lock, timeout, [this] {
        return !prefetch_queue_.empty() || !running_;
    });
    
    if (!running_) {
        return batch;
    }
    
    // Collect up to batch_size requests
    while (!prefetch_queue_.empty() && batch.size() < config_.batch_size) {
        batch.push_back(prefetch_queue_.top());
        prefetch_queue_.pop();
    }
    
    return batch;
}

void PrefetchCoordinator::processPrefetchBatch(const std::vector<PrefetchRequest>& batch) {
    // Process each request in the batch
    for (const auto& request : batch) {
        if (!running_) {
            break;
        }
        
        // A/B testing logic
        if (ab_testing_enabled_) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<> dis(0.0, 1.0);
            
            bool is_test_group = dis(gen) < ab_test_percentage_;
            
            if (is_test_group) {
                // Test group: use ML prefetching
                executePrefetch(request);
                ab_test_total_++;
                // Track hit separately
            } else {
                // Control group: no prefetching
                ab_control_total_++;
            }
        } else {
            // Normal operation
            executePrefetch(request);
        }
    }
}

std::vector<float> PrefetchCoordinator::extractFeaturesForKey(const std::string& key) {
    if (!feature_engineering_) {
        return {};
    }
    
    // Get access history for the key
    std::vector<ml::AccessRecord> records;
    
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        auto it = access_history_.find(key);
        if (it != access_history_.end()) {
            for (const auto& event : it->second) {
                ml::AccessRecord record;
                record.key = event.key;
                record.timestamp = event.timestamp;
                record.access_count = 1;
                record.value_size = event.value_size;
                records.push_back(record);
            }
        }
    }
    
    if (records.empty()) {
        // No history, use default features
        return std::vector<float>(64, 0.0f); // 64-dimensional zero vector
    }
    
    // Extract features using feature engineering
    return feature_engineering_->extractFeatures(records);
}

std::vector<std::vector<float>> PrefetchCoordinator::prepareFeatureBatch(
    const std::vector<std::string>& keys) {
    
    std::vector<std::vector<float>> feature_batch;
    feature_batch.reserve(keys.size());
    
    for (const auto& key : keys) {
        feature_batch.push_back(extractFeaturesForKey(key));
    }
    
    return feature_batch;
}

void PrefetchCoordinator::updateStats(const PrefetchRequest& request, bool success) {
    // Update confidence tracking
    float current_avg_conf = stats_.avg_confidence.load();
    uint64_t count = stats_.predictions_made.load();
    if (count > 0) {
        stats_.avg_confidence = (current_avg_conf * (count - 1) + request.confidence) / count;
    }
    
    // Adapt threshold if enabled
    if (config_.enable_adaptive_threshold) {
        adaptThreshold();
    }
}

void PrefetchCoordinator::adaptThreshold() {
    // Simple adaptive threshold based on hit rate
    float current_hit_rate = getCurrentHitRate();
    float target_hit_rate = 0.8f; // Target 80% hit rate
    
    if (current_hit_rate < target_hit_rate - 0.05f) {
        // Hit rate too low, increase threshold to be more selective
        adjustThreshold(0.02f);
    } else if (current_hit_rate > target_hit_rate + 0.05f) {
        // Hit rate high, can afford to be less selective
        adjustThreshold(-0.01f);
    }
}

bool PrefetchCoordinator::shouldPrefetch(float confidence) const {
    return confidence >= current_threshold_;
}

bool PrefetchCoordinator::checkEvictionPolicy(const std::string& key) {
    // Check with registered eviction callbacks
    for (const auto& callback : eviction_callbacks_) {
        if (!callback(key)) {
            stats_.cache_evictions_prevented++;
            return false;
        }
    }
    
    return true;
}

void PrefetchCoordinator::notifyPrefetchComplete(const std::string& key, bool success) {
    // Notify registered callbacks
    for (const auto& callback : prefetch_callbacks_) {
        callback(key, success);
    }
}

// Factory implementations

std::unique_ptr<PrefetchCoordinator> PrefetchCoordinatorFactory::createDefault() {
    PrefetchCoordinator::PrefetchConfig config;
    config.model_type = PrefetchCoordinator::ModelType::ENSEMBLE;
    config.confidence_threshold = 0.7;
    config.max_prefetch_keys = 100;
    config.enable_background_prefetch = true;
    config.enable_adaptive_threshold = true;
    
    return std::make_unique<PrefetchCoordinator>(config);
}

std::unique_ptr<PrefetchCoordinator> PrefetchCoordinatorFactory::createHighAccuracy() {
    PrefetchCoordinator::PrefetchConfig config;
    config.model_type = PrefetchCoordinator::ModelType::ENSEMBLE;
    config.confidence_threshold = 0.85;
    config.max_prefetch_keys = 50;
    config.batch_size = 128;
    config.enable_adaptive_threshold = false;
    
    return std::make_unique<PrefetchCoordinator>(config);
}

std::unique_ptr<PrefetchCoordinator> PrefetchCoordinatorFactory::createLowLatency() {
    PrefetchCoordinator::PrefetchConfig config;
    config.model_type = PrefetchCoordinator::ModelType::XGBOOST;
    config.confidence_threshold = 0.6;
    config.max_prefetch_keys = 20;
    config.batch_size = 32;
    config.batch_timeout_ms = 5.0f;
    config.prefetch_threads = 4;
    
    return std::make_unique<PrefetchCoordinator>(config);
}

std::unique_ptr<PrefetchCoordinator> PrefetchCoordinatorFactory::createAdaptive() {
    PrefetchCoordinator::PrefetchConfig config;
    config.model_type = PrefetchCoordinator::ModelType::ADAPTIVE;
    config.confidence_threshold = 0.7;
    config.enable_adaptive_threshold = true;
    config.enable_background_prefetch = true;
    config.ab_test_percentage = 0.1f;
    
    return std::make_unique<PrefetchCoordinator>(config);
}

} // namespace ppe
} // namespace predis