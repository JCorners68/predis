#include "adaptive_learning_system.h"
#include "models/model_interfaces.h"
#include "inference_engine.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <iostream>

namespace predis {
namespace ml {

AdaptiveLearningSystem::AdaptiveLearningSystem(const AdaptiveConfig& config)
    : config_(config) {
    stats_.last_update = std::chrono::system_clock::now();
}

AdaptiveLearningSystem::~AdaptiveLearningSystem() {
    shutdown();
}

void AdaptiveLearningSystem::initialize() {
    if (running_) return;
    
    running_ = true;
    
    // Start background threads
    learning_thread_ = std::thread(&AdaptiveLearningSystem::learningThreadLoop, this);
    monitoring_thread_ = std::thread(&AdaptiveLearningSystem::monitoringThreadLoop, this);
    
    std::cout << "Adaptive Learning System initialized\n";
    std::cout << "Learning mode: " << static_cast<int>(config_.learning_mode) << "\n";
    std::cout << "Auto-rollback: " << (config_.auto_rollback ? "enabled" : "disabled") << "\n";
}

void AdaptiveLearningSystem::shutdown() {
    if (!running_) return;
    
    running_ = false;
    learning_cv_.notify_all();
    monitoring_cv_.notify_all();
    
    if (learning_thread_.joinable()) {
        learning_thread_.join();
    }
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    std::cout << "Adaptive Learning System shutdown complete\n";
}

void AdaptiveLearningSystem::setBaseModel(std::shared_ptr<BaseModel> model) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    current_model_ = model;
    
    // Create initial version
    ModelVersion version;
    version.version_id = generateVersionId();
    version.model = model;
    version.created_time = std::chrono::system_clock::now();
    version.validation_accuracy = 0.0;
    version.production_accuracy = 0.0;
    version.samples_trained = 0;
    version.is_active = true;
    
    updateModelHistory(version);
}

void AdaptiveLearningSystem::setInferenceEngine(std::shared_ptr<InferenceEngine> engine) {
    inference_engine_ = engine;
}

std::shared_ptr<BaseModel> AdaptiveLearningSystem::getCurrentModel() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return current_model_;
}

void AdaptiveLearningSystem::addTrainingSample(const std::vector<float>& features, float label) {
    training_buffer_.add(features, label);
    stats_.total_samples++;
    
    // Trigger update if threshold reached
    if (config_.learning_mode == LearningMode::ONLINE &&
        training_buffer_.size() >= config_.mini_batch_size) {
        learning_cv_.notify_one();
    }
}

void AdaptiveLearningSystem::addTrainingBatch(const std::vector<std::vector<float>>& features,
                                             const std::vector<float>& labels) {
    training_buffer_.addBatch(features, labels);
    stats_.total_samples += features.size();
    
    // Check if update needed
    if (training_buffer_.size() >= config_.retraining_threshold) {
        learning_cv_.notify_one();
    }
}

void AdaptiveLearningSystem::triggerModelUpdate() {
    learning_cv_.notify_one();
}

AdaptiveLearningSystem::DriftDetectionResult AdaptiveLearningSystem::detectDrift() {
    DriftDetectionResult result;
    result.drift_type = DriftType::NONE;
    result.drift_magnitude = 0.0;
    result.confidence = 0.0;
    
    if (!drift_detection_enabled_) {
        return result;
    }
    
    // Analyze performance history
    if (performance_history_.size() < 20) {
        result.recommendation = "Insufficient data for drift detection";
        return result;
    }
    
    // Calculate statistics over sliding windows
    size_t window_size = 10;
    size_t recent_start = performance_history_.size() - window_size;
    size_t historical_end = performance_history_.size() - window_size;
    
    double recent_mean = 0.0, historical_mean = 0.0;
    
    for (size_t i = 0; i < window_size && i < historical_end; ++i) {
        historical_mean += performance_history_[i];
    }
    historical_mean /= window_size;
    
    for (size_t i = recent_start; i < performance_history_.size(); ++i) {
        recent_mean += performance_history_[i];
    }
    recent_mean /= window_size;
    
    // Calculate drift magnitude
    result.drift_magnitude = std::abs(recent_mean - historical_mean);
    
    // Classify drift type
    if (result.drift_magnitude > config_.drift_threshold * 2) {
        result.drift_type = DriftType::SUDDEN;
        result.confidence = 0.9;
        result.recommendation = "Immediate model retraining recommended";
    } else if (result.drift_magnitude > config_.drift_threshold) {
        result.drift_type = DriftType::GRADUAL;
        result.confidence = 0.7;
        result.recommendation = "Schedule model retraining";
    } else {
        // Check for recurring patterns
        bool has_pattern = false;
        for (size_t i = 0; i < performance_history_.size() - 20; i += 10) {
            double pattern_diff = std::abs(performance_history_[i] - recent_mean);
            if (pattern_diff < config_.drift_threshold * 0.5) {
                has_pattern = true;
                break;
            }
        }
        
        if (has_pattern) {
            result.drift_type = DriftType::RECURRING;
            result.confidence = 0.6;
            result.recommendation = "Consider seasonal model adjustments";
        }
    }
    
    // Update state
    if (result.drift_type != DriftType::NONE) {
        last_drift_type_ = result.drift_type;
        stats_.drift_detections++;
        stats_.last_drift_detection = std::chrono::system_clock::now();
    }
    
    return result;
}

void AdaptiveLearningSystem::enableDriftDetection(bool enable) {
    drift_detection_enabled_ = enable;
}

std::string AdaptiveLearningSystem::saveCurrentModel() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (!current_model_) {
        return "";
    }
    
    std::string version_id = generateVersionId();
    std::string path = getModelPath(version_id);
    
    if (current_model_->saveModel(path)) {
        // Update version history
        for (auto& version : model_history_) {
            if (version.is_active) {
                version.version_id = version_id;
                break;
            }
        }
        return version_id;
    }
    
    return "";
}

bool AdaptiveLearningSystem::rollbackToVersion(const std::string& version_id) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // Find version in history
    for (const auto& version : model_history_) {
        if (version.version_id == version_id) {
            current_model_ = version.model;
            
            // Update active status
            for (auto& v : model_history_) {
                v.is_active = (v.version_id == version_id);
            }
            
            stats_.rollbacks_triggered++;
            
            // Update inference engine if available
            if (inference_engine_) {
                inference_engine_->updateModel(current_model_);
            }
            
            std::cout << "Rolled back to model version: " << version_id << "\n";
            return true;
        }
    }
    
    return false;
}

bool AdaptiveLearningSystem::deployModel(std::shared_ptr<BaseModel> new_model,
                                        const std::string& version_id) {
    if (!validateModel(new_model)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // Create new version
    ModelVersion version;
    version.version_id = version_id.empty() ? generateVersionId() : version_id;
    version.model = new_model;
    version.created_time = std::chrono::system_clock::now();
    version.validation_accuracy = 0.0;
    version.production_accuracy = 0.0;
    version.samples_trained = training_buffer_.size();
    version.is_active = true;
    
    // Deactivate current version
    for (auto& v : model_history_) {
        v.is_active = false;
    }
    
    // Deploy
    current_model_ = new_model;
    updateModelHistory(version);
    
    // Update inference engine
    if (inference_engine_) {
        inference_engine_->updateModel(new_model);
    }
    
    std::cout << "Deployed new model version: " << version.version_id << "\n";
    return true;
}

void AdaptiveLearningSystem::enableAutoRetraining(bool enable) {
    // Control through config
    if (enable) {
        config_.learning_mode = LearningMode::HYBRID;
    } else {
        config_.learning_mode = LearningMode::OFFLINE;
    }
}

void AdaptiveLearningSystem::setRetrainingSchedule(std::chrono::minutes interval) {
    config_.update_interval = interval;
}

bool AdaptiveLearningSystem::isRetrainingNeeded() const {
    // Check multiple conditions
    if (training_buffer_.size() >= config_.retraining_threshold) {
        return true;
    }
    
    if (isPerformanceDegraded()) {
        return true;
    }
    
    if (last_drift_type_ != DriftType::NONE) {
        return true;
    }
    
    // Check time since last update
    auto now = std::chrono::system_clock::now();
    auto time_since_update = std::chrono::duration_cast<std::chrono::minutes>(
        now - stats_.last_update);
    
    return time_since_update >= config_.update_interval;
}

void AdaptiveLearningSystem::updatePerformanceMetrics(double accuracy, double latency) {
    stats_.current_accuracy = accuracy;
    
    // Update performance history
    updatePerformanceHistory(accuracy);
    
    // Check for performance degradation
    if (config_.auto_rollback && isPerformanceDegraded()) {
        // Find best previous model
        double best_accuracy = 0.0;
        std::string best_version;
        
        for (const auto& version : model_history_) {
            if (!version.is_active && version.production_accuracy > best_accuracy) {
                best_accuracy = version.production_accuracy;
                best_version = version.version_id;
            }
        }
        
        if (!best_version.empty() && best_accuracy > accuracy) {
            std::cout << "Performance degradation detected. Rolling back...\n";
            rollbackToVersion(best_version);
        }
    }
}

void AdaptiveLearningSystem::startABTest(std::shared_ptr<BaseModel> test_model,
                                        double test_percentage) {
    test_model_ = test_model;
    ab_test_percentage_ = test_percentage;
    
    // Reset counters
    ab_control_correct_ = 0;
    ab_control_total_ = 0;
    ab_test_correct_ = 0;
    ab_test_total_ = 0;
    
    std::cout << "Started A/B test with " << (test_percentage * 100) << "% test traffic\n";
}

void AdaptiveLearningSystem::stopABTest(bool deploy_test_model) {
    if (!test_model_) return;
    
    auto [control_accuracy, test_accuracy] = getABTestResults();
    
    std::cout << "A/B Test Results:\n";
    std::cout << "  Control: " << (control_accuracy * 100) << "% accuracy\n";
    std::cout << "  Test: " << (test_accuracy * 100) << "% accuracy\n";
    
    if (deploy_test_model && test_accuracy > control_accuracy) {
        deployModel(test_model_);
    }
    
    test_model_.reset();
    ab_test_percentage_ = 0.0;
}

std::pair<double, double> AdaptiveLearningSystem::getABTestResults() const {
    double control_accuracy = ab_control_total_ > 0 ?
        static_cast<double>(ab_control_correct_) / ab_control_total_ : 0.0;
    double test_accuracy = ab_test_total_ > 0 ?
        static_cast<double>(ab_test_correct_) / ab_test_total_ : 0.0;
    
    return {control_accuracy, test_accuracy};
}

// Private methods

void AdaptiveLearningSystem::learningThreadLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(model_mutex_);
        
        // Wait for training data or timeout
        learning_cv_.wait_for(lock, config_.update_interval, [this] {
            return !running_ || isRetrainingNeeded();
        });
        
        if (!running_) break;
        
        if (isRetrainingNeeded()) {
            lock.unlock();
            performModelUpdate();
        }
    }
}

void AdaptiveLearningSystem::monitoringThreadLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(model_mutex_);
        
        // Check every minute
        monitoring_cv_.wait_for(lock, std::chrono::minutes(1));
        
        if (!running_) break;
        
        // Detect drift
        auto drift_result = detectDrift();
        if (drift_result.drift_type != DriftType::NONE) {
            std::cout << "Drift detected: " << static_cast<int>(drift_result.drift_type)
                     << ", magnitude: " << drift_result.drift_magnitude << "\n";
            
            // Trigger retraining if needed
            if (drift_result.drift_type == DriftType::SUDDEN) {
                learning_cv_.notify_one();
            }
        }
    }
}

void AdaptiveLearningSystem::performModelUpdate() {
    auto [features, labels] = training_buffer_.getAndClear();
    
    if (features.empty()) return;
    
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (!current_model_) return;
    
    std::cout << "Performing model update with " << features.size() << " samples\n";
    
    // Clone current model for safety
    auto new_model = current_model_;  // In practice, would deep clone
    
    // Update based on learning mode
    if (config_.learning_mode == LearningMode::ONLINE ||
        config_.learning_mode == LearningMode::HYBRID) {
        // Incremental update
        new_model->updateIncremental(features, labels);
    } else {
        // Full retraining
        TrainingConfig train_config;
        train_config.num_epochs = 10;
        train_config.batch_size = 32;
        new_model->train(features, labels, train_config);
    }
    
    // Validate new model
    if (validateModel(new_model)) {
        deployModel(new_model);
        stats_.updates_performed++;
        stats_.last_update = std::chrono::system_clock::now();
    }
}

bool AdaptiveLearningSystem::validateModel(std::shared_ptr<BaseModel> model) {
    if (!model) return false;
    
    // Basic validation - in practice would use validation set
    auto metrics = model->getMetrics();
    
    // Check minimum performance
    if (metrics.accuracy < config_.performance_threshold) {
        std::cout << "Model validation failed: accuracy " << metrics.accuracy
                 << " below threshold " << config_.performance_threshold << "\n";
        return false;
    }
    
    return true;
}

void AdaptiveLearningSystem::updateModelHistory(const ModelVersion& version) {
    model_history_.push_back(version);
    
    // Maintain history size
    while (model_history_.size() > config_.model_history_size) {
        model_history_.pop_front();
    }
}

void AdaptiveLearningSystem::updatePerformanceHistory(double accuracy) {
    performance_history_.push_back(accuracy);
    
    // Maintain reasonable history size
    if (performance_history_.size() > 1000) {
        performance_history_.erase(performance_history_.begin(),
                                 performance_history_.begin() + 500);
    }
    
    // Update baseline if needed
    if (performance_history_.size() == 1) {
        stats_.baseline_accuracy = accuracy;
    }
}

bool AdaptiveLearningSystem::isPerformanceDegraded() const {
    if (performance_history_.size() < 10) return false;
    
    // Compare recent performance to baseline
    double recent_avg = 0.0;
    size_t count = std::min(size_t(10), performance_history_.size());
    for (size_t i = performance_history_.size() - count; i < performance_history_.size(); ++i) {
        recent_avg += performance_history_[i];
    }
    recent_avg /= count;
    
    return recent_avg < stats_.baseline_accuracy - config_.performance_threshold;
}

std::string AdaptiveLearningSystem::generateVersionId() const {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "v" << timestamp << "_" << stats_.updates_performed.load();
    return ss.str();
}

std::string AdaptiveLearningSystem::getModelPath(const std::string& version_id) const {
    return "./models/" + version_id + ".model";
}

// TrainingBuffer implementation

void AdaptiveLearningSystem::TrainingBuffer::add(const std::vector<float>& feature, float label) {
    std::lock_guard<std::mutex> lock(mutex);
    features.push_back(feature);
    labels.push_back(label);
}

void AdaptiveLearningSystem::TrainingBuffer::addBatch(
    const std::vector<std::vector<float>>& batch_features,
    const std::vector<float>& batch_labels) {
    
    std::lock_guard<std::mutex> lock(mutex);
    features.insert(features.end(), batch_features.begin(), batch_features.end());
    labels.insert(labels.end(), batch_labels.begin(), batch_labels.end());
}

size_t AdaptiveLearningSystem::TrainingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return features.size();
}

void AdaptiveLearningSystem::TrainingBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    features.clear();
    labels.clear();
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
AdaptiveLearningSystem::TrainingBuffer::getAndClear() {
    std::lock_guard<std::mutex> lock(mutex);
    auto result = std::make_pair(std::move(features), std::move(labels));
    features.clear();
    labels.clear();
    return result;
}

// Incremental Learning Strategies

SGDStrategy::SGDStrategy(double learning_rate, double momentum)
    : learning_rate_(learning_rate), momentum_(momentum) {}

void SGDStrategy::updateModel(BaseModel* model,
                             const std::vector<std::vector<float>>& features,
                             const std::vector<float>& labels) {
    // Simplified SGD update - in practice would update model weights
    model->updateIncremental(features, labels);
}

bool SGDStrategy::supportsModelType(const std::string& model_type) const {
    return model_type == "lstm" || model_type == "mlp";
}

// Concept Drift Detector

ConceptDriftDetector::ConceptDriftDetector(Method method, double sensitivity)
    : method_(method), sensitivity_(sensitivity), drift_detected_(false), drift_level_(0.0) {}

void ConceptDriftDetector::addSample(double error) {
    switch (method_) {
        case Method::DDM:
            detectDDM(error);
            break;
        case Method::EDDM:
            detectEDDM(error);
            break;
        case Method::ADWIN:
            detectADWIN(error);
            break;
        case Method::PAGE_HINKLEY:
            detectPageHinkley(error);
            break;
        case Method::KSWIN:
            detectKSWIN(error);
            break;
    }
}

void ConceptDriftDetector::detectADWIN(double error) {
    // Simplified ADWIN implementation
    window_.push_back(error);
    if (window_.size() > max_window_size_) {
        window_.pop_front();
    }
    
    n_++;
    sum_ += error;
    sum_squared_ += error * error;
    
    if (window_.size() < 10) return;
    
    // Check for significant change in error rate
    double mean = sum_ / n_;
    double variance = (sum_squared_ / n_) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    // Detect if recent errors significantly differ from historical
    double recent_mean = 0.0;
    size_t recent_count = std::min(size_t(10), window_.size());
    for (size_t i = window_.size() - recent_count; i < window_.size(); ++i) {
        recent_mean += window_[i];
    }
    recent_mean /= recent_count;
    
    drift_level_ = std::abs(recent_mean - mean) / (std_dev + 1e-8);
    drift_detected_ = drift_level_ > sensitivity_;
    
    if (drift_detected_) {
        DriftPoint point;
        point.sample_index = n_;
        point.drift_level = drift_level_;
        point.timestamp = std::chrono::system_clock::now();
        drift_history_.push_back(point);
    }
}

void ConceptDriftDetector::reset() {
    drift_detected_ = false;
    drift_level_ = 0.0;
    window_.clear();
    sum_ = 0.0;
    sum_squared_ = 0.0;
    n_ = 0;
}

} // namespace ml
} // namespace predis