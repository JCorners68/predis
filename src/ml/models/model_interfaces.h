#pragma once

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <unordered_map>

namespace predis {
namespace ml {

// Forward declarations
struct FeatureVector;

// Prediction result structure
struct PredictionResult {
    float probability;          // Probability of access in next time window
    float confidence;           // Model confidence in prediction
    bool should_prefetch;       // Binary decision based on thresholds
    float expected_time_ms;     // Expected time until next access
    
    // Additional metadata
    std::string model_name;
    float inference_time_ms;
};

// Training configuration
struct TrainingConfig {
    size_t batch_size = 32;
    size_t epochs = 10;
    float learning_rate = 0.001f;
    float validation_split = 0.2f;
    float early_stopping_patience = 3;
    bool use_gpu = true;
    size_t max_sequence_length = 100;
    
    // Model-specific parameters
    std::unordered_map<std::string, float> hyperparams;
};

// Training result
struct TrainingResult {
    float final_loss;
    float validation_accuracy;
    float training_time_seconds;
    size_t epochs_trained;
    std::vector<float> loss_history;
    std::vector<float> validation_history;
};

// Base interface for all ML models
class IPredictiveModel {
public:
    virtual ~IPredictiveModel() = default;
    
    // Training interface
    virtual TrainingResult train(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        const TrainingConfig& config = TrainingConfig{}) = 0;
    
    // Incremental training for online learning
    virtual void partial_fit(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets) = 0;
    
    // Prediction interface
    virtual PredictionResult predict(
        const FeatureVector& features,
        float confidence_threshold = 0.7f) = 0;
    
    // Batch prediction for efficiency
    virtual std::vector<PredictionResult> predict_batch(
        const std::vector<FeatureVector>& features,
        float confidence_threshold = 0.7f) = 0;
    
    // Model persistence
    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
    
    // Model metadata
    virtual std::string get_name() const = 0;
    virtual size_t get_model_size_bytes() const = 0;
    virtual bool supports_gpu() const = 0;
    
    // Performance metrics
    virtual float get_inference_latency_ms() const = 0;
    virtual void reset_performance_metrics() = 0;
};

// Model factory for creating different model types
class ModelFactory {
public:
    enum class ModelType {
        LSTM,
        XGBoost,
        NGBoost,
        Ensemble
    };
    
    static std::unique_ptr<IPredictiveModel> create_model(
        ModelType type,
        const TrainingConfig& config = TrainingConfig{});
    
    static std::vector<ModelType> get_available_models();
    static std::string model_type_to_string(ModelType type);
};

// Model performance monitor
class ModelPerformanceMonitor {
public:
    struct PerformanceMetrics {
        float avg_inference_latency_ms;
        float p99_inference_latency_ms;
        float prediction_accuracy;
        float false_positive_rate;
        float false_negative_rate;
        size_t total_predictions;
        std::chrono::steady_clock::time_point start_time;
    };
    
    void record_prediction(
        const PredictionResult& prediction,
        bool actual_accessed,
        float inference_time_ms);
    
    PerformanceMetrics get_metrics() const;
    void reset();
    
private:
    mutable std::mutex metrics_mutex_;
    std::vector<float> inference_latencies_;
    size_t true_positives_ = 0;
    size_t true_negatives_ = 0;
    size_t false_positives_ = 0;
    size_t false_negatives_ = 0;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace ml
} // namespace predis