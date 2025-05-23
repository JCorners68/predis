#pragma once

#include "model_interfaces.h"
#include "../feature_engineering.h"
#include <torch/torch.h>
#include <deque>
#include <atomic>

namespace predis {
namespace ml {

// Lightweight LSTM network optimized for RTX 5080
class LightweightLSTMNet : public torch::nn::Module {
public:
    LightweightLSTMNet(int64_t input_size = 64, 
                      int64_t hidden_size = 128, 
                      int64_t num_layers = 2,
                      float dropout = 0.1);
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::ReLU relu_{nullptr};
    
    int64_t hidden_size_;
    int64_t num_layers_;
};

// LSTM model implementation
class LSTMModel : public IPredictiveModel {
public:
    LSTMModel();
    ~LSTMModel() override;
    
    // IPredictiveModel interface
    TrainingResult train(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        const TrainingConfig& config = TrainingConfig{}) override;
    
    void partial_fit(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets) override;
    
    PredictionResult predict(
        const FeatureVector& features,
        float confidence_threshold = 0.7f) override;
    
    std::vector<PredictionResult> predict_batch(
        const std::vector<FeatureVector>& features,
        float confidence_threshold = 0.7f) override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
    std::string get_name() const override { return "LightweightLSTM"; }
    size_t get_model_size_bytes() const override;
    bool supports_gpu() const override { return true; }
    
    float get_inference_latency_ms() const override;
    void reset_performance_metrics() override;
    
private:
    // Model components
    std::shared_ptr<LightweightLSTMNet> model_;
    std::shared_ptr<torch::optim::Adam> optimizer_;
    torch::Device device_;
    
    // Configuration
    int64_t input_size_ = 64;
    int64_t hidden_size_ = 128;
    int64_t num_layers_ = 2;
    int64_t sequence_length_ = 20;
    float learning_rate_ = 0.001f;
    
    // Performance tracking
    mutable std::atomic<float> total_inference_time_ms_{0};
    mutable std::atomic<size_t> inference_count_{0};
    std::deque<float> recent_latencies_;
    mutable std::mutex latency_mutex_;
    
    // Training state
    bool is_trained_ = false;
    size_t epochs_trained_ = 0;
    
    // Helper methods
    torch::Tensor features_to_tensor(const std::vector<FeatureVector>& features);
    torch::Tensor create_sequences(torch::Tensor features, torch::Tensor targets);
    void update_latency_metrics(float latency_ms);
    float calculate_confidence(float probability);
};

// LSTM-specific training utilities
class LSTMTrainingUtils {
public:
    // Create time-series sequences from feature vectors
    static std::pair<torch::Tensor, torch::Tensor> create_training_sequences(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        size_t sequence_length);
    
    // Data augmentation for time series
    static std::pair<std::vector<FeatureVector>, std::vector<float>> augment_data(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        float noise_level = 0.05f);
    
    // Validation split for time series (respects temporal order)
    static std::tuple<std::vector<FeatureVector>, std::vector<float>,
                     std::vector<FeatureVector>, std::vector<float>>
    temporal_train_test_split(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        float test_size = 0.2f);
};

} // namespace ml
} // namespace predis