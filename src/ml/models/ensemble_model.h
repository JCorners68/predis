#ifndef PREDIS_ML_MODELS_ENSEMBLE_MODEL_H_
#define PREDIS_ML_MODELS_ENSEMBLE_MODEL_H_

#include "model_interfaces.h"
#include "lstm_model.h"
#include "xgboost_model.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace predis {
namespace ml {

enum class EnsembleStrategy {
    AVERAGE,           // Simple average of all model predictions
    WEIGHTED_AVERAGE,  // Weighted average based on model performance
    VOTING,           // Majority voting for classification
    STACKING,         // Use meta-model to combine predictions
    DYNAMIC           // Dynamically select based on confidence
};

struct EnsembleConfig {
    EnsembleStrategy strategy = EnsembleStrategy::WEIGHTED_AVERAGE;
    std::vector<std::string> model_types;  // e.g., ["lstm", "xgboost", "ngboost"]
    std::unordered_map<std::string, float> model_weights;  // Model-specific weights
    float confidence_threshold = 0.7f;     // For dynamic strategy
    bool use_gpu = true;
    int update_frequency = 100;            // How often to recalibrate weights
};

class EnsembleModel : public BaseModel {
public:
    explicit EnsembleModel(const EnsembleConfig& config);
    ~EnsembleModel() override = default;
    
    // Core prediction interface
    std::vector<float> predict(const std::vector<std::vector<float>>& features) override;
    std::vector<float> getConfidenceScores(const std::vector<std::vector<float>>& features) override;
    
    // Training interface
    void train(const std::vector<std::vector<float>>& features,
               const std::vector<float>& labels,
               const TrainingConfig& config) override;
    
    void updateIncremental(const std::vector<std::vector<float>>& new_features,
                          const std::vector<float>& new_labels) override;
    
    // Model persistence
    bool saveModel(const std::string& path) override;
    bool loadModel(const std::string& path) override;
    
    // Metrics and monitoring
    ModelMetrics getMetrics() const override;
    std::string getModelType() const override { return "Ensemble"; }
    bool isGPUEnabled() const override { return config_.use_gpu; }
    
    // Ensemble-specific methods
    void addModel(const std::string& name, std::unique_ptr<BaseModel> model);
    void removeModel(const std::string& name);
    void updateWeights(const std::unordered_map<std::string, float>& new_weights);
    std::unordered_map<std::string, ModelMetrics> getIndividualMetrics() const;
    
    // Dynamic weight calibration
    void calibrateWeights(const std::vector<std::vector<float>>& validation_features,
                         const std::vector<float>& validation_labels);
    
    // A/B testing support
    void setABTestMode(bool enabled, const std::string& control_model = "");
    std::unordered_map<std::string, float> getABTestResults() const;
    
private:
    EnsembleConfig config_;
    std::unordered_map<std::string, std::unique_ptr<BaseModel>> models_;
    std::unordered_map<std::string, float> model_weights_;
    std::unique_ptr<BaseModel> meta_model_;  // For stacking strategy
    
    // Performance tracking
    mutable float last_inference_time_ = 0.0f;
    std::unordered_map<std::string, std::vector<float>> model_predictions_cache_;
    
    // A/B testing state
    bool ab_test_mode_ = false;
    std::string ab_control_model_;
    std::unordered_map<std::string, std::vector<float>> ab_test_metrics_;
    
    // Helper methods
    std::vector<float> combineAveragePredictions(
        const std::unordered_map<std::string, std::vector<float>>& all_predictions);
    
    std::vector<float> combineWeightedPredictions(
        const std::unordered_map<std::string, std::vector<float>>& all_predictions);
    
    std::vector<float> combineVotingPredictions(
        const std::unordered_map<std::string, std::vector<float>>& all_predictions);
    
    std::vector<float> combineStackingPredictions(
        const std::unordered_map<std::string, std::vector<float>>& all_predictions,
        const std::vector<std::vector<float>>& original_features);
    
    std::vector<float> combineDynamicPredictions(
        const std::unordered_map<std::string, std::vector<float>>& all_predictions,
        const std::unordered_map<std::string, std::vector<float>>& confidence_scores);
    
    void normalizeWeights();
    float calculateModelWeight(const ModelMetrics& metrics) const;
    
    // GPU optimization helpers
    void predictGPUBatch(const std::vector<std::vector<float>>& features,
                        std::unordered_map<std::string, std::vector<float>>& predictions);
};

// Factory for creating ensemble models with common configurations
class EnsembleModelFactory {
public:
    static std::unique_ptr<EnsembleModel> createDefaultEnsemble();
    static std::unique_ptr<EnsembleModel> createHighAccuracyEnsemble();
    static std::unique_ptr<EnsembleModel> createLowLatencyEnsemble();
    static std::unique_ptr<EnsembleModel> createAdaptiveEnsemble();
};

} // namespace ml
} // namespace predis

#endif // PREDIS_ML_MODELS_ENSEMBLE_MODEL_H_