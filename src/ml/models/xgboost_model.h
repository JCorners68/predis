#pragma once

#include "model_interfaces.h"
#include "../feature_engineering.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>

namespace predis {
namespace ml {

// Decision tree node for gradient boosting
struct TreeNode {
    int feature_index = -1;         // Feature to split on (-1 for leaf)
    float split_value = 0.0f;       // Split threshold
    float leaf_value = 0.0f;        // Prediction value for leaf nodes
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    
    bool is_leaf() const { return feature_index == -1; }
};

// Lightweight gradient boosting tree
class GradientBoostingTree {
public:
    GradientBoostingTree(int max_depth = 5, float learning_rate = 0.1f);
    
    void fit(const std::vector<std::vector<float>>& features,
             const std::vector<float>& gradients,
             const std::vector<float>& hessians);
    
    float predict(const std::vector<float>& features) const;
    
    size_t get_size_bytes() const;
    
private:
    std::unique_ptr<TreeNode> root_;
    int max_depth_;
    float learning_rate_;
    
    std::unique_ptr<TreeNode> build_tree(
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& gradients,
        const std::vector<float>& hessians,
        const std::vector<int>& indices,
        int depth);
    
    std::pair<int, float> find_best_split(
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& gradients,
        const std::vector<float>& hessians,
        const std::vector<int>& indices);
    
    float calculate_leaf_value(
        const std::vector<float>& gradients,
        const std::vector<float>& hessians,
        const std::vector<int>& indices);
    
    float predict_recursive(const TreeNode* node, 
                          const std::vector<float>& features) const;
    
    size_t count_nodes(const TreeNode* node) const;
};

// Lightweight XGBoost-style model implementation
class XGBoostModel : public IPredictiveModel {
public:
    XGBoostModel();
    ~XGBoostModel() override = default;
    
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
    
    std::string get_name() const override { return "LightweightXGBoost"; }
    size_t get_model_size_bytes() const override;
    bool supports_gpu() const override { return false; } // CPU implementation
    
    float get_inference_latency_ms() const override;
    void reset_performance_metrics() override;
    
private:
    // Model components
    std::vector<std::unique_ptr<GradientBoostingTree>> trees_;
    
    // Configuration
    int n_estimators_ = 50;
    int max_depth_ = 5;
    float learning_rate_ = 0.1f;
    float subsample_ = 0.8f;
    float colsample_ = 0.8f;
    
    // Performance tracking
    mutable std::atomic<float> total_inference_time_ms_{0};
    mutable std::atomic<size_t> inference_count_{0};
    
    // Training state
    bool is_trained_ = false;
    std::vector<float> feature_importance_;
    
    // Helper methods
    std::vector<std::vector<float>> features_to_matrix(
        const std::vector<FeatureVector>& features);
    
    std::pair<std::vector<float>, std::vector<float>> compute_gradients_hessians(
        const std::vector<float>& predictions,
        const std::vector<float>& targets);
    
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    
    std::vector<int> subsample_indices(size_t n_samples);
    std::vector<int> subsample_features(size_t n_features);
    
    void update_feature_importance(const GradientBoostingTree& tree);
};

// Training utilities for XGBoost
class XGBoostTrainingUtils {
public:
    // Cross-validation with early stopping
    static std::pair<int, float> find_optimal_n_estimators(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        int max_estimators = 100,
        int cv_folds = 5);
    
    // Feature importance analysis
    static std::vector<std::pair<int, float>> get_top_features(
        const XGBoostModel& model,
        int top_k = 10);
    
    // Hyperparameter tuning
    struct HyperparamSearchResult {
        int n_estimators;
        int max_depth;
        float learning_rate;
        float best_score;
    };
    
    static HyperparamSearchResult grid_search(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        const std::vector<int>& n_estimators_grid,
        const std::vector<int>& max_depth_grid,
        const std::vector<float>& learning_rate_grid);
};

} // namespace ml
} // namespace predis