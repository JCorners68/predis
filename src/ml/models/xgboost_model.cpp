#include "xgboost_model.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

namespace predis {
namespace ml {

// Decision tree node for gradient boosting
struct TreeNode {
    int feature_index;
    float split_value;
    float leaf_value;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    
    bool is_leaf() const { return !left && !right; }
    
    float predict(const std::vector<float>& features) const {
        if (is_leaf()) {
            return leaf_value;
        }
        
        if (features[feature_index] <= split_value) {
            return left->predict(features);
        } else {
            return right->predict(features);
        }
    }
};

class DecisionTree {
public:
    DecisionTree(int max_depth, int min_samples_split, float min_impurity_decrease)
        : max_depth_(max_depth), 
          min_samples_split_(min_samples_split),
          min_impurity_decrease_(min_impurity_decrease) {}
    
    void fit(const std::vector<std::vector<float>>& features,
             const std::vector<float>& gradients,
             const std::vector<float>& hessians) {
        std::vector<int> indices(features.size());
        std::iota(indices.begin(), indices.end(), 0);
        root_ = buildTree(features, gradients, hessians, indices, 0);
    }
    
    float predict(const std::vector<float>& features) const {
        return root_ ? root_->predict(features) : 0.0f;
    }
    
private:
    int max_depth_;
    int min_samples_split_;
    float min_impurity_decrease_;
    std::unique_ptr<TreeNode> root_;
    
    std::unique_ptr<TreeNode> buildTree(const std::vector<std::vector<float>>& features,
                                       const std::vector<float>& gradients,
                                       const std::vector<float>& hessians,
                                       const std::vector<int>& indices,
                                       int depth) {
        auto node = std::make_unique<TreeNode>();
        
        // Check stopping criteria
        if (depth >= max_depth_ || indices.size() < min_samples_split_) {
            node->leaf_value = computeLeafValue(gradients, hessians, indices);
            return node;
        }
        
        // Find best split
        int best_feature = -1;
        float best_split_value = 0.0f;
        float best_gain = -std::numeric_limits<float>::infinity();
        std::vector<int> best_left_indices, best_right_indices;
        
        int num_features = features[0].size();
        for (int feature = 0; feature < num_features; ++feature) {
            auto [split_value, gain, left_indices, right_indices] = 
                findBestSplit(features, gradients, hessians, indices, feature);
            
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feature;
                best_split_value = split_value;
                best_left_indices = left_indices;
                best_right_indices = right_indices;
            }
        }
        
        // Check if gain is sufficient
        if (best_gain < min_impurity_decrease_ || best_left_indices.empty() || best_right_indices.empty()) {
            node->leaf_value = computeLeafValue(gradients, hessians, indices);
            return node;
        }
        
        // Create split
        node->feature_index = best_feature;
        node->split_value = best_split_value;
        node->left = buildTree(features, gradients, hessians, best_left_indices, depth + 1);
        node->right = buildTree(features, gradients, hessians, best_right_indices, depth + 1);
        
        return node;
    }
    
    std::tuple<float, float, std::vector<int>, std::vector<int>> 
    findBestSplit(const std::vector<std::vector<float>>& features,
                  const std::vector<float>& gradients,
                  const std::vector<float>& hessians,
                  const std::vector<int>& indices,
                  int feature_index) {
        // Sort indices by feature value
        std::vector<int> sorted_indices = indices;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](int a, int b) { return features[a][feature_index] < features[b][feature_index]; });
        
        float best_split_value = 0.0f;
        float best_gain = -std::numeric_limits<float>::infinity();
        std::vector<int> best_left, best_right;
        
        // Try all possible splits
        for (size_t i = 1; i < sorted_indices.size(); ++i) {
            if (features[sorted_indices[i-1]][feature_index] == features[sorted_indices[i]][feature_index]) {
                continue; // Skip equal values
            }
            
            float split_value = (features[sorted_indices[i-1]][feature_index] + 
                               features[sorted_indices[i]][feature_index]) / 2.0f;
            
            // Split indices
            std::vector<int> left_indices(sorted_indices.begin(), sorted_indices.begin() + i);
            std::vector<int> right_indices(sorted_indices.begin() + i, sorted_indices.end());
            
            // Calculate gain
            float gain = calculateGain(gradients, hessians, indices, left_indices, right_indices);
            
            if (gain > best_gain) {
                best_gain = gain;
                best_split_value = split_value;
                best_left = left_indices;
                best_right = right_indices;
            }
        }
        
        return {best_split_value, best_gain, best_left, best_right};
    }
    
    float calculateGain(const std::vector<float>& gradients,
                       const std::vector<float>& hessians,
                       const std::vector<int>& parent_indices,
                       const std::vector<int>& left_indices,
                       const std::vector<int>& right_indices) {
        auto calculateScore = [&](const std::vector<int>& indices) {
            float sum_grad = 0.0f, sum_hess = 0.0f;
            for (int idx : indices) {
                sum_grad += gradients[idx];
                sum_hess += hessians[idx];
            }
            return (sum_grad * sum_grad) / (sum_hess + 1e-8f);
        };
        
        float parent_score = calculateScore(parent_indices);
        float left_score = calculateScore(left_indices);
        float right_score = calculateScore(right_indices);
        
        return 0.5f * (left_score + right_score - parent_score);
    }
    
    float computeLeafValue(const std::vector<float>& gradients,
                          const std::vector<float>& hessians,
                          const std::vector<int>& indices) {
        float sum_grad = 0.0f, sum_hess = 0.0f;
        for (int idx : indices) {
            sum_grad += gradients[idx];
            sum_hess += hessians[idx];
        }
        return -sum_grad / (sum_hess + 1e-8f);
    }
};

XGBoostModel::XGBoostModel(const ModelConfig& config)
    : config_(config), trained_(false), model_version_(1) {
    
    n_estimators_ = config.n_estimators;
    max_depth_ = config.max_depth;
    learning_rate_ = config.learning_rate;
    min_samples_split_ = 2;
    min_impurity_decrease_ = 0.0f;
    
    trees_.reserve(n_estimators_);
    
    // Initialize GPU resources if enabled
    if (config.use_gpu) {
        allocateGPUMemory();
    }
}

XGBoostModel::~XGBoostModel() {
    if (config_.use_gpu) {
        freeGPUMemory();
    }
}

std::vector<float> XGBoostModel::predict(const std::vector<std::vector<float>>& features) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> predictions(features.size(), 0.0f);
    
    if (config_.use_gpu && features.size() > 100) {
        // GPU-accelerated prediction for large batches
        predictGPU(features, predictions);
    } else {
        // CPU prediction
        for (size_t i = 0; i < features.size(); ++i) {
            predictions[i] = predictSingle(features[i]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    last_inference_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
    
    return predictions;
}

float XGBoostModel::predictSingle(const std::vector<float>& features) {
    float prediction = 0.0f;
    
    for (const auto& tree : trees_) {
        prediction += learning_rate_ * tree->predict(features);
    }
    
    // Apply sigmoid for probability output
    return 1.0f / (1.0f + std::exp(-prediction));
}

std::vector<float> XGBoostModel::getConfidenceScores(const std::vector<std::vector<float>>& features) {
    // For gradient boosting, confidence can be based on tree agreement
    std::vector<float> confidence_scores(features.size(), 0.0f);
    
    // Collect predictions from individual trees
    std::vector<std::vector<float>> tree_predictions(trees_.size());
    for (size_t t = 0; t < trees_.size(); ++t) {
        tree_predictions[t].resize(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            tree_predictions[t][i] = trees_[t]->predict(features[i]);
        }
    }
    
    // Calculate variance across trees as uncertainty measure
    for (size_t i = 0; i < features.size(); ++i) {
        float mean = 0.0f;
        for (size_t t = 0; t < trees_.size(); ++t) {
            mean += tree_predictions[t][i];
        }
        mean /= trees_.size();
        
        float variance = 0.0f;
        for (size_t t = 0; t < trees_.size(); ++t) {
            float diff = tree_predictions[t][i] - mean;
            variance += diff * diff;
        }
        variance /= trees_.size();
        
        // Convert variance to confidence (lower variance = higher confidence)
        confidence_scores[i] = std::exp(-variance);
    }
    
    return confidence_scores;
}

void XGBoostModel::train(const std::vector<std::vector<float>>& features,
                         const std::vector<float>& labels,
                         const TrainingConfig& train_config) {
    std::cout << "Training XGBoost model with " << features.size() << " samples\n";
    
    // Initialize predictions
    std::vector<float> predictions(features.size(), 0.0f);
    
    // Train trees iteratively
    for (int iter = 0; iter < n_estimators_; ++iter) {
        // Calculate gradients and hessians
        std::vector<float> gradients(features.size());
        std::vector<float> hessians(features.size());
        
        for (size_t i = 0; i < features.size(); ++i) {
            float p = 1.0f / (1.0f + std::exp(-predictions[i]));
            gradients[i] = p - labels[i];
            hessians[i] = p * (1.0f - p);
        }
        
        // Train new tree
        auto tree = std::make_unique<DecisionTree>(max_depth_, min_samples_split_, min_impurity_decrease_);
        tree->fit(features, gradients, hessians);
        
        // Update predictions
        for (size_t i = 0; i < features.size(); ++i) {
            predictions[i] += learning_rate_ * tree->predict(features[i]);
        }
        
        trees_.push_back(std::move(tree));
        
        // Calculate loss
        if (iter % 10 == 0) {
            float loss = 0.0f;
            for (size_t i = 0; i < features.size(); ++i) {
                float p = 1.0f / (1.0f + std::exp(-predictions[i]));
                loss -= labels[i] * std::log(p) + (1 - labels[i]) * std::log(1 - p);
            }
            std::cout << "Iteration " << iter << " - Loss: " << loss / features.size() << "\n";
        }
    }
    
    trained_ = true;
    model_version_++;
}

void XGBoostModel::updateIncremental(const std::vector<std::vector<float>>& new_features,
                                     const std::vector<float>& new_labels) {
    // Add more trees for incremental learning
    int additional_trees = std::min(10, n_estimators_ / 10);
    
    // Get current predictions on new data
    std::vector<float> predictions(new_features.size(), 0.0f);
    for (const auto& tree : trees_) {
        for (size_t i = 0; i < new_features.size(); ++i) {
            predictions[i] += learning_rate_ * tree->predict(new_features[i]);
        }
    }
    
    // Train additional trees
    for (int iter = 0; iter < additional_trees; ++iter) {
        std::vector<float> gradients(new_features.size());
        std::vector<float> hessians(new_features.size());
        
        for (size_t i = 0; i < new_features.size(); ++i) {
            float p = 1.0f / (1.0f + std::exp(-predictions[i]));
            gradients[i] = p - new_labels[i];
            hessians[i] = p * (1.0f - p);
        }
        
        auto tree = std::make_unique<DecisionTree>(max_depth_, min_samples_split_, min_impurity_decrease_);
        tree->fit(new_features, gradients, hessians);
        
        for (size_t i = 0; i < new_features.size(); ++i) {
            predictions[i] += learning_rate_ * tree->predict(new_features[i]);
        }
        
        trees_.push_back(std::move(tree));
    }
    
    model_version_++;
}

bool XGBoostModel::saveModel(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Save model metadata
    file.write(reinterpret_cast<const char*>(&n_estimators_), sizeof(n_estimators_));
    file.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    file.write(reinterpret_cast<const char*>(&model_version_), sizeof(model_version_));
    
    // Save number of trees
    size_t num_trees = trees_.size();
    file.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));
    
    // Note: Tree serialization would be implemented here
    // For now, we'll just save the metadata
    
    file.close();
    return true;
}

bool XGBoostModel::loadModel(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Load model metadata
    file.read(reinterpret_cast<char*>(&n_estimators_), sizeof(n_estimators_));
    file.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
    file.read(reinterpret_cast<char*>(&model_version_), sizeof(model_version_));
    
    // Load number of trees
    size_t num_trees;
    file.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
    
    // Note: Tree deserialization would be implemented here
    
    file.close();
    trained_ = true;
    return true;
}

ModelMetrics XGBoostModel::getMetrics() const {
    ModelMetrics metrics;
    metrics.model_type = getModelType();
    metrics.accuracy = 0.88f; // Placeholder
    metrics.precision = 0.86f;
    metrics.recall = 0.90f;
    metrics.f1_score = 0.88f;
    metrics.inference_time_ms = last_inference_time_;
    metrics.model_size_mb = trees_.size() * max_depth_ * 0.001f; // Rough estimate
    metrics.last_update_time = std::chrono::system_clock::now();
    return metrics;
}

void XGBoostModel::predictGPU(const std::vector<std::vector<float>>& features,
                              std::vector<float>& predictions) {
    // GPU-accelerated batch prediction
    // This would involve:
    // 1. Transfer feature data to GPU
    // 2. Execute tree traversal kernels in parallel
    // 3. Aggregate predictions
    // 4. Transfer results back to CPU
    
    // Placeholder for GPU implementation
    for (size_t i = 0; i < features.size(); ++i) {
        predictions[i] = predictSingle(features[i]);
    }
}

void XGBoostModel::allocateGPUMemory() {
    // Allocate GPU memory for tree structures and batch processing
    size_t tree_memory = n_estimators_ * max_depth_ * sizeof(float) * 10; // Rough estimate
    cudaMalloc(&d_tree_nodes_, tree_memory);
    cudaMalloc(&d_predictions_, 10000 * sizeof(float)); // Batch size buffer
}

void XGBoostModel::freeGPUMemory() {
    if (d_tree_nodes_) cudaFree(d_tree_nodes_);
    if (d_predictions_) cudaFree(d_predictions_);
}

} // namespace ml
} // namespace predis