#include "ensemble_model.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <future>

namespace predis {
namespace ml {

EnsembleModel::EnsembleModel(const EnsembleConfig& config) : config_(config) {
    // Initialize default weights if not provided
    if (config_.model_weights.empty() && !config_.model_types.empty()) {
        float default_weight = 1.0f / config_.model_types.size();
        for (const auto& model_type : config_.model_types) {
            model_weights_[model_type] = default_weight;
        }
    } else {
        model_weights_ = config_.model_weights;
    }
    
    // Create meta-model for stacking strategy
    if (config_.strategy == EnsembleStrategy::STACKING) {
        ModelConfig meta_config;
        meta_config.model_type = "xgboost";
        meta_config.n_estimators = 50;
        meta_config.max_depth = 3;
        meta_config.use_gpu = config_.use_gpu;
        meta_model_ = std::make_unique<XGBoostModel>(meta_config);
    }
}

void EnsembleModel::addModel(const std::string& name, std::unique_ptr<BaseModel> model) {
    models_[name] = std::move(model);
    
    // Initialize weight if not present
    if (model_weights_.find(name) == model_weights_.end()) {
        model_weights_[name] = 1.0f;
        normalizeWeights();
    }
}

void EnsembleModel::removeModel(const std::string& name) {
    models_.erase(name);
    model_weights_.erase(name);
    normalizeWeights();
}

std::vector<float> EnsembleModel::predict(const std::vector<std::vector<float>>& features) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Collect predictions from all models
    std::unordered_map<std::string, std::vector<float>> all_predictions;
    std::unordered_map<std::string, std::vector<float>> confidence_scores;
    
    if (config_.use_gpu && features.size() > 100) {
        // GPU batch prediction
        predictGPUBatch(features, all_predictions);
    } else {
        // Parallel CPU prediction
        std::vector<std::future<std::pair<std::vector<float>, std::vector<float>>>> futures;
        
        for (const auto& [name, model] : models_) {
            futures.push_back(std::async(std::launch::async, [&model, &features]() {
                auto predictions = model->predict(features);
                auto confidences = model->getConfidenceScores(features);
                return std::make_pair(predictions, confidences);
            }));
        }
        
        // Collect results
        size_t idx = 0;
        for (const auto& [name, model] : models_) {
            auto [predictions, confidences] = futures[idx].get();
            all_predictions[name] = predictions;
            confidence_scores[name] = confidences;
            idx++;
        }
    }
    
    // Store predictions for A/B testing if enabled
    if (ab_test_mode_) {
        model_predictions_cache_ = all_predictions;
    }
    
    // Combine predictions based on strategy
    std::vector<float> final_predictions;
    switch (config_.strategy) {
        case EnsembleStrategy::AVERAGE:
            final_predictions = combineAveragePredictions(all_predictions);
            break;
        case EnsembleStrategy::WEIGHTED_AVERAGE:
            final_predictions = combineWeightedPredictions(all_predictions);
            break;
        case EnsembleStrategy::VOTING:
            final_predictions = combineVotingPredictions(all_predictions);
            break;
        case EnsembleStrategy::STACKING:
            final_predictions = combineStackingPredictions(all_predictions, features);
            break;
        case EnsembleStrategy::DYNAMIC:
            final_predictions = combineDynamicPredictions(all_predictions, confidence_scores);
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    last_inference_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
    
    return final_predictions;
}

std::vector<float> EnsembleModel::combineAveragePredictions(
    const std::unordered_map<std::string, std::vector<float>>& all_predictions) {
    
    if (all_predictions.empty()) return {};
    
    size_t num_samples = all_predictions.begin()->second.size();
    std::vector<float> combined(num_samples, 0.0f);
    
    for (const auto& [name, predictions] : all_predictions) {
        for (size_t i = 0; i < num_samples; ++i) {
            combined[i] += predictions[i];
        }
    }
    
    float num_models = all_predictions.size();
    for (auto& pred : combined) {
        pred /= num_models;
    }
    
    return combined;
}

std::vector<float> EnsembleModel::combineWeightedPredictions(
    const std::unordered_map<std::string, std::vector<float>>& all_predictions) {
    
    if (all_predictions.empty()) return {};
    
    size_t num_samples = all_predictions.begin()->second.size();
    std::vector<float> combined(num_samples, 0.0f);
    
    for (const auto& [name, predictions] : all_predictions) {
        float weight = model_weights_.at(name);
        for (size_t i = 0; i < num_samples; ++i) {
            combined[i] += weight * predictions[i];
        }
    }
    
    return combined;
}

std::vector<float> EnsembleModel::combineVotingPredictions(
    const std::unordered_map<std::string, std::vector<float>>& all_predictions) {
    
    if (all_predictions.empty()) return {};
    
    size_t num_samples = all_predictions.begin()->second.size();
    std::vector<float> combined(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        int positive_votes = 0;
        int total_votes = 0;
        
        for (const auto& [name, predictions] : all_predictions) {
            float weight = model_weights_.at(name);
            if (predictions[i] > 0.5f) {
                positive_votes += static_cast<int>(weight * 100); // Weight as vote strength
            }
            total_votes += static_cast<int>(weight * 100);
        }
        
        combined[i] = static_cast<float>(positive_votes) / total_votes;
    }
    
    return combined;
}

std::vector<float> EnsembleModel::combineStackingPredictions(
    const std::unordered_map<std::string, std::vector<float>>& all_predictions,
    const std::vector<std::vector<float>>& original_features) {
    
    if (!meta_model_ || all_predictions.empty()) return {};
    
    size_t num_samples = all_predictions.begin()->second.size();
    
    // Create meta-features: concatenate base model predictions with original features
    std::vector<std::vector<float>> meta_features(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        meta_features[i] = original_features[i];
        
        // Add predictions from each model as additional features
        for (const auto& [name, predictions] : all_predictions) {
            meta_features[i].push_back(predictions[i]);
        }
    }
    
    // Use meta-model to make final predictions
    return meta_model_->predict(meta_features);
}

std::vector<float> EnsembleModel::combineDynamicPredictions(
    const std::unordered_map<std::string, std::vector<float>>& all_predictions,
    const std::unordered_map<std::string, std::vector<float>>& confidence_scores) {
    
    if (all_predictions.empty()) return {};
    
    size_t num_samples = all_predictions.begin()->second.size();
    std::vector<float> combined(num_samples, 0.0f);
    
    for (size_t i = 0; i < num_samples; ++i) {
        float total_weight = 0.0f;
        
        // Use confidence-weighted combination
        for (const auto& [name, predictions] : all_predictions) {
            float confidence = confidence_scores.at(name)[i];
            if (confidence >= config_.confidence_threshold) {
                float dynamic_weight = model_weights_.at(name) * confidence;
                combined[i] += dynamic_weight * predictions[i];
                total_weight += dynamic_weight;
            }
        }
        
        // Fallback to weighted average if no confident predictions
        if (total_weight < 0.01f) {
            for (const auto& [name, predictions] : all_predictions) {
                combined[i] += model_weights_.at(name) * predictions[i];
            }
        } else {
            combined[i] /= total_weight;
        }
    }
    
    return combined;
}

std::vector<float> EnsembleModel::getConfidenceScores(
    const std::vector<std::vector<float>>& features) {
    
    // Collect confidence scores from all models
    std::unordered_map<std::string, std::vector<float>> all_confidences;
    
    for (const auto& [name, model] : models_) {
        all_confidences[name] = model->getConfidenceScores(features);
    }
    
    // Combine confidence scores based on model weights
    size_t num_samples = features.size();
    std::vector<float> combined_confidence(num_samples, 0.0f);
    
    for (size_t i = 0; i < num_samples; ++i) {
        for (const auto& [name, confidences] : all_confidences) {
            combined_confidence[i] += model_weights_.at(name) * confidences[i];
        }
    }
    
    return combined_confidence;
}

void EnsembleModel::train(const std::vector<std::vector<float>>& features,
                         const std::vector<float>& labels,
                         const TrainingConfig& config) {
    std::cout << "Training ensemble model with " << models_.size() << " base models\n";
    
    // Train each model independently
    std::vector<std::thread> training_threads;
    
    for (auto& [name, model] : models_) {
        training_threads.emplace_back([&model, &features, &labels, &config, &name]() {
            std::cout << "Training " << name << " model...\n";
            model->train(features, labels, config);
        });
    }
    
    // Wait for all models to finish training
    for (auto& thread : training_threads) {
        thread.join();
    }
    
    // Train meta-model for stacking
    if (config_.strategy == EnsembleStrategy::STACKING && meta_model_) {
        std::cout << "Training meta-model for stacking...\n";
        
        // Collect predictions from base models
        std::unordered_map<std::string, std::vector<float>> base_predictions;
        for (const auto& [name, model] : models_) {
            base_predictions[name] = model->predict(features);
        }
        
        // Create meta-features
        std::vector<std::vector<float>> meta_features(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            meta_features[i] = features[i];
            for (const auto& [name, predictions] : base_predictions) {
                meta_features[i].push_back(predictions[i]);
            }
        }
        
        // Train meta-model
        meta_model_->train(meta_features, labels, config);
    }
    
    // Calibrate weights if using weighted strategies
    if (config_.strategy == EnsembleStrategy::WEIGHTED_AVERAGE || 
        config_.strategy == EnsembleStrategy::DYNAMIC) {
        // Use a portion of training data for validation
        size_t val_size = features.size() / 5;
        std::vector<std::vector<float>> val_features(features.end() - val_size, features.end());
        std::vector<float> val_labels(labels.end() - val_size, labels.end());
        
        calibrateWeights(val_features, val_labels);
    }
}

void EnsembleModel::updateIncremental(const std::vector<std::vector<float>>& new_features,
                                     const std::vector<float>& new_labels) {
    // Update each model incrementally
    for (auto& [name, model] : models_) {
        model->updateIncremental(new_features, new_labels);
    }
    
    // Update meta-model if using stacking
    if (config_.strategy == EnsembleStrategy::STACKING && meta_model_) {
        std::unordered_map<std::string, std::vector<float>> base_predictions;
        for (const auto& [name, model] : models_) {
            base_predictions[name] = model->predict(new_features);
        }
        
        std::vector<std::vector<float>> meta_features(new_features.size());
        for (size_t i = 0; i < new_features.size(); ++i) {
            meta_features[i] = new_features[i];
            for (const auto& [name, predictions] : base_predictions) {
                meta_features[i].push_back(predictions[i]);
            }
        }
        
        meta_model_->updateIncremental(meta_features, new_labels);
    }
    
    // Periodically recalibrate weights
    static int update_counter = 0;
    if (++update_counter % config_.update_frequency == 0) {
        calibrateWeights(new_features, new_labels);
    }
}

void EnsembleModel::calibrateWeights(const std::vector<std::vector<float>>& validation_features,
                                    const std::vector<float>& validation_labels) {
    std::cout << "Calibrating ensemble weights...\n";
    
    // Get individual model predictions
    std::unordered_map<std::string, std::vector<float>> model_predictions;
    std::unordered_map<std::string, float> model_errors;
    
    for (const auto& [name, model] : models_) {
        auto predictions = model->predict(validation_features);
        model_predictions[name] = predictions;
        
        // Calculate error (MSE)
        float error = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i] - validation_labels[i];
            error += diff * diff;
        }
        model_errors[name] = error / predictions.size();
    }
    
    // Update weights inversely proportional to error
    for (const auto& [name, error] : model_errors) {
        model_weights_[name] = 1.0f / (error + 0.001f); // Add small constant to avoid division by zero
    }
    
    normalizeWeights();
    
    std::cout << "Updated weights: ";
    for (const auto& [name, weight] : model_weights_) {
        std::cout << name << "=" << weight << " ";
    }
    std::cout << "\n";
}

void EnsembleModel::updateWeights(const std::unordered_map<std::string, float>& new_weights) {
    model_weights_ = new_weights;
    normalizeWeights();
}

void EnsembleModel::normalizeWeights() {
    float sum = 0.0f;
    for (const auto& [name, weight] : model_weights_) {
        sum += weight;
    }
    
    if (sum > 0) {
        for (auto& [name, weight] : model_weights_) {
            weight /= sum;
        }
    }
}

bool EnsembleModel::saveModel(const std::string& path) {
    // Save ensemble configuration
    std::ofstream config_file(path + ".ensemble", std::ios::binary);
    if (!config_file.is_open()) return false;
    
    // Save strategy and weights
    int strategy = static_cast<int>(config_.strategy);
    config_file.write(reinterpret_cast<const char*>(&strategy), sizeof(strategy));
    
    size_t num_models = models_.size();
    config_file.write(reinterpret_cast<const char*>(&num_models), sizeof(num_models));
    
    // Save each model
    for (const auto& [name, model] : models_) {
        size_t name_len = name.length();
        config_file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        config_file.write(name.c_str(), name_len);
        
        float weight = model_weights_.at(name);
        config_file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        
        // Save individual model
        model->saveModel(path + "." + name);
    }
    
    config_file.close();
    
    // Save meta-model if exists
    if (meta_model_) {
        meta_model_->saveModel(path + ".meta");
    }
    
    return true;
}

bool EnsembleModel::loadModel(const std::string& path) {
    std::ifstream config_file(path + ".ensemble", std::ios::binary);
    if (!config_file.is_open()) return false;
    
    // Load strategy
    int strategy;
    config_file.read(reinterpret_cast<char*>(&strategy), sizeof(strategy));
    config_.strategy = static_cast<EnsembleStrategy>(strategy);
    
    size_t num_models;
    config_file.read(reinterpret_cast<char*>(&num_models), sizeof(num_models));
    
    // Clear existing models
    models_.clear();
    model_weights_.clear();
    
    // Load each model
    for (size_t i = 0; i < num_models; ++i) {
        size_t name_len;
        config_file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        
        std::string name(name_len, '\0');
        config_file.read(&name[0], name_len);
        
        float weight;
        config_file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        model_weights_[name] = weight;
        
        // Note: Model loading would need factory pattern to create correct model type
        // For now, this is a placeholder
    }
    
    config_file.close();
    
    // Load meta-model if exists
    if (config_.strategy == EnsembleStrategy::STACKING) {
        ModelConfig meta_config;
        meta_config.model_type = "xgboost";
        meta_model_ = std::make_unique<XGBoostModel>(meta_config);
        meta_model_->loadModel(path + ".meta");
    }
    
    return true;
}

ModelMetrics EnsembleModel::getMetrics() const {
    ModelMetrics ensemble_metrics;
    ensemble_metrics.model_type = getModelType();
    
    // Aggregate metrics from all models
    float total_accuracy = 0.0f;
    float total_precision = 0.0f;
    float total_recall = 0.0f;
    float min_inference_time = std::numeric_limits<float>::max();
    float total_model_size = 0.0f;
    
    for (const auto& [name, model] : models_) {
        auto metrics = model->getMetrics();
        float weight = model_weights_.at(name);
        
        total_accuracy += weight * metrics.accuracy;
        total_precision += weight * metrics.precision;
        total_recall += weight * metrics.recall;
        min_inference_time = std::min(min_inference_time, metrics.inference_time_ms);
        total_model_size += metrics.model_size_mb;
    }
    
    ensemble_metrics.accuracy = total_accuracy;
    ensemble_metrics.precision = total_precision;
    ensemble_metrics.recall = total_recall;
    ensemble_metrics.f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall);
    ensemble_metrics.inference_time_ms = last_inference_time_;
    ensemble_metrics.model_size_mb = total_model_size;
    ensemble_metrics.last_update_time = std::chrono::system_clock::now();
    
    return ensemble_metrics;
}

std::unordered_map<std::string, ModelMetrics> EnsembleModel::getIndividualMetrics() const {
    std::unordered_map<std::string, ModelMetrics> individual_metrics;
    
    for (const auto& [name, model] : models_) {
        individual_metrics[name] = model->getMetrics();
    }
    
    return individual_metrics;
}

void EnsembleModel::setABTestMode(bool enabled, const std::string& control_model) {
    ab_test_mode_ = enabled;
    ab_control_model_ = control_model;
    
    if (enabled) {
        ab_test_metrics_.clear();
        for (const auto& [name, model] : models_) {
            ab_test_metrics_[name] = std::vector<float>();
        }
    }
}

std::unordered_map<std::string, float> EnsembleModel::getABTestResults() const {
    std::unordered_map<std::string, float> results;
    
    if (!ab_test_mode_ || ab_test_metrics_.empty()) {
        return results;
    }
    
    // Calculate average performance for each model
    for (const auto& [name, metrics] : ab_test_metrics_) {
        if (!metrics.empty()) {
            float avg = std::accumulate(metrics.begin(), metrics.end(), 0.0f) / metrics.size();
            results[name] = avg;
        }
    }
    
    return results;
}

void EnsembleModel::predictGPUBatch(const std::vector<std::vector<float>>& features,
                                   std::unordered_map<std::string, std::vector<float>>& predictions) {
    // GPU-optimized batch prediction
    // This would coordinate GPU predictions across all models
    for (const auto& [name, model] : models_) {
        predictions[name] = model->predict(features);
    }
}

// Factory implementations
std::unique_ptr<EnsembleModel> EnsembleModelFactory::createDefaultEnsemble() {
    EnsembleConfig config;
    config.strategy = EnsembleStrategy::WEIGHTED_AVERAGE;
    config.model_types = {"lstm", "xgboost"};
    config.use_gpu = true;
    
    auto ensemble = std::make_unique<EnsembleModel>(config);
    
    // Add LSTM model
    ModelConfig lstm_config;
    lstm_config.model_type = "lstm";
    lstm_config.hidden_units = 128;
    lstm_config.num_layers = 2;
    lstm_config.use_gpu = true;
    ensemble->addModel("lstm", std::make_unique<LSTMModel>(lstm_config));
    
    // Add XGBoost model
    ModelConfig xgb_config;
    xgb_config.model_type = "xgboost";
    xgb_config.n_estimators = 100;
    xgb_config.max_depth = 6;
    xgb_config.use_gpu = true;
    ensemble->addModel("xgboost", std::make_unique<XGBoostModel>(xgb_config));
    
    return ensemble;
}

std::unique_ptr<EnsembleModel> EnsembleModelFactory::createHighAccuracyEnsemble() {
    EnsembleConfig config;
    config.strategy = EnsembleStrategy::STACKING;
    config.model_types = {"lstm", "xgboost", "lstm_deep"};
    config.use_gpu = true;
    
    auto ensemble = std::make_unique<EnsembleModel>(config);
    
    // Add multiple model variants for better accuracy
    ModelConfig lstm_config;
    lstm_config.model_type = "lstm";
    lstm_config.hidden_units = 256;
    lstm_config.num_layers = 3;
    lstm_config.use_gpu = true;
    ensemble->addModel("lstm", std::make_unique<LSTMModel>(lstm_config));
    
    ModelConfig xgb_config;
    xgb_config.model_type = "xgboost";
    xgb_config.n_estimators = 200;
    xgb_config.max_depth = 8;
    xgb_config.use_gpu = true;
    ensemble->addModel("xgboost", std::make_unique<XGBoostModel>(xgb_config));
    
    // Deep LSTM variant
    lstm_config.hidden_units = 512;
    lstm_config.num_layers = 4;
    ensemble->addModel("lstm_deep", std::make_unique<LSTMModel>(lstm_config));
    
    return ensemble;
}

std::unique_ptr<EnsembleModel> EnsembleModelFactory::createLowLatencyEnsemble() {
    EnsembleConfig config;
    config.strategy = EnsembleStrategy::DYNAMIC;
    config.model_types = {"xgboost_small"};
    config.confidence_threshold = 0.8f;
    config.use_gpu = true;
    
    auto ensemble = std::make_unique<EnsembleModel>(config);
    
    // Small, fast XGBoost model
    ModelConfig xgb_config;
    xgb_config.model_type = "xgboost";
    xgb_config.n_estimators = 50;
    xgb_config.max_depth = 4;
    xgb_config.use_gpu = true;
    ensemble->addModel("xgboost_small", std::make_unique<XGBoostModel>(xgb_config));
    
    return ensemble;
}

std::unique_ptr<EnsembleModel> EnsembleModelFactory::createAdaptiveEnsemble() {
    EnsembleConfig config;
    config.strategy = EnsembleStrategy::DYNAMIC;
    config.model_types = {"lstm", "xgboost"};
    config.update_frequency = 50;
    config.use_gpu = true;
    
    auto ensemble = std::make_unique<EnsembleModel>(config);
    
    // Models that adapt quickly
    ModelConfig lstm_config;
    lstm_config.model_type = "lstm";
    lstm_config.hidden_units = 128;
    lstm_config.num_layers = 2;
    lstm_config.learning_rate = 0.01f;
    lstm_config.use_gpu = true;
    ensemble->addModel("lstm", std::make_unique<LSTMModel>(lstm_config));
    
    ModelConfig xgb_config;
    xgb_config.model_type = "xgboost";
    xgb_config.n_estimators = 100;
    xgb_config.max_depth = 5;
    xgb_config.learning_rate = 0.1f;
    xgb_config.use_gpu = true;
    ensemble->addModel("xgboost", std::make_unique<XGBoostModel>(xgb_config));
    
    return ensemble;
}

} // namespace ml
} // namespace predis