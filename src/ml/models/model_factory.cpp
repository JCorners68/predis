#include "model_interfaces.h"
#include "lstm_model.h"
#include "xgboost_model.h"
#include "ensemble_model.h"
#include <memory>
#include <stdexcept>
#include <iostream>

namespace predis {
namespace ml {

std::unique_ptr<BaseModel> ModelFactory::createModel(const ModelConfig& config) {
    std::cout << "Creating model of type: " << config.model_type << "\n";
    
    if (config.model_type == "lstm") {
        return std::make_unique<LSTMModel>(config);
    } else if (config.model_type == "xgboost") {
        return std::make_unique<XGBoostModel>(config);
    } else if (config.model_type == "ensemble") {
        // Create ensemble with default configuration
        EnsembleConfig ensemble_config;
        ensemble_config.strategy = EnsembleStrategy::WEIGHTED_AVERAGE;
        ensemble_config.use_gpu = config.use_gpu;
        return std::make_unique<EnsembleModel>(ensemble_config);
    } else {
        throw std::invalid_argument("Unknown model type: " + config.model_type);
    }
}

std::unique_ptr<BaseModel> ModelFactory::createDefaultModel() {
    // Default to XGBoost for best balance of speed and accuracy
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 100;
    config.max_depth = 6;
    config.learning_rate = 0.1f;
    config.use_gpu = true;
    
    return createModel(config);
}

std::unique_ptr<BaseModel> ModelFactory::loadModel(const std::string& path, 
                                                   const std::string& model_type) {
    ModelConfig config;
    config.model_type = model_type;
    
    auto model = createModel(config);
    if (!model->loadModel(path)) {
        throw std::runtime_error("Failed to load model from: " + path);
    }
    
    return model;
}

ModelConfig ModelFactory::getOptimizedConfig(const std::string& model_type,
                                            const std::string& optimization_target) {
    ModelConfig config;
    config.model_type = model_type;
    
    if (model_type == "lstm") {
        if (optimization_target == "accuracy") {
            config.hidden_units = 256;
            config.num_layers = 3;
            config.sequence_length = 100;
            config.learning_rate = 0.001f;
        } else if (optimization_target == "speed") {
            config.hidden_units = 64;
            config.num_layers = 1;
            config.sequence_length = 50;
            config.learning_rate = 0.01f;
        } else { // balanced
            config.hidden_units = 128;
            config.num_layers = 2;
            config.sequence_length = 75;
            config.learning_rate = 0.005f;
        }
    } else if (model_type == "xgboost") {
        if (optimization_target == "accuracy") {
            config.n_estimators = 200;
            config.max_depth = 8;
            config.learning_rate = 0.05f;
        } else if (optimization_target == "speed") {
            config.n_estimators = 50;
            config.max_depth = 4;
            config.learning_rate = 0.3f;
        } else { // balanced
            config.n_estimators = 100;
            config.max_depth = 6;
            config.learning_rate = 0.1f;
        }
    }
    
    config.use_gpu = true;
    return config;
}

} // namespace ml
} // namespace predis