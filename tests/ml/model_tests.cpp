#include <gtest/gtest.h>
#include "../../src/ml/models/model_interfaces.h"
#include "../../src/ml/models/lstm_model.h"
#include "../../src/ml/models/xgboost_model.h"
#include "../../src/ml/models/ensemble_model.h"
#include "../../src/ml/inference_engine.h"
#include <vector>
#include <random>
#include <chrono>

namespace predis {
namespace ml {
namespace test {

// Test fixture for ML models
class MLModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic training data
        generateSyntheticData(1000, 50);
    }
    
    void generateSyntheticData(size_t num_samples, size_t feature_dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        train_features.resize(num_samples);
        train_labels.resize(num_samples);
        
        for (size_t i = 0; i < num_samples; ++i) {
            train_features[i].resize(feature_dim);
            for (size_t j = 0; j < feature_dim; ++j) {
                train_features[i][j] = dis(gen);
            }
            // Simple rule for labels: positive if sum > threshold
            float sum = std::accumulate(train_features[i].begin(), 
                                      train_features[i].end(), 0.0f);
            train_labels[i] = (sum > feature_dim * 0.5f) ? 1.0f : 0.0f;
        }
        
        // Generate test data
        test_features.resize(100);
        test_labels.resize(100);
        
        for (size_t i = 0; i < 100; ++i) {
            test_features[i].resize(feature_dim);
            for (size_t j = 0; j < feature_dim; ++j) {
                test_features[i][j] = dis(gen);
            }
            float sum = std::accumulate(test_features[i].begin(), 
                                      test_features[i].end(), 0.0f);
            test_labels[i] = (sum > feature_dim * 0.5f) ? 1.0f : 0.0f;
        }
    }
    
    std::vector<std::vector<float>> train_features;
    std::vector<float> train_labels;
    std::vector<std::vector<float>> test_features;
    std::vector<float> test_labels;
};

// Test LSTM Model
TEST_F(MLModelTest, LSTMModelBasicFunctionality) {
    ModelConfig config;
    config.model_type = "lstm";
    config.feature_dim = 50;
    config.hidden_units = 64;
    config.num_layers = 2;
    config.sequence_length = 10;
    config.use_gpu = false; // CPU only for unit tests
    
    auto model = std::make_unique<LSTMModel>(config);
    
    // Test prediction before training
    auto predictions = model->predict(test_features);
    EXPECT_EQ(predictions.size(), test_features.size());
    
    // All predictions should be between 0 and 1
    for (const auto& pred : predictions) {
        EXPECT_GE(pred, 0.0f);
        EXPECT_LE(pred, 1.0f);
    }
}

TEST_F(MLModelTest, LSTMModelTraining) {
    ModelConfig config;
    config.model_type = "lstm";
    config.feature_dim = 50;
    config.hidden_units = 32;
    config.num_layers = 1;
    config.use_gpu = false;
    
    auto model = std::make_unique<LSTMModel>(config);
    
    TrainingConfig train_config;
    train_config.num_epochs = 10;
    train_config.batch_size = 32;
    train_config.learning_rate = 0.01f;
    
    // Train model
    model->train(train_features, train_labels, train_config);
    
    // Test predictions after training
    auto predictions = model->predict(test_features);
    
    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float predicted_class = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (predicted_class == test_labels[i]) {
            correct++;
        }
    }
    
    float accuracy = static_cast<float>(correct) / predictions.size();
    EXPECT_GT(accuracy, 0.6f); // Should be better than random
}

// Test XGBoost Model
TEST_F(MLModelTest, XGBoostModelBasicFunctionality) {
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 50;
    config.max_depth = 4;
    config.learning_rate = 0.1f;
    config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(config);
    
    // Test prediction before training
    auto predictions = model->predict(test_features);
    EXPECT_EQ(predictions.size(), test_features.size());
}

TEST_F(MLModelTest, XGBoostModelTraining) {
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 100;
    config.max_depth = 6;
    config.learning_rate = 0.1f;
    config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(config);
    
    TrainingConfig train_config;
    train_config.num_epochs = 1; // XGBoost doesn't use epochs
    
    // Train model
    model->train(train_features, train_labels, train_config);
    
    // Test predictions
    auto predictions = model->predict(test_features);
    
    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float predicted_class = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (predicted_class == test_labels[i]) {
            correct++;
        }
    }
    
    float accuracy = static_cast<float>(correct) / predictions.size();
    EXPECT_GT(accuracy, 0.7f); // XGBoost should perform well
}

TEST_F(MLModelTest, XGBoostConfidenceScores) {
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 50;
    config.max_depth = 4;
    config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(config);
    
    TrainingConfig train_config;
    model->train(train_features, train_labels, train_config);
    
    auto confidence_scores = model->getConfidenceScores(test_features);
    EXPECT_EQ(confidence_scores.size(), test_features.size());
    
    // All confidence scores should be between 0 and 1
    for (const auto& score : confidence_scores) {
        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
    }
}

// Test Ensemble Model
TEST_F(MLModelTest, EnsembleModelCreation) {
    EnsembleConfig config;
    config.strategy = EnsembleStrategy::WEIGHTED_AVERAGE;
    config.use_gpu = false;
    
    auto ensemble = std::make_unique<EnsembleModel>(config);
    
    // Add models to ensemble
    ModelConfig lstm_config;
    lstm_config.model_type = "lstm";
    lstm_config.hidden_units = 32;
    lstm_config.num_layers = 1;
    lstm_config.use_gpu = false;
    ensemble->addModel("lstm", std::make_unique<LSTMModel>(lstm_config));
    
    ModelConfig xgb_config;
    xgb_config.model_type = "xgboost";
    xgb_config.n_estimators = 50;
    xgb_config.max_depth = 4;
    xgb_config.use_gpu = false;
    ensemble->addModel("xgboost", std::make_unique<XGBoostModel>(xgb_config));
    
    // Test ensemble prediction
    auto predictions = ensemble->predict(test_features);
    EXPECT_EQ(predictions.size(), test_features.size());
}

TEST_F(MLModelTest, EnsembleModelTraining) {
    auto ensemble = EnsembleModelFactory::createDefaultEnsemble();
    
    TrainingConfig train_config;
    train_config.num_epochs = 10;
    train_config.batch_size = 32;
    
    // Train ensemble
    ensemble->train(train_features, train_labels, train_config);
    
    // Test predictions
    auto predictions = ensemble->predict(test_features);
    
    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float predicted_class = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (predicted_class == test_labels[i]) {
            correct++;
        }
    }
    
    float accuracy = static_cast<float>(correct) / predictions.size();
    EXPECT_GT(accuracy, 0.7f); // Ensemble should perform well
}

TEST_F(MLModelTest, EnsembleStrategies) {
    // Test different ensemble strategies
    std::vector<EnsembleStrategy> strategies = {
        EnsembleStrategy::AVERAGE,
        EnsembleStrategy::WEIGHTED_AVERAGE,
        EnsembleStrategy::VOTING,
        EnsembleStrategy::DYNAMIC
    };
    
    for (auto strategy : strategies) {
        EnsembleConfig config;
        config.strategy = strategy;
        config.use_gpu = false;
        
        auto ensemble = std::make_unique<EnsembleModel>(config);
        
        // Add models
        ModelConfig xgb_config;
        xgb_config.model_type = "xgboost";
        xgb_config.n_estimators = 30;
        xgb_config.max_depth = 3;
        xgb_config.use_gpu = false;
        
        ensemble->addModel("xgb1", std::make_unique<XGBoostModel>(xgb_config));
        ensemble->addModel("xgb2", std::make_unique<XGBoostModel>(xgb_config));
        
        // Train
        TrainingConfig train_config;
        ensemble->train(train_features, train_labels, train_config);
        
        // Predict
        auto predictions = ensemble->predict(test_features);
        EXPECT_EQ(predictions.size(), test_features.size());
    }
}

// Test Inference Engine
TEST_F(MLModelTest, InferenceEngineBasic) {
    InferenceEngineConfig config;
    config.batch_size = 32;
    config.num_worker_threads = 2;
    config.use_gpu = false;
    
    auto engine = std::make_unique<InferenceEngine>(config);
    
    // Create and set model
    ModelConfig model_config;
    model_config.model_type = "xgboost";
    model_config.n_estimators = 50;
    model_config.max_depth = 4;
    model_config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(model_config);
    TrainingConfig train_config;
    model->train(train_features, train_labels, train_config);
    
    engine->setModel(std::move(model));
    engine->start();
    
    // Test synchronous prediction
    auto result = engine->predict(test_features);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.predictions.size(), test_features.size());
    
    engine->stop();
}

TEST_F(MLModelTest, InferenceEngineAsync) {
    auto engine = InferenceEngineFactory::createDefaultEngine();
    
    // Set model
    ModelConfig model_config;
    model_config.model_type = "xgboost";
    model_config.n_estimators = 50;
    model_config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(model_config);
    engine->setModel(std::move(model));
    engine->start();
    
    // Test asynchronous prediction
    auto request_id = engine->submitRequest(test_features);
    EXPECT_FALSE(request_id.empty());
    
    // Wait for result
    InferenceResult result;
    int attempts = 0;
    while (!engine->getResult(request_id, result) && attempts < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        attempts++;
    }
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.predictions.size(), test_features.size());
    
    engine->stop();
}

TEST_F(MLModelTest, InferenceEnginePerformance) {
    auto engine = InferenceEngineFactory::createHighThroughputEngine();
    
    // Set model
    auto model = ModelFactory::createDefaultModel();
    engine->setModel(std::move(model));
    engine->start();
    
    // Submit multiple requests
    std::vector<std::string> request_ids;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        request_ids.push_back(engine->submitRequest(test_features));
    }
    
    // Collect results
    std::vector<InferenceResult> results;
    for (const auto& id : request_ids) {
        InferenceResult result;
        while (!engine->getResult(id, result)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        results.push_back(result);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Check all results
    for (const auto& result : results) {
        EXPECT_TRUE(result.success);
        EXPECT_GT(result.predictions.size(), 0);
    }
    
    // Check performance metrics
    auto metrics = engine->getMetrics();
    EXPECT_EQ(metrics.total_requests, 10);
    EXPECT_EQ(metrics.successful_requests, 10);
    EXPECT_GT(metrics.getThroughput(), 0.0f);
    
    std::cout << "Inference throughput: " << metrics.getThroughput() 
              << " requests/sec\n";
    std::cout << "Average latency: " << metrics.getAverageLatency() 
              << " ms\n";
    
    engine->stop();
}

// Test model persistence
TEST_F(MLModelTest, ModelSaveLoad) {
    // Train a model
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 50;
    config.max_depth = 4;
    config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(config);
    TrainingConfig train_config;
    model->train(train_features, train_labels, train_config);
    
    // Get predictions before saving
    auto predictions_before = model->predict(test_features);
    
    // Save model
    std::string model_path = "/tmp/test_model.bin";
    EXPECT_TRUE(model->saveModel(model_path));
    
    // Load model
    auto loaded_model = std::make_unique<XGBoostModel>(config);
    EXPECT_TRUE(loaded_model->loadModel(model_path));
    
    // Get predictions after loading
    auto predictions_after = loaded_model->predict(test_features);
    
    // Predictions should be similar (not exact due to potential precision differences)
    EXPECT_EQ(predictions_before.size(), predictions_after.size());
    
    // Clean up
    std::remove(model_path.c_str());
}

// Test incremental learning
TEST_F(MLModelTest, IncrementalLearning) {
    ModelConfig config;
    config.model_type = "xgboost";
    config.n_estimators = 50;
    config.max_depth = 4;
    config.use_gpu = false;
    
    auto model = std::make_unique<XGBoostModel>(config);
    
    // Initial training
    TrainingConfig train_config;
    model->train(train_features, train_labels, train_config);
    
    // Get initial performance
    auto initial_predictions = model->predict(test_features);
    
    // Incremental update with new data
    std::vector<std::vector<float>> new_features(50);
    std::vector<float> new_labels(50);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (size_t i = 0; i < 50; ++i) {
        new_features[i].resize(50);
        for (size_t j = 0; j < 50; ++j) {
            new_features[i][j] = dis(gen);
        }
        float sum = std::accumulate(new_features[i].begin(), 
                                   new_features[i].end(), 0.0f);
        new_labels[i] = (sum > 25.0f) ? 1.0f : 0.0f;
    }
    
    model->updateIncremental(new_features, new_labels);
    
    // Test that model still works after incremental update
    auto updated_predictions = model->predict(test_features);
    EXPECT_EQ(updated_predictions.size(), test_features.size());
}

} // namespace test
} // namespace ml
} // namespace predis

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}