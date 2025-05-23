#include <gtest/gtest.h>
#include "../../src/ml/adaptive_learning_system.h"
#include "../../src/ml/models/ensemble_model.h"
#include "../../src/ml/inference_engine.h"
#include <thread>
#include <chrono>
#include <random>

namespace predis {
namespace ml {
namespace test {

class AdaptiveLearningTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create adaptive learning system
        AdaptiveLearningSystem::AdaptiveConfig config;
        config.learning_mode = AdaptiveLearningSystem::LearningMode::HYBRID;
        config.mini_batch_size = 50;
        config.retraining_threshold = 500;
        config.performance_threshold = 0.7;
        config.drift_threshold = 0.05;
        config.auto_rollback = true;
        
        adaptive_system_ = std::make_unique<AdaptiveLearningSystem>(config);
        
        // Create and set base model
        auto model = EnsembleModelFactory::createDefaultEnsemble();
        adaptive_system_->setBaseModel(model);
        
        // Initialize system
        adaptive_system_->initialize();
    }
    
    void TearDown() override {
        adaptive_system_->shutdown();
    }
    
    // Generate synthetic training data
    std::pair<std::vector<float>, float> generateSample(bool drift = false) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<> dis(0.0, 1.0);
        
        std::vector<float> features(64);  // 64-dimensional features
        for (auto& f : features) {
            f = dis(gen);
        }
        
        // Simple linear relationship with noise
        float label = 0.0;
        for (size_t i = 0; i < 10; ++i) {
            label += features[i] * (drift ? 0.5 : 1.0);  // Change weights if drift
        }
        label = 1.0 / (1.0 + std::exp(-label));  // Sigmoid
        
        return {features, label};
    }
    
    std::unique_ptr<AdaptiveLearningSystem> adaptive_system_;
};

TEST_F(AdaptiveLearningTest, BasicOnlineLearning) {
    // Add training samples one by one
    for (int i = 0; i < 100; ++i) {
        auto [features, label] = generateSample();
        adaptive_system_->addTrainingSample(features, label);
    }
    
    // Wait for potential update
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check stats
    auto stats = adaptive_system_->getStats();
    EXPECT_EQ(stats.total_samples, 100);
    EXPECT_GE(stats.updates_performed, 1);  // At least one update
}

TEST_F(AdaptiveLearningTest, BatchLearning) {
    // Generate batch of training data
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    
    for (int i = 0; i < 500; ++i) {
        auto [f, l] = generateSample();
        features.push_back(f);
        labels.push_back(l);
    }
    
    // Add batch
    adaptive_system_->addTrainingBatch(features, labels);
    
    // Trigger update
    adaptive_system_->triggerModelUpdate();
    
    // Wait for update
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto stats = adaptive_system_->getStats();
    EXPECT_EQ(stats.total_samples, 500);
    EXPECT_GE(stats.updates_performed, 1);
}

TEST_F(AdaptiveLearningTest, ConceptDriftDetection) {
    // Train on normal data
    for (int i = 0; i < 200; ++i) {
        auto [features, label] = generateSample(false);
        adaptive_system_->addTrainingSample(features, label);
        
        // Simulate good performance
        adaptive_system_->updatePerformanceMetrics(0.85, 5.0);
    }
    
    // Introduce drift
    for (int i = 0; i < 100; ++i) {
        auto [features, label] = generateSample(true);  // With drift
        adaptive_system_->addTrainingSample(features, label);
        
        // Simulate degraded performance
        adaptive_system_->updatePerformanceMetrics(0.65, 5.0);
    }
    
    // Check drift detection
    auto drift_result = adaptive_system_->detectDrift();
    EXPECT_NE(drift_result.drift_type, AdaptiveLearningSystem::DriftType::NONE);
    EXPECT_GT(drift_result.drift_magnitude, 0.0);
    
    // Check if drift was detected
    EXPECT_TRUE(adaptive_system_->isDriftDetected());
}

TEST_F(AdaptiveLearningTest, ModelVersioning) {
    // Save initial version
    std::string v1 = adaptive_system_->saveCurrentModel();
    EXPECT_FALSE(v1.empty());
    
    // Add training data and update
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    for (int i = 0; i < 500; ++i) {
        auto [f, l] = generateSample();
        features.push_back(f);
        labels.push_back(l);
    }
    adaptive_system_->addTrainingBatch(features, labels);
    adaptive_system_->triggerModelUpdate();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Save new version
    std::string v2 = adaptive_system_->saveCurrentModel();
    EXPECT_FALSE(v2.empty());
    EXPECT_NE(v1, v2);
    
    // Check model history
    auto history = adaptive_system_->getModelHistory();
    EXPECT_GE(history.size(), 2);
}

TEST_F(AdaptiveLearningTest, AutomaticRollback) {
    // Get initial stats
    auto initial_stats = adaptive_system_->getStats();
    
    // Simulate good performance initially
    for (int i = 0; i < 100; ++i) {
        adaptive_system_->updatePerformanceMetrics(0.85, 5.0);
    }
    
    // Update with bad data (simulate model degradation)
    std::vector<std::vector<float>> bad_features;
    std::vector<float> bad_labels;
    for (int i = 0; i < 100; ++i) {
        bad_features.push_back(std::vector<float>(64, 0.0));  // All zeros
        bad_labels.push_back(0.5);
    }
    adaptive_system_->addTrainingBatch(bad_features, bad_labels);
    adaptive_system_->triggerModelUpdate();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Simulate very poor performance
    for (int i = 0; i < 50; ++i) {
        adaptive_system_->updatePerformanceMetrics(0.4, 5.0);  // Below threshold
    }
    
    // Check if rollback was triggered
    auto final_stats = adaptive_system_->getStats();
    EXPECT_GT(final_stats.rollbacks_triggered, initial_stats.rollbacks_triggered);
}

TEST_F(AdaptiveLearningTest, ABTesting) {
    // Create test model
    auto test_model = EnsembleModelFactory::createLowLatencyEnsemble();
    
    // Start A/B test
    adaptive_system_->startABTest(test_model, 0.5);  // 50/50 split
    
    // Simulate predictions and results
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution use_test(0.5);
    
    for (int i = 0; i < 1000; ++i) {
        // In real implementation, this would be done internally
        // Here we're just checking the API works
    }
    
    // Get results
    auto [control_accuracy, test_accuracy] = adaptive_system_->getABTestResults();
    
    // Stop test
    adaptive_system_->stopABTest(false);  // Don't deploy
    
    // Results should be initialized
    EXPECT_GE(control_accuracy, 0.0);
    EXPECT_LE(control_accuracy, 1.0);
    EXPECT_GE(test_accuracy, 0.0);
    EXPECT_LE(test_accuracy, 1.0);
}

TEST_F(AdaptiveLearningTest, RetrainingSchedule) {
    // Set short retraining interval for testing
    adaptive_system_->setRetrainingSchedule(std::chrono::minutes(1));
    
    // Initially should not need retraining
    EXPECT_FALSE(adaptive_system_->isRetrainingNeeded());
    
    // Add enough samples to trigger threshold
    for (int i = 0; i < 600; ++i) {
        auto [features, label] = generateSample();
        adaptive_system_->addTrainingSample(features, label);
    }
    
    // Now should need retraining
    EXPECT_TRUE(adaptive_system_->isRetrainingNeeded());
}

TEST_F(AdaptiveLearningTest, IncrementalLearning) {
    // Test incremental updates
    auto config = adaptive_system_->getConfig();
    config.learning_mode = AdaptiveLearningSystem::LearningMode::ONLINE;
    config.mini_batch_size = 10;
    adaptive_system_->updateConfig(config);
    
    // Add samples in small batches
    for (int batch = 0; batch < 10; ++batch) {
        for (int i = 0; i < 10; ++i) {
            auto [features, label] = generateSample();
            adaptive_system_->addTrainingSample(features, label);
        }
        
        // Small delay between batches
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Should have multiple incremental updates
    auto stats = adaptive_system_->getStats();
    EXPECT_GT(stats.updates_performed, 5);
}

TEST_F(AdaptiveLearningTest, DriftTypeClassification) {
    ConceptDriftDetector detector(ConceptDriftDetector::Method::ADWIN, 0.05);
    
    // Test gradual drift
    for (int i = 0; i < 100; ++i) {
        double error = 0.1 + (i * 0.001);  // Slowly increasing error
        detector.addSample(error);
    }
    
    EXPECT_TRUE(detector.isDriftDetected());
    EXPECT_GT(detector.getDriftLevel(), 0.0);
    
    // Reset and test sudden drift
    detector.reset();
    for (int i = 0; i < 50; ++i) {
        detector.addSample(0.1);  // Low error
    }
    for (int i = 0; i < 10; ++i) {
        detector.addSample(0.8);  // Sudden high error
    }
    
    EXPECT_TRUE(detector.isDriftDetected());
}

TEST_F(AdaptiveLearningTest, ModelDeployment) {
    // Create new model
    auto new_model = EnsembleModelFactory::createHighAccuracyEnsemble();
    
    // Deploy with custom version ID
    bool deployed = adaptive_system_->deployModel(new_model, "custom_v1");
    EXPECT_TRUE(deployed);
    
    // Check current model
    auto current = adaptive_system_->getCurrentModel();
    EXPECT_NE(current, nullptr);
    
    // Check version in history
    auto history = adaptive_system_->getModelHistory();
    bool found = false;
    for (const auto& version : history) {
        if (version.version_id == "custom_v1") {
            found = true;
            EXPECT_TRUE(version.is_active);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(AdaptiveLearningTest, PerformanceUnderContinuousLearning) {
    // Simulate continuous stream of data
    auto start_time = std::chrono::steady_clock::now();
    
    // Run for 5 seconds
    while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(5)) {
        // Generate and add samples
        auto [features, label] = generateSample();
        adaptive_system_->addTrainingSample(features, label);
        
        // Simulate performance updates
        adaptive_system_->updatePerformanceMetrics(0.8 + (rand() % 10) / 100.0, 5.0);
        
        // Small delay to simulate real-time stream
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Check final stats
    auto stats = adaptive_system_->getStats();
    EXPECT_GT(stats.total_samples, 0);
    EXPECT_GT(stats.updates_performed, 0);
    
    std::cout << "Continuous learning test results:\n";
    std::cout << "  Total samples: " << stats.total_samples << "\n";
    std::cout << "  Updates performed: " << stats.updates_performed << "\n";
    std::cout << "  Rollbacks: " << stats.rollbacks_triggered << "\n";
    std::cout << "  Drift detections: " << stats.drift_detections << "\n";
}

} // namespace test
} // namespace ml
} // namespace predis

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}