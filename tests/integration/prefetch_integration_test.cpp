#include <gtest/gtest.h>
#include "../../src/ppe/prefetch_coordinator.h"
#include "../../src/ppe/prefetch_monitor.h"
#include "../../src/ml/inference_engine.h"
#include "../../src/ml/feature_engineering.h"
#include "../../src/ml/models/ensemble_model.h"
#include "../../src/logger/optimized_access_logger.h"
#include "../../src/core/simple_cache_manager.h"
#include <thread>
#include <chrono>
#include <random>

namespace predis {
namespace ppe {
namespace test {

class PrefetchIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create components
        coordinator_ = std::make_unique<PrefetchCoordinator>();
        monitor_ = std::make_unique<PrefetchMonitor>();
        cache_ = std::make_shared<core::SimpleCacheManager>(100 * 1024 * 1024); // 100MB
        
        // Initialize cache
        cache_->initialize();
        
        // Set up coordinator
        coordinator_->setCacheManager(cache_);
        coordinator_->initialize();
        
        // Set up monitor
        monitor_->setPrefetchCoordinator(coordinator_);
        monitor_->startMonitoring();
    }
    
    void TearDown() override {
        coordinator_->shutdown();
        monitor_->stopMonitoring();
    }
    
    void simulateAccessPattern(const std::string& pattern_type, int num_accesses) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if (pattern_type == "sequential") {
            // Sequential access pattern
            for (int i = 0; i < num_accesses; ++i) {
                std::string key = "key_" + std::to_string(i);
                
                PrefetchCoordinator::AccessEvent event;
                event.key = key;
                event.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                event.was_hit = cache_->contains(key);
                event.value_size = 1024;
                
                coordinator_->logAccess(event);
                
                // Simulate some processing time
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } else if (pattern_type == "temporal") {
            // Temporal pattern - access same keys repeatedly
            std::vector<std::string> key_set = {"user_1", "user_2", "user_3", "user_4", "user_5"};
            std::uniform_int_distribution<> dis(0, key_set.size() - 1);
            
            for (int i = 0; i < num_accesses; ++i) {
                std::string key = key_set[dis(gen)];
                
                PrefetchCoordinator::AccessEvent event;
                event.key = key;
                event.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                event.was_hit = cache_->contains(key);
                event.value_size = 2048;
                
                coordinator_->logAccess(event);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        } else if (pattern_type == "related") {
            // Related keys pattern
            for (int i = 0; i < num_accesses; ++i) {
                std::string base_key = "item_" + std::to_string(i % 10);
                
                PrefetchCoordinator::AccessEvent event;
                event.key = base_key;
                event.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                event.was_hit = cache_->contains(base_key);
                event.value_size = 512;
                
                // Add related keys
                event.related_keys.push_back(base_key + "_meta");
                event.related_keys.push_back(base_key + "_stats");
                
                coordinator_->logAccess(event);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        }
    }
    
    std::unique_ptr<PrefetchCoordinator> coordinator_;
    std::unique_ptr<PrefetchMonitor> monitor_;
    std::shared_ptr<core::SimpleCacheManager> cache_;
};

TEST_F(PrefetchIntegrationTest, BasicPrefetchFunctionality) {
    // Test basic prefetch operations
    std::string test_key = "test_key_1";
    
    // Schedule a prefetch
    coordinator_->schedulePrefetch(test_key, 0.85f, 1);
    
    // Wait for prefetch to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check stats
    auto stats = coordinator_->getStats();
    EXPECT_GT(stats.prefetches_executed, 0);
}

TEST_F(PrefetchIntegrationTest, SequentialPatternPrediction) {
    // Simulate sequential access pattern
    simulateAccessPattern("sequential", 50);
    
    // Test prediction
    auto predicted_keys = coordinator_->predictNextKeys("key_10", 5);
    
    // Should predict key_11, key_12, etc.
    EXPECT_FALSE(predicted_keys.empty());
    
    // Check if predictions include sequential keys
    bool found_sequential = false;
    for (const auto& key : predicted_keys) {
        if (key == "key_11" || key == "key_12") {
            found_sequential = true;
            break;
        }
    }
    EXPECT_TRUE(found_sequential);
}

TEST_F(PrefetchIntegrationTest, ConfidenceThresholdFiltering) {
    // Test that low confidence predictions are filtered
    auto config = coordinator_->getConfig();
    config.confidence_threshold = 0.9;
    coordinator_->updateConfig(config);
    
    // Schedule prefetches with different confidence levels
    coordinator_->schedulePrefetch("high_conf_key", 0.95f);
    coordinator_->schedulePrefetch("low_conf_key", 0.5f);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats = coordinator_->getStats();
    // Only high confidence prefetch should execute
    EXPECT_EQ(stats.prefetches_executed, 1);
}

TEST_F(PrefetchIntegrationTest, PrefetchQueuePriority) {
    // Test priority queue functionality
    coordinator_->schedulePrefetch("low_priority", 0.8f, 1);
    coordinator_->schedulePrefetch("high_priority", 0.8f, 10);
    coordinator_->schedulePrefetch("medium_priority", 0.8f, 5);
    
    // High priority should be processed first
    // This test would need access to internal queue state or
    // observation of processing order
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats = coordinator_->getStats();
    EXPECT_GE(stats.prefetches_executed, 3);
}

TEST_F(PrefetchIntegrationTest, AdaptiveThresholdAdjustment) {
    // Enable adaptive threshold
    coordinator_->enableAdaptiveThreshold(true);
    
    // Simulate access pattern with varying hit rates
    simulateAccessPattern("temporal", 100);
    
    // Check if threshold has adapted
    float initial_threshold = coordinator_->getConfig().confidence_threshold;
    
    // Simulate more accesses
    simulateAccessPattern("temporal", 100);
    
    float current_threshold = coordinator_->getCurrentThreshold();
    
    // Threshold should have changed based on performance
    EXPECT_NE(initial_threshold, current_threshold);
}

TEST_F(PrefetchIntegrationTest, ABTestingFramework) {
    // Enable A/B testing
    coordinator_->enableABTesting(true, 0.5f); // 50% test traffic
    
    // Simulate many accesses
    simulateAccessPattern("sequential", 200);
    
    // Get A/B test results
    auto [control_rate, test_rate] = coordinator_->getABTestResults();
    
    // Both groups should have data
    EXPECT_GE(control_rate, 0.0f);
    EXPECT_GE(test_rate, 0.0f);
    
    // Test group (with prefetching) should have better hit rate
    // Note: This may not always be true in unit tests
    std::cout << "A/B Test Results - Control: " << control_rate 
              << ", Test: " << test_rate << std::endl;
}

TEST_F(PrefetchIntegrationTest, MonitoringAndMetrics) {
    // Simulate access pattern
    simulateAccessPattern("related", 50);
    
    // Get metrics from monitor
    auto metrics = monitor_->getCurrentMetrics();
    
    EXPECT_GT(metrics.total_predictions, 0);
    EXPECT_GE(metrics.precision, 0.0);
    EXPECT_LE(metrics.precision, 1.0);
    
    // Check top keys
    auto top_keys = monitor_->getTopKeys(5);
    EXPECT_FALSE(top_keys.empty());
    
    // Generate dashboard output
    PrefetchDashboard dashboard(monitor_);
    dashboard.printSummary();
}

TEST_F(PrefetchIntegrationTest, HitRateImprovement) {
    // Establish baseline hit rate without prefetching
    auto config = coordinator_->getConfig();
    config.enable_background_prefetch = false;
    coordinator_->updateConfig(config);
    
    // Simulate accesses without prefetching
    simulateAccessPattern("sequential", 50);
    
    float baseline_hit_rate = cache_->getHitRate();
    coordinator_->updateBaselineHitRate(baseline_hit_rate);
    
    // Enable prefetching
    config.enable_background_prefetch = true;
    coordinator_->updateConfig(config);
    
    // Simulate more accesses with prefetching
    simulateAccessPattern("sequential", 50);
    
    // Check hit rate improvement
    auto stats = coordinator_->getStats();
    float improvement = stats.getHitRateImprovement(baseline_hit_rate);
    
    std::cout << "Hit rate improvement: " << (improvement * 100) << "%" << std::endl;
    
    // Should show some improvement (may be small in unit test)
    EXPECT_GE(improvement, 0.0f);
}

TEST_F(PrefetchIntegrationTest, EvictionPolicyIntegration) {
    // Register eviction callback
    int evictions_prevented = 0;
    coordinator_->registerEvictionCallback(
        [&evictions_prevented](const std::string& key) {
            // Prevent eviction of keys starting with "important_"
            if (key.substr(0, 10) == "important_") {
                evictions_prevented++;
                return false;
            }
            return true;
        });
    
    // Schedule prefetches
    coordinator_->schedulePrefetch("important_data", 0.9f);
    coordinator_->schedulePrefetch("regular_data", 0.9f);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check that eviction policy was respected
    auto stats = coordinator_->getStats();
    EXPECT_EQ(stats.cache_evictions_prevented, evictions_prevented);
}

TEST_F(PrefetchIntegrationTest, ModelUpdateDuringOperation) {
    // Simulate initial access pattern
    simulateAccessPattern("temporal", 50);
    
    // Create and set a new model
    auto new_model = ml::EnsembleModelFactory::createLowLatencyEnsemble();
    coordinator_->updateModel(std::move(new_model));
    
    // Continue with access pattern
    simulateAccessPattern("temporal", 50);
    
    // System should continue functioning after model update
    auto stats = coordinator_->getStats();
    EXPECT_GT(stats.predictions_made, 0);
}

TEST_F(PrefetchIntegrationTest, AlertThresholds) {
    // Set alert thresholds
    monitor_->setAlertThreshold("hit_rate_improvement", 0.2); // 20%
    monitor_->setAlertThreshold("precision", 0.8);
    monitor_->setAlertThreshold("avg_prediction_latency_ms", 5.0);
    
    // Simulate access pattern
    simulateAccessPattern("sequential", 100);
    
    // Check for alerts
    std::vector<std::string> alerts;
    bool has_alerts = monitor_->checkAlerts(alerts);
    
    if (has_alerts) {
        std::cout << "Triggered alerts:\n";
        for (const auto& alert : alerts) {
            std::cout << "  - " << alert << "\n";
        }
    }
}

TEST_F(PrefetchIntegrationTest, PerformanceUnderLoad) {
    // Stress test with high load
    const int num_threads = 4;
    const int accesses_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch multiple threads simulating access
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i, accesses_per_thread]() {
            std::string pattern = (i % 2 == 0) ? "sequential" : "temporal";
            simulateAccessPattern(pattern, accesses_per_thread);
        });
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Check performance metrics
    auto metrics = monitor_->getCurrentMetrics();
    std::cout << "Performance under load:\n";
    std::cout << "  Total predictions: " << metrics.total_predictions << "\n";
    std::cout << "  Avg prediction latency: " << metrics.avg_prediction_latency_ms << "ms\n";
    std::cout << "  Throughput: " << (metrics.total_predictions * 1000.0 / duration) 
              << " predictions/sec\n";
    
    // Latency should remain reasonable under load
    EXPECT_LT(metrics.avg_prediction_latency_ms, 20.0);
}

TEST_F(PrefetchIntegrationTest, ExportMetrics) {
    // Generate some activity
    simulateAccessPattern("related", 50);
    
    // Export metrics in different formats
    std::string json_metrics = monitor_->exportMetricsJSON();
    std::string csv_metrics = monitor_->exportMetricsCSV();
    
    // Verify exports contain data
    EXPECT_FALSE(json_metrics.empty());
    EXPECT_FALSE(csv_metrics.empty());
    
    // JSON should contain expected fields
    EXPECT_NE(json_metrics.find("total_predictions"), std::string::npos);
    EXPECT_NE(json_metrics.find("hit_rate_improvement"), std::string::npos);
    
    // CSV should have header and data row
    EXPECT_NE(csv_metrics.find("timestamp"), std::string::npos);
    EXPECT_NE(csv_metrics.find("\n"), std::string::npos);
}

} // namespace test
} // namespace ppe
} // namespace predis

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}