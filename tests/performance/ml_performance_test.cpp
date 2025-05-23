#include <gtest/gtest.h>
#include "../../src/benchmarks/ml_performance_validator.h"
#include "../../src/ppe/prefetch_coordinator.h"
#include "../../src/ppe/prefetch_monitor.h"
#include "../../src/core/simple_cache_manager.h"
#include "../../src/ml/inference_engine.h"
#include "../../src/ml/models/ensemble_model.h"
#include <iostream>
#include <iomanip>

namespace predis {
namespace benchmarks {
namespace test {

class MLPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create components
        cache_ = std::make_shared<core::SimpleCacheManager>(100 * 1024 * 1024); // 100MB
        coordinator_ = ppe::PrefetchCoordinatorFactory::createDefault();
        monitor_ = std::make_shared<ppe::PrefetchMonitor>();
        validator_ = std::make_shared<MLPerformanceValidator>();
        
        // Initialize
        cache_->initialize();
        coordinator_->setCacheManager(cache_);
        coordinator_->initialize();
        
        monitor_->setPrefetchCoordinator(coordinator_);
        monitor_->startMonitoring();
        
        validator_->setCacheManager(cache_);
        validator_->setPrefetchCoordinator(coordinator_);
        validator_->setPrefetchMonitor(monitor_);
    }
    
    void TearDown() override {
        coordinator_->shutdown();
        monitor_->stopMonitoring();
    }
    
    std::shared_ptr<core::SimpleCacheManager> cache_;
    std::shared_ptr<ppe::PrefetchCoordinator> coordinator_;
    std::shared_ptr<ppe::PrefetchMonitor> monitor_;
    std::shared_ptr<MLPerformanceValidator> validator_;
};

TEST_F(MLPerformanceTest, ValidateHitRateImprovement) {
    std::cout << "\n========================================\n";
    std::cout << "Epic 3 Success Criteria: Hit Rate Test\n";
    std::cout << "Target: 20%+ improvement\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 50000;
    config.num_keys = 5000;
    config.workload_type = MLPerformanceValidator::WorkloadType::SEQUENTIAL;
    config.enable_ml = true;
    config.warmup_operations = 1000;
    
    auto result = validator_->validateHitRateImprovement(config);
    
    // Print detailed results
    std::cout << "\n--- Detailed Results ---\n";
    std::cout << "Baseline hit rate: " << std::fixed << std::setprecision(2) 
              << (result.baseline_hit_rate * 100) << "%\n";
    std::cout << "ML-enhanced hit rate: " << (result.ml_hit_rate * 100) << "%\n";
    std::cout << "Absolute improvement: " << (result.hit_rate_improvement * 100) << "%\n";
    std::cout << "Relative improvement: " << result.hit_rate_improvement_percentage << "%\n";
    std::cout << "Statistical significance: " << (result.statistically_significant ? "Yes" : "No") 
              << " (p=" << result.p_value << ")\n";
    
    // Epic 3 success criteria
    EXPECT_TRUE(result.meets_hit_rate_target) 
        << "Hit rate improvement (" << result.hit_rate_improvement_percentage 
        << "%) does not meet 20% target";
}

TEST_F(MLPerformanceTest, ValidateInferenceLatency) {
    std::cout << "\n========================================\n";
    std::cout << "Epic 3 Success Criteria: Latency Test\n";
    std::cout << "Target: <10ms inference latency\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 10000;
    config.num_keys = 1000;
    
    auto result = validator_->validateInferenceLatency(config);
    
    // Print latency distribution
    std::cout << "\n--- Latency Distribution ---\n";
    std::cout << "Min: " << result.p50_latency_ms << "ms (estimated)\n";
    std::cout << "P50: " << result.p50_latency_ms << "ms\n";
    std::cout << "P95: " << result.p95_latency_ms << "ms\n";
    std::cout << "P99: " << result.p99_latency_ms << "ms\n";
    std::cout << "Max: " << result.max_inference_latency_ms << "ms\n";
    
    // Epic 3 success criteria
    EXPECT_TRUE(result.meets_latency_target)
        << "Average inference latency (" << result.avg_inference_latency_ms 
        << "ms) exceeds 10ms target";
}

TEST_F(MLPerformanceTest, ValidateSystemOverhead) {
    std::cout << "\n========================================\n";
    std::cout << "Epic 3 Success Criteria: Overhead Test\n";
    std::cout << "Target: <1% system overhead\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 100000;
    config.num_keys = 10000;
    config.enable_ml = true;
    
    auto result = validator_->validateSystemOverhead(config);
    
    // Epic 3 success criteria
    EXPECT_TRUE(result.meets_overhead_target)
        << "CPU overhead (" << result.cpu_overhead_percentage 
        << "%) exceeds 1% target";
}

TEST_F(MLPerformanceTest, WorkloadComparison) {
    std::cout << "\n========================================\n";
    std::cout << "Workload Pattern Comparison\n";
    std::cout << "========================================\n\n";
    
    std::vector<MLPerformanceValidator::WorkloadType> workloads = {
        MLPerformanceValidator::WorkloadType::SEQUENTIAL,
        MLPerformanceValidator::WorkloadType::TEMPORAL,
        MLPerformanceValidator::WorkloadType::RANDOM,
        MLPerformanceValidator::WorkloadType::ZIPFIAN,
        MLPerformanceValidator::WorkloadType::MIXED
    };
    
    std::vector<std::string> workload_names = {
        "Sequential", "Temporal", "Random", "Zipfian", "Mixed"
    };
    
    std::cout << std::setw(15) << "Workload" 
              << std::setw(15) << "Baseline" 
              << std::setw(15) << "With ML"
              << std::setw(20) << "Improvement %"
              << std::setw(15) << "Meets Target\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (size_t i = 0; i < workloads.size(); ++i) {
        MLPerformanceValidator::ValidationConfig config;
        config.num_operations = 20000;
        config.num_keys = 2000;
        config.workload_type = workloads[i];
        config.enable_ml = true;
        
        auto result = validator_->runValidation(config);
        
        std::cout << std::setw(15) << workload_names[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) 
                  << (result.baseline_hit_rate * 100) << "%"
                  << std::setw(14) << (result.ml_hit_rate * 100) << "%"
                  << std::setw(19) << result.hit_rate_improvement_percentage << "%"
                  << std::setw(15) << (result.meets_hit_rate_target ? "Yes" : "No")
                  << "\n";
    }
}

TEST_F(MLPerformanceTest, ABTestValidation) {
    std::cout << "\n========================================\n";
    std::cout << "A/B Test Validation\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 50000;
    config.num_keys = 5000;
    config.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
    
    auto ab_result = validator_->runABTest(config, 0.5);
    
    std::cout << "Control Group (No ML):\n";
    std::cout << "  Hit rate: " << std::fixed << std::setprecision(2) 
              << (ab_result.control_group.baseline_hit_rate * 100) << "%\n";
    
    std::cout << "\nTest Group (With ML):\n";
    std::cout << "  Hit rate: " << (ab_result.test_group.ml_hit_rate * 100) << "%\n";
    std::cout << "  Inference latency: " << ab_result.test_group.avg_inference_latency_ms << "ms\n";
    
    std::cout << "\nImprovement: " << ab_result.improvement_percentage << "%\n";
    std::cout << "Statistical Power: " << ab_result.statistical_power << "\n";
    std::cout << "\nRecommendation: " << ab_result.recommendation << "\n";
    
    // Validate A/B test shows improvement
    EXPECT_GT(ab_result.improvement_percentage, 0) 
        << "A/B test shows no improvement with ML";
}

TEST_F(MLPerformanceTest, ScalabilityTest) {
    std::cout << "\n========================================\n";
    std::cout << "Scalability Test\n";
    std::cout << "========================================\n\n";
    
    std::vector<size_t> operation_counts = {10000, 50000, 100000};
    std::vector<int> thread_counts = {1, 4, 8};
    
    std::cout << std::setw(15) << "Operations" 
              << std::setw(15) << "Threads"
              << std::setw(20) << "Throughput (ops/s)"
              << std::setw(15) << "Latency (ms)\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (auto ops : operation_counts) {
        for (auto threads : thread_counts) {
            MLPerformanceValidator::ValidationConfig config;
            config.num_operations = ops;
            config.num_keys = ops / 10;
            config.num_threads = threads;
            config.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
            
            auto result = validator_->runValidation(config);
            
            std::cout << std::setw(15) << ops
                      << std::setw(15) << threads
                      << std::setw(20) << std::fixed << std::setprecision(0) 
                      << result.throughput_ops_sec
                      << std::setw(15) << std::setprecision(2) 
                      << result.avg_latency_ms
                      << "\n";
        }
    }
}

TEST_F(MLPerformanceTest, ModelComparisonTest) {
    std::cout << "\n========================================\n";
    std::cout << "ML Model Comparison\n";
    std::cout << "========================================\n\n";
    
    std::vector<ppe::PrefetchCoordinator::ModelType> models = {
        ppe::PrefetchCoordinator::ModelType::LSTM,
        ppe::PrefetchCoordinator::ModelType::XGBOOST,
        ppe::PrefetchCoordinator::ModelType::ENSEMBLE
    };
    
    std::vector<std::string> model_names = {"LSTM", "XGBoost", "Ensemble"};
    
    MLPerformanceValidator::ValidationConfig base_config;
    base_config.num_operations = 20000;
    base_config.num_keys = 2000;
    base_config.workload_type = MLPerformanceValidator::WorkloadType::TEMPORAL;
    
    std::cout << std::setw(15) << "Model" 
              << std::setw(20) << "Hit Rate Improve %"
              << std::setw(20) << "Inference Time (ms)"
              << std::setw(15) << "Accuracy\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (size_t i = 0; i < models.size(); ++i) {
        // Update coordinator with new model
        auto config = coordinator_->getConfig();
        config.model_type = models[i];
        coordinator_->updateConfig(config);
        
        // Clear cache and reset for fair comparison
        cache_->clear();
        monitor_->resetMetrics();
        
        auto result = validator_->runValidation(base_config);
        
        std::cout << std::setw(15) << model_names[i]
                  << std::setw(20) << std::fixed << std::setprecision(1) 
                  << result.hit_rate_improvement_percentage
                  << std::setw(20) << std::setprecision(2) 
                  << result.avg_inference_latency_ms
                  << std::setw(15) << std::setprecision(1) 
                  << (result.prefetch_accuracy * 100) << "%"
                  << "\n";
    }
}

TEST_F(MLPerformanceTest, RegressionTest) {
    std::cout << "\n========================================\n";
    std::cout << "Performance Regression Test\n";
    std::cout << "========================================\n\n";
    
    // Define regression test configurations
    std::vector<MLPerformanceValidator::ValidationConfig> configs;
    
    // Config 1: Small workload
    MLPerformanceValidator::ValidationConfig config1;
    config1.num_operations = 10000;
    config1.num_keys = 1000;
    config1.workload_type = MLPerformanceValidator::WorkloadType::SEQUENTIAL;
    configs.push_back(config1);
    
    // Config 2: Large workload
    MLPerformanceValidator::ValidationConfig config2;
    config2.num_operations = 50000;
    config2.num_keys = 5000;
    config2.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
    configs.push_back(config2);
    
    // Config 3: High concurrency
    MLPerformanceValidator::ValidationConfig config3;
    config3.num_operations = 20000;
    config3.num_keys = 2000;
    config3.num_threads = 8;
    config3.workload_type = MLPerformanceValidator::WorkloadType::TEMPORAL;
    configs.push_back(config3);
    
    bool all_passed = validator_->runRegressionTests(configs);
    
    EXPECT_TRUE(all_passed) << "Performance regression detected";
}

TEST_F(MLPerformanceTest, GeneratePerformanceReport) {
    std::cout << "\n========================================\n";
    std::cout << "Generating Performance Report\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 100000;
    config.num_keys = 10000;
    config.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
    config.enable_ml = true;
    
    auto result = validator_->runValidation(config);
    
    // Generate text report
    std::string report = validator_->generateReport(result);
    std::cout << report << "\n";
    
    // Export results
    validator_->exportResultsJSON(result, "ml_performance_results.json");
    validator_->exportResultsCSV(result, "ml_performance_results.csv");
    
    // Create benchmark suite
    auto benchmark = std::make_shared<MLPerformanceBenchmark>(validator_);
    auto suite = benchmark->runQuickValidation();
    benchmark->generateHTMLReport(suite, "ml_performance_report.html");
    
    std::cout << "\nReports generated:\n";
    std::cout << "  - ml_performance_results.json\n";
    std::cout << "  - ml_performance_results.csv\n";
    std::cout << "  - ml_performance_report.html\n";
}

// Main test that validates all Epic 3 success criteria
TEST_F(MLPerformanceTest, Epic3SuccessCriteriaValidation) {
    std::cout << "\n========================================\n";
    std::cout << "EPIC 3 COMPLETE SUCCESS VALIDATION\n";
    std::cout << "========================================\n\n";
    
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 100000;
    config.num_keys = 10000;
    config.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
    config.enable_ml = true;
    config.warmup_operations = 5000;
    
    auto result = validator_->runValidation(config);
    
    std::cout << "=== EPIC 3 SUCCESS CRITERIA ===\n\n";
    
    // Criterion 1: Hit Rate Improvement
    std::cout << "1. Cache Hit Rate Improvement\n";
    std::cout << "   Target: 20%+ improvement\n";
    std::cout << "   Achieved: " << std::fixed << std::setprecision(1) 
              << result.hit_rate_improvement_percentage << "%\n";
    std::cout << "   Status: " << (result.meets_hit_rate_target ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Criterion 2: ML Inference Latency
    std::cout << "2. ML Inference Latency\n";
    std::cout << "   Target: <10ms\n";
    std::cout << "   Achieved: " << std::setprecision(2) 
              << result.avg_inference_latency_ms << "ms\n";
    std::cout << "   Status: " << (result.meets_latency_target ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Criterion 3: System Overhead
    std::cout << "3. System Overhead\n";
    std::cout << "   Target: <1% CPU overhead\n";
    std::cout << "   Achieved: " << result.cpu_overhead_percentage << "%\n";
    std::cout << "   Status: " << (result.meets_overhead_target ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Overall Result
    std::cout << "=== OVERALL RESULT ===\n";
    std::cout << "Epic 3 Success: " << (result.overall_success ? "✅ YES" : "❌ NO") << "\n\n";
    
    // Additional metrics
    std::cout << "Additional Performance Metrics:\n";
    std::cout << "- Throughput: " << std::fixed << std::setprecision(0) 
              << result.throughput_ops_sec << " ops/sec\n";
    std::cout << "- Prefetch Accuracy: " << std::setprecision(1) 
              << (result.prefetch_accuracy * 100) << "%\n";
    std::cout << "- Statistical Significance: " 
              << (result.statistically_significant ? "Yes" : "No") << "\n";
    
    // Epic 3 final validation
    EXPECT_TRUE(result.overall_success) 
        << "Epic 3 success criteria not met. See detailed results above.";
}

} // namespace test
} // namespace benchmarks
} // namespace predis

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}