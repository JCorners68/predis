/*
 * Copyright 2025 Predis Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../../src/benchmarks/performance_benchmark_suite.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace predis::benchmarks;

class BenchmarkSuiteTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize benchmark suite with test configuration
        PerformanceBenchmarkSuite::BenchmarkConfig config;
        config.num_operations = 1000;      // Smaller for testing
        config.num_iterations = 3;         // Fewer iterations for speed
        config.warmup_iterations = 100;
        config.key_size = 16;
        config.value_size = 64;
        config.confidence_level = 0.95;
        config.minimum_improvement_factor = 5.0;   // Lower for testing
        config.target_improvement_factor = 15.0;   // Lower for testing
        config.use_dedicated_redis_instance = false;  // Skip Redis for unit tests
        
        benchmark_suite = std::make_unique<PerformanceBenchmarkSuite>();
        
        // Note: For unit tests, we'll test the framework without actual Redis/Predis
        // Real integration tests would require Redis server and Predis initialization
    }
    
    void TearDown() override {
        if (benchmark_suite) {
            benchmark_suite->shutdown();
        }
    }
    
    std::unique_ptr<PerformanceBenchmarkSuite> benchmark_suite;
};

TEST_F(BenchmarkSuiteTest, WorkloadGeneratorTest) {
    std::cout << "\n==== Testing Workload Generator ====" << std::endl;
    
    WorkloadGenerator generator;
    
    // Test different workload parameters
    WorkloadGenerator::WorkloadParameters params;
    params.total_operations = 1000;
    params.read_write_ratio = 0.8;  // 80% reads
    params.key_space_size = 10000;
    params.key_size = 16;
    params.value_size = 64;
    params.batch_probability = 0.3;  // 30% batch operations
    params.max_batch_size = 50;
    
    auto operations = generator.generate_workload(params);
    
    EXPECT_EQ(operations.size(), params.total_operations);
    
    // Analyze generated workload
    size_t read_ops = 0, write_ops = 0, batch_ops = 0;
    
    for (const auto& op : operations) {
        switch (op.type) {
            case WorkloadGenerator::Operation::Type::GET:
            case WorkloadGenerator::Operation::Type::EXISTS:
                read_ops++;
                break;
            case WorkloadGenerator::Operation::Type::SET:
            case WorkloadGenerator::Operation::Type::DELETE:
                write_ops++;
                break;
            case WorkloadGenerator::Operation::Type::BATCH_GET:
            case WorkloadGenerator::Operation::Type::BATCH_SET:
            case WorkloadGenerator::Operation::Type::BATCH_DELETE:
                batch_ops++;
                break;
        }
        
        // Verify key and value sizes
        for (const auto& key : op.keys) {
            EXPECT_LE(key.length(), params.key_size * 2);  // Allow some variation
        }
        for (const auto& value : op.values) {
            EXPECT_EQ(value.length(), params.value_size);
        }
    }
    
    double read_ratio = static_cast<double>(read_ops) / (read_ops + write_ops + batch_ops);
    double batch_ratio = static_cast<double>(batch_ops) / operations.size();
    
    std::cout << "Generated workload analysis:" << std::endl;
    std::cout << "  Total operations: " << operations.size() << std::endl;
    std::cout << "  Read operations: " << read_ops << " (" << (read_ratio * 100) << "%)" << std::endl;
    std::cout << "  Write operations: " << write_ops << std::endl;
    std::cout << "  Batch operations: " << batch_ops << " (" << (batch_ratio * 100) << "%)" << std::endl;
    
    // Verify ratios are approximately correct (within 20% tolerance)
    EXPECT_NEAR(read_ratio, 0.6, 0.3);  // Relaxed for batch operations affecting ratio
    EXPECT_NEAR(batch_ratio, params.batch_probability, 0.2);
}

TEST_F(BenchmarkSuiteTest, StatisticalAnalysisTest) {
    std::cout << "\n==== Testing Statistical Analysis ====" << std::endl;
    
    // Create test data sets
    std::vector<double> baseline_samples = {100, 105, 95, 110, 90, 108, 102, 97, 103, 99};
    std::vector<double> improved_samples = {1500, 1600, 1450, 1550, 1480, 1520, 1590, 1470, 1510, 1540};
    
    auto stats = PerformanceBenchmarkSuite::perform_statistical_analysis(
        baseline_samples, improved_samples, 0.95);
    
    std::cout << "Statistical analysis results:" << std::endl;
    std::cout << "  Statistical significance: " << (stats.statistically_significant ? "YES" : "NO") << std::endl;
    std::cout << "  P-value: " << stats.p_value << std::endl;
    std::cout << "  Effect size (Cohen's d): " << stats.effect_size << std::endl;
    std::cout << "  Confidence interval: [" << stats.confidence_interval_lower 
              << ", " << stats.confidence_interval_upper << "]" << std::endl;
    std::cout << "  Summary: " << stats.statistical_summary << std::endl;
    
    // Verify statistical significance is detected for large improvement
    EXPECT_TRUE(stats.statistically_significant);
    EXPECT_LT(stats.p_value, 0.05);  // Should be statistically significant
    EXPECT_GT(stats.effect_size, 2.0);  // Large effect size
    
    // Test improvement factor calculation
    PerformanceBenchmarkSuite::BenchmarkResult::PerformanceMetrics baseline_metrics;
    baseline_metrics.average_ops_per_second = 100.0;
    
    PerformanceBenchmarkSuite::BenchmarkResult::PerformanceMetrics improved_metrics;
    improved_metrics.average_ops_per_second = 1500.0;
    
    double improvement = PerformanceBenchmarkSuite::calculate_improvement_factor(baseline_metrics, improved_metrics);
    
    std::cout << "Improvement factor: " << improvement << "x" << std::endl;
    EXPECT_NEAR(improvement, 15.0, 0.1);
}

TEST_F(BenchmarkSuiteTest, BenchmarkConfigurationTest) {
    std::cout << "\n==== Testing Benchmark Configuration ====" << std::endl;
    
    // Test default configuration
    PerformanceBenchmarkSuite::BenchmarkConfig default_config;
    
    EXPECT_EQ(default_config.num_operations, 100000);
    EXPECT_EQ(default_config.num_iterations, 10);
    EXPECT_EQ(default_config.confidence_level, 0.95);
    EXPECT_EQ(default_config.minimum_improvement_factor, 10.0);
    EXPECT_EQ(default_config.target_improvement_factor, 25.0);
    
    // Test custom configuration
    PerformanceBenchmarkSuite::BenchmarkConfig custom_config;
    custom_config.num_operations = 50000;
    custom_config.num_iterations = 5;
    custom_config.read_ratio = 0.7;
    custom_config.batch_operation_ratio = 0.4;
    custom_config.batch_size = 150;
    
    benchmark_suite->set_benchmark_config(custom_config);
    auto retrieved_config = benchmark_suite->get_benchmark_config();
    
    EXPECT_EQ(retrieved_config.num_operations, 50000);
    EXPECT_EQ(retrieved_config.num_iterations, 5);
    EXPECT_EQ(retrieved_config.read_ratio, 0.7);
    EXPECT_EQ(retrieved_config.batch_operation_ratio, 0.4);
    EXPECT_EQ(retrieved_config.batch_size, 150);
    
    std::cout << "Configuration test passed" << std::endl;
}

TEST_F(BenchmarkSuiteTest, ReportGenerationTest) {
    std::cout << "\n==== Testing Report Generation ====" << std::endl;
    
    // Create mock benchmark results for testing
    std::vector<PerformanceBenchmarkSuite::BenchmarkResult> mock_results;
    
    // Create a high-performance result
    PerformanceBenchmarkSuite::BenchmarkResult good_result;
    good_result.test_name = "High Performance Test";
    good_result.workload_description = "Optimized workload with batch operations";
    good_result.redis_metrics.average_ops_per_second = 50000;
    good_result.redis_metrics.average_latency_ms = 2.0;
    good_result.redis_metrics.p99_latency_ms = 5.0;
    good_result.predis_metrics.average_ops_per_second = 1250000;  // 25x improvement
    good_result.predis_metrics.average_latency_ms = 0.08;
    good_result.predis_metrics.p99_latency_ms = 0.2;
    good_result.improvement_factor = 25.0;
    good_result.latency_improvement_factor = 25.0;
    good_result.meets_minimum_target = true;
    good_result.meets_epic2_target = true;
    good_result.statistics.statistically_significant = true;
    good_result.statistics.p_value = 0.001;
    good_result.statistics.effect_size = 5.5;
    good_result.statistics.statistical_summary = "Improvement: 25.0x (95% CI: 22.5-27.5), p=0.001, Cohen's d=5.5 [SIGNIFICANT]";
    
    mock_results.push_back(good_result);
    
    // Create a moderate performance result
    PerformanceBenchmarkSuite::BenchmarkResult moderate_result;
    moderate_result.test_name = "Moderate Performance Test";
    moderate_result.workload_description = "Mixed read/write workload";
    moderate_result.redis_metrics.average_ops_per_second = 75000;
    moderate_result.redis_metrics.average_latency_ms = 1.33;
    moderate_result.predis_metrics.average_ops_per_second = 900000;  // 12x improvement
    moderate_result.predis_metrics.average_latency_ms = 0.11;
    moderate_result.improvement_factor = 12.0;
    moderate_result.latency_improvement_factor = 12.1;
    moderate_result.meets_minimum_target = true;
    moderate_result.meets_epic2_target = false;  // Below 25x target
    moderate_result.statistics.statistically_significant = true;
    moderate_result.statistics.p_value = 0.02;
    moderate_result.statistics.effect_size = 3.2;
    moderate_result.statistics.statistical_summary = "Improvement: 12.0x (95% CI: 10.8-13.2), p=0.02, Cohen's d=3.2 [SIGNIFICANT]";
    
    mock_results.push_back(moderate_result);
    
    // Test HTML report generation
    bool html_success = benchmark_suite->generate_benchmark_report(mock_results, "test_benchmark_report.html");
    EXPECT_TRUE(html_success);
    
    // Verify HTML file was created and contains expected content
    std::ifstream html_file("test_benchmark_report.html");
    EXPECT_TRUE(html_file.is_open());
    
    std::string html_content((std::istreambuf_iterator<char>(html_file)),
                            std::istreambuf_iterator<char>());
    html_file.close();
    
    EXPECT_NE(html_content.find("Epic 2 Story 2.4"), std::string::npos);
    EXPECT_NE(html_content.find("High Performance Test"), std::string::npos);
    EXPECT_NE(html_content.find("25.0x"), std::string::npos);
    
    // Test CSV export
    bool csv_success = benchmark_suite->export_results_to_csv(mock_results, "test_benchmark_results.csv");
    EXPECT_TRUE(csv_success);
    
    // Verify CSV file content
    std::ifstream csv_file("test_benchmark_results.csv");
    EXPECT_TRUE(csv_file.is_open());
    
    std::string csv_line;
    std::getline(csv_file, csv_line);  // Header
    EXPECT_NE(csv_line.find("Test_Name"), std::string::npos);
    EXPECT_NE(csv_line.find("Improvement_Factor"), std::string::npos);
    
    std::getline(csv_file, csv_line);  // First data row
    EXPECT_NE(csv_line.find("High Performance Test"), std::string::npos);
    csv_file.close();
    
    // Test JSON export
    bool json_success = benchmark_suite->export_results_to_json(mock_results, "test_benchmark_results.json");
    EXPECT_TRUE(json_success);
    
    // Verify JSON file content
    std::ifstream json_file("test_benchmark_results.json");
    EXPECT_TRUE(json_file.is_open());
    
    std::string json_content((std::istreambuf_iterator<char>(json_file)),
                            std::istreambuf_iterator<char>());
    json_file.close();
    
    EXPECT_NE(json_content.find("\"benchmark_suite\""), std::string::npos);
    EXPECT_NE(json_content.find("\"Epic 2 Story 2.4\""), std::string::npos);
    EXPECT_NE(json_content.find("\"improvement_factor\": 25.0"), std::string::npos);
    
    std::cout << "Report generation test passed" << std::endl;
    
    // Cleanup test files
    std::filesystem::remove("test_benchmark_report.html");
    std::filesystem::remove("test_benchmark_results.csv");
    std::filesystem::remove("test_benchmark_results.json");
}

TEST_F(BenchmarkSuiteTest, CIIntegrationTest) {
    std::cout << "\n==== Testing CI Integration ====" << std::endl;
    
    // Create mock results for CI validation
    std::vector<PerformanceBenchmarkSuite::BenchmarkResult> ci_results;
    
    // Passing result
    PerformanceBenchmarkSuite::BenchmarkResult passing_result;
    passing_result.test_name = "CI Passing Test";
    passing_result.improvement_factor = 15.0;
    passing_result.meets_minimum_target = true;
    passing_result.statistics.statistically_significant = true;
    ci_results.push_back(passing_result);
    
    // Failing result
    PerformanceBenchmarkSuite::BenchmarkResult failing_result;
    failing_result.test_name = "CI Failing Test";
    failing_result.improvement_factor = 8.0;  // Below minimum
    failing_result.meets_minimum_target = false;
    failing_result.statistics.statistically_significant = false;
    ci_results.push_back(failing_result);
    
    auto ci_result = benchmark_suite->validate_for_ci_pipeline(ci_results);
    
    std::cout << "CI validation results:" << std::endl;
    std::cout << "  All tests passed: " << (ci_result.all_tests_passed ? "YES" : "NO") << std::endl;
    std::cout << "  Total benchmarks: " << ci_result.total_benchmarks << std::endl;
    std::cout << "  Passed benchmarks: " << ci_result.passed_benchmarks << std::endl;
    std::cout << "  Failed benchmarks: " << ci_result.failed_benchmarks << std::endl;
    std::cout << "  Summary: " << ci_result.summary_message << std::endl;
    
    EXPECT_FALSE(ci_result.all_tests_passed);
    EXPECT_EQ(ci_result.total_benchmarks, 2);
    EXPECT_EQ(ci_result.passed_benchmarks, 1);
    EXPECT_EQ(ci_result.failed_benchmarks, 1);
    EXPECT_EQ(ci_result.failure_reasons.size(), 1);
    
    // Test with all passing results
    std::vector<PerformanceBenchmarkSuite::BenchmarkResult> all_passing;
    passing_result.test_name = "Test 1";
    all_passing.push_back(passing_result);
    passing_result.test_name = "Test 2";
    all_passing.push_back(passing_result);
    
    auto all_passing_result = benchmark_suite->validate_for_ci_pipeline(all_passing);
    
    EXPECT_TRUE(all_passing_result.all_tests_passed);
    EXPECT_EQ(all_passing_result.failed_benchmarks, 0);
    
    std::cout << "CI integration test passed" << std::endl;
}

TEST_F(BenchmarkSuiteTest, SystemInformationTest) {
    std::cout << "\n==== Testing System Information Collection ====" << std::endl;
    
    std::string system_info = benchmark_suite->get_system_info();
    std::string gpu_info = benchmark_suite->get_gpu_info();
    
    std::cout << "System info: " << system_info << std::endl;
    std::cout << "GPU info: " << gpu_info << std::endl;
    
    EXPECT_FALSE(system_info.empty());
    EXPECT_FALSE(gpu_info.empty());
    
    // Verify expected content
    EXPECT_NE(system_info.find("cores"), std::string::npos);
    EXPECT_NE(gpu_info.find("RTX 5080"), std::string::npos);
    
    std::cout << "System information test passed" << std::endl;
}

// Integration test that would require actual Redis and Predis setup
TEST_F(BenchmarkSuiteTest, DISABLED_FullIntegrationTest) {
    std::cout << "\n==== Full Integration Test (Requires Redis/Predis) ====" << std::endl;
    
    // This test is disabled by default as it requires:
    // 1. Redis server running on localhost:6379
    // 2. Predis system fully initialized
    // 3. GPU hardware available
    
    PerformanceBenchmarkSuite::BenchmarkConfig config;
    config.num_operations = 10000;
    config.num_iterations = 3;
    config.redis_host = "localhost";
    config.redis_port = 6379;
    
    bool initialized = benchmark_suite->initialize(config);
    if (!initialized) {
        GTEST_SKIP() << "Redis server not available for integration test";
    }
    
    // Run a simple workload test
    auto result = benchmark_suite->run_workload_benchmark(
        PerformanceBenchmarkSuite::WorkloadType::READ_HEAVY, 
        "Integration Test - Read Heavy");
    
    std::cout << "Integration test results:" << std::endl;
    std::cout << "  Redis performance: " << result.redis_metrics.average_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Predis performance: " << result.predis_metrics.average_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement factor: " << result.improvement_factor << "x" << std::endl;
    
    EXPECT_GT(result.improvement_factor, 1.0);
    EXPECT_TRUE(result.statistics.statistically_significant);
    
    // Generate full report
    std::vector<PerformanceBenchmarkSuite::BenchmarkResult> results = {result};
    EXPECT_TRUE(benchmark_suite->generate_benchmark_report(results, "integration_test_report.html"));
    EXPECT_TRUE(benchmark_suite->export_results_to_csv(results, "integration_test_results.csv"));
}