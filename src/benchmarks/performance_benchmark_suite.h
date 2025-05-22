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

#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <functional>
#include <map>
#include <atomic>
#include <thread>

namespace predis {
namespace benchmarks {

/**
 * Comprehensive Performance Benchmark Suite for Epic 2 Story 2.4
 * 
 * Provides automated Redis vs Predis comparison benchmarking with:
 * - Statistical significance validation of performance claims
 * - Multiple workload scenarios (read-heavy, write-heavy, mixed)
 * - Performance regression testing and CI integration capabilities
 * - Comprehensive data collection and automated report generation
 * 
 * Target: Demonstrate consistent 10-25x performance improvements over Redis
 */
class PerformanceBenchmarkSuite {
public:
    /**
     * Benchmark configuration for different testing scenarios
     */
    struct BenchmarkConfig {
        // Test parameters
        size_t num_operations = 100000;
        size_t key_size = 16;
        size_t value_size = 64;
        size_t num_threads = 1;
        size_t warmup_iterations = 1000;
        
        // Statistical validation
        size_t num_iterations = 10;
        double confidence_level = 0.95;  // 95% confidence interval
        double significance_threshold = 0.05;  // p < 0.05 for significance
        
        // Workload patterns
        double read_ratio = 0.8;  // 80% reads, 20% writes (default)
        double batch_operation_ratio = 0.5;  // 50% of operations are batched
        size_t batch_size = 100;
        
        // Performance targets
        double minimum_improvement_factor = 10.0;  // 10x minimum improvement
        double target_improvement_factor = 25.0;   // 25x target improvement
        
        // Environment settings
        bool use_dedicated_redis_instance = true;
        std::string redis_host = "localhost";
        int redis_port = 6379;
        bool flush_redis_before_test = true;
        
        BenchmarkConfig() = default;
    };

    /**
     * Comprehensive benchmark results with statistical analysis
     */
    struct BenchmarkResult {
        std::string test_name;
        std::string workload_description;
        
        // Performance metrics
        struct PerformanceMetrics {
            double average_ops_per_second = 0.0;
            double median_ops_per_second = 0.0;
            double p95_ops_per_second = 0.0;
            double p99_ops_per_second = 0.0;
            
            double average_latency_ms = 0.0;
            double median_latency_ms = 0.0;
            double p95_latency_ms = 0.0;
            double p99_latency_ms = 0.0;
            
            double standard_deviation = 0.0;
            double coefficient_of_variation = 0.0;
            
            size_t total_operations = 0;
            size_t successful_operations = 0;
            double success_rate_percent = 0.0;
        };
        
        PerformanceMetrics redis_metrics;
        PerformanceMetrics predis_metrics;
        
        // Improvement analysis
        double improvement_factor = 0.0;
        double latency_improvement_factor = 0.0;
        
        // Statistical validation
        struct StatisticalAnalysis {
            double confidence_interval_lower = 0.0;
            double confidence_interval_upper = 0.0;
            double p_value = 0.0;
            bool statistically_significant = false;
            double effect_size = 0.0;  // Cohen's d
            
            std::string statistical_summary;
        } statistics;
        
        // Epic 2 target validation
        bool meets_minimum_target = false;  // 10x improvement
        bool meets_epic2_target = false;    // 25x improvement
        bool regression_detected = false;
        
        // Metadata
        std::chrono::system_clock::time_point timestamp;
        std::string environment_info;
        std::string git_commit_hash;
        
        BenchmarkResult() {
            timestamp = std::chrono::system_clock::now();
        }
    };

    /**
     * Workload scenario definitions for comprehensive testing
     */
    enum class WorkloadType {
        READ_HEAVY,      // 90% reads, 10% writes
        WRITE_HEAVY,     // 10% reads, 90% writes  
        BALANCED,        // 50% reads, 50% writes
        BATCH_INTENSIVE, // Heavy use of batch operations
        MIXED_REALISTIC, // Real-world mixed workload
        STRESS_TEST,     // Maximum load testing
        LATENCY_FOCUSED  // Low-latency optimized workload
    };

    PerformanceBenchmarkSuite() = default;
    ~PerformanceBenchmarkSuite() = default;

    /**
     * Initialize the benchmark suite with configuration
     */
    bool initialize(const BenchmarkConfig& config = BenchmarkConfig{});

    /**
     * Shutdown and cleanup benchmark resources
     */
    void shutdown();

    /**
     * Run comprehensive benchmark suite with all workload scenarios
     */
    std::vector<BenchmarkResult> run_comprehensive_benchmark_suite();

    /**
     * Run specific workload scenario benchmark
     */
    BenchmarkResult run_workload_benchmark(WorkloadType workload_type, 
                                          const std::string& custom_name = "");

    /**
     * Run custom benchmark with specific configuration
     */
    BenchmarkResult run_custom_benchmark(const BenchmarkConfig& config,
                                        const std::string& test_name);

    /**
     * Performance regression testing
     */
    struct RegressionTestResult {
        bool regression_detected = false;
        double performance_change_percent = 0.0;
        std::string baseline_version;
        std::string current_version;
        std::vector<std::string> failing_benchmarks;
        std::string analysis_summary;
    };
    
    RegressionTestResult run_regression_test(const std::string& baseline_results_file);

    /**
     * Statistical analysis utilities
     */
    static double calculate_improvement_factor(const BenchmarkResult::PerformanceMetrics& baseline,
                                             const BenchmarkResult::PerformanceMetrics& improved);
    
    static BenchmarkResult::StatisticalAnalysis perform_statistical_analysis(
        const std::vector<double>& baseline_samples,
        const std::vector<double>& improved_samples,
        double confidence_level = 0.95);

    /**
     * Report generation and visualization
     */
    bool generate_benchmark_report(const std::vector<BenchmarkResult>& results,
                                  const std::string& output_file = "benchmark_report.html");
    
    bool export_results_to_csv(const std::vector<BenchmarkResult>& results,
                               const std::string& csv_file = "benchmark_results.csv");
    
    bool export_results_to_json(const std::vector<BenchmarkResult>& results,
                                const std::string& json_file = "benchmark_results.json");

    /**
     * CI/CD integration utilities
     */
    struct CIResult {
        bool all_tests_passed = false;
        size_t total_benchmarks = 0;
        size_t passed_benchmarks = 0;
        size_t failed_benchmarks = 0;
        std::vector<std::string> failure_reasons;
        std::string summary_message;
    };
    
    CIResult validate_for_ci_pipeline(const std::vector<BenchmarkResult>& results);
    
    /**
     * Configuration management
     */
    void set_benchmark_config(const BenchmarkConfig& config);
    BenchmarkConfig get_benchmark_config() const;
    
    /**
     * Environment and system information
     */
    std::string get_system_info() const;
    std::string get_gpu_info() const;
    std::string get_redis_version() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Redis client wrapper for benchmark comparisons
 */
class RedisBenchmarkClient {
public:
    RedisBenchmarkClient() = default;
    ~RedisBenchmarkClient() = default;

    bool connect(const std::string& host = "localhost", int port = 6379);
    void disconnect();
    
    // Single operations
    bool set(const std::string& key, const std::string& value);
    std::string get(const std::string& key);
    bool del(const std::string& key);
    bool exists(const std::string& key);
    
    // Batch operations (using pipelining)
    std::vector<bool> batch_set(const std::vector<std::string>& keys,
                               const std::vector<std::string>& values);
    std::vector<std::string> batch_get(const std::vector<std::string>& keys);
    std::vector<bool> batch_del(const std::vector<std::string>& keys);
    std::vector<bool> batch_exists(const std::vector<std::string>& keys);
    
    // Utility operations
    void flush_db();
    size_t get_memory_usage();
    std::string get_info();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Predis client wrapper for benchmark testing
 */
class PredisBenchmarkClient {
public:
    PredisBenchmarkClient() = default;
    ~PredisBenchmarkClient() = default;

    bool initialize();
    void shutdown();
    
    // Single operations
    bool set(const std::string& key, const std::string& value);
    std::string get(const std::string& key);
    bool del(const std::string& key);
    bool exists(const std::string& key);
    
    // Batch operations
    std::vector<bool> batch_set(const std::vector<std::string>& keys,
                               const std::vector<std::string>& values);
    std::vector<std::string> batch_get(const std::vector<std::string>& keys);
    std::vector<bool> batch_del(const std::vector<std::string>& keys);
    std::vector<bool> batch_exists(const std::vector<std::string>& keys);
    
    // Performance monitoring
    struct PerformanceStats {
        double gpu_utilization_percent = 0.0;
        double memory_bandwidth_gbps = 0.0;
        size_t cache_hit_count = 0;
        size_t cache_miss_count = 0;
        double pipeline_efficiency = 0.0;
    };
    
    PerformanceStats get_performance_stats();
    void reset_stats();
    
    // Memory management
    void flush_cache();
    size_t get_memory_usage();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Workload generator for realistic benchmark scenarios
 */
class WorkloadGenerator {
public:
    struct WorkloadParameters {
        size_t total_operations = 10000;
        double read_write_ratio = 0.8;  // 80% reads
        size_t key_space_size = 100000;
        size_t key_size = 16;
        size_t value_size = 64;
        
        // Distribution patterns
        enum class AccessPattern { UNIFORM, ZIPFIAN, HOTSPOT } access_pattern = AccessPattern::ZIPFIAN;
        double zipfian_constant = 1.0;
        double hotspot_ratio = 0.1;  // 10% of keys are hot
        
        // Temporal patterns
        double batch_probability = 0.3;  // 30% chance of batch operation
        size_t max_batch_size = 100;
        
        WorkloadParameters() = default;
    };

    struct Operation {
        enum class Type { GET, SET, DELETE, EXISTS, BATCH_GET, BATCH_SET, BATCH_DELETE } type;
        std::vector<std::string> keys;
        std::vector<std::string> values;
        std::chrono::steady_clock::time_point timestamp;
    };

    WorkloadGenerator() = default;
    ~WorkloadGenerator() = default;

    /**
     * Generate workload based on specified parameters
     */
    std::vector<Operation> generate_workload(const WorkloadParameters& params);
    
    /**
     * Generate predefined workload scenarios
     */
    std::vector<Operation> generate_read_heavy_workload(size_t operations);
    std::vector<Operation> generate_write_heavy_workload(size_t operations);
    std::vector<Operation> generate_balanced_workload(size_t operations);
    std::vector<Operation> generate_batch_intensive_workload(size_t operations);
    std::vector<Operation> generate_realistic_mixed_workload(size_t operations);

private:
    std::string generate_key(size_t index, size_t key_size);
    std::string generate_value(size_t value_size);
    size_t select_key_with_distribution(size_t key_space_size, 
                                       WorkloadParameters::AccessPattern pattern,
                                       double parameter);
};

} // namespace benchmarks
} // namespace predis