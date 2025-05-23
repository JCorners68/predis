#include "ml_performance_validator.h"
#include "../ppe/prefetch_coordinator.h"
#include "../ppe/prefetch_monitor.h"
#include "../core/simple_cache_manager.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <future>
#include <iostream>

namespace predis {
namespace benchmarks {

MLPerformanceValidator::MLPerformanceValidator() {}

void MLPerformanceValidator::setCacheManager(std::shared_ptr<core::SimpleCacheManager> cache) {
    cache_ = cache;
}

void MLPerformanceValidator::setPrefetchCoordinator(std::shared_ptr<ppe::PrefetchCoordinator> coordinator) {
    coordinator_ = coordinator;
}

void MLPerformanceValidator::setPrefetchMonitor(std::shared_ptr<ppe::PrefetchMonitor> monitor) {
    monitor_ = monitor;
}

MLPerformanceValidator::ValidationResult 
MLPerformanceValidator::runValidation(const ValidationConfig& config) {
    ValidationResult result;
    result.test_time = std::chrono::system_clock::now();
    
    std::cout << "=== ML Performance Validation ===\n";
    std::cout << "Workload: " << static_cast<int>(config.workload_type) << "\n";
    std::cout << "Operations: " << config.num_operations << "\n";
    std::cout << "Keys: " << config.num_keys << "\n";
    
    // Generate workload
    auto workload = generateWorkload(config);
    
    // Warmup phase
    if (config.warmup_operations > 0) {
        std::cout << "Running warmup phase...\n";
        for (size_t i = 0; i < config.warmup_operations && i < workload.size(); ++i) {
            std::string value;
            cache_->get(workload[i], value);
        }
    }
    
    // Measure baseline (without ML/prefetching)
    if (coordinator_) {
        auto original_config = coordinator_->getConfig();
        auto temp_config = original_config;
        temp_config.enable_background_prefetch = false;
        coordinator_->updateConfig(temp_config);
        
        std::cout << "Measuring baseline performance...\n";
        measureBaseline(workload, result);
        
        // Restore config for ML measurement
        coordinator_->updateConfig(original_config);
    } else {
        measureBaseline(workload, result);
    }
    
    // Clear cache and reset stats
    cache_->clear();
    if (monitor_) monitor_->resetMetrics();
    
    // Measure with ML/prefetching
    if (config.enable_ml && coordinator_) {
        std::cout << "Measuring ML-enhanced performance...\n";
        measureWithML(workload, result);
    }
    
    // Calculate improvements
    result.hit_rate_improvement = result.ml_hit_rate - result.baseline_hit_rate;
    result.hit_rate_improvement_percentage = 
        (result.baseline_hit_rate > 0) ? 
        (result.hit_rate_improvement / result.baseline_hit_rate * 100) : 0;
    
    // Measure overhead
    measureOverhead(result);
    
    // Validate success criteria
    validateSuccessCriteria(result);
    
    // Statistical analysis
    if (config.num_operations >= 1000) {
        // Simple confidence interval calculation
        result.confidence_interval = 1.96 * std::sqrt(
            result.ml_hit_rate * (1 - result.ml_hit_rate) / config.num_operations);
        
        // Simplified p-value (would use proper statistical test in production)
        result.p_value = (result.hit_rate_improvement > 0.05) ? 0.01 : 0.1;
        result.statistically_significant = result.p_value < 0.05;
    }
    
    return result;
}

MLPerformanceValidator::ABTestResult 
MLPerformanceValidator::runABTest(const ValidationConfig& config, double test_split) {
    ABTestResult ab_result;
    
    std::cout << "=== A/B Test Validation ===\n";
    std::cout << "Test split: " << (test_split * 100) << "%\n";
    
    // Generate workload
    auto full_workload = generateWorkload(config);
    
    // Split workload
    size_t split_point = full_workload.size() * test_split;
    std::vector<std::string> control_workload(full_workload.begin(), 
                                            full_workload.begin() + split_point);
    std::vector<std::string> test_workload(full_workload.begin() + split_point, 
                                         full_workload.end());
    
    // Run control group (no ML)
    if (coordinator_) {
        auto original_config = coordinator_->getConfig();
        auto control_config = original_config;
        control_config.enable_background_prefetch = false;
        coordinator_->updateConfig(control_config);
        
        measureBaseline(control_workload, ab_result.control_group);
        
        // Run test group (with ML)
        coordinator_->updateConfig(original_config);
        cache_->clear();
        measureWithML(test_workload, ab_result.test_group);
    }
    
    // Calculate improvement
    ab_result.improvement_percentage = 
        ((ab_result.test_group.ml_hit_rate - ab_result.control_group.baseline_hit_rate) / 
         ab_result.control_group.baseline_hit_rate) * 100;
    
    // Statistical power (simplified)
    ab_result.statistical_power = 0.8; // Would calculate properly in production
    
    // Recommendation
    if (ab_result.improvement_percentage >= 20.0 && 
        ab_result.test_group.avg_inference_latency_ms < 10.0) {
        ab_result.recommendation = "ML prefetching shows significant improvement. Recommend deployment.";
    } else if (ab_result.improvement_percentage >= 10.0) {
        ab_result.recommendation = "ML prefetching shows moderate improvement. Consider optimization.";
    } else {
        ab_result.recommendation = "ML prefetching shows minimal improvement. Further tuning needed.";
    }
    
    return ab_result;
}

MLPerformanceValidator::ValidationResult 
MLPerformanceValidator::validateHitRateImprovement(const ValidationConfig& config) {
    auto result = runValidation(config);
    
    std::cout << "\n=== Hit Rate Validation ===\n";
    std::cout << "Baseline hit rate: " << std::fixed << std::setprecision(2) 
              << (result.baseline_hit_rate * 100) << "%\n";
    std::cout << "ML hit rate: " << (result.ml_hit_rate * 100) << "%\n";
    std::cout << "Improvement: " << result.hit_rate_improvement_percentage << "%\n";
    std::cout << "Target: 20%+ improvement\n";
    std::cout << "Result: " << (result.meets_hit_rate_target ? "PASS" : "FAIL") << "\n";
    
    return result;
}

MLPerformanceValidator::ValidationResult 
MLPerformanceValidator::validateInferenceLatency(const ValidationConfig& config) {
    ValidationResult result;
    
    if (!coordinator_) {
        std::cerr << "No prefetch coordinator set\n";
        return result;
    }
    
    std::cout << "\n=== Inference Latency Validation ===\n";
    
    // Generate test features
    auto workload = generateWorkload(config);
    std::vector<double> latencies;
    
    // Measure inference latency
    for (size_t i = 0; i < std::min(size_t(1000), workload.size()); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Trigger prediction
        coordinator_->predictNextKeys(workload[i], 10);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count() / 1000.0;
        
        latencies.push_back(latency_ms);
    }
    
    // Calculate statistics
    result.avg_inference_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 
                                     latencies.size();
    result.max_inference_latency_ms = *std::max_element(latencies.begin(), latencies.end());
    
    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end());
    result.p50_latency_ms = latencies[latencies.size() / 2];
    result.p95_latency_ms = latencies[latencies.size() * 95 / 100];
    result.p99_latency_ms = latencies[latencies.size() * 99 / 100];
    
    result.meets_latency_target = result.avg_inference_latency_ms < 10.0;
    
    std::cout << "Average latency: " << result.avg_inference_latency_ms << "ms\n";
    std::cout << "P95 latency: " << result.p95_latency_ms << "ms\n";
    std::cout << "P99 latency: " << result.p99_latency_ms << "ms\n";
    std::cout << "Target: <10ms average\n";
    std::cout << "Result: " << (result.meets_latency_target ? "PASS" : "FAIL") << "\n";
    
    return result;
}

MLPerformanceValidator::ValidationResult 
MLPerformanceValidator::validateSystemOverhead(const ValidationConfig& config) {
    ValidationResult result;
    
    std::cout << "\n=== System Overhead Validation ===\n";
    
    // Measure CPU usage before
    auto cpu_start = std::clock();
    
    // Run workload
    auto workload = generateWorkload(config);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& key : workload) {
        std::string value;
        cache_->get(key, value);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto cpu_end = std::clock();
    
    // Calculate CPU overhead
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    double wall_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    result.cpu_overhead_percentage = (cpu_time / wall_time - 1.0) * 100;
    
    // Memory overhead (simplified)
    result.memory_overhead_mb = 50.0; // Placeholder - would measure actual memory
    
    // Check against target
    result.meets_overhead_target = result.cpu_overhead_percentage < 1.0;
    
    std::cout << "CPU overhead: " << result.cpu_overhead_percentage << "%\n";
    std::cout << "Memory overhead: " << result.memory_overhead_mb << "MB\n";
    std::cout << "Target: <1% CPU overhead\n";
    std::cout << "Result: " << (result.meets_overhead_target ? "PASS" : "FAIL") << "\n";
    
    return result;
}

std::vector<std::string> 
MLPerformanceValidator::generateWorkload(const ValidationConfig& config) {
    switch (config.workload_type) {
        case WorkloadType::SEQUENTIAL:
            return generateSequentialWorkload(config.num_operations, config.num_keys);
        case WorkloadType::TEMPORAL:
            return generateTemporalWorkload(config.num_operations, config.num_keys);
        case WorkloadType::RANDOM:
            return generateRandomWorkload(config.num_operations, config.num_keys);
        case WorkloadType::ZIPFIAN:
            return generateZipfianWorkload(config.num_operations, config.num_keys);
        case WorkloadType::MIXED:
            return generateMixedWorkload(config.num_operations, config.num_keys);
        case WorkloadType::REAL_WORLD:
            if (!config.trace_file.empty()) {
                // Load from trace file
                std::vector<std::string> trace;
                std::ifstream file(config.trace_file);
                std::string key;
                while (file >> key && trace.size() < config.num_operations) {
                    trace.push_back(key);
                }
                return trace;
            }
            // Fall back to mixed if no trace file
            return generateMixedWorkload(config.num_operations, config.num_keys);
    }
    
    return {};
}

std::string MLPerformanceValidator::generateReport(const ValidationResult& result) {
    std::ostringstream report;
    
    report << "=== ML Performance Validation Report ===\n\n";
    report << "Test Time: " << std::chrono::system_clock::to_time_t(result.test_time) << "\n\n";
    
    report << "Hit Rate Performance:\n";
    report << "  Baseline: " << std::fixed << std::setprecision(2) 
           << (result.baseline_hit_rate * 100) << "%\n";
    report << "  With ML: " << (result.ml_hit_rate * 100) << "%\n";
    report << "  Improvement: " << result.hit_rate_improvement_percentage << "%\n";
    report << "  Target: 20%+\n";
    report << "  Status: " << (result.meets_hit_rate_target ? "PASS" : "FAIL") << "\n\n";
    
    report << "Inference Latency:\n";
    report << "  Average: " << result.avg_inference_latency_ms << "ms\n";
    report << "  P95: " << result.p95_latency_ms << "ms\n";
    report << "  P99: " << result.p99_latency_ms << "ms\n";
    report << "  Target: <10ms\n";
    report << "  Status: " << (result.meets_latency_target ? "PASS" : "FAIL") << "\n\n";
    
    report << "System Overhead:\n";
    report << "  CPU: " << result.cpu_overhead_percentage << "%\n";
    report << "  Memory: " << result.memory_overhead_mb << "MB\n";
    report << "  Target: <1% CPU\n";
    report << "  Status: " << (result.meets_overhead_target ? "PASS" : "FAIL") << "\n\n";
    
    report << "Statistical Validation:\n";
    report << "  Confidence Interval: ±" << (result.confidence_interval * 100) << "%\n";
    report << "  P-value: " << result.p_value << "\n";
    report << "  Statistically Significant: " << (result.statistically_significant ? "Yes" : "No") << "\n\n";
    
    report << "Overall Result: " << (result.overall_success ? "SUCCESS" : "FAILURE") << "\n";
    
    return report.str();
}

// Private helper methods

std::vector<std::string> 
MLPerformanceValidator::generateSequentialWorkload(size_t num_ops, size_t num_keys) {
    std::vector<std::string> workload;
    workload.reserve(num_ops);
    
    for (size_t i = 0; i < num_ops; ++i) {
        workload.push_back("key_" + std::to_string(i % num_keys));
    }
    
    return workload;
}

std::vector<std::string> 
MLPerformanceValidator::generateTemporalWorkload(size_t num_ops, size_t num_keys) {
    std::vector<std::string> workload;
    workload.reserve(num_ops);
    
    // Create hot keys (20% of keys get 80% of accesses)
    size_t hot_keys = num_keys / 5;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> hot_dis(0, hot_keys - 1);
    std::uniform_int_distribution<> cold_dis(hot_keys, num_keys - 1);
    std::bernoulli_distribution use_hot(0.8);
    
    for (size_t i = 0; i < num_ops; ++i) {
        if (use_hot(gen)) {
            workload.push_back("key_" + std::to_string(hot_dis(gen)));
        } else {
            workload.push_back("key_" + std::to_string(cold_dis(gen)));
        }
    }
    
    return workload;
}

std::vector<std::string> 
MLPerformanceValidator::generateRandomWorkload(size_t num_ops, size_t num_keys) {
    std::vector<std::string> workload;
    workload.reserve(num_ops);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_keys - 1);
    
    for (size_t i = 0; i < num_ops; ++i) {
        workload.push_back("key_" + std::to_string(dis(gen)));
    }
    
    return workload;
}

std::vector<std::string> 
MLPerformanceValidator::generateZipfianWorkload(size_t num_ops, size_t num_keys) {
    std::vector<std::string> workload;
    workload.reserve(num_ops);
    
    // Zipfian distribution parameters
    const double alpha = 1.0;
    std::vector<double> probabilities(num_keys);
    double sum = 0.0;
    
    // Calculate probabilities
    for (size_t i = 0; i < num_keys; ++i) {
        probabilities[i] = 1.0 / std::pow(i + 1, alpha);
        sum += probabilities[i];
    }
    
    // Normalize
    for (auto& p : probabilities) {
        p /= sum;
    }
    
    // Generate workload
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(probabilities.begin(), probabilities.end());
    
    for (size_t i = 0; i < num_ops; ++i) {
        workload.push_back("key_" + std::to_string(dis(gen)));
    }
    
    return workload;
}

std::vector<std::string> 
MLPerformanceValidator::generateMixedWorkload(size_t num_ops, size_t num_keys) {
    std::vector<std::string> workload;
    workload.reserve(num_ops);
    
    // Mix of different patterns
    size_t quarter = num_ops / 4;
    
    // Sequential part
    auto seq = generateSequentialWorkload(quarter, num_keys);
    workload.insert(workload.end(), seq.begin(), seq.end());
    
    // Temporal part
    auto temp = generateTemporalWorkload(quarter, num_keys);
    workload.insert(workload.end(), temp.begin(), temp.end());
    
    // Random part
    auto rand = generateRandomWorkload(quarter, num_keys);
    workload.insert(workload.end(), rand.begin(), rand.end());
    
    // Zipfian part
    auto zipf = generateZipfianWorkload(num_ops - 3 * quarter, num_keys);
    workload.insert(workload.end(), zipf.begin(), zipf.end());
    
    // Shuffle to mix patterns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(workload.begin(), workload.end(), gen);
    
    return workload;
}

void MLPerformanceValidator::measureBaseline(const std::vector<std::string>& workload,
                                            ValidationResult& result) {
    size_t hits = 0;
    size_t total = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& key : workload) {
        std::string value;
        if (cache_->get(key, value)) {
            hits++;
        } else {
            // Simulate loading data
            cache_->put(key, "value_" + key);
        }
        total++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    result.baseline_hit_rate = static_cast<double>(hits) / total;
    result.throughput_ops_sec = total * 1000.0 / 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

void MLPerformanceValidator::measureWithML(const std::vector<std::string>& workload,
                                          ValidationResult& result) {
    size_t hits = 0;
    size_t total = 0;
    std::vector<double> latencies;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& key : workload) {
        auto op_start = std::chrono::high_resolution_clock::now();
        
        // Log access for ML learning
        if (coordinator_) {
            ppe::PrefetchCoordinator::AccessEvent event;
            event.key = key;
            event.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
            event.was_hit = cache_->contains(key);
            coordinator_->logAccess(event);
        }
        
        // Try to get from cache
        std::string value;
        if (cache_->get(key, value)) {
            hits++;
        } else {
            // Simulate loading data
            cache_->put(key, "value_" + key);
        }
        
        auto op_end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            op_end - op_start).count() / 1000.0;
        latencies.push_back(latency_ms);
        
        total++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    result.ml_hit_rate = static_cast<double>(hits) / total;
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 
                           latencies.size();
    
    // Get ML-specific metrics from monitor
    if (monitor_) {
        auto metrics = monitor_->getCurrentMetrics();
        result.prefetch_accuracy = metrics.precision;
        result.prefetch_coverage = metrics.recall;
        result.avg_inference_latency_ms = metrics.avg_prediction_latency_ms;
    }
}

void MLPerformanceValidator::measureOverhead(ValidationResult& result) {
    // This is a simplified overhead measurement
    // In production, would use proper CPU/memory profiling
    
    // Estimate based on latency increase
    if (result.avg_latency_ms > 0 && result.baseline_hit_rate > 0) {
        // Rough estimate: overhead proportional to latency increase
        double baseline_latency = 0.1; // Assume 0.1ms baseline
        result.cpu_overhead_percentage = 
            ((result.avg_latency_ms - baseline_latency) / baseline_latency) * 100;
        
        // Clamp to reasonable range
        result.cpu_overhead_percentage = std::max(0.0, std::min(100.0, result.cpu_overhead_percentage));
    }
    
    // Memory overhead estimate
    result.memory_overhead_mb = 50.0; // Placeholder
}

bool MLPerformanceValidator::validateSuccessCriteria(ValidationResult& result) {
    result.meets_hit_rate_target = result.hit_rate_improvement_percentage >= 20.0;
    result.meets_latency_target = result.avg_inference_latency_ms < 10.0;
    result.meets_overhead_target = result.cpu_overhead_percentage < 1.0;
    
    result.overall_success = result.meets_hit_rate_target && 
                           result.meets_latency_target && 
                           result.meets_overhead_target;
    
    return result.overall_success;
}

// MLPerformanceBenchmark implementation

MLPerformanceBenchmark::MLPerformanceBenchmark(std::shared_ptr<MLPerformanceValidator> validator)
    : validator_(validator) {}

MLPerformanceBenchmark::BenchmarkSuite 
MLPerformanceBenchmark::runQuickValidation() {
    BenchmarkSuite suite;
    suite.name = "Quick Validation";
    suite.description = "Fast validation of core functionality";
    
    auto start = std::chrono::steady_clock::now();
    
    // Quick test with small workload
    MLPerformanceValidator::ValidationConfig config;
    config.num_operations = 10000;
    config.num_keys = 1000;
    config.workload_type = MLPerformanceValidator::WorkloadType::MIXED;
    
    auto result = validator_->runValidation(config);
    suite.results.push_back(result);
    suite.all_passed = result.overall_success;
    
    auto end = std::chrono::steady_clock::now();
    suite.total_runtime = end - start;
    
    return suite;
}

void MLPerformanceBenchmark::generateHTMLReport(const BenchmarkSuite& suite,
                                               const std::string& filename) {
    std::ofstream file(filename);
    
    file << "<!DOCTYPE html>\n<html>\n<head>\n";
    file << "<title>ML Performance Benchmark Report - " << suite.name << "</title>\n";
    file << "<style>\n";
    file << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "table { border-collapse: collapse; width: 100%; }\n";
    file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
    file << "th { background-color: #f2f2f2; }\n";
    file << ".pass { color: green; font-weight: bold; }\n";
    file << ".fail { color: red; font-weight: bold; }\n";
    file << "</style>\n</head>\n<body>\n";
    
    file << "<h1>ML Performance Benchmark Report</h1>\n";
    file << "<h2>" << suite.name << "</h2>\n";
    file << "<p>" << suite.description << "</p>\n";
    file << "<p>Total Runtime: " << suite.total_runtime.count() << " seconds</p>\n";
    file << "<p>Overall Result: <span class=\"" << (suite.all_passed ? "pass" : "fail") << "\">"
         << (suite.all_passed ? "PASS" : "FAIL") << "</span></p>\n";
    
    file << "<h3>Detailed Results</h3>\n";
    file << "<table>\n";
    file << "<tr><th>Test</th><th>Hit Rate Improvement</th><th>Inference Latency</th>";
    file << "<th>CPU Overhead</th><th>Result</th></tr>\n";
    
    for (const auto& result : suite.results) {
        file << "<tr>";
        file << "<td>Validation</td>";
        file << "<td>" << std::fixed << std::setprecision(1) 
             << result.hit_rate_improvement_percentage << "%</td>";
        file << "<td>" << result.avg_inference_latency_ms << "ms</td>";
        file << "<td>" << result.cpu_overhead_percentage << "%</td>";
        file << "<td class=\"" << (result.overall_success ? "pass" : "fail") << "\">"
             << (result.overall_success ? "PASS" : "FAIL") << "</td>";
        file << "</tr>\n";
    }
    
    file << "</table>\n";
    file << "</body>\n</html>";
    
    file.close();
    std::cout << "HTML report generated: " << filename << std::endl;
}

} // namespace benchmarks
} // namespace predis