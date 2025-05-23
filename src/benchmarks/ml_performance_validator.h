#ifndef PREDIS_BENCHMARKS_ML_PERFORMANCE_VALIDATOR_H_
#define PREDIS_BENCHMARKS_ML_PERFORMANCE_VALIDATOR_H_

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <random>

namespace predis {

// Forward declarations
namespace ppe {
    class PrefetchCoordinator;
    class PrefetchMonitor;
}

namespace core {
    class SimpleCacheManager;
}

namespace benchmarks {

/**
 * @brief Comprehensive performance validation framework for ML prefetching
 * 
 * Validates Epic 3 success criteria:
 * - 20%+ cache hit rate improvement
 * - <10ms ML inference latency
 * - <1% system overhead
 */
class MLPerformanceValidator {
public:
    enum class WorkloadType {
        SEQUENTIAL,      // Sequential key access (e.g., key_1, key_2, key_3)
        TEMPORAL,        // Repeated access to same keys over time
        RANDOM,          // Random access pattern
        ZIPFIAN,         // Zipfian distribution (80/20 rule)
        MIXED,           // Mix of different patterns
        REAL_WORLD      // Based on actual Redis traces
    };
    
    struct ValidationConfig {
        size_t num_operations = 100000;
        size_t num_keys = 10000;
        size_t value_size = 1024;
        WorkloadType workload_type = WorkloadType::MIXED;
        bool enable_prefetching = true;
        bool enable_ml = true;
        double ml_confidence_threshold = 0.7;
        int num_threads = 1;
        size_t warmup_operations = 1000;
        std::string trace_file;  // For REAL_WORLD workload
    };
    
    struct ValidationResult {
        // Core metrics
        double baseline_hit_rate = 0.0;
        double ml_hit_rate = 0.0;
        double hit_rate_improvement = 0.0;
        double hit_rate_improvement_percentage = 0.0;
        
        // Performance metrics
        double avg_latency_ms = 0.0;
        double p50_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        double p99_latency_ms = 0.0;
        double throughput_ops_sec = 0.0;
        
        // ML metrics
        double avg_inference_latency_ms = 0.0;
        double max_inference_latency_ms = 0.0;
        double prefetch_accuracy = 0.0;
        double prefetch_coverage = 0.0;
        
        // Resource metrics
        double cpu_overhead_percentage = 0.0;
        double memory_overhead_mb = 0.0;
        double gpu_utilization_percentage = 0.0;
        
        // Statistical validation
        double confidence_interval = 0.0;
        double p_value = 0.0;
        bool statistically_significant = false;
        
        // Success criteria
        bool meets_hit_rate_target = false;      // 20%+ improvement
        bool meets_latency_target = false;       // <10ms inference
        bool meets_overhead_target = false;      // <1% overhead
        bool overall_success = false;
        
        // Detailed breakdown
        std::unordered_map<std::string, double> workload_breakdown;
        std::vector<double> latency_histogram;
        
        // Timestamp
        std::chrono::time_point<std::chrono::system_clock> test_time;
    };
    
    struct ABTestResult {
        ValidationResult control_group;   // Without ML/prefetching
        ValidationResult test_group;      // With ML/prefetching
        double improvement_percentage = 0.0;
        double statistical_power = 0.0;
        std::string recommendation;
    };

public:
    MLPerformanceValidator();
    ~MLPerformanceValidator() = default;
    
    // Set components
    void setCacheManager(std::shared_ptr<core::SimpleCacheManager> cache);
    void setPrefetchCoordinator(std::shared_ptr<ppe::PrefetchCoordinator> coordinator);
    void setPrefetchMonitor(std::shared_ptr<ppe::PrefetchMonitor> monitor);
    
    // Run validation tests
    ValidationResult runValidation(const ValidationConfig& config);
    ABTestResult runABTest(const ValidationConfig& config, double test_split = 0.5);
    
    // Specific validations
    ValidationResult validateHitRateImprovement(const ValidationConfig& config);
    ValidationResult validateInferenceLatency(const ValidationConfig& config);
    ValidationResult validateSystemOverhead(const ValidationConfig& config);
    
    // Workload generation
    std::vector<std::string> generateWorkload(const ValidationConfig& config);
    void simulateRealWorldTrace(const std::string& trace_file);
    
    // Performance comparison
    ValidationResult compareWithBaseline(const ValidationConfig& config);
    ValidationResult compareWithRedis(const ValidationConfig& config);
    
    // Regression testing
    bool runRegressionTests(const std::vector<ValidationConfig>& configs);
    void saveRegressionBaseline(const std::string& filename);
    bool compareWithRegressionBaseline(const std::string& filename);
    
    // Reporting
    std::string generateReport(const ValidationResult& result);
    std::string generateABTestReport(const ABTestResult& result);
    void exportResultsJSON(const ValidationResult& result, const std::string& filename);
    void exportResultsCSV(const ValidationResult& result, const std::string& filename);
    
    // Statistical analysis
    double calculateConfidenceInterval(const std::vector<double>& samples);
    double calculatePValue(const std::vector<double>& control, 
                          const std::vector<double>& test);
    bool isStatisticallySignificant(double p_value, double alpha = 0.05);
    
private:
    std::shared_ptr<core::SimpleCacheManager> cache_;
    std::shared_ptr<ppe::PrefetchCoordinator> coordinator_;
    std::shared_ptr<ppe::PrefetchMonitor> monitor_;
    
    // Workload generators
    std::vector<std::string> generateSequentialWorkload(size_t num_ops, size_t num_keys);
    std::vector<std::string> generateTemporalWorkload(size_t num_ops, size_t num_keys);
    std::vector<std::string> generateRandomWorkload(size_t num_ops, size_t num_keys);
    std::vector<std::string> generateZipfianWorkload(size_t num_ops, size_t num_keys);
    std::vector<std::string> generateMixedWorkload(size_t num_ops, size_t num_keys);
    
    // Measurement helpers
    void measureBaseline(const std::vector<std::string>& workload, 
                        ValidationResult& result);
    void measureWithML(const std::vector<std::string>& workload,
                      ValidationResult& result);
    void measureOverhead(ValidationResult& result);
    
    // Statistical helpers
    std::vector<double> collectLatencySamples(const std::vector<std::string>& workload);
    void calculatePercentiles(const std::vector<double>& samples,
                            ValidationResult& result);
    
    // Validation helpers
    bool validateSuccessCriteria(ValidationResult& result);
    void populateWorkloadBreakdown(const std::vector<std::string>& workload,
                                  ValidationResult& result);
};

/**
 * @brief Automated performance benchmark suite
 */
class MLPerformanceBenchmark {
public:
    struct BenchmarkConfig {
        std::vector<MLPerformanceValidator::WorkloadType> workloads;
        std::vector<size_t> operation_counts = {10000, 50000, 100000};
        std::vector<size_t> key_counts = {1000, 5000, 10000};
        std::vector<int> thread_counts = {1, 4, 8};
        bool enable_gpu = true;
        bool save_results = true;
        std::string output_dir = "./benchmark_results/";
    };
    
    struct BenchmarkSuite {
        std::string name;
        std::string description;
        std::vector<MLPerformanceValidator::ValidationResult> results;
        std::chrono::duration<double> total_runtime;
        bool all_passed = true;
    };

public:
    MLPerformanceBenchmark(std::shared_ptr<MLPerformanceValidator> validator);
    ~MLPerformanceBenchmark() = default;
    
    // Run benchmark suites
    BenchmarkSuite runCompleteSuite(const BenchmarkConfig& config);
    BenchmarkSuite runQuickValidation();
    BenchmarkSuite runNightlyBenchmark();
    BenchmarkSuite runReleaseBenchmark();
    
    // Specific benchmarks
    void benchmarkHitRateImprovement();
    void benchmarkInferenceLatency();
    void benchmarkScalability();
    void benchmarkMemoryUsage();
    
    // Reporting
    void generateHTMLReport(const BenchmarkSuite& suite, const std::string& filename);
    void generateMarkdownReport(const BenchmarkSuite& suite, const std::string& filename);
    void publishResults(const BenchmarkSuite& suite);
    
private:
    std::shared_ptr<MLPerformanceValidator> validator_;
    
    void runWorkloadBenchmark(const MLPerformanceValidator::ValidationConfig& base_config,
                             BenchmarkSuite& suite);
};

} // namespace benchmarks
} // namespace predis

#endif // PREDIS_BENCHMARKS_ML_PERFORMANCE_VALIDATOR_H_