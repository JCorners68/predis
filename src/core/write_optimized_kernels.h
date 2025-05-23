#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>

namespace predis {

// Forward declarations
struct WriteOptimizedBucket;

// Write optimization strategies
enum class WriteStrategy {
    BASELINE,           // Original implementation
    WARP_COOPERATIVE,   // Warp-level cooperation
    LOCK_FREE,          // Lock-free with CAS
    MEMORY_OPTIMIZED,   // Prefetching and coalescing
    WRITE_COMBINING     // Combine small writes
};

// Performance metrics for write operations
struct WritePerformanceMetrics {
    double throughput_ops_sec;
    double latency_ms;
    double memory_bandwidth_gbps;
    uint64_t conflicts_detected;
    uint64_t operations_completed;
    float gpu_utilization;
};

// Write optimization configuration
struct WriteOptimizationConfig {
    WriteStrategy strategy = WriteStrategy::MEMORY_OPTIMIZED;
    size_t batch_size = 1024;
    size_t shared_memory_size = 48 * 1024;  // 48KB shared memory
    int num_streams = 4;
    bool enable_prefetching = true;
    bool enable_write_combining = true;
};

// Optimized write kernels
__global__ void optimized_batch_write_kernel(WriteOptimizedBucket* buckets,
                                            size_t num_buckets,
                                            const uint8_t* keys,
                                            const uint8_t* values,
                                            const size_t* key_sizes,
                                            const size_t* value_sizes,
                                            size_t num_operations,
                                            uint32_t* success_flags);

__global__ void memory_optimized_write_kernel(WriteOptimizedBucket* buckets,
                                            size_t num_buckets,
                                            const uint8_t* __restrict__ keys,
                                            const uint8_t* __restrict__ values,
                                            size_t num_operations,
                                            size_t key_stride,
                                            size_t value_stride);

__global__ void write_combining_kernel(WriteOptimizedBucket* buckets,
                                     size_t num_buckets,
                                     const uint8_t* keys,
                                     const uint8_t* values,
                                     const size_t* offsets,
                                     size_t num_operations);

// Device functions
__device__ uint64_t optimized_hash(const uint8_t* key, size_t size);

__device__ void warp_cooperative_write(WriteOptimizedBucket* buckets,
                                      size_t num_buckets,
                                      const uint8_t* key,
                                      size_t key_size,
                                      const uint8_t* value,
                                      size_t value_size);

__device__ bool lock_free_insert(WriteOptimizedBucket* bucket,
                                const uint8_t* key,
                                size_t key_size,
                                const uint8_t* value,
                                size_t value_size);

// Host-side write optimization manager
class WriteOptimizationManager {
private:
    void* d_buckets_;  // Device memory for optimized buckets
    size_t num_buckets_;
    WriteOptimizationConfig config_;
    cudaStream_t streams_[4];
    
    // Performance tracking
    WritePerformanceMetrics current_metrics_;
    std::vector<WritePerformanceMetrics> historical_metrics_;
    
public:
    explicit WriteOptimizationManager(size_t num_buckets);
    ~WriteOptimizationManager();
    
    // Batch write operations
    void batch_write(const std::vector<std::pair<std::string, std::string>>& kv_pairs);
    
    // Single write operation
    bool write(const std::string& key, const std::string& value);
    
    // Configuration
    void set_config(const WriteOptimizationConfig& config) { config_ = config; }
    WriteOptimizationConfig get_config() const { return config_; }
    
    // Performance metrics
    WritePerformanceMetrics get_current_metrics() const { return current_metrics_; }
    std::vector<WritePerformanceMetrics> get_historical_metrics() const { return historical_metrics_; }
    
    // Strategy selection
    void set_write_strategy(WriteStrategy strategy) { config_.strategy = strategy; }
    
    // Performance analysis
    void analyze_performance();
    void reset_metrics();
};

// Write pattern analyzer for optimization decisions
class WritePatternAnalyzer {
private:
    struct WritePattern {
        size_t avg_key_size;
        size_t avg_value_size;
        double write_frequency;
        double temporal_locality;
        double spatial_locality;
    };
    
    WritePattern current_pattern_;
    
public:
    WritePatternAnalyzer() = default;
    
    // Analyze write patterns
    void analyze(const std::vector<std::pair<std::string, std::string>>& recent_writes);
    
    // Get optimization recommendations
    WriteStrategy recommend_strategy() const;
    WriteOptimizationConfig recommend_config() const;
    
    // Pattern metrics
    WritePattern get_current_pattern() const { return current_pattern_; }
};

// Benchmark utilities for write optimization
class WriteOptimizationBenchmark {
private:
    WriteOptimizationManager* manager_;
    size_t num_operations_;
    
public:
    WriteOptimizationBenchmark(WriteOptimizationManager* manager, size_t num_ops)
        : manager_(manager), num_operations_(num_ops) {}
    
    // Run benchmarks for different strategies
    std::vector<WritePerformanceMetrics> benchmark_all_strategies();
    WritePerformanceMetrics benchmark_strategy(WriteStrategy strategy);
    
    // Compare with baseline
    double calculate_improvement(const WritePerformanceMetrics& optimized,
                               const WritePerformanceMetrics& baseline);
    
    // Generate report
    void generate_report(const std::string& filename);
};

} // namespace predis