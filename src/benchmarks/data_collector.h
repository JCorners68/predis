#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <map>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace predis {
namespace benchmarks {

struct TimeSeriesDataPoint {
    std::chrono::high_resolution_clock::time_point timestamp;
    double latency_us;
    size_t operations_per_second;
    size_t memory_usage_bytes;
    double cpu_utilization;
    double gpu_utilization;
    bool operation_success;
    std::string operation_type;
};

struct SystemMetrics {
    double cpu_usage_percent;
    double memory_usage_percent;
    double gpu_memory_usage_percent;
    double gpu_utilization_percent;
    size_t network_bytes_sent;
    size_t network_bytes_received;
    double disk_io_read_mbps;
    double disk_io_write_mbps;
    std::chrono::high_resolution_clock::time_point collection_time;
};

struct PerformanceWindow {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::vector<double> latencies_us;
    size_t total_operations;
    size_t successful_operations;
    size_t failed_operations;
    double average_latency_us;
    double p50_latency_us;
    double p95_latency_us;
    double p99_latency_us;
    double throughput_ops_per_sec;
    SystemMetrics system_metrics;
};

class RealTimeDataCollector {
public:
    struct CollectionConfig {
        std::chrono::milliseconds collection_interval{100};
        size_t max_data_points{10000};
        bool enable_system_metrics{true};
        bool enable_gpu_metrics{true};
        bool enable_network_metrics{true};
        std::string output_file_prefix{"benchmark_data"};
        bool auto_export_csv{true};
        size_t sliding_window_size{1000};
    };

    explicit RealTimeDataCollector(const CollectionConfig& config = {});
    ~RealTimeDataCollector();

    void start_collection();
    void stop_collection();
    void record_operation(const std::string& operation_type, 
                         std::chrono::microseconds latency,
                         bool success = true);
    void record_system_metrics(const SystemMetrics& metrics);
    
    std::vector<TimeSeriesDataPoint> get_time_series_data() const;
    std::vector<PerformanceWindow> get_sliding_windows() const;
    SystemMetrics get_current_system_metrics() const;
    
    void export_to_csv(const std::string& filename) const;
    void export_to_json(const std::string& filename) const;
    void clear_data();
    
    size_t get_total_operations() const;
    double get_current_throughput() const;
    double get_average_latency_us() const;

private:
    CollectionConfig config_;
    mutable std::mutex data_mutex_;
    std::vector<TimeSeriesDataPoint> time_series_data_;
    std::queue<PerformanceWindow> sliding_windows_;
    
    std::thread collection_thread_;
    std::atomic<bool> is_collecting_{false};
    std::condition_variable collection_cv_;
    std::mutex collection_mutex_;
    
    SystemMetrics current_system_metrics_;
    std::chrono::high_resolution_clock::time_point last_metrics_update_;
    
    void collection_worker();
    SystemMetrics collect_system_metrics();
    void update_sliding_windows();
    void export_data_if_needed();
};

class BenchmarkDataAnalyzer {
public:
    struct AnalysisResults {
        struct PerformanceComparison {
            double improvement_factor;
            double statistical_significance;
            double confidence_interval_lower;
            double confidence_interval_upper;
            bool meets_epic2_targets;
            std::string performance_category;
        };
        
        PerformanceComparison redis_vs_predis;
        std::map<std::string, double> operation_type_improvements;
        std::vector<double> throughput_over_time;
        std::vector<double> latency_distribution;
        
        struct TrendAnalysis {
            double performance_trend_slope;
            double r_squared;
            bool is_performance_stable;
            std::vector<std::string> performance_anomalies;
        };
        TrendAnalysis trend_analysis;
        
        struct ResourceUtilization {
            double avg_cpu_utilization;
            double avg_gpu_utilization;
            double avg_memory_utilization;
            double resource_efficiency_score;
        };
        ResourceUtilization resource_utilization;
    };

    static AnalysisResults analyze_benchmark_data(
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        double target_improvement_factor = 15.0);
    
    static std::vector<double> calculate_percentiles(
        const std::vector<double>& data,
        const std::vector<double>& percentiles = {50.0, 95.0, 99.0, 99.9});
    
    static double calculate_cohens_d(const std::vector<double>& group1,
                                   const std::vector<double>& group2);
    
    static bool validate_epic2_performance_targets(const AnalysisResults& results);
    
    static std::string generate_performance_summary(const AnalysisResults& results);

private:
    static double calculate_improvement_factor(const std::vector<double>& baseline,
                                             const std::vector<double>& improved);
    static double perform_statistical_test(const std::vector<double>& group1,
                                          const std::vector<double>& group2);
    static std::pair<double, double> calculate_confidence_interval(
        const std::vector<double>& data, double confidence_level = 0.95);
};

} // namespace benchmarks
} // namespace predis