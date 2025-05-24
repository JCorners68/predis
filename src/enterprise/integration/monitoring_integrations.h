#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace predis {
namespace enterprise {

// Base metrics structure
struct Metric {
    std::string name;
    double value;
    std::unordered_map<std::string, std::string> labels;
    std::chrono::system_clock::time_point timestamp;
    
    enum Type {
        COUNTER,
        GAUGE,
        HISTOGRAM,
        SUMMARY
    } type;
};

// Base monitoring integration interface
class MonitoringIntegration {
public:
    virtual ~MonitoringIntegration() = default;
    
    // Export a single metric
    virtual void ExportMetric(const Metric& metric) = 0;
    
    // Export batch of metrics
    virtual void ExportBatch(const std::vector<Metric>& metrics) = 0;
    
    // Integration name
    virtual std::string GetName() const = 0;
    
    // Check if connection is healthy
    virtual bool IsHealthy() const = 0;
};

// Prometheus integration
class PrometheusExporter : public MonitoringIntegration {
public:
    struct Config {
        int port = 9090;
        std::string endpoint = "/metrics";
        bool enable_histogram_buckets = true;
        std::vector<double> default_buckets = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0};
    };
    
    explicit PrometheusExporter(const Config& config);
    ~PrometheusExporter();
    
    void ExportMetric(const Metric& metric) override;
    void ExportBatch(const std::vector<Metric>& metrics) override;
    std::string GetName() const override { return "prometheus"; }
    bool IsHealthy() const override;
    
    // Prometheus-specific methods
    void RegisterCounter(const std::string& name, const std::string& help);
    void RegisterGauge(const std::string& name, const std::string& help);
    void RegisterHistogram(const std::string& name, const std::string& help,
                          const std::vector<double>& buckets = {});
    
    // HTTP handler for /metrics endpoint
    std::string GenerateMetricsResponse() const;
    
private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// DataDog integration
class DataDogExporter : public MonitoringIntegration {
public:
    struct Config {
        std::string api_key;
        std::string app_key;
        std::string host = "https://api.datadoghq.com";
        std::string site = "datadoghq.com";  // or datadoghq.eu
        std::vector<std::string> global_tags;
        size_t batch_size = 100;
        std::chrono::seconds flush_interval{10};
    };
    
    explicit DataDogExporter(const Config& config);
    ~DataDogExporter();
    
    void ExportMetric(const Metric& metric) override;
    void ExportBatch(const std::vector<Metric>& metrics) override;
    std::string GetName() const override { return "datadog"; }
    bool IsHealthy() const override;
    
    // DataDog-specific methods
    void SendEvent(const std::string& title, const std::string& text,
                   const std::string& alert_type = "info");
    void SendServiceCheck(const std::string& check_name, int status,
                         const std::string& message = "");
    
private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// New Relic integration
class NewRelicExporter : public MonitoringIntegration {
public:
    struct Config {
        std::string license_key;
        std::string app_name = "Predis";
        std::string host = "https://metric-api.newrelic.com";
        std::string region = "US";  // or "EU"
        size_t batch_size = 100;
        std::chrono::seconds flush_interval{10};
        bool enable_distributed_tracing = true;
    };
    
    explicit NewRelicExporter(const Config& config);
    ~NewRelicExporter();
    
    void ExportMetric(const Metric& metric) override;
    void ExportBatch(const std::vector<Metric>& metrics) override;
    std::string GetName() const override { return "newrelic"; }
    bool IsHealthy() const override;
    
    // New Relic-specific methods
    void RecordCustomEvent(const std::string& event_type,
                          const std::unordered_map<std::string, std::string>& attributes);
    std::string StartTransaction(const std::string& name);
    void EndTransaction(const std::string& transaction_id);
    
private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// AWS CloudWatch integration
class CloudWatchExporter : public MonitoringIntegration {
public:
    struct Config {
        std::string region = "us-east-1";
        std::string namespace_prefix = "Predis";
        std::string access_key_id;  // Optional, uses IAM role if not provided
        std::string secret_access_key;
        size_t batch_size = 20;  // CloudWatch limit
        std::chrono::seconds flush_interval{60};
        bool enable_detailed_metrics = false;
    };
    
    explicit CloudWatchExporter(const Config& config);
    ~CloudWatchExporter();
    
    void ExportMetric(const Metric& metric) override;
    void ExportBatch(const std::vector<Metric>& metrics) override;
    std::string GetName() const override { return "cloudwatch"; }
    bool IsHealthy() const override;
    
    // CloudWatch-specific methods
    void PutMetricAlarm(const std::string& alarm_name,
                       const std::string& metric_name,
                       double threshold,
                       const std::string& comparison_operator);
    
private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Monitoring manager that handles multiple integrations
class MonitoringManager {
public:
    MonitoringManager();
    ~MonitoringManager();
    
    // Add integration
    void AddIntegration(std::unique_ptr<MonitoringIntegration> integration);
    
    // Remove integration by name
    void RemoveIntegration(const std::string& name);
    
    // Export to all integrations
    void ExportMetric(const Metric& metric);
    void ExportBatch(const std::vector<Metric>& metrics);
    
    // Predis-specific metric helpers
    void RecordCacheHit(const std::string& operation, double latency_ms);
    void RecordCacheMiss(const std::string& operation, double latency_ms);
    void RecordGPUUtilization(double utilization_pct);
    void RecordMLInference(double inference_time_ms, const std::string& model_name);
    void RecordMemoryUsage(size_t bytes_used, size_t bytes_total);
    void RecordBatchOperation(size_t batch_size, double total_time_ms);
    
    // Get health status
    struct HealthStatus {
        bool overall_healthy;
        std::unordered_map<std::string, bool> integration_health;
        std::chrono::system_clock::time_point last_check;
    };
    
    HealthStatus GetHealthStatus() const;
    
    // Metrics aggregation
    struct AggregatedMetrics {
        double avg_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double cache_hit_rate;
        double gpu_utilization;
        size_t total_operations;
        size_t total_hits;
        size_t total_misses;
        std::chrono::system_clock::time_point period_start;
        std::chrono::system_clock::time_point period_end;
    };
    
    AggregatedMetrics GetAggregatedMetrics(std::chrono::minutes period = std::chrono::minutes(5));
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Custom metric collectors
class PredisMetricCollector {
public:
    PredisMetricCollector();
    
    // Cache metrics
    void CollectCacheMetrics(MonitoringManager& manager);
    
    // GPU metrics
    void CollectGPUMetrics(MonitoringManager& manager);
    
    // ML metrics
    void CollectMLMetrics(MonitoringManager& manager);
    
    // System metrics
    void CollectSystemMetrics(MonitoringManager& manager);
    
    // Start automatic collection
    void StartCollection(MonitoringManager& manager,
                        std::chrono::seconds interval = std::chrono::seconds(10));
    void StopCollection();
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace enterprise
}  // namespace predis