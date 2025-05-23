#ifndef PREDIS_PPE_PREFETCH_MONITOR_H_
#define PREDIS_PPE_PREFETCH_MONITOR_H_

#include <atomic>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace predis {
namespace ppe {

// Forward declaration
class PrefetchCoordinator;

/**
 * @brief Real-time monitoring and analytics for prefetch performance
 * 
 * Tracks prefetch effectiveness, ML model accuracy, and cache hit improvements
 */
class PrefetchMonitor {
public:
    struct TimeWindow {
        std::chrono::seconds duration;
        std::string name;
    };
    
    struct PrefetchMetrics {
        // Core metrics
        uint64_t total_predictions = 0;
        uint64_t accurate_predictions = 0;
        uint64_t prefetch_requests = 0;
        uint64_t successful_prefetches = 0;
        uint64_t cache_hits_from_prefetch = 0;
        uint64_t wasted_prefetches = 0;
        
        // Timing metrics
        double avg_prediction_latency_ms = 0.0;
        double avg_prefetch_latency_ms = 0.0;
        double p95_prediction_latency_ms = 0.0;
        double p99_prediction_latency_ms = 0.0;
        
        // Effectiveness metrics
        double precision = 0.0;  // Accurate predictions / Total predictions
        double recall = 0.0;     // Cache hits from prefetch / Total cache hits
        double f1_score = 0.0;   // Harmonic mean of precision and recall
        double hit_rate_improvement = 0.0;  // % improvement over baseline
        
        // Resource utilization
        double gpu_utilization = 0.0;
        double memory_usage_mb = 0.0;
        double bandwidth_usage_mbps = 0.0;
        
        // Model performance
        double model_accuracy = 0.0;
        double model_confidence_avg = 0.0;
        uint64_t model_updates = 0;
        
        // Time window
        std::chrono::time_point<std::chrono::system_clock> window_start;
        std::chrono::time_point<std::chrono::system_clock> window_end;
    };
    
    struct KeyMetrics {
        uint64_t access_count = 0;
        uint64_t prefetch_count = 0;
        uint64_t hit_count = 0;
        double avg_time_to_access_ms = 0.0;  // Time between prefetch and actual access
        std::chrono::time_point<std::chrono::system_clock> last_access;
        std::chrono::time_point<std::chrono::system_clock> last_prefetch;
    };
    
    struct ModelPerformance {
        std::string model_type;
        uint64_t predictions_made = 0;
        uint64_t correct_predictions = 0;
        double avg_confidence = 0.0;
        double avg_latency_ms = 0.0;
        std::vector<double> confidence_distribution;  // Histogram buckets
    };

public:
    PrefetchMonitor();
    ~PrefetchMonitor() = default;
    
    // Set the coordinator to monitor
    void setPrefetchCoordinator(std::shared_ptr<PrefetchCoordinator> coordinator);
    
    // Start/stop monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return monitoring_; }
    
    // Record events
    void recordPrediction(const std::string& key, float confidence, 
                         bool was_used, double latency_ms);
    void recordPrefetch(const std::string& key, bool success, double latency_ms);
    void recordCacheAccess(const std::string& key, bool was_hit, bool from_prefetch);
    void recordModelUpdate(const std::string& model_type);
    
    // Get metrics for different time windows
    PrefetchMetrics getMetrics(const TimeWindow& window) const;
    PrefetchMetrics getCurrentMetrics() const;
    PrefetchMetrics getLifetimeMetrics() const;
    
    // Get per-key analytics
    KeyMetrics getKeyMetrics(const std::string& key) const;
    std::vector<std::pair<std::string, KeyMetrics>> getTopKeys(size_t n) const;
    
    // Model performance tracking
    ModelPerformance getModelPerformance(const std::string& model_type) const;
    std::vector<ModelPerformance> getAllModelPerformance() const;
    
    // Real-time monitoring
    double getCurrentHitRate() const;
    double getBaselineHitRate() const { return baseline_hit_rate_; }
    double getHitRateImprovement() const;
    
    // Alerting thresholds
    void setAlertThreshold(const std::string& metric, double threshold);
    bool checkAlerts(std::vector<std::string>& triggered_alerts) const;
    
    // Export metrics
    std::string exportMetricsJSON() const;
    std::string exportMetricsCSV() const;
    void exportToPrometheus(const std::string& endpoint) const;
    
    // Reset metrics
    void resetMetrics();
    void resetKeyMetrics();
    
private:
    std::shared_ptr<PrefetchCoordinator> coordinator_;
    std::atomic<bool> monitoring_{false};
    
    // Metrics storage
    mutable std::mutex metrics_mutex_;
    PrefetchMetrics current_metrics_;
    PrefetchMetrics lifetime_metrics_;
    std::unordered_map<std::string, KeyMetrics> key_metrics_;
    std::unordered_map<std::string, ModelPerformance> model_performance_;
    
    // Time-series data for windowed metrics
    struct TimedEvent {
        std::chrono::time_point<std::chrono::system_clock> timestamp;
        std::string event_type;
        std::string key;
        double value;
    };
    std::vector<TimedEvent> event_history_;
    size_t max_history_size_ = 100000;
    
    // Baseline tracking
    double baseline_hit_rate_ = 0.0;
    uint64_t baseline_sample_count_ = 0;
    
    // Alert thresholds
    std::unordered_map<std::string, double> alert_thresholds_;
    
    // Helper methods
    void updateMetrics(PrefetchMetrics& metrics, const TimedEvent& event);
    void pruneEventHistory();
    double calculatePercentile(const std::vector<double>& values, double percentile) const;
    void updateKeyMetrics(const std::string& key, const std::string& event_type);
};

/**
 * @brief Dashboard for real-time prefetch performance visualization
 */
class PrefetchDashboard {
public:
    PrefetchDashboard(std::shared_ptr<PrefetchMonitor> monitor);
    ~PrefetchDashboard() = default;
    
    // Generate HTML dashboard
    std::string generateHTML() const;
    
    // Generate JSON for web API
    std::string generateJSON() const;
    
    // Console output
    void printSummary() const;
    void printDetailedMetrics() const;
    
    // Real-time updates
    void startRealtimeUpdates(int update_interval_ms = 1000);
    void stopRealtimeUpdates();
    
private:
    std::shared_ptr<PrefetchMonitor> monitor_;
    std::thread update_thread_;
    std::atomic<bool> updating_{false};
    
    std::string formatMetrics(const PrefetchMonitor::PrefetchMetrics& metrics) const;
    std::string generateChartData() const;
};

} // namespace ppe
} // namespace predis

#endif // PREDIS_PPE_PREFETCH_MONITOR_H_