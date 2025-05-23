#ifndef PREDIS_ML_MODEL_PERFORMANCE_MONITOR_H_
#define PREDIS_ML_MODEL_PERFORMANCE_MONITOR_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <deque>

namespace predis {
namespace ml {

// Forward declarations
class BaseModel;
class AdaptiveLearningSystem;

/**
 * @brief Real-time monitoring for ML model performance in production
 * 
 * Tracks model accuracy, drift, and triggers adaptive learning
 */
class ModelPerformanceMonitor {
public:
    struct PerformanceMetrics {
        // Accuracy metrics
        double accuracy = 0.0;
        double precision = 0.0;
        double recall = 0.0;
        double f1_score = 0.0;
        
        // Error analysis
        double mean_absolute_error = 0.0;
        double mean_squared_error = 0.0;
        double error_variance = 0.0;
        
        // Latency metrics
        double avg_inference_time_ms = 0.0;
        double p50_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        double p99_latency_ms = 0.0;
        
        // Throughput
        double predictions_per_second = 0.0;
        uint64_t total_predictions = 0;
        
        // Time window
        std::chrono::time_point<std::chrono::system_clock> window_start;
        std::chrono::time_point<std::chrono::system_clock> window_end;
    };
    
    struct ModelComparison {
        std::string model_a_id;
        std::string model_b_id;
        PerformanceMetrics metrics_a;
        PerformanceMetrics metrics_b;
        double improvement_percentage = 0.0;
        bool statistically_significant = false;
        double p_value = 0.0;
    };
    
    struct AlertConfig {
        double accuracy_threshold = 0.8;
        double latency_threshold_ms = 10.0;
        double drift_threshold = 0.05;
        bool enable_email_alerts = false;
        bool enable_slack_alerts = false;
        std::string alert_webhook_url;
    };
    
    struct Alert {
        enum class Type {
            ACCURACY_DEGRADATION,
            LATENCY_SPIKE,
            DRIFT_DETECTED,
            MODEL_FAILURE,
            RETRAINING_NEEDED
        };
        
        Type type;
        std::string message;
        double severity;  // 0-1
        std::chrono::time_point<std::chrono::system_clock> timestamp;
        std::unordered_map<std::string, std::string> metadata;
    };

public:
    ModelPerformanceMonitor();
    ~ModelPerformanceMonitor() = default;
    
    // Set components
    void setModel(std::shared_ptr<BaseModel> model);
    void setAdaptiveLearningSystem(std::shared_ptr<AdaptiveLearningSystem> als);
    
    // Record predictions
    void recordPrediction(const std::vector<float>& features,
                         float predicted_value,
                         float actual_value,
                         double inference_time_ms);
    
    void recordBatchPredictions(const std::vector<std::vector<float>>& features,
                               const std::vector<float>& predicted_values,
                               const std::vector<float>& actual_values,
                               double batch_inference_time_ms);
    
    // Get metrics
    PerformanceMetrics getCurrentMetrics() const;
    PerformanceMetrics getMetricsForWindow(std::chrono::minutes window_size) const;
    std::vector<PerformanceMetrics> getHistoricalMetrics(size_t num_windows) const;
    
    // Model comparison
    ModelComparison compareModels(const std::string& model_a_id,
                                 const std::string& model_b_id);
    
    // Drift detection
    bool isDriftDetected() const { return drift_detected_; }
    double getDriftMagnitude() const { return drift_magnitude_; }
    void resetDriftDetection();
    
    // Alerting
    void setAlertConfig(const AlertConfig& config);
    std::vector<Alert> getActiveAlerts() const;
    void acknowledgeAlert(size_t alert_id);
    
    // Reporting
    std::string generatePerformanceReport() const;
    void exportMetricsCSV(const std::string& filename) const;
    void exportMetricsJSON(const std::string& filename) const;
    
    // Real-time dashboard data
    std::string getDashboardJSON() const;
    
private:
    std::shared_ptr<BaseModel> model_;
    std::shared_ptr<AdaptiveLearningSystem> adaptive_learning_;
    AlertConfig alert_config_;
    
    // Metrics storage
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    std::deque<PerformanceMetrics> historical_metrics_;
    size_t max_history_size_ = 100;
    
    // Prediction tracking
    struct PredictionRecord {
        std::vector<float> features;
        float predicted;
        float actual;
        double inference_time_ms;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    std::deque<PredictionRecord> prediction_history_;
    size_t max_predictions_ = 10000;
    
    // Drift detection
    std::atomic<bool> drift_detected_{false};
    std::atomic<double> drift_magnitude_{0.0};
    std::vector<double> error_history_;
    
    // Alerts
    std::vector<Alert> active_alerts_;
    std::atomic<size_t> alert_counter_{0};
    
    // Internal methods
    void updateMetrics(const PredictionRecord& record);
    void checkForAlerts();
    void sendAlert(const Alert& alert);
    void updateDriftDetection(double error);
    double calculateStatistic(const std::vector<double>& values, 
                            const std::string& statistic) const;
    void pruneHistory();
};

/**
 * @brief Production monitoring dashboard for ML system
 */
class MLMonitoringDashboard {
public:
    struct DashboardConfig {
        std::string host = "localhost";
        int port = 8080;
        bool enable_auto_refresh = true;
        int refresh_interval_seconds = 5;
        bool enable_historical_charts = true;
        bool enable_alerts_panel = true;
    };
    
    MLMonitoringDashboard(std::shared_ptr<ModelPerformanceMonitor> monitor,
                         const DashboardConfig& config = DashboardConfig());
    ~MLMonitoringDashboard();
    
    // Start/stop dashboard server
    void start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Get dashboard URL
    std::string getURL() const;
    
private:
    std::shared_ptr<ModelPerformanceMonitor> monitor_;
    DashboardConfig config_;
    std::atomic<bool> running_{false};
    
    // Dashboard components
    std::string generateHTML() const;
    std::string generateMetricsJSON() const;
    std::string generateChartsJSON() const;
    std::string generateAlertsJSON() const;
};

} // namespace ml
} // namespace predis

#endif // PREDIS_ML_MODEL_PERFORMANCE_MONITOR_H_