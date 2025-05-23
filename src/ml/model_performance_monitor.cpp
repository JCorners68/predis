#include "model_performance_monitor.h"
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace predis {
namespace ml {

ModelPerformanceMonitor::ModelPerformanceMonitor()
    : monitoring_enabled_(true),
      alert_callback_(nullptr) {
    // Initialize thresholds
    alert_thresholds_.min_accuracy = 0.7;
    alert_thresholds_.max_latency_ms = 10.0;
    alert_thresholds_.max_memory_mb = 500.0;
    alert_thresholds_.min_throughput_qps = 1000.0;
}

void ModelPerformanceMonitor::recordPrediction(
    const std::string& model_id,
    double confidence,
    double latency_ms,
    bool was_correct) {
    
    if (!monitoring_enabled_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = model_metrics_[model_id];
    auto now = std::chrono::steady_clock::now();
    
    // Update basic metrics
    metrics.total_predictions++;
    if (was_correct) {
        metrics.correct_predictions++;
    }
    metrics.total_latency_ms += latency_ms;
    
    // Update confidence distribution
    int confidence_bucket = static_cast<int>(confidence * 10);
    confidence_bucket = std::min(9, std::max(0, confidence_bucket));
    metrics.confidence_distribution[confidence_bucket]++;
    
    // Update latency percentiles (simple implementation)
    metrics.latency_samples.push_back(latency_ms);
    if (metrics.latency_samples.size() > 10000) {
        // Keep only recent samples
        metrics.latency_samples.erase(
            metrics.latency_samples.begin(),
            metrics.latency_samples.begin() + 5000
        );
    }
    
    // Update time series data
    auto time_bucket = std::chrono::duration_cast<std::chrono::minutes>(
        now - metrics.start_time).count();
    metrics.accuracy_over_time[time_bucket] = 
        static_cast<double>(metrics.correct_predictions) / metrics.total_predictions;
    
    // Check for alerts
    checkAlerts(model_id, metrics, latency_ms);
}

void ModelPerformanceMonitor::recordTraining(
    const std::string& model_id,
    double training_time_ms,
    double validation_accuracy,
    size_t dataset_size) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = model_metrics_[model_id];
    metrics.total_trainings++;
    metrics.total_training_time_ms += training_time_ms;
    metrics.last_validation_accuracy = validation_accuracy;
    metrics.total_training_samples += dataset_size;
}

void ModelPerformanceMonitor::recordDriftDetection(
    const std::string& model_id,
    double drift_score,
    bool drift_detected) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = model_metrics_[model_id];
    metrics.drift_checks++;
    if (drift_detected) {
        metrics.drift_detections++;
        metrics.last_drift_time = std::chrono::steady_clock::now();
    }
    
    // Record drift score history
    metrics.drift_scores.push_back(drift_score);
    if (metrics.drift_scores.size() > 1000) {
        metrics.drift_scores.erase(
            metrics.drift_scores.begin(),
            metrics.drift_scores.begin() + 500
        );
    }
}

ModelMetrics ModelPerformanceMonitor::getMetrics(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto it = model_metrics_.find(model_id);
    if (it != model_metrics_.end()) {
        return it->second;
    }
    return ModelMetrics();
}

PerformanceReport ModelPerformanceMonitor::generateReport(
    const std::string& model_id) const {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PerformanceReport report;
    report.model_id = model_id;
    report.timestamp = std::chrono::system_clock::now();
    
    auto it = model_metrics_.find(model_id);
    if (it == model_metrics_.end()) {
        return report;
    }
    
    const auto& metrics = it->second;
    
    // Calculate accuracy
    if (metrics.total_predictions > 0) {
        report.accuracy = static_cast<double>(metrics.correct_predictions) / 
                         metrics.total_predictions;
    }
    
    // Calculate average latency
    if (metrics.total_predictions > 0) {
        report.avg_latency_ms = metrics.total_latency_ms / metrics.total_predictions;
    }
    
    // Calculate latency percentiles
    if (!metrics.latency_samples.empty()) {
        auto sorted_latencies = metrics.latency_samples;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        
        size_t p50_idx = sorted_latencies.size() * 0.5;
        size_t p95_idx = sorted_latencies.size() * 0.95;
        size_t p99_idx = sorted_latencies.size() * 0.99;
        
        report.p50_latency_ms = sorted_latencies[p50_idx];
        report.p95_latency_ms = sorted_latencies[p95_idx];
        report.p99_latency_ms = sorted_latencies[p99_idx];
    }
    
    // Calculate throughput
    auto duration = std::chrono::steady_clock::now() - metrics.start_time;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    if (seconds > 0) {
        report.throughput_qps = static_cast<double>(metrics.total_predictions) / seconds;
    }
    
    // Drift information
    report.drift_rate = metrics.drift_checks > 0 ? 
        static_cast<double>(metrics.drift_detections) / metrics.drift_checks : 0.0;
    
    // Training information
    if (metrics.total_trainings > 0) {
        report.avg_training_time_ms = metrics.total_training_time_ms / 
                                     metrics.total_trainings;
    }
    report.last_validation_accuracy = metrics.last_validation_accuracy;
    
    // Generate summary
    std::stringstream ss;
    ss << "Model: " << model_id << "\n"
       << "Accuracy: " << std::fixed << std::setprecision(2) 
       << (report.accuracy * 100) << "%\n"
       << "Avg Latency: " << std::setprecision(1) << report.avg_latency_ms << "ms\n"
       << "Throughput: " << std::setprecision(0) << report.throughput_qps << " QPS\n"
       << "Drift Rate: " << std::setprecision(1) << (report.drift_rate * 100) << "%";
    
    report.summary = ss.str();
    
    return report;
}

void ModelPerformanceMonitor::setAlertThresholds(const AlertThresholds& thresholds) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    alert_thresholds_ = thresholds;
}

void ModelPerformanceMonitor::setAlertCallback(AlertCallback callback) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    alert_callback_ = callback;
}

void ModelPerformanceMonitor::reset(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (model_id.empty()) {
        model_metrics_.clear();
    } else {
        model_metrics_.erase(model_id);
    }
}

void ModelPerformanceMonitor::checkAlerts(
    const std::string& model_id,
    const ModelMetrics& metrics,
    double latest_latency_ms) {
    
    if (!alert_callback_) return;
    
    // Check accuracy
    double accuracy = metrics.total_predictions > 0 ?
        static_cast<double>(metrics.correct_predictions) / metrics.total_predictions : 0.0;
    
    if (accuracy < alert_thresholds_.min_accuracy && metrics.total_predictions > 100) {
        alert_callback_(AlertType::LOW_ACCURACY, model_id,
            "Accuracy below threshold: " + std::to_string(accuracy));
    }
    
    // Check latency
    if (latest_latency_ms > alert_thresholds_.max_latency_ms) {
        alert_callback_(AlertType::HIGH_LATENCY, model_id,
            "Latency above threshold: " + std::to_string(latest_latency_ms) + "ms");
    }
    
    // Check throughput
    auto duration = std::chrono::steady_clock::now() - metrics.start_time;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    if (seconds > 10) {  // Only check after 10 seconds
        double throughput = static_cast<double>(metrics.total_predictions) / seconds;
        if (throughput < alert_thresholds_.min_throughput_qps) {
            alert_callback_(AlertType::LOW_THROUGHPUT, model_id,
                "Throughput below threshold: " + std::to_string(throughput) + " QPS");
        }
    }
    
    // Check drift rate
    if (metrics.drift_checks > 10) {  // Only check after sufficient samples
        double drift_rate = static_cast<double>(metrics.drift_detections) / 
                           metrics.drift_checks;
        if (drift_rate > 0.2) {  // Alert if >20% drift rate
            alert_callback_(AlertType::DRIFT_DETECTED, model_id,
                "High drift rate: " + std::to_string(drift_rate * 100) + "%");
        }
    }
}

std::string ModelPerformanceMonitor::exportMetrics(
    const std::string& model_id,
    const std::string& format) const {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (format == "json") {
        return exportMetricsJson(model_id);
    } else if (format == "csv") {
        return exportMetricsCsv(model_id);
    }
    
    return "Unsupported format: " + format;
}

std::string ModelPerformanceMonitor::exportMetricsJson(
    const std::string& model_id) const {
    
    std::stringstream ss;
    ss << "{\n";
    
    auto it = model_metrics_.find(model_id);
    if (it != model_metrics_.end()) {
        const auto& metrics = it->second;
        
        ss << "  \"model_id\": \"" << model_id << "\",\n";
        ss << "  \"total_predictions\": " << metrics.total_predictions << ",\n";
        ss << "  \"correct_predictions\": " << metrics.correct_predictions << ",\n";
        ss << "  \"accuracy\": " << std::fixed << std::setprecision(4)
           << (metrics.total_predictions > 0 ? 
               static_cast<double>(metrics.correct_predictions) / metrics.total_predictions : 0.0) 
           << ",\n";
        ss << "  \"avg_latency_ms\": " << std::setprecision(2)
           << (metrics.total_predictions > 0 ? 
               metrics.total_latency_ms / metrics.total_predictions : 0.0) 
           << ",\n";
        ss << "  \"total_trainings\": " << metrics.total_trainings << ",\n";
        ss << "  \"drift_detections\": " << metrics.drift_detections << ",\n";
        ss << "  \"drift_checks\": " << metrics.drift_checks << "\n";
    }
    
    ss << "}";
    return ss.str();
}

std::string ModelPerformanceMonitor::exportMetricsCsv(
    const std::string& model_id) const {
    
    std::stringstream ss;
    ss << "metric,value\n";
    
    auto it = model_metrics_.find(model_id);
    if (it != model_metrics_.end()) {
        const auto& metrics = it->second;
        
        ss << "total_predictions," << metrics.total_predictions << "\n";
        ss << "correct_predictions," << metrics.correct_predictions << "\n";
        ss << "accuracy," << std::fixed << std::setprecision(4)
           << (metrics.total_predictions > 0 ? 
               static_cast<double>(metrics.correct_predictions) / metrics.total_predictions : 0.0) 
           << "\n";
        ss << "avg_latency_ms," << std::setprecision(2)
           << (metrics.total_predictions > 0 ? 
               metrics.total_latency_ms / metrics.total_predictions : 0.0) 
           << "\n";
        ss << "total_trainings," << metrics.total_trainings << "\n";
        ss << "drift_detections," << metrics.drift_detections << "\n";
        ss << "drift_checks," << metrics.drift_checks << "\n";
    }
    
    return ss.str();
}

} // namespace ml
} // namespace predis