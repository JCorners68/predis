#include "prefetch_monitor.h"
#include "prefetch_coordinator.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>

namespace predis {
namespace ppe {

PrefetchMonitor::PrefetchMonitor() {
    current_metrics_.window_start = std::chrono::system_clock::now();
    lifetime_metrics_.window_start = std::chrono::system_clock::now();
    
    // Set default alert thresholds
    alert_thresholds_["hit_rate_improvement"] = 0.15;  // Alert if < 15% improvement
    alert_thresholds_["precision"] = 0.7;              // Alert if < 70% precision
    alert_thresholds_["avg_prediction_latency_ms"] = 10.0;  // Alert if > 10ms
}

void PrefetchMonitor::setPrefetchCoordinator(std::shared_ptr<PrefetchCoordinator> coordinator) {
    coordinator_ = coordinator;
}

void PrefetchMonitor::startMonitoring() {
    monitoring_ = true;
    current_metrics_.window_start = std::chrono::system_clock::now();
}

void PrefetchMonitor::stopMonitoring() {
    monitoring_ = false;
}

void PrefetchMonitor::recordPrediction(const std::string& key, float confidence, 
                                      bool was_used, double latency_ms) {
    if (!monitoring_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Update metrics
    current_metrics_.total_predictions++;
    lifetime_metrics_.total_predictions++;
    
    if (was_used) {
        current_metrics_.accurate_predictions++;
        lifetime_metrics_.accurate_predictions++;
    }
    
    // Update latency tracking
    double n = current_metrics_.total_predictions;
    current_metrics_.avg_prediction_latency_ms = 
        (current_metrics_.avg_prediction_latency_ms * (n - 1) + latency_ms) / n;
    
    // Update confidence tracking
    current_metrics_.model_confidence_avg = 
        (current_metrics_.model_confidence_avg * (n - 1) + confidence) / n;
    
    // Record event
    TimedEvent event;
    event.timestamp = std::chrono::system_clock::now();
    event.event_type = "prediction";
    event.key = key;
    event.value = confidence;
    event_history_.push_back(event);
    
    // Update key metrics
    updateKeyMetrics(key, "prediction");
    
    // Prune old events if needed
    if (event_history_.size() > max_history_size_) {
        pruneEventHistory();
    }
}

void PrefetchMonitor::recordPrefetch(const std::string& key, bool success, double latency_ms) {
    if (!monitoring_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.prefetch_requests++;
    lifetime_metrics_.prefetch_requests++;
    
    if (success) {
        current_metrics_.successful_prefetches++;
        lifetime_metrics_.successful_prefetches++;
    }
    
    // Update latency
    double n = current_metrics_.prefetch_requests;
    current_metrics_.avg_prefetch_latency_ms = 
        (current_metrics_.avg_prefetch_latency_ms * (n - 1) + latency_ms) / n;
    
    // Record event
    TimedEvent event;
    event.timestamp = std::chrono::system_clock::now();
    event.event_type = "prefetch";
    event.key = key;
    event.value = success ? 1.0 : 0.0;
    event_history_.push_back(event);
    
    // Update key metrics
    auto& km = key_metrics_[key];
    km.prefetch_count++;
    km.last_prefetch = event.timestamp;
}

void PrefetchMonitor::recordCacheAccess(const std::string& key, bool was_hit, bool from_prefetch) {
    if (!monitoring_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (was_hit && from_prefetch) {
        current_metrics_.cache_hits_from_prefetch++;
        lifetime_metrics_.cache_hits_from_prefetch++;
    }
    
    // Update baseline hit rate
    if (!from_prefetch) {
        baseline_sample_count_++;
        baseline_hit_rate_ = (baseline_hit_rate_ * (baseline_sample_count_ - 1) + 
                             (was_hit ? 1.0 : 0.0)) / baseline_sample_count_;
    }
    
    // Update key metrics
    auto& km = key_metrics_[key];
    km.access_count++;
    if (was_hit) {
        km.hit_count++;
    }
    km.last_access = std::chrono::system_clock::now();
    
    // Calculate time to access if this was prefetched
    if (from_prefetch && km.last_prefetch != std::chrono::time_point<std::chrono::system_clock>()) {
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            km.last_access - km.last_prefetch).count();
        km.avg_time_to_access_ms = (km.avg_time_to_access_ms * (km.prefetch_count - 1) + 
                                   time_diff) / km.prefetch_count;
    }
}

void PrefetchMonitor::recordModelUpdate(const std::string& model_type) {
    if (!monitoring_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.model_updates++;
    lifetime_metrics_.model_updates++;
    
    model_performance_[model_type].model_type = model_type;
}

PrefetchMonitor::PrefetchMetrics PrefetchMonitor::getMetrics(const TimeWindow& window) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PrefetchMetrics metrics;
    metrics.window_start = std::chrono::system_clock::now() - window.duration;
    metrics.window_end = std::chrono::system_clock::now();
    
    // Filter events within the time window
    for (const auto& event : event_history_) {
        if (event.timestamp >= metrics.window_start) {
            updateMetrics(metrics, event);
        }
    }
    
    // Calculate derived metrics
    if (metrics.total_predictions > 0) {
        metrics.precision = static_cast<double>(metrics.accurate_predictions) / 
                          metrics.total_predictions;
    }
    
    metrics.hit_rate_improvement = getHitRateImprovement();
    
    // Calculate F1 score
    if (metrics.precision > 0 || metrics.recall > 0) {
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / 
                          (metrics.precision + metrics.recall);
    }
    
    return metrics;
}

PrefetchMonitor::PrefetchMetrics PrefetchMonitor::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PrefetchMetrics metrics = current_metrics_;
    metrics.window_end = std::chrono::system_clock::now();
    
    // Calculate derived metrics
    if (metrics.total_predictions > 0) {
        metrics.precision = static_cast<double>(metrics.accurate_predictions) / 
                          metrics.total_predictions;
    }
    
    metrics.hit_rate_improvement = getHitRateImprovement();
    metrics.model_accuracy = metrics.precision;  // Simplified
    
    if (metrics.precision > 0 || metrics.recall > 0) {
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / 
                          (metrics.precision + metrics.recall);
    }
    
    return metrics;
}

PrefetchMonitor::PrefetchMetrics PrefetchMonitor::getLifetimeMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PrefetchMetrics metrics = lifetime_metrics_;
    metrics.window_end = std::chrono::system_clock::now();
    
    // Calculate wasted prefetches
    metrics.wasted_prefetches = metrics.successful_prefetches - metrics.cache_hits_from_prefetch;
    
    return metrics;
}

PrefetchMonitor::KeyMetrics PrefetchMonitor::getKeyMetrics(const std::string& key) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto it = key_metrics_.find(key);
    if (it != key_metrics_.end()) {
        return it->second;
    }
    
    return KeyMetrics();
}

std::vector<std::pair<std::string, PrefetchMonitor::KeyMetrics>> 
PrefetchMonitor::getTopKeys(size_t n) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::vector<std::pair<std::string, KeyMetrics>> all_keys(
        key_metrics_.begin(), key_metrics_.end());
    
    // Sort by access count
    std::sort(all_keys.begin(), all_keys.end(),
        [](const auto& a, const auto& b) {
            return a.second.access_count > b.second.access_count;
        });
    
    // Return top n
    if (all_keys.size() > n) {
        all_keys.resize(n);
    }
    
    return all_keys;
}

double PrefetchMonitor::getCurrentHitRate() const {
    if (!coordinator_) return 0.0;
    
    return coordinator_->getCurrentHitRate();
}

double PrefetchMonitor::getHitRateImprovement() const {
    double current_rate = getCurrentHitRate();
    return current_rate - baseline_hit_rate_;
}

void PrefetchMonitor::setAlertThreshold(const std::string& metric, double threshold) {
    alert_thresholds_[metric] = threshold;
}

bool PrefetchMonitor::checkAlerts(std::vector<std::string>& triggered_alerts) const {
    triggered_alerts.clear();
    
    auto metrics = getCurrentMetrics();
    
    // Check hit rate improvement
    if (alert_thresholds_.count("hit_rate_improvement") && 
        metrics.hit_rate_improvement < alert_thresholds_.at("hit_rate_improvement")) {
        triggered_alerts.push_back("Hit rate improvement below threshold");
    }
    
    // Check precision
    if (alert_thresholds_.count("precision") && 
        metrics.precision < alert_thresholds_.at("precision")) {
        triggered_alerts.push_back("Prefetch precision below threshold");
    }
    
    // Check latency
    if (alert_thresholds_.count("avg_prediction_latency_ms") && 
        metrics.avg_prediction_latency_ms > alert_thresholds_.at("avg_prediction_latency_ms")) {
        triggered_alerts.push_back("Prediction latency above threshold");
    }
    
    return !triggered_alerts.empty();
}

std::string PrefetchMonitor::exportMetricsJSON() const {
    auto metrics = getCurrentMetrics();
    
    std::ostringstream json;
    json << "{\n";
    json << "  \"window_start\": \"" << std::chrono::system_clock::to_time_t(metrics.window_start) << "\",\n";
    json << "  \"window_end\": \"" << std::chrono::system_clock::to_time_t(metrics.window_end) << "\",\n";
    json << "  \"predictions\": {\n";
    json << "    \"total\": " << metrics.total_predictions << ",\n";
    json << "    \"accurate\": " << metrics.accurate_predictions << ",\n";
    json << "    \"precision\": " << std::fixed << std::setprecision(3) << metrics.precision << "\n";
    json << "  },\n";
    json << "  \"prefetching\": {\n";
    json << "    \"requests\": " << metrics.prefetch_requests << ",\n";
    json << "    \"successful\": " << metrics.successful_prefetches << ",\n";
    json << "    \"cache_hits\": " << metrics.cache_hits_from_prefetch << ",\n";
    json << "    \"wasted\": " << metrics.wasted_prefetches << "\n";
    json << "  },\n";
    json << "  \"performance\": {\n";
    json << "    \"hit_rate_improvement\": " << std::fixed << std::setprecision(3) 
         << metrics.hit_rate_improvement << ",\n";
    json << "    \"f1_score\": " << metrics.f1_score << ",\n";
    json << "    \"avg_prediction_latency_ms\": " << metrics.avg_prediction_latency_ms << ",\n";
    json << "    \"avg_prefetch_latency_ms\": " << metrics.avg_prefetch_latency_ms << "\n";
    json << "  }\n";
    json << "}";
    
    return json.str();
}

std::string PrefetchMonitor::exportMetricsCSV() const {
    auto metrics = getCurrentMetrics();
    
    std::ostringstream csv;
    csv << "timestamp,total_predictions,accurate_predictions,precision,";
    csv << "prefetch_requests,successful_prefetches,cache_hits_from_prefetch,";
    csv << "hit_rate_improvement,f1_score,avg_prediction_latency_ms\n";
    
    csv << std::chrono::system_clock::to_time_t(metrics.window_end) << ",";
    csv << metrics.total_predictions << ",";
    csv << metrics.accurate_predictions << ",";
    csv << std::fixed << std::setprecision(3) << metrics.precision << ",";
    csv << metrics.prefetch_requests << ",";
    csv << metrics.successful_prefetches << ",";
    csv << metrics.cache_hits_from_prefetch << ",";
    csv << metrics.hit_rate_improvement << ",";
    csv << metrics.f1_score << ",";
    csv << metrics.avg_prediction_latency_ms << "\n";
    
    return csv.str();
}

void PrefetchMonitor::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_ = PrefetchMetrics();
    current_metrics_.window_start = std::chrono::system_clock::now();
    
    event_history_.clear();
}

void PrefetchMonitor::resetKeyMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    key_metrics_.clear();
}

void PrefetchMonitor::updateMetrics(PrefetchMetrics& metrics, const TimedEvent& event) const {
    if (event.event_type == "prediction") {
        metrics.total_predictions++;
        // More complex logic would track if prediction was used
    } else if (event.event_type == "prefetch") {
        metrics.prefetch_requests++;
        if (event.value > 0.5) {  // Success
            metrics.successful_prefetches++;
        }
    }
}

void PrefetchMonitor::pruneEventHistory() {
    // Remove oldest 20% of events
    size_t remove_count = event_history_.size() / 5;
    event_history_.erase(event_history_.begin(), event_history_.begin() + remove_count);
}

void PrefetchMonitor::updateKeyMetrics(const std::string& key, const std::string& event_type) {
    auto& km = key_metrics_[key];
    // Update is handled in specific record methods
}

// PrefetchDashboard implementation

PrefetchDashboard::PrefetchDashboard(std::shared_ptr<PrefetchMonitor> monitor)
    : monitor_(monitor) {}

std::string PrefetchDashboard::generateHTML() const {
    auto metrics = monitor_->getCurrentMetrics();
    
    std::ostringstream html;
    html << "<!DOCTYPE html>\n<html>\n<head>\n";
    html << "<title>Predis Prefetch Performance Dashboard</title>\n";
    html << "<meta http-equiv=\"refresh\" content=\"5\">\n";  // Auto-refresh
    html << "<style>\n";
    html << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
    html << ".metric { display: inline-block; margin: 10px; padding: 10px; ";
    html << "border: 1px solid #ccc; border-radius: 5px; }\n";
    html << ".metric-value { font-size: 24px; font-weight: bold; }\n";
    html << ".metric-label { font-size: 14px; color: #666; }\n";
    html << ".good { color: #28a745; }\n";
    html << ".warning { color: #ffc107; }\n";
    html << ".bad { color: #dc3545; }\n";
    html << "</style>\n</head>\n<body>\n";
    
    html << "<h1>Predis Prefetch Performance Dashboard</h1>\n";
    
    // Key metrics
    html << "<div class=\"metrics\">\n";
    
    // Hit rate improvement
    html << "<div class=\"metric\">\n";
    html << "<div class=\"metric-value ";
    if (metrics.hit_rate_improvement >= 0.2) html << "good";
    else if (metrics.hit_rate_improvement >= 0.1) html << "warning";
    else html << "bad";
    html << "\">" << std::fixed << std::setprecision(1) 
         << (metrics.hit_rate_improvement * 100) << "%</div>\n";
    html << "<div class=\"metric-label\">Hit Rate Improvement</div>\n";
    html << "</div>\n";
    
    // Precision
    html << "<div class=\"metric\">\n";
    html << "<div class=\"metric-value ";
    if (metrics.precision >= 0.8) html << "good";
    else if (metrics.precision >= 0.6) html << "warning";
    else html << "bad";
    html << "\">" << std::fixed << std::setprecision(1) 
         << (metrics.precision * 100) << "%</div>\n";
    html << "<div class=\"metric-label\">Prefetch Precision</div>\n";
    html << "</div>\n";
    
    // Prediction latency
    html << "<div class=\"metric\">\n";
    html << "<div class=\"metric-value ";
    if (metrics.avg_prediction_latency_ms <= 5) html << "good";
    else if (metrics.avg_prediction_latency_ms <= 10) html << "warning";
    else html << "bad";
    html << "\">" << std::fixed << std::setprecision(1) 
         << metrics.avg_prediction_latency_ms << "ms</div>\n";
    html << "<div class=\"metric-label\">Avg Prediction Latency</div>\n";
    html << "</div>\n";
    
    html << "</div>\n";  // metrics
    
    // Detailed statistics
    html << "<h2>Detailed Statistics</h2>\n";
    html << "<table border=\"1\" cellpadding=\"5\">\n";
    html << "<tr><th>Metric</th><th>Value</th></tr>\n";
    html << "<tr><td>Total Predictions</td><td>" << metrics.total_predictions << "</td></tr>\n";
    html << "<tr><td>Accurate Predictions</td><td>" << metrics.accurate_predictions << "</td></tr>\n";
    html << "<tr><td>Prefetch Requests</td><td>" << metrics.prefetch_requests << "</td></tr>\n";
    html << "<tr><td>Successful Prefetches</td><td>" << metrics.successful_prefetches << "</td></tr>\n";
    html << "<tr><td>Cache Hits from Prefetch</td><td>" << metrics.cache_hits_from_prefetch << "</td></tr>\n";
    html << "<tr><td>F1 Score</td><td>" << std::fixed << std::setprecision(3) 
         << metrics.f1_score << "</td></tr>\n";
    html << "</table>\n";
    
    // Top keys
    html << "<h2>Top Accessed Keys</h2>\n";
    auto top_keys = monitor_->getTopKeys(10);
    html << "<table border=\"1\" cellpadding=\"5\">\n";
    html << "<tr><th>Key</th><th>Accesses</th><th>Prefetches</th><th>Hits</th><th>Avg Time to Access</th></tr>\n";
    
    for (const auto& [key, km] : top_keys) {
        html << "<tr>";
        html << "<td>" << key << "</td>";
        html << "<td>" << km.access_count << "</td>";
        html << "<td>" << km.prefetch_count << "</td>";
        html << "<td>" << km.hit_count << "</td>";
        html << "<td>" << std::fixed << std::setprecision(1) 
             << km.avg_time_to_access_ms << "ms</td>";
        html << "</tr>\n";
    }
    html << "</table>\n";
    
    html << "</body>\n</html>";
    
    return html.str();
}

void PrefetchDashboard::printSummary() const {
    auto metrics = monitor_->getCurrentMetrics();
    
    std::cout << "\n=== Prefetch Performance Summary ===\n";
    std::cout << "Hit Rate Improvement: " << std::fixed << std::setprecision(1) 
              << (metrics.hit_rate_improvement * 100) << "%\n";
    std::cout << "Prefetch Precision: " << (metrics.precision * 100) << "%\n";
    std::cout << "F1 Score: " << std::setprecision(3) << metrics.f1_score << "\n";
    std::cout << "Avg Prediction Latency: " << std::setprecision(1) 
              << metrics.avg_prediction_latency_ms << "ms\n";
    std::cout << "Total Predictions: " << metrics.total_predictions << "\n";
    std::cout << "Cache Hits from Prefetch: " << metrics.cache_hits_from_prefetch << "\n";
    std::cout << "==================================\n";
}

} // namespace ppe
} // namespace predis