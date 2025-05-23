#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>

namespace predis {
namespace benchmarks {

// Data structure for confusion matrix
struct ConfusionMatrix {
    size_t true_positives = 0;
    size_t true_negatives = 0;
    size_t false_positives = 0;
    size_t false_negatives = 0;
    
    double accuracy() const {
        size_t total = true_positives + true_negatives + false_positives + false_negatives;
        return total > 0 ? static_cast<double>(true_positives + true_negatives) / total : 0.0;
    }
    
    double precision() const {
        size_t predicted_positive = true_positives + false_positives;
        return predicted_positive > 0 ? static_cast<double>(true_positives) / predicted_positive : 0.0;
    }
    
    double recall() const {
        size_t actual_positive = true_positives + false_negatives;
        return actual_positive > 0 ? static_cast<double>(true_positives) / actual_positive : 0.0;
    }
    
    double f1_score() const {
        double p = precision();
        double r = recall();
        return (p + r) > 0 ? 2.0 * (p * r) / (p + r) : 0.0;
    }
    
    double specificity() const {
        size_t actual_negative = true_negatives + false_positives;
        return actual_negative > 0 ? static_cast<double>(true_negatives) / actual_negative : 0.0;
    }
};

// Comprehensive ML model metrics
struct MLModelMetrics {
    // Basic classification metrics
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double specificity = 0.0;
    
    // Advanced metrics
    double auc_roc = 0.0;  // Area Under ROC Curve
    double auc_pr = 0.0;   // Area Under Precision-Recall Curve
    double log_loss = 0.0;
    double matthews_correlation_coefficient = 0.0;
    
    // Multi-class metrics
    std::map<std::string, double> per_class_precision;
    std::map<std::string, double> per_class_recall;
    std::map<std::string, double> per_class_f1;
    double macro_avg_precision = 0.0;
    double macro_avg_recall = 0.0;
    double macro_avg_f1 = 0.0;
    double weighted_avg_precision = 0.0;
    double weighted_avg_recall = 0.0;
    double weighted_avg_f1 = 0.0;
    
    // Regression metrics (for confidence scores)
    double mean_absolute_error = 0.0;
    double mean_squared_error = 0.0;
    double root_mean_squared_error = 0.0;
    double r_squared = 0.0;
    
    // Time series specific metrics
    double mean_absolute_percentage_error = 0.0;
    double symmetric_mean_absolute_percentage_error = 0.0;
};

// Training metrics
struct TrainingMetrics {
    // Loss tracking
    std::vector<double> training_loss_history;
    std::vector<double> validation_loss_history;
    double final_training_loss = 0.0;
    double final_validation_loss = 0.0;
    
    // Accuracy tracking
    std::vector<double> training_accuracy_history;
    std::vector<double> validation_accuracy_history;
    double final_training_accuracy = 0.0;
    double final_validation_accuracy = 0.0;
    
    // Training efficiency
    double total_training_time_seconds = 0.0;
    double average_epoch_time_seconds = 0.0;
    size_t total_epochs = 0;
    size_t early_stopping_epoch = 0;
    bool early_stopped = false;
    
    // Resource utilization
    double peak_memory_usage_mb = 0.0;
    double average_gpu_utilization = 0.0;
    double average_cpu_utilization = 0.0;
    
    // Convergence metrics
    double learning_rate_final = 0.0;
    double gradient_norm_final = 0.0;
    size_t plateau_count = 0;
};

// Inference performance metrics
struct InferenceMetrics {
    // Latency statistics
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p90_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    
    // Throughput
    double avg_throughput_qps = 0.0;
    double peak_throughput_qps = 0.0;
    
    // Batch processing
    double avg_batch_size = 0.0;
    double batch_efficiency = 0.0;  // Ratio of actual to optimal batch size
    
    // Resource efficiency
    double gpu_memory_usage_mb = 0.0;
    double model_size_mb = 0.0;
    double cache_memory_overhead_mb = 0.0;
};

// Prefetching-specific metrics
struct PrefetchingMetrics {
    // Hit rate improvements
    double baseline_hit_rate = 0.0;
    double ml_enhanced_hit_rate = 0.0;
    double hit_rate_improvement_percentage = 0.0;
    double hit_rate_improvement_absolute = 0.0;
    
    // Prefetch effectiveness
    double prefetch_accuracy = 0.0;  // Ratio of used prefetches
    double prefetch_coverage = 0.0;  // Ratio of hits from prefetch
    double prefetch_timeliness = 0.0;  // Ratio of timely prefetches
    double wasted_prefetch_ratio = 0.0;
    
    // Pattern-specific performance
    std::map<std::string, double> pattern_hit_rates;
    std::map<std::string, double> pattern_improvements;
    
    // Resource impact
    double prefetch_bandwidth_usage_mbps = 0.0;
    double prefetch_cpu_overhead_percentage = 0.0;
    double prefetch_memory_overhead_mb = 0.0;
};

// Model comparison metrics
struct ModelComparisonMetrics {
    std::string model_a_name;
    std::string model_b_name;
    
    // Performance comparison
    double accuracy_difference = 0.0;
    double latency_ratio = 0.0;
    double throughput_ratio = 0.0;
    double memory_ratio = 0.0;
    
    // Statistical significance
    double p_value = 0.0;
    double cohen_d_effect_size = 0.0;
    bool statistically_significant = false;
    
    // A/B test results
    size_t model_a_samples = 0;
    size_t model_b_samples = 0;
    double model_a_win_rate = 0.0;
    double confidence_interval_lower = 0.0;
    double confidence_interval_upper = 0.0;
};

// Comprehensive DS metrics report
struct DSMetricsReport {
    std::string experiment_id;
    std::chrono::system_clock::time_point timestamp;
    std::string model_name;
    std::string model_version;
    std::string dataset_name;
    
    // Core metrics
    MLModelMetrics model_metrics;
    TrainingMetrics training_metrics;
    InferenceMetrics inference_metrics;
    PrefetchingMetrics prefetching_metrics;
    
    // Model comparisons
    std::vector<ModelComparisonMetrics> comparisons;
    
    // Workload-specific results
    std::map<std::string, MLModelMetrics> workload_metrics;
    
    // Feature importance
    std::map<std::string, double> feature_importance;
    
    // Hyperparameters
    std::map<std::string, std::string> hyperparameters;
    
    // Summary statistics
    std::string executive_summary;
    std::vector<std::string> key_findings;
    std::vector<std::string> recommendations;
};

// Main DS metrics collector interface
class DSMetricsCollector {
public:
    virtual ~DSMetricsCollector() = default;
    
    // Model evaluation
    virtual void evaluateModel(
        const std::string& model_name,
        const std::vector<std::string>& predictions,
        const std::vector<std::string>& ground_truth,
        const std::vector<double>& confidence_scores = {}
    ) = 0;
    
    // Training metrics collection
    virtual void recordTrainingEpoch(
        double loss,
        double accuracy,
        double validation_loss,
        double validation_accuracy,
        double epoch_time_seconds
    ) = 0;
    
    // Inference metrics collection
    virtual void recordInference(
        double latency_ms,
        size_t batch_size,
        double gpu_memory_mb
    ) = 0;
    
    // Prefetching metrics collection
    virtual void recordPrefetchResult(
        bool was_hit,
        bool was_prefetched,
        const std::string& access_pattern
    ) = 0;
    
    // Generate comprehensive report
    virtual DSMetricsReport generateReport() = 0;
    
    // Export functions
    virtual std::string exportJSON() const = 0;
    virtual std::string exportCSV() const = 0;
    virtual std::string exportHTML() const = 0;
    virtual std::string exportLatex() const = 0;
};

// Factory for creating metrics collectors
std::unique_ptr<DSMetricsCollector> createDSMetricsCollector(
    const std::string& experiment_name,
    const std::string& model_name
);

} // namespace benchmarks
} // namespace predis