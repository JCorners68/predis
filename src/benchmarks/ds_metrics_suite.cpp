#include "ds_metrics_suite.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>

namespace predis {
namespace benchmarks {

class DSMetricsCollectorImpl : public DSMetricsCollector {
private:
    DSMetricsReport report_;
    
    // Temporary storage for incremental calculations
    ConfusionMatrix confusion_matrix_;
    std::vector<double> all_predictions_;
    std::vector<double> all_ground_truth_;
    std::vector<double> inference_latencies_;
    std::vector<std::pair<bool, bool>> prefetch_results_; // (was_hit, was_prefetched)
    std::map<std::string, std::vector<std::pair<bool, bool>>> pattern_prefetch_results_;
    
    // Training state
    size_t epoch_count_ = 0;
    std::chrono::steady_clock::time_point training_start_;
    
public:
    DSMetricsCollectorImpl(const std::string& experiment_name, const std::string& model_name) {
        report_.experiment_id = experiment_name;
        report_.model_name = model_name;
        report_.timestamp = std::chrono::system_clock::now();
        training_start_ = std::chrono::steady_clock::now();
    }
    
    void evaluateModel(
        const std::string& model_name,
        const std::vector<std::string>& predictions,
        const std::vector<std::string>& ground_truth,
        const std::vector<double>& confidence_scores) override {
        
        if (predictions.size() != ground_truth.size()) {
            throw std::invalid_argument("Predictions and ground truth must have same size");
        }
        
        // Reset confusion matrix
        confusion_matrix_ = ConfusionMatrix();
        
        // Calculate confusion matrix (binary classification for now)
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred_positive = (predictions[i] == "1" || predictions[i] == "true");
            bool true_positive = (ground_truth[i] == "1" || ground_truth[i] == "true");
            
            if (pred_positive && true_positive) {
                confusion_matrix_.true_positives++;
            } else if (!pred_positive && !true_positive) {
                confusion_matrix_.true_negatives++;
            } else if (pred_positive && !true_positive) {
                confusion_matrix_.false_positives++;
            } else {
                confusion_matrix_.false_negatives++;
            }
        }
        
        // Update model metrics
        auto& metrics = report_.model_metrics;
        metrics.accuracy = confusion_matrix_.accuracy();
        metrics.precision = confusion_matrix_.precision();
        metrics.recall = confusion_matrix_.recall();
        metrics.f1_score = confusion_matrix_.f1_score();
        metrics.specificity = confusion_matrix_.specificity();
        
        // Calculate Matthews Correlation Coefficient
        double tp = confusion_matrix_.true_positives;
        double tn = confusion_matrix_.true_negatives;
        double fp = confusion_matrix_.false_positives;
        double fn = confusion_matrix_.false_negatives;
        
        double numerator = (tp * tn) - (fp * fn);
        double denominator = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        metrics.matthews_correlation_coefficient = denominator > 0 ? numerator / denominator : 0.0;
        
        // Calculate log loss if confidence scores provided
        if (!confidence_scores.empty()) {
            double log_loss = 0.0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                bool true_positive = (ground_truth[i] == "1" || ground_truth[i] == "true");
                double p = confidence_scores[i];
                p = std::max(1e-15, std::min(1 - 1e-15, p)); // Clip to avoid log(0)
                
                if (true_positive) {
                    log_loss -= std::log(p);
                } else {
                    log_loss -= std::log(1 - p);
                }
            }
            metrics.log_loss = log_loss / predictions.size();
        }
    }
    
    void recordTrainingEpoch(
        double loss,
        double accuracy,
        double validation_loss,
        double validation_accuracy,
        double epoch_time_seconds) override {
        
        auto& training = report_.training_metrics;
        
        training.training_loss_history.push_back(loss);
        training.validation_loss_history.push_back(validation_loss);
        training.training_accuracy_history.push_back(accuracy);
        training.validation_accuracy_history.push_back(validation_accuracy);
        
        training.final_training_loss = loss;
        training.final_validation_loss = validation_loss;
        training.final_training_accuracy = accuracy;
        training.final_validation_accuracy = validation_accuracy;
        
        epoch_count_++;
        training.total_epochs = epoch_count_;
        
        // Update timing
        auto now = std::chrono::steady_clock::now();
        training.total_training_time_seconds = 
            std::chrono::duration<double>(now - training_start_).count();
        training.average_epoch_time_seconds = 
            training.total_training_time_seconds / epoch_count_;
        
        // Check for early stopping (simple patience-based)
        if (training.validation_loss_history.size() > 5) {
            bool improving = false;
            double best_loss = validation_loss;
            for (int i = 1; i <= 5; ++i) {
                size_t idx = training.validation_loss_history.size() - i - 1;
                if (training.validation_loss_history[idx] < best_loss) {
                    improving = true;
                    break;
                }
            }
            if (!improving) {
                training.plateau_count++;
            }
        }
    }
    
    void recordInference(
        double latency_ms,
        size_t batch_size,
        double gpu_memory_mb) override {
        
        inference_latencies_.push_back(latency_ms);
        
        auto& inference = report_.inference_metrics;
        
        // Update latency statistics
        inference.avg_latency_ms = std::accumulate(
            inference_latencies_.begin(), 
            inference_latencies_.end(), 
            0.0) / inference_latencies_.size();
        
        // Calculate percentiles
        if (!inference_latencies_.empty()) {
            auto sorted_latencies = inference_latencies_;
            std::sort(sorted_latencies.begin(), sorted_latencies.end());
            
            size_t n = sorted_latencies.size();
            inference.p50_latency_ms = sorted_latencies[n * 0.50];
            inference.p90_latency_ms = sorted_latencies[n * 0.90];
            inference.p95_latency_ms = sorted_latencies[n * 0.95];
            inference.p99_latency_ms = sorted_latencies[n * 0.99];
            inference.min_latency_ms = sorted_latencies.front();
            inference.max_latency_ms = sorted_latencies.back();
        }
        
        // Update batch and memory metrics
        inference.avg_batch_size = 
            (inference.avg_batch_size * (inference_latencies_.size() - 1) + batch_size) 
            / inference_latencies_.size();
        inference.gpu_memory_usage_mb = std::max(inference.gpu_memory_usage_mb, gpu_memory_mb);
        
        // Calculate throughput
        if (inference.avg_latency_ms > 0) {
            inference.avg_throughput_qps = 1000.0 / inference.avg_latency_ms * inference.avg_batch_size;
        }
    }
    
    void recordPrefetchResult(
        bool was_hit,
        bool was_prefetched,
        const std::string& access_pattern) override {
        
        prefetch_results_.push_back({was_hit, was_prefetched});
        pattern_prefetch_results_[access_pattern].push_back({was_hit, was_prefetched});
        
        // Update prefetching metrics
        auto& prefetch = report_.prefetching_metrics;
        
        size_t total_accesses = prefetch_results_.size();
        size_t hits = 0;
        size_t prefetched_hits = 0;
        size_t total_prefetched = 0;
        size_t useful_prefetches = 0;
        
        for (const auto& [hit, prefetched] : prefetch_results_) {
            if (hit) hits++;
            if (prefetched) total_prefetched++;
            if (hit && prefetched) {
                prefetched_hits++;
                useful_prefetches++;
            }
        }
        
        // Calculate baseline (non-prefetched hits)
        size_t baseline_hits = hits - prefetched_hits;
        prefetch.baseline_hit_rate = static_cast<double>(baseline_hits) / total_accesses;
        prefetch.ml_enhanced_hit_rate = static_cast<double>(hits) / total_accesses;
        
        // Calculate improvements
        prefetch.hit_rate_improvement_absolute = 
            prefetch.ml_enhanced_hit_rate - prefetch.baseline_hit_rate;
        prefetch.hit_rate_improvement_percentage = 
            prefetch.baseline_hit_rate > 0 ? 
            (prefetch.hit_rate_improvement_absolute / prefetch.baseline_hit_rate * 100) : 0;
        
        // Prefetch effectiveness
        prefetch.prefetch_accuracy = 
            total_prefetched > 0 ? 
            static_cast<double>(useful_prefetches) / total_prefetched : 0;
        prefetch.prefetch_coverage = 
            hits > 0 ? static_cast<double>(prefetched_hits) / hits : 0;
        prefetch.wasted_prefetch_ratio = 
            total_prefetched > 0 ? 
            1.0 - static_cast<double>(useful_prefetches) / total_prefetched : 0;
        
        // Pattern-specific metrics
        for (const auto& [pattern, results] : pattern_prefetch_results_) {
            size_t pattern_hits = 0;
            size_t pattern_total = results.size();
            for (const auto& [hit, _] : results) {
                if (hit) pattern_hits++;
            }
            prefetch.pattern_hit_rates[pattern] = 
                static_cast<double>(pattern_hits) / pattern_total;
        }
    }
    
    DSMetricsReport generateReport() override {
        // Add executive summary
        std::stringstream summary;
        summary << "Model: " << report_.model_name << "\n";
        summary << "Accuracy: " << std::fixed << std::setprecision(1) 
                << (report_.model_metrics.accuracy * 100) << "%\n";
        summary << "F1 Score: " << std::setprecision(3) << report_.model_metrics.f1_score << "\n";
        summary << "Avg Inference Latency: " << std::setprecision(1) 
                << report_.inference_metrics.avg_latency_ms << "ms\n";
        summary << "Hit Rate Improvement: " << std::setprecision(1) 
                << report_.prefetching_metrics.hit_rate_improvement_percentage << "%";
        
        report_.executive_summary = summary.str();
        
        // Add key findings
        report_.key_findings.clear();
        
        if (report_.model_metrics.accuracy > 0.8) {
            report_.key_findings.push_back("Model achieves high accuracy (>80%)");
        }
        
        if (report_.inference_metrics.avg_latency_ms < 10.0) {
            report_.key_findings.push_back("Inference latency meets <10ms target");
        }
        
        if (report_.prefetching_metrics.hit_rate_improvement_percentage > 20.0) {
            report_.key_findings.push_back("Significant hit rate improvement (>20%)");
        }
        
        if (report_.prefetching_metrics.prefetch_accuracy > 0.75) {
            report_.key_findings.push_back("High prefetch accuracy (>75%)");
        }
        
        // Add recommendations
        report_.recommendations.clear();
        
        if (report_.model_metrics.precision < 0.7) {
            report_.recommendations.push_back("Consider increasing confidence threshold to reduce false positives");
        }
        
        if (report_.prefetching_metrics.wasted_prefetch_ratio > 0.3) {
            report_.recommendations.push_back("Optimize prefetch strategy to reduce wasted prefetches");
        }
        
        if (report_.inference_metrics.p99_latency_ms > 20.0) {
            report_.recommendations.push_back("Investigate P99 latency spikes for optimization");
        }
        
        return report_;
    }
    
    std::string exportJSON() const override {
        std::stringstream json;
        json << "{\n";
        json << "  \"experiment_id\": \"" << report_.experiment_id << "\",\n";
        json << "  \"model_name\": \"" << report_.model_name << "\",\n";
        json << "  \"model_metrics\": {\n";
        json << "    \"accuracy\": " << report_.model_metrics.accuracy << ",\n";
        json << "    \"precision\": " << report_.model_metrics.precision << ",\n";
        json << "    \"recall\": " << report_.model_metrics.recall << ",\n";
        json << "    \"f1_score\": " << report_.model_metrics.f1_score << ",\n";
        json << "    \"specificity\": " << report_.model_metrics.specificity << ",\n";
        json << "    \"mcc\": " << report_.model_metrics.matthews_correlation_coefficient << "\n";
        json << "  },\n";
        json << "  \"inference_metrics\": {\n";
        json << "    \"avg_latency_ms\": " << report_.inference_metrics.avg_latency_ms << ",\n";
        json << "    \"p50_latency_ms\": " << report_.inference_metrics.p50_latency_ms << ",\n";
        json << "    \"p95_latency_ms\": " << report_.inference_metrics.p95_latency_ms << ",\n";
        json << "    \"p99_latency_ms\": " << report_.inference_metrics.p99_latency_ms << ",\n";
        json << "    \"throughput_qps\": " << report_.inference_metrics.avg_throughput_qps << "\n";
        json << "  },\n";
        json << "  \"prefetching_metrics\": {\n";
        json << "    \"baseline_hit_rate\": " << report_.prefetching_metrics.baseline_hit_rate << ",\n";
        json << "    \"ml_enhanced_hit_rate\": " << report_.prefetching_metrics.ml_enhanced_hit_rate << ",\n";
        json << "    \"improvement_percentage\": " << report_.prefetching_metrics.hit_rate_improvement_percentage << ",\n";
        json << "    \"prefetch_accuracy\": " << report_.prefetching_metrics.prefetch_accuracy << ",\n";
        json << "    \"prefetch_coverage\": " << report_.prefetching_metrics.prefetch_coverage << "\n";
        json << "  },\n";
        json << "  \"training_metrics\": {\n";
        json << "    \"final_accuracy\": " << report_.training_metrics.final_validation_accuracy << ",\n";
        json << "    \"total_epochs\": " << report_.training_metrics.total_epochs << ",\n";
        json << "    \"training_time_seconds\": " << report_.training_metrics.total_training_time_seconds << "\n";
        json << "  }\n";
        json << "}";
        return json.str();
    }
    
    std::string exportCSV() const override {
        std::stringstream csv;
        csv << "metric,value\n";
        csv << "accuracy," << report_.model_metrics.accuracy << "\n";
        csv << "precision," << report_.model_metrics.precision << "\n";
        csv << "recall," << report_.model_metrics.recall << "\n";
        csv << "f1_score," << report_.model_metrics.f1_score << "\n";
        csv << "avg_latency_ms," << report_.inference_metrics.avg_latency_ms << "\n";
        csv << "p95_latency_ms," << report_.inference_metrics.p95_latency_ms << "\n";
        csv << "throughput_qps," << report_.inference_metrics.avg_throughput_qps << "\n";
        csv << "hit_rate_improvement_%," << report_.prefetching_metrics.hit_rate_improvement_percentage << "\n";
        csv << "prefetch_accuracy," << report_.prefetching_metrics.prefetch_accuracy << "\n";
        return csv.str();
    }
    
    std::string exportHTML() const override {
        std::stringstream html;
        html << "<!DOCTYPE html>\n<html>\n<head>\n";
        html << "<title>DS Metrics Report - " << report_.model_name << "</title>\n";
        html << "<style>\n";
        html << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
        html << "table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n";
        html << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
        html << "th { background-color: #4CAF50; color: white; }\n";
        html << ".metric-good { color: green; font-weight: bold; }\n";
        html << ".metric-bad { color: red; font-weight: bold; }\n";
        html << "</style>\n</head>\n<body>\n";
        
        html << "<h1>Data Science Metrics Report</h1>\n";
        html << "<h2>Model: " << report_.model_name << "</h2>\n";
        html << "<p>Experiment: " << report_.experiment_id << "</p>\n";
        
        // Model Performance
        html << "<h3>Model Performance</h3>\n";
        html << "<table>\n";
        html << "<tr><th>Metric</th><th>Value</th></tr>\n";
        html << "<tr><td>Accuracy</td><td class='" 
             << (report_.model_metrics.accuracy > 0.8 ? "metric-good" : "") << "'>"
             << std::fixed << std::setprecision(1) << (report_.model_metrics.accuracy * 100) << "%</td></tr>\n";
        html << "<tr><td>Precision</td><td>" << std::setprecision(3) << report_.model_metrics.precision << "</td></tr>\n";
        html << "<tr><td>Recall</td><td>" << report_.model_metrics.recall << "</td></tr>\n";
        html << "<tr><td>F1 Score</td><td>" << report_.model_metrics.f1_score << "</td></tr>\n";
        html << "</table>\n";
        
        // Inference Performance
        html << "<h3>Inference Performance</h3>\n";
        html << "<table>\n";
        html << "<tr><th>Metric</th><th>Value</th></tr>\n";
        html << "<tr><td>Average Latency</td><td class='"
             << (report_.inference_metrics.avg_latency_ms < 10 ? "metric-good" : "") << "'>"
             << std::setprecision(1) << report_.inference_metrics.avg_latency_ms << " ms</td></tr>\n";
        html << "<tr><td>P95 Latency</td><td>" << report_.inference_metrics.p95_latency_ms << " ms</td></tr>\n";
        html << "<tr><td>Throughput</td><td>" << std::setprecision(0) 
             << report_.inference_metrics.avg_throughput_qps << " QPS</td></tr>\n";
        html << "</table>\n";
        
        // Prefetching Performance
        html << "<h3>Prefetching Performance</h3>\n";
        html << "<table>\n";
        html << "<tr><th>Metric</th><th>Value</th></tr>\n";
        html << "<tr><td>Hit Rate Improvement</td><td class='"
             << (report_.prefetching_metrics.hit_rate_improvement_percentage > 20 ? "metric-good" : "") << "'>"
             << std::setprecision(1) << report_.prefetching_metrics.hit_rate_improvement_percentage << "%</td></tr>\n";
        html << "<tr><td>Prefetch Accuracy</td><td>" 
             << std::setprecision(1) << (report_.prefetching_metrics.prefetch_accuracy * 100) << "%</td></tr>\n";
        html << "<tr><td>Prefetch Coverage</td><td>" 
             << (report_.prefetching_metrics.prefetch_coverage * 100) << "%</td></tr>\n";
        html << "</table>\n";
        
        html << "</body>\n</html>";
        return html.str();
    }
    
    std::string exportLatex() const override {
        std::stringstream latex;
        latex << "\\documentclass{article}\n";
        latex << "\\usepackage{booktabs}\n";
        latex << "\\begin{document}\n\n";
        
        latex << "\\section{Data Science Metrics Report}\n";
        latex << "\\subsection{" << report_.model_name << "}\n\n";
        
        latex << "\\begin{table}[h]\n";
        latex << "\\centering\n";
        latex << "\\begin{tabular}{lr}\n";
        latex << "\\toprule\n";
        latex << "Metric & Value \\\\\n";
        latex << "\\midrule\n";
        latex << "Accuracy & " << std::fixed << std::setprecision(1) 
              << (report_.model_metrics.accuracy * 100) << "\\% \\\\\n";
        latex << "F1 Score & " << std::setprecision(3) << report_.model_metrics.f1_score << " \\\\\n";
        latex << "Avg Latency & " << std::setprecision(1) 
              << report_.inference_metrics.avg_latency_ms << " ms \\\\\n";
        latex << "Hit Rate Improvement & " 
              << report_.prefetching_metrics.hit_rate_improvement_percentage << "\\% \\\\\n";
        latex << "\\bottomrule\n";
        latex << "\\end{tabular}\n";
        latex << "\\caption{Model Performance Summary}\n";
        latex << "\\end{table}\n\n";
        
        latex << "\\end{document}";
        return latex.str();
    }
};

std::unique_ptr<DSMetricsCollector> createDSMetricsCollector(
    const std::string& experiment_name,
    const std::string& model_name) {
    return std::make_unique<DSMetricsCollectorImpl>(experiment_name, model_name);
}

} // namespace benchmarks
} // namespace predis