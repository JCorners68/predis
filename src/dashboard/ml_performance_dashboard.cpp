#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include "../benchmarks/ml_performance_validator.h"
#include "../ppe/prefetch_monitor.h"

namespace predis {
namespace dashboard {

class MLPerformanceDashboard {
public:
    struct DashboardConfig {
        std::string output_path = "./dashboard/";
        bool auto_refresh = true;
        int refresh_interval_seconds = 30;
        bool include_charts = true;
        bool export_data = true;
    };
    
    MLPerformanceDashboard(const DashboardConfig& config = DashboardConfig())
        : config_(config) {}
    
    void generateDashboard(const benchmarks::MLPerformanceValidator::ValidationResult& result,
                          const ppe::PrefetchMonitor::PrefetchMetrics& realtime_metrics) {
        generateHTML(result, realtime_metrics);
        generateJSON(result, realtime_metrics);
        
        if (config_.include_charts) {
            generateChartData(result, realtime_metrics);
        }
    }
    
    void generateComparisonDashboard(
        const std::vector<benchmarks::MLPerformanceValidator::ValidationResult>& results,
        const std::vector<std::string>& labels) {
        
        std::ofstream html(config_.output_path + "comparison_dashboard.html");
        
        html << generateHTMLHeader("ML Performance Comparison Dashboard");
        html << "<body>\n";
        html << "<div class=\"container\">\n";
        html << "<h1>ML Performance Comparison Dashboard</h1>\n";
        html << "<p>Generated: " << getCurrentTimestamp() << "</p>\n";
        
        // Summary table
        html << "<h2>Performance Summary</h2>\n";
        html << "<table class=\"summary-table\">\n";
        html << "<tr>\n";
        html << "<th>Configuration</th>\n";
        html << "<th>Hit Rate Improvement</th>\n";
        html << "<th>Inference Latency</th>\n";
        html << "<th>CPU Overhead</th>\n";
        html << "<th>Overall Success</th>\n";
        html << "</tr>\n";
        
        for (size_t i = 0; i < results.size() && i < labels.size(); ++i) {
            const auto& r = results[i];
            html << "<tr>\n";
            html << "<td>" << labels[i] << "</td>\n";
            html << "<td class=\"" << getColorClass(r.hit_rate_improvement_percentage, 20.0, 10.0) 
                 << "\">" << std::fixed << std::setprecision(1) 
                 << r.hit_rate_improvement_percentage << "%</td>\n";
            html << "<td class=\"" << getColorClass(10.0 - r.avg_inference_latency_ms, 5.0, 0.0) 
                 << "\">" << std::setprecision(2) << r.avg_inference_latency_ms << "ms</td>\n";
            html << "<td class=\"" << getColorClass(1.0 - r.cpu_overhead_percentage, 0.5, 0.0) 
                 << "\">" << r.cpu_overhead_percentage << "%</td>\n";
            html << "<td class=\"" << (r.overall_success ? "success" : "failure") << "\">"
                 << (r.overall_success ? "PASS" : "FAIL") << "</td>\n";
            html << "</tr>\n";
        }
        
        html << "</table>\n";
        
        // Charts
        if (config_.include_charts) {
            html << generateComparisonCharts(results, labels);
        }
        
        html << "</div>\n";
        html << "</body>\n</html>";
        html.close();
    }
    
private:
    DashboardConfig config_;
    
    std::string generateHTMLHeader(const std::string& title) {
        std::ostringstream html;
        html << "<!DOCTYPE html>\n<html>\n<head>\n";
        html << "<title>" << title << "</title>\n";
        if (config_.auto_refresh) {
            html << "<meta http-equiv=\"refresh\" content=\"" 
                 << config_.refresh_interval_seconds << "\">\n";
        }
        html << "<style>\n";
        html << generateCSS();
        html << "</style>\n";
        html << "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n";
        html << "</head>\n";
        return html.str();
    }
    
    std::string generateCSS() {
        return R"(
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }
        .metric-target {
            color: #999;
            font-size: 12px;
            margin-top: 5px;
        }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .failure { color: #dc3545; }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-pass {
            background-color: #d4edda;
            color: #155724;
        }
        .status-fail {
            background-color: #f8d7da;
            color: #721c24;
        }
        )";
    }
    
    void generateHTML(const benchmarks::MLPerformanceValidator::ValidationResult& result,
                     const ppe::PrefetchMonitor::PrefetchMetrics& realtime_metrics) {
        std::ofstream html(config_.output_path + "ml_performance_dashboard.html");
        
        html << generateHTMLHeader("Predis ML Performance Dashboard");
        html << "<body>\n";
        html << "<div class=\"container\">\n";
        html << "<h1>Predis ML Performance Dashboard</h1>\n";
        html << "<p style=\"text-align: center; color: #666;\">Real-time Performance Monitoring</p>\n";
        html << "<p style=\"text-align: center; color: #999;\">Last updated: " 
             << getCurrentTimestamp() << "</p>\n";
        
        // Key metrics cards
        html << "<div class=\"metrics-grid\">\n";
        
        // Hit Rate Improvement
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Hit Rate Improvement</div>\n";
        html << "<div class=\"metric-value " 
             << getColorClass(result.hit_rate_improvement_percentage, 20.0, 10.0) << "\">"
             << std::fixed << std::setprecision(1) << result.hit_rate_improvement_percentage 
             << "%</div>\n";
        html << "<div class=\"metric-target\">Target: ≥20%</div>\n";
        html << "</div>\n";
        
        // Inference Latency
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Avg Inference Latency</div>\n";
        html << "<div class=\"metric-value " 
             << getColorClass(10.0 - result.avg_inference_latency_ms, 5.0, 0.0) << "\">"
             << std::setprecision(2) << result.avg_inference_latency_ms << "ms</div>\n";
        html << "<div class=\"metric-target\">Target: <10ms</div>\n";
        html << "</div>\n";
        
        // System Overhead
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">CPU Overhead</div>\n";
        html << "<div class=\"metric-value " 
             << getColorClass(1.0 - result.cpu_overhead_percentage, 0.5, 0.0) << "\">"
             << result.cpu_overhead_percentage << "%</div>\n";
        html << "<div class=\"metric-target\">Target: <1%</div>\n";
        html << "</div>\n";
        
        // Overall Status
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Overall Status</div>\n";
        html << "<div class=\"metric-value\">\n";
        html << "<span class=\"status-badge " 
             << (result.overall_success ? "status-pass" : "status-fail") << "\">"
             << (result.overall_success ? "PASS" : "FAIL") << "</span>\n";
        html << "</div>\n";
        html << "<div class=\"metric-target\">All criteria met</div>\n";
        html << "</div>\n";
        
        html << "</div>\n"; // metrics-grid
        
        // Real-time metrics
        html << "<h2>Real-time Prefetch Metrics</h2>\n";
        html << "<div class=\"metrics-grid\">\n";
        
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Predictions Made</div>\n";
        html << "<div class=\"metric-value\">" << realtime_metrics.total_predictions << "</div>\n";
        html << "</div>\n";
        
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Prefetch Accuracy</div>\n";
        html << "<div class=\"metric-value\">" << std::fixed << std::setprecision(1) 
             << (realtime_metrics.precision * 100) << "%</div>\n";
        html << "</div>\n";
        
        html << "<div class=\"metric-card\">\n";
        html << "<div class=\"metric-label\">Cache Hits from Prefetch</div>\n";
        html << "<div class=\"metric-value\">" << realtime_metrics.cache_hits_from_prefetch << "</div>\n";
        html << "</div>\n";
        
        html << "</div>\n";
        
        // Performance charts
        if (config_.include_charts) {
            html << generatePerformanceCharts(result, realtime_metrics);
        }
        
        // Detailed statistics table
        html << "<h2>Detailed Performance Statistics</h2>\n";
        html << "<table>\n";
        html << "<tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>\n";
        
        addTableRow(html, "Baseline Hit Rate", 
                   std::to_string(result.baseline_hit_rate * 100) + "%", "-", "");
        addTableRow(html, "ML-Enhanced Hit Rate", 
                   std::to_string(result.ml_hit_rate * 100) + "%", "-", "");
        addTableRow(html, "Hit Rate Improvement", 
                   std::to_string(result.hit_rate_improvement_percentage) + "%", 
                   "≥20%", result.meets_hit_rate_target);
        addTableRow(html, "Average Inference Latency", 
                   std::to_string(result.avg_inference_latency_ms) + "ms", 
                   "<10ms", result.meets_latency_target);
        addTableRow(html, "P95 Latency", 
                   std::to_string(result.p95_latency_ms) + "ms", "-", "");
        addTableRow(html, "P99 Latency", 
                   std::to_string(result.p99_latency_ms) + "ms", "-", "");
        addTableRow(html, "CPU Overhead", 
                   std::to_string(result.cpu_overhead_percentage) + "%", 
                   "<1%", result.meets_overhead_target);
        addTableRow(html, "Throughput", 
                   std::to_string(static_cast<int>(result.throughput_ops_sec)) + " ops/s", 
                   "-", "");
        
        html << "</table>\n";
        
        html << "</div>\n"; // container
        html << "</body>\n</html>";
        
        html.close();
    }
    
    void generateJSON(const benchmarks::MLPerformanceValidator::ValidationResult& result,
                     const ppe::PrefetchMonitor::PrefetchMetrics& realtime_metrics) {
        std::ofstream json(config_.output_path + "ml_performance_data.json");
        
        json << "{\n";
        json << "  \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
        json << "  \"validation_results\": {\n";
        json << "    \"hit_rate_improvement_percentage\": " << result.hit_rate_improvement_percentage << ",\n";
        json << "    \"avg_inference_latency_ms\": " << result.avg_inference_latency_ms << ",\n";
        json << "    \"cpu_overhead_percentage\": " << result.cpu_overhead_percentage << ",\n";
        json << "    \"overall_success\": " << (result.overall_success ? "true" : "false") << ",\n";
        json << "    \"meets_hit_rate_target\": " << (result.meets_hit_rate_target ? "true" : "false") << ",\n";
        json << "    \"meets_latency_target\": " << (result.meets_latency_target ? "true" : "false") << ",\n";
        json << "    \"meets_overhead_target\": " << (result.meets_overhead_target ? "true" : "false") << "\n";
        json << "  },\n";
        json << "  \"realtime_metrics\": {\n";
        json << "    \"total_predictions\": " << realtime_metrics.total_predictions << ",\n";
        json << "    \"prefetch_accuracy\": " << realtime_metrics.precision << ",\n";
        json << "    \"cache_hits_from_prefetch\": " << realtime_metrics.cache_hits_from_prefetch << ",\n";
        json << "    \"f1_score\": " << realtime_metrics.f1_score << "\n";
        json << "  }\n";
        json << "}\n";
        
        json.close();
    }
    
    std::string generatePerformanceCharts(const benchmarks::MLPerformanceValidator::ValidationResult& result,
                                         const ppe::PrefetchMonitor::PrefetchMetrics& realtime_metrics) {
        std::ostringstream html;
        
        // Hit rate comparison chart
        html << "<div class=\"chart-container\">\n";
        html << "<h3>Hit Rate Comparison</h3>\n";
        html << "<canvas id=\"hitRateChart\" width=\"400\" height=\"200\"></canvas>\n";
        html << "</div>\n";
        
        // Latency distribution chart
        html << "<div class=\"chart-container\">\n";
        html << "<h3>Latency Distribution</h3>\n";
        html << "<canvas id=\"latencyChart\" width=\"400\" height=\"200\"></canvas>\n";
        html << "</div>\n";
        
        // JavaScript for charts
        html << "<script>\n";
        
        // Hit rate chart
        html << "const hitRateCtx = document.getElementById('hitRateChart').getContext('2d');\n";
        html << "new Chart(hitRateCtx, {\n";
        html << "  type: 'bar',\n";
        html << "  data: {\n";
        html << "    labels: ['Baseline', 'With ML', 'Improvement'],\n";
        html << "    datasets: [{\n";
        html << "      label: 'Hit Rate %',\n";
        html << "      data: [" << (result.baseline_hit_rate * 100) << ", " 
             << (result.ml_hit_rate * 100) << ", " 
             << result.hit_rate_improvement_percentage << "],\n";
        html << "      backgroundColor: ['#6c757d', '#007bff', '#28a745']\n";
        html << "    }]\n";
        html << "  },\n";
        html << "  options: {\n";
        html << "    scales: { y: { beginAtZero: true } }\n";
        html << "  }\n";
        html << "});\n";
        
        // Latency chart
        html << "const latencyCtx = document.getElementById('latencyChart').getContext('2d');\n";
        html << "new Chart(latencyCtx, {\n";
        html << "  type: 'line',\n";
        html << "  data: {\n";
        html << "    labels: ['P50', 'Average', 'P95', 'P99'],\n";
        html << "    datasets: [{\n";
        html << "      label: 'Latency (ms)',\n";
        html << "      data: [" << result.p50_latency_ms << ", " 
             << result.avg_inference_latency_ms << ", " 
             << result.p95_latency_ms << ", " 
             << result.p99_latency_ms << "],\n";
        html << "      borderColor: '#007bff',\n";
        html << "      tension: 0.1\n";
        html << "    }]\n";
        html << "  },\n";
        html << "  options: {\n";
        html << "    scales: { y: { beginAtZero: true } }\n";
        html << "  }\n";
        html << "});\n";
        
        html << "</script>\n";
        
        return html.str();
    }
    
    std::string generateComparisonCharts(
        const std::vector<benchmarks::MLPerformanceValidator::ValidationResult>& results,
        const std::vector<std::string>& labels) {
        
        std::ostringstream html;
        
        html << "<div class=\"chart-container\">\n";
        html << "<h3>Hit Rate Improvement Comparison</h3>\n";
        html << "<canvas id=\"comparisonChart\" width=\"800\" height=\"400\"></canvas>\n";
        html << "</div>\n";
        
        html << "<script>\n";
        html << "const ctx = document.getElementById('comparisonChart').getContext('2d');\n";
        html << "new Chart(ctx, {\n";
        html << "  type: 'bar',\n";
        html << "  data: {\n";
        html << "    labels: [";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) html << ", ";
            html << "'" << labels[i] << "'";
        }
        html << "],\n";
        html << "    datasets: [{\n";
        html << "      label: 'Hit Rate Improvement %',\n";
        html << "      data: [";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) html << ", ";
            html << results[i].hit_rate_improvement_percentage;
        }
        html << "],\n";
        html << "      backgroundColor: '#007bff'\n";
        html << "    }]\n";
        html << "  },\n";
        html << "  options: {\n";
        html << "    scales: { y: { beginAtZero: true } },\n";
        html << "    plugins: {\n";
        html << "      annotation: {\n";
        html << "        annotations: {\n";
        html << "          line1: {\n";
        html << "            type: 'line',\n";
        html << "            yMin: 20,\n";
        html << "            yMax: 20,\n";
        html << "            borderColor: 'rgb(255, 99, 132)',\n";
        html << "            borderWidth: 2,\n";
        html << "            label: {\n";
        html << "              content: 'Target: 20%',\n";
        html << "              enabled: true\n";
        html << "            }\n";
        html << "          }\n";
        html << "        }\n";
        html << "      }\n";
        html << "    }\n";
        html << "  }\n";
        html << "});\n";
        html << "</script>\n";
        
        return html.str();
    }
    
    void generateChartData(const benchmarks::MLPerformanceValidator::ValidationResult& result,
                          const ppe::PrefetchMonitor::PrefetchMetrics& realtime_metrics) {
        // Export data for external charting tools
        std::ofstream data(config_.output_path + "chart_data.csv");
        
        data << "metric,value,target\n";
        data << "hit_rate_improvement," << result.hit_rate_improvement_percentage << ",20\n";
        data << "inference_latency," << result.avg_inference_latency_ms << ",10\n";
        data << "cpu_overhead," << result.cpu_overhead_percentage << ",1\n";
        data << "prefetch_accuracy," << (realtime_metrics.precision * 100) << ",80\n";
        
        data.close();
    }
    
    std::string getColorClass(double value, double good_threshold, double warning_threshold) {
        if (value >= good_threshold) return "success";
        if (value >= warning_threshold) return "warning";
        return "failure";
    }
    
    void addTableRow(std::ofstream& html, const std::string& metric, 
                    const std::string& value, const std::string& target, bool status) {
        html << "<tr>\n";
        html << "<td>" << metric << "</td>\n";
        html << "<td>" << value << "</td>\n";
        html << "<td>" << target << "</td>\n";
        html << "<td><span class=\"status-badge " 
             << (status ? "status-pass" : "status-fail") << "\">"
             << (status ? "PASS" : "FAIL") << "</span></td>\n";
        html << "</tr>\n";
    }
    
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

} // namespace dashboard
} // namespace predis