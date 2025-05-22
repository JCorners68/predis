#pragma once

#include "data_collector.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

namespace predis {
namespace benchmarks {

struct ChartData {
    std::vector<double> x_values;
    std::vector<double> y_values;
    std::string label;
    std::string color;
    std::string chart_type; // "line", "bar", "scatter", "histogram"
};

struct VisualizationConfig {
    struct ChartConfig {
        std::string title;
        std::string x_axis_label;
        std::string y_axis_label;
        size_t width = 800;
        size_t height = 600;
        bool show_legend = true;
        bool show_grid = true;
        std::string background_color = "#ffffff";
    };
    
    ChartConfig throughput_chart;
    ChartConfig latency_distribution_chart;
    ChartConfig performance_comparison_chart;
    ChartConfig resource_utilization_chart;
    ChartConfig timeline_chart;
    
    std::string output_directory = "./visualization_output/";
    std::string output_format = "html"; // "html", "svg", "png"
    bool enable_interactive_charts = true;
    bool generate_executive_dashboard = true;
};

class BenchmarkVisualizationGenerator {
public:
    explicit BenchmarkVisualizationGenerator(const VisualizationConfig& config = {});
    
    void generate_performance_comparison_chart(
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        const std::string& output_filename = "performance_comparison.html");
    
    void generate_latency_distribution_chart(
        const std::vector<double>& redis_latencies,
        const std::vector<double>& predis_latencies,
        const std::string& output_filename = "latency_distribution.html");
    
    void generate_throughput_timeline_chart(
        const std::vector<TimeSeriesDataPoint>& data,
        const std::string& output_filename = "throughput_timeline.html");
    
    void generate_resource_utilization_chart(
        const std::vector<TimeSeriesDataPoint>& data,
        const std::string& output_filename = "resource_utilization.html");
    
    void generate_epic2_dashboard(
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results,
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        const std::string& output_filename = "epic2_dashboard.html");
    
    void generate_statistical_validation_chart(
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results,
        const std::string& output_filename = "statistical_validation.html");
    
    void generate_all_visualizations(
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results);

private:
    VisualizationConfig config_;
    
    std::string generate_html_chart(const std::vector<ChartData>& chart_data,
                                   const VisualizationConfig::ChartConfig& chart_config,
                                   const std::string& chart_id = "chart");
    
    std::string generate_plotly_js_chart(const std::vector<ChartData>& chart_data,
                                        const VisualizationConfig::ChartConfig& chart_config);
    
    std::string generate_d3_js_chart(const std::vector<ChartData>& chart_data,
                                    const VisualizationConfig::ChartConfig& chart_config);
    
    std::string generate_css_styles() const;
    std::string generate_dashboard_header() const;
    std::string generate_epic2_summary_section(const BenchmarkDataAnalyzer::AnalysisResults& results) const;
    
    void ensure_output_directory() const;
    std::vector<double> calculate_histogram_bins(const std::vector<double>& data, size_t num_bins = 50) const;
    std::pair<std::vector<double>, std::vector<double>> create_histogram_data(
        const std::vector<double>& data, size_t num_bins = 50) const;
};

class InteractiveReportGenerator {
public:
    struct ReportConfig {
        std::string company_name = "Predis GPU Cache";
        std::string report_title = "Epic 2: Performance Benchmarking Results";
        std::string report_version = "1.0";
        bool include_executive_summary = true;
        bool include_technical_details = true;
        bool include_statistical_analysis = true;
        bool include_recommendations = true;
        std::string css_theme = "professional"; // "professional", "dark", "minimal"
    };
    
    explicit InteractiveReportGenerator(const ReportConfig& config = {});
    
    std::string generate_full_report(
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results,
        const std::string& output_filename = "epic2_full_report.html");
    
    std::string generate_executive_summary(
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results) const;
    
    std::string generate_technical_section(
        const std::vector<TimeSeriesDataPoint>& redis_data,
        const std::vector<TimeSeriesDataPoint>& predis_data,
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results) const;
    
    std::string generate_statistical_analysis_section(
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results) const;
    
    std::string generate_recommendations_section(
        const BenchmarkDataAnalyzer::AnalysisResults& analysis_results) const;

private:
    ReportConfig config_;
    BenchmarkVisualizationGenerator visualization_generator_;
    
    std::string get_theme_css() const;
    std::string format_number(double value, int precision = 2) const;
    std::string format_percentage(double value, int precision = 1) const;
    std::string get_performance_rating(double improvement_factor) const;
    std::string get_statistical_significance_interpretation(double p_value) const;
};

} // namespace benchmarks
} // namespace predis