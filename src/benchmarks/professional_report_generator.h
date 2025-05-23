#pragma once

#include "data_collector.h"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace predis {
namespace benchmarks {

class ProfessionalReportGenerator {
public:
    struct BenchmarkResults {
        std::string test_name;
        double improvement_factor;
        double throughput_ops_per_sec;
        double average_latency_ms;
        double p99_latency_ms;
        double statistical_significance;
        std::string performance_category;
        bool meets_epic2_targets;
        std::string notes;
    };

    struct ReportConfig {
        std::string company_name = "Predis GPU Cache";
        std::string report_title = "Epic 2: Performance Benchmark Results";
        std::string execution_date;
        std::string version = "v2.4.1";
        bool include_executive_summary = true;
        bool include_charts = true;
        bool include_technical_details = true;
        std::string theme = "professional"; // professional, investor, technical
    };

    explicit ProfessionalReportGenerator(const ReportConfig& config = {});
    
    std::string generate_professional_html_report(
        const std::vector<BenchmarkResults>& results,
        const std::string& output_filename = "epic2_professional_report.html");
    
    std::string generate_investor_presentation(
        const std::vector<BenchmarkResults>& results,
        const std::string& output_filename = "epic2_investor_presentation.html");

private:
    ReportConfig config_;
    
    std::string generate_html_header() const;
    std::string generate_css_styles() const;
    std::string generate_javascript_charts(const std::vector<BenchmarkResults>& results) const;
    std::string generate_executive_summary(const std::vector<BenchmarkResults>& results) const;
    std::string generate_performance_dashboard(const std::vector<BenchmarkResults>& results) const;
    std::string generate_charts_section(const std::vector<BenchmarkResults>& results) const;
    std::string generate_technical_details(const std::vector<BenchmarkResults>& results) const;
    std::string generate_footer() const;
    
    double calculate_average_improvement(const std::vector<BenchmarkResults>& results) const;
    std::string format_number(double value, int precision = 1) const;
    std::string get_performance_color(const std::string& category) const;
};

} // namespace benchmarks
} // namespace predis