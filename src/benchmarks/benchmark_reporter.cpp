/*
 * Copyright 2025 Predis Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "performance_benchmark_suite.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>

namespace predis {
namespace benchmarks {

// Report generation implementation for PerformanceBenchmarkSuite
bool PerformanceBenchmarkSuite::generate_benchmark_report(const std::vector<BenchmarkResult>& results,
                                                          const std::string& output_file) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to create report file: " << output_file << std::endl;
        return false;
    }
    
    // Generate comprehensive HTML report
    file << R"html(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predis vs Redis Performance Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }
        .epic-badge { background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
        .summary { background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .metric-card { background: white; border: 1px solid #bdc3c7; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .improvement-good { color: #27ae60; font-weight: bold; }
        .improvement-excellent { color: #2ecc71; font-weight: bold; }
        .improvement-poor { color: #e74c3c; font-weight: bold; }
        .target-met { background: #d5f4e6; border-left: 4px solid #27ae60; }
        .target-missed { background: #fdf2f2; border-left: 4px solid #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .stats-significant { background: #d5f4e6; padding: 4px 8px; border-radius: 4px; }
        .stats-not-significant { background: #fff3cd; padding: 4px 8px; border-radius: 4px; }
        .chart-container { margin: 20px 0; text-align: center; }
        .performance-chart { width: 100%; height: 400px; border: 1px solid #ddd; background: #f9f9f9; display: flex; align-items: center; justify-content: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Predis vs Redis Performance Benchmark Report</h1>
            <p class="epic-badge">Epic 2 Story 2.4: Performance Validation Suite</p>
            <p>Target: Demonstrate 10-25x performance improvements over Redis</p>
        </div>
)html";
    
    // Generate executive summary
    file << generate_executive_summary(results);
    
    // Generate detailed results
    file << "<h2>📊 Detailed Benchmark Results</h2>\n";
    file << "<table>\n";
    file << "<tr><th>Test Name</th><th>Workload</th><th>Redis (ops/sec)</th><th>Predis (ops/sec)</th>";
    file << "<th>Improvement</th><th>Latency Improvement</th><th>Statistical Significance</th><th>Epic 2 Target</th></tr>\n";
    
    for (const auto& result : results) {
        file << "<tr class=\"" << (result.meets_epic2_target ? "target-met" : "target-missed") << "\">\n";
        file << "<td>" << result.test_name << "</td>\n";
        file << "<td>" << result.workload_description << "</td>\n";
        file << "<td>" << std::fixed << std::setprecision(0) << result.redis_metrics.average_ops_per_second << "</td>\n";
        file << "<td>" << std::fixed << std::setprecision(0) << result.predis_metrics.average_ops_per_second << "</td>\n";
        
        std::string improvement_class = "improvement-poor";
        if (result.improvement_factor >= 25.0) improvement_class = "improvement-excellent";
        else if (result.improvement_factor >= 10.0) improvement_class = "improvement-good";
        
        file << "<td class=\"" << improvement_class << "\">" << std::fixed << std::setprecision(1) << result.improvement_factor << "x</td>\n";
        file << "<td>" << std::fixed << std::setprecision(1) << result.latency_improvement_factor << "x</td>\n";
        
        std::string stats_class = result.statistics.statistically_significant ? "stats-significant" : "stats-not-significant";
        file << "<td class=\"" << stats_class << "\">" << result.statistics.statistical_summary << "</td>\n";
        
        file << "<td>" << (result.meets_epic2_target ? "✅ 25x+" : (result.meets_minimum_target ? "⚠️ 10x+" : "❌ <10x")) << "</td>\n";
        file << "</tr>\n";
    }
    
    file << "</table>\n";
    
    // Generate performance analysis
    file << generate_performance_analysis(results);
    
    // Generate system information
    file << "<h2>🖥️ System Information</h2>\n";
    file << "<div class=\"metric-card\">\n";
    file << "<p><strong>System:</strong> " << get_system_info() << "</p>\n";
    file << "<p><strong>GPU:</strong> " << get_gpu_info() << "</p>\n";
    file << "<p><strong>Redis Version:</strong> " << get_redis_version() << "</p>\n";
    file << "<p><strong>Report Generated:</strong> " << generate_timestamp() << "</p>\n";
    file << "</div>\n";
    
    // Close HTML
    file << R"html(
    </div>
</body>
</html>
)html";
    
    file.close();
    
    std::cout << "Comprehensive benchmark report generated: " << output_file << std::endl;
    return true;
}

std::string PerformanceBenchmarkSuite::generate_executive_summary(const std::vector<BenchmarkResult>& results) {
    std::ostringstream summary;
    
    if (results.empty()) {
        summary << "<div class=\"summary\"><h2>❌ No Results Available</h2></div>\n";
        return summary.str();
    }
    
    // Calculate overall statistics
    size_t tests_meeting_epic2_target = 0;
    size_t tests_meeting_minimum_target = 0;
    size_t statistically_significant_tests = 0;
    
    double avg_improvement = 0.0;
    double max_improvement = 0.0;
    double min_improvement = std::numeric_limits<double>::max();
    
    for (const auto& result : results) {
        if (result.meets_epic2_target) tests_meeting_epic2_target++;
        if (result.meets_minimum_target) tests_meeting_minimum_target++;
        if (result.statistics.statistically_significant) statistically_significant_tests++;
        
        avg_improvement += result.improvement_factor;
        max_improvement = std::max(max_improvement, result.improvement_factor);
        min_improvement = std::min(min_improvement, result.improvement_factor);
    }
    
    avg_improvement /= results.size();
    
    summary << "<div class=\"summary\">\n";
    summary << "<h2>📈 Executive Summary</h2>\n";
    
    // Overall Epic 2 success assessment
    bool epic2_success = (tests_meeting_epic2_target >= results.size() * 0.7);  // 70% of tests meet 25x target
    bool epic2_partial = (tests_meeting_minimum_target >= results.size() * 0.8);  // 80% meet 10x minimum
    
    if (epic2_success) {
        summary << "<div class=\"metric-card target-met\">\n";
        summary << "<h3>🎯 EPIC 2 SUCCESS: Performance Targets Achieved</h3>\n";
        summary << "<p><strong>" << tests_meeting_epic2_target << "/" << results.size() 
                << " tests achieve 25x+ improvement target</strong></p>\n";
    } else if (epic2_partial) {
        summary << "<div class=\"metric-card target-missed\">\n";
        summary << "<h3>⚠️ EPIC 2 PARTIAL: Minimum Targets Met</h3>\n";
        summary << "<p><strong>" << tests_meeting_minimum_target << "/" << results.size() 
                << " tests achieve 10x+ minimum improvement</strong></p>\n";
    } else {
        summary << "<div class=\"metric-card target-missed\">\n";
        summary << "<h3>❌ EPIC 2 NEEDS WORK: Targets Not Met</h3>\n";
        summary << "<p><strong>Only " << tests_meeting_minimum_target << "/" << results.size() 
                << " tests achieve minimum 10x improvement</strong></p>\n";
    }
    
    summary << "<p><strong>Average Improvement:</strong> " << std::fixed << std::setprecision(1) 
            << avg_improvement << "x</p>\n";
    summary << "<p><strong>Best Performance:</strong> " << std::fixed << std::setprecision(1) 
            << max_improvement << "x improvement</p>\n";
    summary << "<p><strong>Statistical Significance:</strong> " << statistically_significant_tests 
            << "/" << results.size() << " tests statistically significant</p>\n";
    summary << "</div>\n";
    
    // Key findings
    summary << "<div class=\"metric-card\">\n";
    summary << "<h3>🔍 Key Findings</h3>\n";
    summary << "<ul>\n";
    
    // Find best and worst performing workloads
    auto best_result = *std::max_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.improvement_factor < b.improvement_factor; });
    auto worst_result = *std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.improvement_factor < b.improvement_factor; });
    
    summary << "<li><strong>Best Workload:</strong> " << best_result.test_name 
            << " with " << std::fixed << std::setprecision(1) << best_result.improvement_factor << "x improvement</li>\n";
    summary << "<li><strong>Most Challenging:</strong> " << worst_result.test_name 
            << " with " << std::fixed << std::setprecision(1) << worst_result.improvement_factor << "x improvement</li>\n";
    
    // Analyze batch vs single operation performance
    double batch_avg = 0.0, single_avg = 0.0;
    size_t batch_count = 0, single_count = 0;
    
    for (const auto& result : results) {
        if (result.test_name.find("Batch") != std::string::npos) {
            batch_avg += result.improvement_factor;
            batch_count++;
        } else {
            single_avg += result.improvement_factor;
            single_count++;
        }
    }
    
    if (batch_count > 0 && single_count > 0) {
        batch_avg /= batch_count;
        single_avg /= single_count;
        
        summary << "<li><strong>Batch Operations:</strong> " << std::fixed << std::setprecision(1) 
                << batch_avg << "x average improvement</li>\n";
        summary << "<li><strong>Single Operations:</strong> " << std::fixed << std::setprecision(1) 
                << single_avg << "x average improvement</li>\n";
    }
    
    summary << "</ul>\n";
    summary << "</div>\n";
    summary << "</div>\n";
    
    return summary.str();
}

std::string PerformanceBenchmarkSuite::generate_performance_analysis(const std::vector<BenchmarkResult>& results) {
    std::ostringstream analysis;
    
    analysis << "<h2>📊 Performance Analysis</h2>\n";
    
    // Latency analysis
    analysis << "<div class=\"metric-card\">\n";
    analysis << "<h3>⏱️ Latency Analysis</h3>\n";
    analysis << "<table>\n";
    analysis << "<tr><th>Test</th><th>Redis Avg (ms)</th><th>Predis Avg (ms)</th><th>Redis P99 (ms)</th><th>Predis P99 (ms)</th><th>Improvement</th></tr>\n";
    
    for (const auto& result : results) {
        analysis << "<tr>\n";
        analysis << "<td>" << result.test_name << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.redis_metrics.average_latency_ms << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.predis_metrics.average_latency_ms << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.redis_metrics.p99_latency_ms << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.predis_metrics.p99_latency_ms << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(1) << result.latency_improvement_factor << "x</td>\n";
        analysis << "</tr>\n";
    }
    
    analysis << "</table>\n";
    analysis << "</div>\n";
    
    // Performance consistency analysis
    analysis << "<div class=\"metric-card\">\n";
    analysis << "<h3>📈 Performance Consistency</h3>\n";
    analysis << "<table>\n";
    analysis << "<tr><th>Test</th><th>Redis CV</th><th>Predis CV</th><th>Consistency Improvement</th></tr>\n";
    
    for (const auto& result : results) {
        double consistency_improvement = (result.redis_metrics.coefficient_of_variation > 0) ?
            result.redis_metrics.coefficient_of_variation / result.predis_metrics.coefficient_of_variation : 1.0;
        
        analysis << "<tr>\n";
        analysis << "<td>" << result.test_name << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.redis_metrics.coefficient_of_variation << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(3) << result.predis_metrics.coefficient_of_variation << "</td>\n";
        analysis << "<td>" << std::fixed << std::setprecision(1) << consistency_improvement << "x</td>\n";
        analysis << "</tr>\n";
    }
    
    analysis << "</table>\n";
    analysis << "<p><em>CV = Coefficient of Variation (lower is more consistent)</em></p>\n";
    analysis << "</div>\n";
    
    return analysis.str();
}

bool PerformanceBenchmarkSuite::export_results_to_csv(const std::vector<BenchmarkResult>& results,
                                                      const std::string& csv_file) {
    std::ofstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Failed to create CSV file: " << csv_file << std::endl;
        return false;
    }
    
    // CSV header
    file << "Test_Name,Workload_Description,";
    file << "Redis_Avg_OpsPerSec,Predis_Avg_OpsPerSec,Improvement_Factor,";
    file << "Redis_Avg_Latency_Ms,Predis_Avg_Latency_Ms,Latency_Improvement,";
    file << "Redis_P99_Latency_Ms,Predis_P99_Latency_Ms,";
    file << "Statistical_Significance,P_Value,Effect_Size,";
    file << "Meets_Minimum_Target,Meets_Epic2_Target,";
    file << "Timestamp\n";
    
    // Data rows
    for (const auto& result : results) {
        file << "\"" << result.test_name << "\",";
        file << "\"" << result.workload_description << "\",";
        file << std::fixed << std::setprecision(2) << result.redis_metrics.average_ops_per_second << ",";
        file << std::fixed << std::setprecision(2) << result.predis_metrics.average_ops_per_second << ",";
        file << std::fixed << std::setprecision(3) << result.improvement_factor << ",";
        file << std::fixed << std::setprecision(3) << result.redis_metrics.average_latency_ms << ",";
        file << std::fixed << std::setprecision(3) << result.predis_metrics.average_latency_ms << ",";
        file << std::fixed << std::setprecision(3) << result.latency_improvement_factor << ",";
        file << std::fixed << std::setprecision(3) << result.redis_metrics.p99_latency_ms << ",";
        file << std::fixed << std::setprecision(3) << result.predis_metrics.p99_latency_ms << ",";
        file << (result.statistics.statistically_significant ? "TRUE" : "FALSE") << ",";
        file << std::fixed << std::setprecision(4) << result.statistics.p_value << ",";
        file << std::fixed << std::setprecision(3) << result.statistics.effect_size << ",";
        file << (result.meets_minimum_target ? "TRUE" : "FALSE") << ",";
        file << (result.meets_epic2_target ? "TRUE" : "FALSE") << ",";
        
        // Timestamp
        auto time = std::chrono::system_clock::to_time_t(result.timestamp);
        file << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "\n";
    }
    
    file.close();
    std::cout << "Benchmark results exported to CSV: " << csv_file << std::endl;
    return true;
}

bool PerformanceBenchmarkSuite::export_results_to_json(const std::vector<BenchmarkResult>& results,
                                                       const std::string& json_file) {
    std::ofstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "Failed to create JSON file: " << json_file << std::endl;
        return false;
    }
    
    file << "{\n";
    file << "  \"benchmark_suite\": \"Predis vs Redis Performance Comparison\",\n";
    file << "  \"epic\": \"Epic 2 Story 2.4\",\n";
    file << "  \"target\": \"10-25x performance improvement over Redis\",\n";
    file << "  \"timestamp\": \"" << generate_timestamp() << "\",\n";
    file << "  \"system_info\": \"" << get_system_info() << "\",\n";
    file << "  \"gpu_info\": \"" << get_gpu_info() << "\",\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        file << "    {\n";
        file << "      \"test_name\": \"" << result.test_name << "\",\n";
        file << "      \"workload_description\": \"" << result.workload_description << "\",\n";
        file << "      \"redis_metrics\": {\n";
        file << "        \"average_ops_per_second\": " << std::fixed << std::setprecision(2) << result.redis_metrics.average_ops_per_second << ",\n";
        file << "        \"average_latency_ms\": " << std::fixed << std::setprecision(3) << result.redis_metrics.average_latency_ms << ",\n";
        file << "        \"p99_latency_ms\": " << std::fixed << std::setprecision(3) << result.redis_metrics.p99_latency_ms << "\n";
        file << "      },\n";
        file << "      \"predis_metrics\": {\n";
        file << "        \"average_ops_per_second\": " << std::fixed << std::setprecision(2) << result.predis_metrics.average_ops_per_second << ",\n";
        file << "        \"average_latency_ms\": " << std::fixed << std::setprecision(3) << result.predis_metrics.average_latency_ms << ",\n";
        file << "        \"p99_latency_ms\": " << std::fixed << std::setprecision(3) << result.predis_metrics.p99_latency_ms << "\n";
        file << "      },\n";
        file << "      \"improvement_factor\": " << std::fixed << std::setprecision(3) << result.improvement_factor << ",\n";
        file << "      \"latency_improvement_factor\": " << std::fixed << std::setprecision(3) << result.latency_improvement_factor << ",\n";
        file << "      \"statistical_analysis\": {\n";
        file << "        \"statistically_significant\": " << (result.statistics.statistically_significant ? "true" : "false") << ",\n";
        file << "        \"p_value\": " << std::fixed << std::setprecision(4) << result.statistics.p_value << ",\n";
        file << "        \"effect_size\": " << std::fixed << std::setprecision(3) << result.statistics.effect_size << "\n";
        file << "      },\n";
        file << "      \"epic2_targets\": {\n";
        file << "        \"meets_minimum_target\": " << (result.meets_minimum_target ? "true" : "false") << ",\n";
        file << "        \"meets_epic2_target\": " << (result.meets_epic2_target ? "true" : "false") << "\n";
        file << "      }\n";
        file << "    }";
        
        if (i < results.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "Benchmark results exported to JSON: " << json_file << std::endl;
    return true;
}

PerformanceBenchmarkSuite::CIResult PerformanceBenchmarkSuite::validate_for_ci_pipeline(const std::vector<BenchmarkResult>& results) {
    CIResult ci_result;
    ci_result.total_benchmarks = results.size();
    
    for (const auto& result : results) {
        if (result.meets_minimum_target && result.statistics.statistically_significant) {
            ci_result.passed_benchmarks++;
        } else {
            ci_result.failed_benchmarks++;
            
            std::string failure_reason = result.test_name + ": ";
            if (!result.meets_minimum_target) {
                failure_reason += "Below 10x improvement (" + std::to_string(result.improvement_factor) + "x)";
            }
            if (!result.statistics.statistically_significant) {
                failure_reason += " Not statistically significant";
            }
            
            ci_result.failure_reasons.push_back(failure_reason);
        }
    }
    
    ci_result.all_tests_passed = (ci_result.failed_benchmarks == 0);
    
    // Generate summary message
    std::ostringstream summary;
    summary << "Benchmark Results: " << ci_result.passed_benchmarks << "/" << ci_result.total_benchmarks << " passed";
    
    if (ci_result.all_tests_passed) {
        summary << " ✅ All Epic 2 performance targets met";
    } else {
        summary << " ❌ " << ci_result.failed_benchmarks << " benchmarks failed Epic 2 targets";
    }
    
    ci_result.summary_message = summary.str();
    
    return ci_result;
}

std::string PerformanceBenchmarkSuite::get_redis_version() const {
    return "Redis 7.0+";  // Simplified for this implementation
}

std::string PerformanceBenchmarkSuite::generate_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return timestamp.str();
}

} // namespace benchmarks
} // namespace predis