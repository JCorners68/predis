#pragma once

#include "../benchmarks/data_collector.h"
#include "../benchmarks/performance_benchmark_suite.h"
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <map>
#include <functional>

namespace predis {
namespace dashboard {

struct DashboardMetrics {
    struct PerformanceMetrics {
        double current_throughput_ops_per_sec;
        double average_latency_us;
        double p95_latency_us;
        double p99_latency_us;
        double improvement_factor_vs_redis;
        bool meets_epic2_targets;
        std::string performance_category;
    };
    
    struct SystemMetrics {
        double cpu_utilization_percent;
        double gpu_utilization_percent;
        double gpu_memory_usage_percent;
        double system_memory_usage_percent;
        double pcie_bandwidth_utilization_percent;
        size_t active_connections;
    };
    
    struct ComparisonMetrics {
        PerformanceMetrics redis_metrics;
        PerformanceMetrics predis_metrics;
        double improvement_ratio;
        double statistical_significance;
        std::vector<double> throughput_timeline;
        std::vector<double> latency_timeline;
    };
    
    PerformanceMetrics current_performance;
    SystemMetrics system_status;
    ComparisonMetrics redis_vs_predis;
    std::chrono::high_resolution_clock::time_point last_update;
};

struct DashboardConfig {
    struct DisplayConfig {
        size_t refresh_interval_ms = 250;
        size_t data_history_points = 100;
        bool enable_3d_visualization = true;
        bool show_technical_details = true;
        std::string theme = "professional"; // "professional", "dark", "investor"
    };
    
    struct DemoConfig {
        std::vector<std::string> available_workloads = {
            "READ_HEAVY", "WRITE_HEAVY", "MIXED", "BATCH_INTENSIVE", 
            "HIGH_CONCURRENCY", "ZIPFIAN_DISTRIBUTION"
        };
        size_t default_operations_count = 100000;
        size_t default_concurrent_clients = 10;
        bool auto_run_demos = false;
        bool enable_workload_switching = true;
    };
    
    DisplayConfig display;
    DemoConfig demo;
    std::string web_server_port = "8080";
    std::string dashboard_title = "Epic 2: Predis GPU Cache Performance Dashboard";
    bool enable_investor_mode = true;
    bool enable_technical_mode = true;
};

class RealTimeDashboard {
public:
    explicit RealTimeDashboard(const DashboardConfig& config = {});
    ~RealTimeDashboard();
    
    void start_dashboard();
    void stop_dashboard();
    
    void update_performance_metrics(const DashboardMetrics::PerformanceMetrics& metrics);
    void update_system_metrics(const DashboardMetrics::SystemMetrics& metrics);
    void update_comparison_metrics(const DashboardMetrics::ComparisonMetrics& metrics);
    
    void run_demo_scenario(const std::string& workload_type, 
                          size_t num_operations = 100000,
                          size_t concurrent_clients = 10);
    
    void switch_display_mode(const std::string& mode); // "investor", "technical", "presentation"
    
    std::string get_dashboard_url() const;
    DashboardMetrics get_current_metrics() const;
    
    void export_demo_results(const std::string& filename = "epic2_demo_results.html");

private:
    DashboardConfig config_;
    std::atomic<bool> is_running_{false};
    
    mutable std::mutex metrics_mutex_;
    DashboardMetrics current_metrics_;
    std::queue<DashboardMetrics> metrics_history_;
    
    std::thread dashboard_thread_;
    std::thread web_server_thread_;
    
    std::unique_ptr<predis::benchmarks::RealTimeDataCollector> data_collector_;
    std::unique_ptr<predis::benchmarks::PerformanceBenchmarkSuite> benchmark_suite_;
    
    void dashboard_worker();
    void web_server_worker();
    void collect_system_metrics();
    void run_continuous_benchmarks();
    
    std::string generate_dashboard_html() const;
    std::string generate_investor_view() const;
    std::string generate_technical_view() const;
    std::string generate_presentation_view() const;
    
    std::string generate_real_time_charts() const;
    std::string generate_comparison_charts() const;
    std::string generate_system_monitoring_charts() const;
    std::string generate_epic2_summary_panel() const;
    
    void update_metrics_history();
    std::string format_performance_summary() const;
};

class InteractiveDemoController {
public:
    struct DemoScenario {
        std::string name;
        std::string description;
        std::string workload_type;
        size_t operations_count;
        size_t concurrent_clients;
        std::map<std::string, std::string> parameters;
        std::function<void()> setup_callback;
        std::function<void()> teardown_callback;
    };
    
    explicit InteractiveDemoController(RealTimeDashboard* dashboard);
    
    void register_demo_scenario(const DemoScenario& scenario);
    void run_demo_scenario(const std::string& scenario_name);
    void run_all_scenarios();
    
    void create_epic2_presentation_sequence();
    void run_investor_demonstration();
    
    std::vector<std::string> get_available_scenarios() const;
    DemoScenario get_scenario(const std::string& name) const;

private:
    RealTimeDashboard* dashboard_;
    std::map<std::string, DemoScenario> scenarios_;
    
    void setup_standard_scenarios();
    void setup_epic2_validation_scenarios();
    void setup_investor_presentation_scenarios();
    
    void run_scenario_with_monitoring(const DemoScenario& scenario);
    void validate_epic2_targets_during_demo(const std::string& scenario_name);
};

class DashboardWebServer {
public:
    explicit DashboardWebServer(const std::string& port, RealTimeDashboard* dashboard);
    ~DashboardWebServer();
    
    void start_server();
    void stop_server();
    
    void register_endpoint(const std::string& path, 
                          std::function<std::string(const std::map<std::string, std::string>&)> handler);

private:
    std::string port_;
    RealTimeDashboard* dashboard_;
    std::atomic<bool> server_running_{false};
    std::thread server_thread_;
    
    std::map<std::string, std::function<std::string(const std::map<std::string, std::string>&)>> endpoints_;
    
    void server_worker();
    void setup_default_endpoints();
    
    std::string handle_dashboard_request(const std::map<std::string, std::string>& params);
    std::string handle_metrics_api_request(const std::map<std::string, std::string>& params);
    std::string handle_demo_control_request(const std::map<std::string, std::string>& params);
    std::string handle_export_request(const std::map<std::string, std::string>& params);
    
    std::string parse_http_request(const std::string& request);
    std::string generate_http_response(const std::string& content, const std::string& content_type = "text/html");
};

} // namespace dashboard
} // namespace predis