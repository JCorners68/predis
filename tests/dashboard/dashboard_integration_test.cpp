#include "../../src/dashboard/real_time_dashboard.h"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

namespace predis {
namespace dashboard {
namespace test {

class DashboardIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        DashboardConfig config;
        config.web_server_port = "8081"; // Use different port for testing
        config.display.refresh_interval_ms = 100; // Faster refresh for testing
        config.demo.auto_run_demos = false;
        
        dashboard_ = std::make_unique<RealTimeDashboard>(config);
    }
    
    void TearDown() override {
        if (dashboard_) {
            dashboard_->stop_dashboard();
        }
    }
    
    std::unique_ptr<RealTimeDashboard> dashboard_;
};

TEST_F(DashboardIntegrationTest, DashboardStartStop) {
    EXPECT_NO_THROW(dashboard_->start_dashboard());
    
    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_NO_THROW(dashboard_->stop_dashboard());
}

TEST_F(DashboardIntegrationTest, MetricsUpdate) {
    dashboard_->start_dashboard();
    
    // Update performance metrics
    DashboardMetrics::PerformanceMetrics perf_metrics;
    perf_metrics.current_throughput_ops_per_sec = 1500000.0;
    perf_metrics.average_latency_us = 0.8;
    perf_metrics.p95_latency_us = 2.1;
    perf_metrics.p99_latency_us = 4.2;
    perf_metrics.improvement_factor_vs_redis = 20.5;
    perf_metrics.meets_epic2_targets = true;
    perf_metrics.performance_category = "EXCELLENT";
    
    EXPECT_NO_THROW(dashboard_->update_performance_metrics(perf_metrics));
    
    // Update system metrics
    DashboardMetrics::SystemMetrics sys_metrics;
    sys_metrics.cpu_utilization_percent = 45.0;
    sys_metrics.gpu_utilization_percent = 78.0;
    sys_metrics.gpu_memory_usage_percent = 65.0;
    sys_metrics.system_memory_usage_percent = 55.0;
    sys_metrics.pcie_bandwidth_utilization_percent = 82.0;
    sys_metrics.active_connections = 10;
    
    EXPECT_NO_THROW(dashboard_->update_system_metrics(sys_metrics));
    
    // Verify metrics are retrievable
    auto current_metrics = dashboard_->get_current_metrics();
    EXPECT_DOUBLE_EQ(current_metrics.current_performance.improvement_factor_vs_redis, 20.5);
    EXPECT_TRUE(current_metrics.current_performance.meets_epic2_targets);
    EXPECT_EQ(current_metrics.current_performance.performance_category, "EXCELLENT");
    
    dashboard_->stop_dashboard();
}

TEST_F(DashboardIntegrationTest, DemoScenarioExecution) {
    dashboard_->start_dashboard();
    
    // Test different demo scenarios
    EXPECT_NO_THROW(dashboard_->run_demo_scenario("READ_HEAVY", 10000, 5));
    EXPECT_NO_THROW(dashboard_->run_demo_scenario("WRITE_HEAVY", 10000, 5));
    EXPECT_NO_THROW(dashboard_->run_demo_scenario("BATCH_INTENSIVE", 10000, 5));
    
    dashboard_->stop_dashboard();
}

TEST_F(DashboardIntegrationTest, DisplayModeSwitch) {
    dashboard_->start_dashboard();
    
    EXPECT_NO_THROW(dashboard_->switch_display_mode("investor"));
    EXPECT_NO_THROW(dashboard_->switch_display_mode("technical"));
    EXPECT_NO_THROW(dashboard_->switch_display_mode("presentation"));
    
    dashboard_->stop_dashboard();
}

TEST_F(DashboardIntegrationTest, DashboardHTMLGeneration) {
    // Set up some test metrics
    DashboardMetrics::PerformanceMetrics perf_metrics;
    perf_metrics.current_throughput_ops_per_sec = 1600000.0;
    perf_metrics.average_latency_us = 0.75;
    perf_metrics.improvement_factor_vs_redis = 21.3;
    perf_metrics.meets_epic2_targets = true;
    perf_metrics.performance_category = "EXCELLENT";
    
    dashboard_->update_performance_metrics(perf_metrics);
    
    // Test HTML export
    EXPECT_NO_THROW(dashboard_->export_demo_results("test_dashboard.html"));
    
    // Verify the file was created and contains expected content
    std::ifstream file("test_dashboard.html");
    EXPECT_TRUE(file.good());
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(content.find("Epic 2: Predis GPU Cache Performance Dashboard") != std::string::npos);
    EXPECT_TRUE(content.find("21.3x") != std::string::npos);
    EXPECT_TRUE(content.find("EXCELLENT") != std::string::npos);
    
    // Clean up
    std::remove("test_dashboard.html");
}

TEST_F(DashboardIntegrationTest, Epic2TargetValidation) {
    dashboard_->start_dashboard();
    
    // Test Epic 2 target validation with different performance levels
    DashboardMetrics::PerformanceMetrics metrics;
    
    // Test with performance that meets Epic 2 targets
    metrics.improvement_factor_vs_redis = 15.5;
    metrics.meets_epic2_targets = true;
    dashboard_->update_performance_metrics(metrics);
    
    auto current = dashboard_->get_current_metrics();
    EXPECT_TRUE(current.current_performance.meets_epic2_targets);
    EXPECT_GE(current.current_performance.improvement_factor_vs_redis, 10.0);
    
    // Test with performance that doesn't meet Epic 2 targets
    metrics.improvement_factor_vs_redis = 8.5;
    metrics.meets_epic2_targets = false;
    dashboard_->update_performance_metrics(metrics);
    
    current = dashboard_->get_current_metrics();
    EXPECT_FALSE(current.current_performance.meets_epic2_targets);
    
    dashboard_->stop_dashboard();
}

class InteractiveDemoControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        DashboardConfig config;
        config.web_server_port = "8082";
        dashboard_ = std::make_unique<RealTimeDashboard>(config);
        demo_controller_ = std::make_unique<InteractiveDemoController>(dashboard_.get());
    }
    
    void TearDown() override {
        if (dashboard_) {
            dashboard_->stop_dashboard();
        }
    }
    
    std::unique_ptr<RealTimeDashboard> dashboard_;
    std::unique_ptr<InteractiveDemoController> demo_controller_;
};

TEST_F(InteractiveDemoControllerTest, ScenarioRegistration) {
    auto available_scenarios = demo_controller_->get_available_scenarios();
    
    // Should have default scenarios
    EXPECT_GT(available_scenarios.size(), 0);
    
    // Register a custom scenario
    InteractiveDemoController::DemoScenario custom_scenario = {
        "Custom_Test_Scenario",
        "Test scenario for validation",
        "MIXED",
        50000,
        8,
        {},
        nullptr,
        nullptr
    };
    
    EXPECT_NO_THROW(demo_controller_->register_demo_scenario(custom_scenario));
    
    auto updated_scenarios = demo_controller_->get_available_scenarios();
    EXPECT_GT(updated_scenarios.size(), available_scenarios.size());
    
    // Verify we can retrieve the custom scenario
    auto retrieved_scenario = demo_controller_->get_scenario("Custom_Test_Scenario");
    EXPECT_EQ(retrieved_scenario.name, "Custom_Test_Scenario");
    EXPECT_EQ(retrieved_scenario.operations_count, 50000);
}

TEST_F(InteractiveDemoControllerTest, Epic2PresentationSequence) {
    dashboard_->start_dashboard();
    
    // Test Epic 2 presentation sequence
    EXPECT_NO_THROW(demo_controller_->create_epic2_presentation_sequence());
    
    // Test investor demonstration
    EXPECT_NO_THROW(demo_controller_->run_investor_demonstration());
    
    dashboard_->stop_dashboard();
}

TEST_F(InteractiveDemoControllerTest, ScenarioExecution) {
    dashboard_->start_dashboard();
    
    auto available_scenarios = demo_controller_->get_available_scenarios();
    
    // Test running individual scenarios
    for (const auto& scenario_name : available_scenarios) {
        if (scenario_name.find("Test") == std::string::npos) { // Skip test scenarios
            EXPECT_NO_THROW(demo_controller_->run_demo_scenario(scenario_name));
        }
    }
    
    dashboard_->stop_dashboard();
}

class DashboardPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        DashboardConfig config;
        config.display.refresh_interval_ms = 50; // High frequency for performance testing
        config.display.data_history_points = 1000;
        dashboard_ = std::make_unique<RealTimeDashboard>(config);
    }
    
    void TearDown() override {
        if (dashboard_) {
            dashboard_->stop_dashboard();
        }
    }
    
    std::unique_ptr<RealTimeDashboard> dashboard_;
};

TEST_F(DashboardPerformanceTest, HighFrequencyUpdates) {
    dashboard_->start_dashboard();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Perform many rapid updates
    for (int i = 0; i < 1000; ++i) {
        DashboardMetrics::PerformanceMetrics metrics;
        metrics.current_throughput_ops_per_sec = 1500000.0 + (i % 100000);
        metrics.average_latency_us = 0.8 + (i % 100) / 1000.0;
        metrics.improvement_factor_vs_redis = 18.0 + (i % 10);
        metrics.meets_epic2_targets = true;
        metrics.performance_category = "EXCELLENT";
        
        dashboard_->update_performance_metrics(metrics);
        
        if (i % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete 1000 updates in reasonable time (< 5 seconds)
    EXPECT_LT(duration.count(), 5000);
    
    // Verify the dashboard is still responsive
    auto current_metrics = dashboard_->get_current_metrics();
    EXPECT_TRUE(current_metrics.current_performance.meets_epic2_targets);
    
    dashboard_->stop_dashboard();
}

TEST_F(DashboardPerformanceTest, ConcurrentAccess) {
    dashboard_->start_dashboard();
    
    std::vector<std::thread> threads;
    std::atomic<int> successful_updates{0};
    const int num_threads = 5;
    const int updates_per_thread = 100;
    
    // Launch multiple threads updating metrics concurrently
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < updates_per_thread; ++i) {
                try {
                    DashboardMetrics::PerformanceMetrics metrics;
                    metrics.current_throughput_ops_per_sec = 1500000.0 + t * 10000 + i;
                    metrics.improvement_factor_vs_redis = 15.0 + t;
                    metrics.meets_epic2_targets = true;
                    
                    dashboard_->update_performance_metrics(metrics);
                    successful_updates.fetch_add(1);
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                } catch (...) {
                    // Count any exceptions as failures
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all updates were successful
    EXPECT_EQ(successful_updates.load(), num_threads * updates_per_thread);
    
    // Verify dashboard is still operational
    auto current_metrics = dashboard_->get_current_metrics();
    EXPECT_TRUE(current_metrics.current_performance.meets_epic2_targets);
    
    dashboard_->stop_dashboard();
}

} // namespace test
} // namespace dashboard
} // namespace predis