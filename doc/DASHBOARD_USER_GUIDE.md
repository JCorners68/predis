# Epic 2 Performance Dashboard - User Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Dashboard Interface](#dashboard-interface)
4. [Running Demo Scenarios](#running-demo-scenarios)
5. [Display Modes](#display-modes)
6. [Exporting Results](#exporting-results)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Epic 2 Performance Dashboard is a real-time visualization tool that demonstrates Predis GPU cache performance improvements over Redis. It provides:

- **Real-time metrics** showing 10-25x performance improvements
- **Interactive demo scenarios** with configurable workloads
- **Professional presentation modes** for investor demonstrations
- **Statistical validation** with confidence intervals and significance testing
- **Exportable reports** suitable for presentations and documentation

### System Requirements
- RTX 5080 GPU (16GB VRAM) or compatible GPU
- WSL2 + Docker environment (recommended)
- Web browser with JavaScript enabled
- Network access to localhost:8080 (default port)

---

## Quick Start

### 1. Starting the Dashboard

```cpp
#include "src/dashboard/real_time_dashboard.h"

using namespace predis::dashboard;

// Create dashboard with default configuration
RealTimeDashboard dashboard;

// Start the dashboard
dashboard.start_dashboard();

// Access at: http://localhost:8080/dashboard
std::cout << "Dashboard URL: " << dashboard.get_dashboard_url() << std::endl;
```

### 2. Basic Demo Execution

```cpp
// Run a basic read-heavy demonstration
dashboard.run_demo_scenario("READ_HEAVY", 100000, 10);

// Switch to investor presentation mode
dashboard.switch_display_mode("investor");

// Export results for presentation
dashboard.export_demo_results("epic2_demo.html");

// Stop when done
dashboard.stop_dashboard();
```

### 3. Custom Configuration

```cpp
// Configure dashboard for specific needs
DashboardConfig config;
config.web_server_port = "8080";
config.dashboard_title = "Epic 2: Predis Performance Demo";
config.display.refresh_interval_ms = 250;
config.display.theme = "professional";
config.demo.default_operations_count = 100000;
config.enable_investor_mode = true;

RealTimeDashboard dashboard(config);
```

---

## Dashboard Interface

### Main Dashboard Components

#### 1. **Epic 2 Summary Panel** (Top Section)
```
┌─────────────────────────────────────────────────────────────┐
│ Performance Improvement: 20.5x (vs Redis, Target: 10-25x)  │
│ Current Throughput: 1.6M ops/sec                           │
│ Average Latency: 0.8ms (P99: 4.2ms)                       │
│ Performance Category: EXCELLENT                            │
│ GPU Utilization: 78% (Memory: 65%)                        │
│ System Status: OPTIMAL                                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Metrics Explained:**
- **Performance Improvement**: Real-time calculation of Predis vs Redis performance ratio
- **Current Throughput**: Operations per second (target: >1M ops/sec)
- **Latency**: Average and P99 latency in milliseconds (target: <1ms)
- **Performance Category**: GOOD (10-15x), EXCELLENT (15-25x), EXCEPTIONAL (>25x)
- **GPU Utilization**: Real-time GPU and memory usage monitoring
- **System Status**: Overall system health indicator

#### 2. **Real-Time Performance Charts** (Middle Section)

**a) Performance Timeline Chart**
- Shows throughput over time (last 100 data points)
- Updates every 250ms with live data
- Visualizes performance consistency and trends

**b) Redis vs Predis Comparison Chart**
- Side-by-side bar chart showing performance difference
- Includes improvement factor annotation (e.g., "20x Faster")
- Color-coded: Redis (red), Predis (green)

#### 3. **System Resource Monitoring** (Bottom Section)
- **CPU Utilization**: Host system CPU usage
- **GPU Utilization**: RTX 5080 compute utilization
- **Memory Usage**: System and GPU memory consumption
- **PCIe Bandwidth**: Data transfer utilization

#### 4. **Interactive Demo Controls**
```
┌─────────────────────────────────────────────────────────────┐
│ [Read-Heavy] [Write-Heavy] [Batch Ops] [High Concurrency]  │
│ [Investor View] [Technical View] [Export Results]          │
└─────────────────────────────────────────────────────────────┘
```

---

## Running Demo Scenarios

### Available Demo Scenarios

#### 1. **READ_HEAVY Workload**
```cpp
dashboard.run_demo_scenario("READ_HEAVY", 100000, 10);
```
- **Purpose**: Demonstrates read operation optimization
- **Configuration**: 90% read operations, 10% writes
- **Expected Result**: 15-20x improvement over Redis
- **Use Case**: Read-intensive applications, caching layers

#### 2. **WRITE_HEAVY Workload**
```cpp
dashboard.run_demo_scenario("WRITE_HEAVY", 100000, 10);
```
- **Purpose**: Shows write operation performance
- **Configuration**: 10% read operations, 90% writes
- **Expected Result**: 10-15x improvement over Redis
- **Use Case**: Data ingestion, real-time analytics

#### 3. **BATCH_INTENSIVE Workload**
```cpp
dashboard.run_demo_scenario("BATCH_INTENSIVE", 50000, 5);
```
- **Purpose**: Demonstrates GPU parallelism advantages
- **Configuration**: 80% batch operations, bulk transfers
- **Expected Result**: 20-25x improvement over Redis
- **Use Case**: Bulk data processing, ETL operations

#### 4. **HIGH_CONCURRENCY Workload**
```cpp
dashboard.run_demo_scenario("HIGH_CONCURRENCY", 200000, 50);
```
- **Purpose**: Tests performance under concurrent load
- **Configuration**: Mixed operations, high client count
- **Expected Result**: Consistent 15x+ improvement
- **Use Case**: Multi-tenant applications, high-traffic scenarios

### Custom Demo Scenarios

```cpp
// Create custom demo controller
InteractiveDemoController demo_controller(&dashboard);

// Define custom scenario
InteractiveDemoController::DemoScenario custom_scenario = {
    "ML_Training_Workload",
    "Simulates ML training data access patterns",
    "MIXED",
    500000,  // operations
    20,      // concurrent clients
    {{"zipfian_distribution", "true"}, {"hotspot_ratio", "0.2"}},
    nullptr, // setup callback
    nullptr  // teardown callback
};

// Register and run
demo_controller.register_demo_scenario(custom_scenario);
demo_controller.run_demo_scenario("ML_Training_Workload");
```

### Epic 2 Presentation Sequence

```cpp
// Automated investor demonstration
demo_controller.create_epic2_presentation_sequence();
demo_controller.run_investor_demonstration();
```

**Presentation Flow:**
1. **Baseline Performance** - Shows Redis performance baseline
2. **Batch Operations Demo** - Demonstrates Story 2.1 improvements
3. **GPU Optimization Demo** - Shows Story 2.2 kernel optimizations
4. **Memory Pipeline Demo** - Displays Story 2.3 memory improvements
5. **Full Performance Validation** - Complete Epic 2 validation

---

## Display Modes

### 1. **Technical Mode** (Default)
```cpp
dashboard.switch_display_mode("technical");
```
**Features:**
- Detailed technical metrics and system information
- Resource utilization graphs and performance breakdowns
- Statistical significance indicators and confidence intervals
- Full Epic 2 implementation details

**Best For:** Technical reviews, engineering presentations, debugging

### 2. **Investor Mode**
```cpp
dashboard.switch_display_mode("investor");
```
**Features:**
- Simplified, business-focused metrics
- Emphasis on performance improvements and competitive advantages
- Clean, professional styling with minimal technical details
- ROI-focused performance categories

**Best For:** Investor presentations, executive demonstrations, funding pitches

### 3. **Presentation Mode**
```cpp
dashboard.switch_display_mode("presentation");
```
**Features:**
- Enhanced 3D visualizations and animations
- Dark theme optimized for projectors
- Large fonts and high-contrast colors
- Full-screen friendly layout

**Best For:** Conference presentations, large audience demonstrations

---

## Exporting Results

### 1. **Basic HTML Export**
```cpp
dashboard.export_demo_results("epic2_demo.html");
```
Creates a standalone HTML file with:
- Complete dashboard snapshot
- Interactive Plotly.js charts
- Current performance metrics
- Professional styling

### 2. **Timestamped Export**
```cpp
dashboard.export_demo_results("epic2_results_" + 
    std::to_string(std::time(nullptr)) + ".html");
```

### 3. **Custom Export with Metadata**
```cpp
// Configure export with additional information
auto metrics = dashboard.get_current_metrics();
std::ostringstream filename;
filename << "epic2_" << metrics.current_performance.performance_category 
         << "_" << std::fixed << std::setprecision(1) 
         << metrics.current_performance.improvement_factor_vs_redis 
         << "x_improvement.html";
dashboard.export_demo_results(filename.str());
```

### Export File Structure
```html
epic2_demo.html
├── CSS Styling (embedded)
├── Plotly.js Library (CDN)
├── Performance Metrics Summary
├── Interactive Charts
│   ├── Real-time Timeline
│   ├── Redis vs Predis Comparison
│   └── System Resource Monitoring
└── Demo Controls (JavaScript)
```

---

## API Usage

### 1. **Manual Metrics Update**
```cpp
// Update performance metrics manually
DashboardMetrics::PerformanceMetrics perf_metrics;
perf_metrics.current_throughput_ops_per_sec = 1600000.0;
perf_metrics.average_latency_us = 0.75;
perf_metrics.improvement_factor_vs_redis = 21.3;
perf_metrics.meets_epic2_targets = true;
perf_metrics.performance_category = "EXCELLENT";

dashboard.update_performance_metrics(perf_metrics);
```

### 2. **System Metrics Integration**
```cpp
// Update system metrics from external monitoring
DashboardMetrics::SystemMetrics sys_metrics;
sys_metrics.cpu_utilization_percent = 45.0;
sys_metrics.gpu_utilization_percent = 78.0;
sys_metrics.gpu_memory_usage_percent = 65.0;
sys_metrics.system_memory_usage_percent = 55.0;
sys_metrics.pcie_bandwidth_utilization_percent = 82.0;
sys_metrics.active_connections = 10;

dashboard.update_system_metrics(sys_metrics);
```

### 3. **Comparison Metrics**
```cpp
// Update Redis vs Predis comparison data
DashboardMetrics::ComparisonMetrics comparison;
comparison.improvement_ratio = 20.5;
comparison.statistical_significance = 0.001; // p-value
comparison.throughput_timeline = {1.5, 1.6, 1.7, 1.8, 1.6}; // M ops/sec
comparison.latency_timeline = {0.8, 0.7, 0.9, 0.8, 0.75};    // ms

dashboard.update_comparison_metrics(comparison);
```

### 4. **Retrieving Current State**
```cpp
// Get current dashboard state
auto current_metrics = dashboard.get_current_metrics();

std::cout << "Current Performance: " 
          << current_metrics.current_performance.improvement_factor_vs_redis 
          << "x improvement\n";
std::cout << "Epic 2 Targets Met: " 
          << (current_metrics.current_performance.meets_epic2_targets ? "YES" : "NO") 
          << "\n";
```

---

## Troubleshooting

### Common Issues

#### 1. **Dashboard Won't Start**
```
Error: Port 8080 already in use
```
**Solution:**
```cpp
DashboardConfig config;
config.web_server_port = "8081"; // Use different port
RealTimeDashboard dashboard(config);
```

#### 2. **No Performance Data Showing**
```
Dashboard shows "No data available"
```
**Solutions:**
- Ensure `start_dashboard()` is called before accessing
- Check that demo scenarios are running
- Verify GPU is accessible and properly initialized

#### 3. **Charts Not Loading**
```
Browser shows empty chart containers
```
**Solutions:**
- Ensure internet access for Plotly.js CDN
- Check browser JavaScript console for errors
- Try refreshing the page (auto-refresh every 1 second)

#### 4. **Performance Numbers Seem Wrong**
```
Improvement factor shows < 10x
```
**Diagnostic Steps:**
```cpp
// Check Epic 2 target validation
auto metrics = dashboard.get_current_metrics();
if (!metrics.current_performance.meets_epic2_targets) {
    std::cout << "Epic 2 targets not met. Current: " 
              << metrics.current_performance.improvement_factor_vs_redis 
              << "x\n";
    // Check GPU utilization, system load, etc.
}
```

### Performance Optimization

#### 1. **Reduce Dashboard Overhead**
```cpp
DashboardConfig config;
config.display.refresh_interval_ms = 500; // Slower refresh
config.display.data_history_points = 50;  // Less history
RealTimeDashboard dashboard(config);
```

#### 2. **Optimize for Large Demos**
```cpp
// For long-running demonstrations
config.demo.auto_run_demos = false;     // Manual control
config.display.enable_3d_visualization = false; // Reduce GPU load
```

### Debugging Mode

```cpp
// Enable verbose logging
DashboardConfig config;
config.display.show_technical_details = true;
config.enable_technical_mode = true;

RealTimeDashboard dashboard(config);
dashboard.switch_display_mode("technical");
```

---

## Integration Examples

### 1. **Integration with Benchmarking Suite**
```cpp
#include "src/benchmarks/performance_benchmark_suite.h"

// Run benchmarks and update dashboard
predis::benchmarks::PerformanceBenchmarkSuite benchmark_suite;
auto results = benchmark_suite.run_comparison_benchmark(config);

// Update dashboard with benchmark results
DashboardMetrics::ComparisonMetrics comparison;
comparison.improvement_ratio = results.performance_improvement_factor;
comparison.statistical_significance = results.statistical_significance_p_value;
dashboard.update_comparison_metrics(comparison);
```

### 2. **Automated Demo Loop**
```cpp
// Continuous demonstration loop
std::vector<std::string> demo_sequence = {
    "READ_HEAVY", "WRITE_HEAVY", "BATCH_INTENSIVE", "HIGH_CONCURRENCY"
};

for (const auto& scenario : demo_sequence) {
    dashboard.run_demo_scenario(scenario, 100000, 10);
    std::this_thread::sleep_for(std::chrono::seconds(30));
    
    // Export results for each scenario
    dashboard.export_demo_results("epic2_" + scenario + "_demo.html");
}
```

### 3. **Custom Monitoring Integration**
```cpp
// Monitor actual GPU performance
class GPUMonitor {
public:
    void update_dashboard(RealTimeDashboard& dashboard) {
        auto gpu_metrics = collect_gpu_metrics();
        
        DashboardMetrics::SystemMetrics sys_metrics;
        sys_metrics.gpu_utilization_percent = gpu_metrics.utilization;
        sys_metrics.gpu_memory_usage_percent = gpu_metrics.memory_usage;
        
        dashboard.update_system_metrics(sys_metrics);
    }
    
private:
    GPUMetrics collect_gpu_metrics() {
        // Actual GPU monitoring implementation
        return GPUMetrics{};
    }
};
```

---

## Best Practices

### 1. **For Investor Presentations**
- Use `investor` display mode
- Run Epic 2 presentation sequence
- Export results with meaningful filenames
- Focus on improvement factors and business metrics

### 2. **For Technical Reviews**
- Use `technical` display mode
- Show statistical significance and confidence intervals
- Include system resource monitoring
- Provide detailed performance breakdowns

### 3. **For Development/Testing**
- Use shorter refresh intervals (100-250ms)
- Enable all technical details
- Monitor system resources closely
- Use custom demo scenarios for specific testing

### 4. **For Production Monitoring**
- Longer refresh intervals (1-5 seconds)
- Focus on key performance indicators
- Set up automated alerting for Epic 2 target misses
- Regular export of performance reports

---

*This user guide covers the Epic 2 Performance Dashboard functionality. For additional technical details, see the implementation files in `src/dashboard/` and test examples in `tests/dashboard/`.*