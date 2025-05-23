#include "write_performance_profiler.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

namespace predis {
namespace benchmarks {

// Global device variables for profiling
__device__ unsigned int g_atomic_retry_count = 0;
__device__ unsigned int g_hash_collision_count = 0;

__global__ void resetProfilingCounters() {
    g_atomic_retry_count = 0;
    g_hash_collision_count = 0;
}

__global__ void collectProfilingStats(unsigned int* retry_count, unsigned int* collision_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *retry_count = g_atomic_retry_count;
        *collision_count = g_hash_collision_count;
    }
}

WritePerformanceProfiler::WritePerformanceProfiler() {
    // Pre-allocate CUDA events
    for (int i = 0; i < 100; ++i) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        start_events_.push_back(start);
        end_events_.push_back(end);
    }
    
    cudaEventCreate(&kernel_start_);
    cudaEventCreate(&kernel_end_);
    cudaEventCreate(&memcpy_start_);
    cudaEventCreate(&memcpy_end_);
}

WritePerformanceProfiler::~WritePerformanceProfiler() {
    for (auto& event : start_events_) {
        cudaEventDestroy(event);
    }
    for (auto& event : end_events_) {
        cudaEventDestroy(event);
    }
    
    cudaEventDestroy(kernel_start_);
    cudaEventDestroy(kernel_end_);
    cudaEventDestroy(memcpy_start_);
    cudaEventDestroy(memcpy_end_);
}

bool WritePerformanceProfiler::initialize(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    err = cudaGetDeviceProperties(&device_props_, device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    reset();
    return true;
}

void WritePerformanceProfiler::startProfiling() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    profiling_enabled_ = true;
    reset();
    
    // Reset GPU counters
    resetProfilingCounters<<<1, 1>>>();
    cudaDeviceSynchronize();
}

void WritePerformanceProfiler::stopProfiling() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    profiling_enabled_ = false;
    
    // Collect final GPU statistics
    unsigned int *d_retry_count, *d_collision_count;
    cudaMalloc(&d_retry_count, sizeof(unsigned int));
    cudaMalloc(&d_collision_count, sizeof(unsigned int));
    
    collectProfilingStats<<<1, 1>>>(d_retry_count, d_collision_count);
    cudaDeviceSynchronize();
    
    unsigned int retry_count, collision_count;
    cudaMemcpy(&retry_count, d_retry_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&collision_count, d_collision_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    current_metrics_.atomic_retries = retry_count;
    current_metrics_.hash_collisions = collision_count;
    
    cudaFree(d_retry_count);
    cudaFree(d_collision_count);
    
    updateLatencyStatistics();
}

void WritePerformanceProfiler::profileBatchWrite(
    const void* keys,
    const void* values,
    const size_t* key_sizes,
    const size_t* value_sizes,
    size_t num_items,
    size_t total_bytes) {
    
    if (!profiling_enabled_) return;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Record start of operation
    cudaEventRecord(start_events_[sample_count_ % start_events_.size()]);
    
    // Profile is called from outside - we just record the timing
    
    // Record end of operation
    cudaEventRecord(end_events_[sample_count_ % end_events_.size()]);
    cudaEventSynchronize(end_events_[sample_count_ % end_events_.size()]);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, 
                        start_events_[sample_count_ % start_events_.size()],
                        end_events_[sample_count_ % end_events_.size()]);
    
    double cpu_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update metrics
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.total_writes += num_items;
    current_metrics_.total_time_ms += cpu_time_ms;
    current_metrics_.kernel_time_ms += gpu_time_ms;
    
    latency_samples_.push_back(gpu_time_ms / num_items);
    
    // Calculate bandwidth
    double bandwidth_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    double time_sec = gpu_time_ms / 1000.0;
    current_metrics_.bandwidth_mb_per_sec = (bandwidth_gb * 1024.0) / time_sec;
    
    sample_count_++;
}

void WritePerformanceProfiler::startPhase(const std::string& phase_name) {
    if (!profiling_enabled_) return;
    
    if (phase_name == "kernel") {
        cudaEventRecord(kernel_start_);
    } else if (phase_name == "memcpy") {
        cudaEventRecord(memcpy_start_);
    }
}

void WritePerformanceProfiler::endPhase(const std::string& phase_name) {
    if (!profiling_enabled_) return;
    
    if (phase_name == "kernel") {
        cudaEventRecord(kernel_end_);
        cudaEventSynchronize(kernel_end_);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, kernel_start_, kernel_end_);
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        current_metrics_.kernel_time_ms += elapsed_ms;
    } else if (phase_name == "memcpy") {
        cudaEventRecord(memcpy_end_);
        cudaEventSynchronize(memcpy_end_);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, memcpy_start_, memcpy_end_);
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        current_metrics_.memory_copy_time_ms += elapsed_ms;
    }
}

void WritePerformanceProfiler::recordAtomicRetries(size_t retries) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.atomic_retries += retries;
}

void WritePerformanceProfiler::recordHashCollisions(size_t collisions) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.hash_collisions += collisions;
}

BottleneckAnalysis WritePerformanceProfiler::analyzeBottlenecks() const {
    BottleneckAnalysis analysis;
    
    // Calculate bottleneck scores
    double total_time = current_metrics_.total_time_ms;
    if (total_time <= 0) {
        return analysis;
    }
    
    // Atomic contention score
    double atomic_score = 0.0;
    if (current_metrics_.total_writes > 0) {
        double retries_per_write = static_cast<double>(current_metrics_.atomic_retries) / 
                                  current_metrics_.total_writes;
        atomic_score = std::min(1.0, retries_per_write / 10.0); // Normalize to 0-1
    }
    analysis.bottleneck_scores[BottleneckAnalysis::ATOMIC_CONTENTION] = atomic_score;
    
    // Memory bandwidth score
    double expected_bandwidth = device_props_.memoryBusWidth / 8.0 * 
                               device_props_.memoryClockRate * 2.0 / 1e6; // GB/s
    double actual_bandwidth = current_metrics_.bandwidth_mb_per_sec / 1024.0; // GB/s
    double bandwidth_score = 1.0 - (actual_bandwidth / expected_bandwidth);
    analysis.bottleneck_scores[BottleneckAnalysis::MEMORY_BANDWIDTH] = bandwidth_score;
    
    // Hash collision score
    double collision_score = 0.0;
    if (current_metrics_.total_writes > 0) {
        double collisions_per_write = static_cast<double>(current_metrics_.hash_collisions) / 
                                     current_metrics_.total_writes;
        collision_score = std::min(1.0, collisions_per_write / 5.0);
    }
    analysis.bottleneck_scores[BottleneckAnalysis::HASH_COLLISION] = collision_score;
    
    // Kernel launch overhead score
    double kernel_overhead = (current_metrics_.total_time_ms - current_metrics_.kernel_time_ms) / 
                            total_time;
    analysis.bottleneck_scores[BottleneckAnalysis::KERNEL_LAUNCH_OVERHEAD] = kernel_overhead;
    
    // PCIe transfer score
    double pcie_score = current_metrics_.memory_copy_time_ms / total_time;
    analysis.bottleneck_scores[BottleneckAnalysis::PCIe_TRANSFER] = pcie_score;
    
    // Identify primary bottleneck
    analysis.primary_bottleneck = identifyPrimaryBottleneck();
    
    // Generate recommendations
    analysis.recommendations = generateRecommendations(analysis);
    
    return analysis;
}

WriteMetrics WritePerformanceProfiler::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    WriteMetrics metrics = current_metrics_;
    
    // Calculate throughput
    if (metrics.total_time_ms > 0) {
        metrics.writes_per_second = (metrics.total_writes * 1000.0) / metrics.total_time_ms;
    }
    
    // Calculate latency statistics
    if (!latency_samples_.empty()) {
        metrics.min_latency_ms = *std::min_element(latency_samples_.begin(), latency_samples_.end());
        metrics.max_latency_ms = *std::max_element(latency_samples_.begin(), latency_samples_.end());
        metrics.avg_latency_ms = std::accumulate(latency_samples_.begin(), latency_samples_.end(), 0.0) / 
                                 latency_samples_.size();
        
        metrics.p50_latency_ms = calculatePercentile(0.50);
        metrics.p95_latency_ms = calculatePercentile(0.95);
        metrics.p99_latency_ms = calculatePercentile(0.99);
    }
    
    // Estimate GPU utilization
    metrics.gpu_utilization = estimateGPUUtilization();
    
    return metrics;
}

std::string WritePerformanceProfiler::generateReport() const {
    auto metrics = getMetrics();
    auto analysis = analyzeBottlenecks();
    
    std::stringstream report;
    report << std::fixed << std::setprecision(2);
    
    report << "=== Write Performance Profile Report ===\n\n";
    
    report << "Performance Metrics:\n";
    report << "  Total Writes: " << metrics.total_writes << "\n";
    report << "  Throughput: " << metrics.writes_per_second << " ops/sec\n";
    report << "  Bandwidth: " << metrics.bandwidth_mb_per_sec << " MB/s\n";
    report << "  GPU Utilization: " << (metrics.gpu_utilization * 100) << "%\n\n";
    
    report << "Latency Distribution:\n";
    report << "  Min: " << metrics.min_latency_ms << " ms\n";
    report << "  Avg: " << metrics.avg_latency_ms << " ms\n";
    report << "  P50: " << metrics.p50_latency_ms << " ms\n";
    report << "  P95: " << metrics.p95_latency_ms << " ms\n";
    report << "  P99: " << metrics.p99_latency_ms << " ms\n";
    report << "  Max: " << metrics.max_latency_ms << " ms\n\n";
    
    report << "Time Breakdown:\n";
    report << "  Total Time: " << metrics.total_time_ms << " ms\n";
    report << "  Kernel Time: " << metrics.kernel_time_ms << " ms ("
           << (metrics.kernel_time_ms / metrics.total_time_ms * 100) << "%)\n";
    report << "  Memory Copy: " << metrics.memory_copy_time_ms << " ms ("
           << (metrics.memory_copy_time_ms / metrics.total_time_ms * 100) << "%)\n\n";
    
    report << "Contention Analysis:\n";
    report << "  Atomic Retries: " << metrics.atomic_retries << "\n";
    report << "  Hash Collisions: " << metrics.hash_collisions << "\n\n";
    
    report << "Bottleneck Analysis:\n";
    const char* bottleneck_names[] = {
        "None", "Atomic Contention", "Memory Bandwidth", 
        "Hash Collision", "Kernel Launch Overhead", "PCIe Transfer"
    };
    report << "  Primary Bottleneck: " << bottleneck_names[analysis.primary_bottleneck] << "\n";
    
    report << "  Bottleneck Scores:\n";
    for (const auto& [type, score] : analysis.bottleneck_scores) {
        report << "    " << bottleneck_names[type] << ": " << (score * 100) << "%\n";
    }
    
    report << "\nRecommendations:\n";
    for (const auto& rec : analysis.recommendations) {
        report << "  - " << rec << "\n";
    }
    
    return report.str();
}

void WritePerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_ = WriteMetrics();
    latency_samples_.clear();
    sample_count_ = 0;
}

void WritePerformanceProfiler::updateLatencyStatistics() {
    // Statistics are updated in getMetrics()
}

double WritePerformanceProfiler::calculatePercentile(double percentile) const {
    if (latency_samples_.empty()) return 0.0;
    
    std::vector<double> sorted_samples = latency_samples_;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    
    size_t index = static_cast<size_t>(percentile * (sorted_samples.size() - 1));
    return sorted_samples[index];
}

double WritePerformanceProfiler::estimateGPUUtilization() const {
    if (current_metrics_.total_time_ms <= 0) return 0.0;
    
    // Simple utilization estimate based on kernel time vs total time
    double kernel_ratio = current_metrics_.kernel_time_ms / current_metrics_.total_time_ms;
    
    // Factor in memory bandwidth utilization
    double expected_bandwidth = device_props_.memoryBusWidth / 8.0 * 
                               device_props_.memoryClockRate * 2.0 / 1e6; // GB/s
    double actual_bandwidth = current_metrics_.bandwidth_mb_per_sec / 1024.0; // GB/s
    double bandwidth_ratio = actual_bandwidth / expected_bandwidth;
    
    // Combined utilization estimate
    return std::min(1.0, (kernel_ratio * 0.7 + bandwidth_ratio * 0.3));
}

BottleneckAnalysis::BottleneckType WritePerformanceProfiler::identifyPrimaryBottleneck() const {
    auto analysis = const_cast<WritePerformanceProfiler*>(this)->analyzeBottlenecks();
    
    BottleneckAnalysis::BottleneckType primary = BottleneckAnalysis::NONE;
    double max_score = 0.0;
    
    for (const auto& [type, score] : analysis.bottleneck_scores) {
        if (score > max_score) {
            max_score = score;
            primary = type;
        }
    }
    
    return primary;
}

std::vector<std::string> WritePerformanceProfiler::generateRecommendations(
    const BottleneckAnalysis& analysis) const {
    
    std::vector<std::string> recommendations;
    
    switch (analysis.primary_bottleneck) {
        case BottleneckAnalysis::ATOMIC_CONTENTION:
            recommendations.push_back("Use warp-cooperative writes to reduce atomic conflicts");
            recommendations.push_back("Implement lock-free algorithms where possible");
            recommendations.push_back("Consider write-combining for small values");
            break;
            
        case BottleneckAnalysis::MEMORY_BANDWIDTH:
            recommendations.push_back("Optimize memory access patterns for coalescing");
            recommendations.push_back("Use shared memory for frequently accessed data");
            recommendations.push_back("Consider compression for large values");
            break;
            
        case BottleneckAnalysis::HASH_COLLISION:
            recommendations.push_back("Improve hash function distribution");
            recommendations.push_back("Increase hash table size to reduce load factor");
            recommendations.push_back("Implement cuckoo hashing or other collision-resistant schemes");
            break;
            
        case BottleneckAnalysis::KERNEL_LAUNCH_OVERHEAD:
            recommendations.push_back("Batch more operations per kernel launch");
            recommendations.push_back("Use persistent kernels for small operations");
            recommendations.push_back("Optimize grid/block dimensions");
            break;
            
        case BottleneckAnalysis::PCIe_TRANSFER:
            recommendations.push_back("Use pinned memory for faster transfers");
            recommendations.push_back("Overlap transfers with computation");
            recommendations.push_back("Batch transfers to amortize overhead");
            break;
            
        default:
            recommendations.push_back("Performance is well-balanced");
            break;
    }
    
    return recommendations;
}

} // namespace benchmarks
} // namespace predis