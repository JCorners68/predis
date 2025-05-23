#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <iostream>
#include <iomanip>

namespace predis {

// Performance profiling data structure
struct WriteProfileData {
    // Timing breakdowns
    double total_time_ms = 0.0;
    double memory_allocation_ms = 0.0;
    double hash_computation_ms = 0.0;
    double atomic_operations_ms = 0.0;
    double memory_copy_ms = 0.0;
    double kernel_launch_ms = 0.0;
    
    // Conflict metrics
    uint64_t hash_conflicts = 0;
    uint64_t atomic_retries = 0;
    uint64_t memory_stalls = 0;
    
    // Throughput metrics
    uint64_t operations_completed = 0;
    double bytes_written = 0.0;
    double memory_bandwidth_gbps = 0.0;
    
    // GPU metrics
    float gpu_utilization = 0.0f;
    float memory_utilization = 0.0f;
    float sm_efficiency = 0.0f;
};

class WritePerformanceProfiler {
private:
    std::unordered_map<std::string, WriteProfileData> profile_data_;
    std::mutex data_mutex_;
    
    // CUDA events for precise timing
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    std::vector<cudaEvent_t> stage_events_;
    
    // Profiling state
    bool profiling_enabled_ = true;
    std::atomic<uint64_t> total_operations_{0};
    
public:
    WritePerformanceProfiler() {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&end_event_);
        
        // Create events for each stage
        for (int i = 0; i < 10; i++) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            stage_events_.push_back(event);
        }
    }
    
    ~WritePerformanceProfiler() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(end_event_);
        for (auto& event : stage_events_) {
            cudaEventDestroy(event);
        }
    }
    
    // Profile a write operation with detailed timing
    void profileWriteOperation(const std::string& operation_name,
                             size_t key_size,
                             size_t value_size,
                             std::function<void()> operation) {
        if (!profiling_enabled_) {
            operation();
            return;
        }
        
        WriteProfileData local_data;
        
        // Record start time
        cudaEventRecord(start_event_);
        
        // Execute operation with stage timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        operation();
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        // Record end time
        cudaEventRecord(end_event_);
        cudaEventSynchronize(end_event_);
        
        // Calculate elapsed time
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start_event_, end_event_);
        
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_end - cpu_start).count() / 1000.0;
        
        // Update profile data
        local_data.total_time_ms = gpu_time_ms;
        local_data.operations_completed = 1;
        local_data.bytes_written = key_size + value_size;
        
        // Calculate bandwidth
        local_data.memory_bandwidth_gbps = 
            (local_data.bytes_written / 1e9) / (gpu_time_ms / 1000.0);
        
        // Update global data
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            auto& global_data = profile_data_[operation_name];
            global_data.total_time_ms += local_data.total_time_ms;
            global_data.operations_completed += local_data.operations_completed;
            global_data.bytes_written += local_data.bytes_written;
            global_data.memory_bandwidth_gbps = 
                (global_data.bytes_written / 1e9) / (global_data.total_time_ms / 1000.0);
        }
        
        total_operations_.fetch_add(1);
    }
    
    // Profile memory allocation overhead
    void profileMemoryAllocation(size_t size, std::function<void()> alloc_func) {
        cudaEventRecord(stage_events_[0]);
        
        alloc_func();
        
        cudaEventRecord(stage_events_[1]);
        cudaEventSynchronize(stage_events_[1]);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, stage_events_[0], stage_events_[1]);
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_["memory_allocation"].memory_allocation_ms += elapsed_ms;
        profile_data_["memory_allocation"].operations_completed++;
    }
    
    // Profile hash computation
    void profileHashComputation(std::function<void()> hash_func) {
        cudaEventRecord(stage_events_[2]);
        
        hash_func();
        
        cudaEventRecord(stage_events_[3]);
        cudaEventSynchronize(stage_events_[3]);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, stage_events_[2], stage_events_[3]);
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_["hash_computation"].hash_computation_ms += elapsed_ms;
        profile_data_["hash_computation"].operations_completed++;
    }
    
    // Profile atomic operations
    void profileAtomicOperations(std::function<uint32_t()> atomic_func) {
        cudaEventRecord(stage_events_[4]);
        
        uint32_t retries = atomic_func();
        
        cudaEventRecord(stage_events_[5]);
        cudaEventSynchronize(stage_events_[5]);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, stage_events_[4], stage_events_[5]);
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_["atomic_operations"].atomic_operations_ms += elapsed_ms;
        profile_data_["atomic_operations"].atomic_retries += retries;
        profile_data_["atomic_operations"].operations_completed++;
    }
    
    // Record hash conflicts
    void recordHashConflict(const std::string& operation_name) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_[operation_name].hash_conflicts++;
    }
    
    // Record memory stalls
    void recordMemoryStall(const std::string& operation_name) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_[operation_name].memory_stalls++;
    }
    
    // Generate performance report
    void generateReport() {
        std::cout << "\n=== Write Performance Analysis Report ===" << std::endl;
        std::cout << "Total operations profiled: " << total_operations_.load() << std::endl;
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Aggregate data
        WriteProfileData aggregate;
        for (const auto& [name, data] : profile_data_) {
            aggregate.total_time_ms += data.total_time_ms;
            aggregate.memory_allocation_ms += data.memory_allocation_ms;
            aggregate.hash_computation_ms += data.hash_computation_ms;
            aggregate.atomic_operations_ms += data.atomic_operations_ms;
            aggregate.memory_copy_ms += data.memory_copy_ms;
            aggregate.kernel_launch_ms += data.kernel_launch_ms;
            aggregate.hash_conflicts += data.hash_conflicts;
            aggregate.atomic_retries += data.atomic_retries;
            aggregate.memory_stalls += data.memory_stalls;
            aggregate.operations_completed += data.operations_completed;
            aggregate.bytes_written += data.bytes_written;
        }
        
        // Calculate percentages
        auto calc_percentage = [&](double value) -> double {
            return aggregate.total_time_ms > 0 ? (value / aggregate.total_time_ms * 100.0) : 0.0;
        };
        
        std::cout << "\n--- Timing Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total time: " << aggregate.total_time_ms << " ms" << std::endl;
        std::cout << "Memory allocation: " << aggregate.memory_allocation_ms 
                  << " ms (" << calc_percentage(aggregate.memory_allocation_ms) << "%)" << std::endl;
        std::cout << "Hash computation: " << aggregate.hash_computation_ms 
                  << " ms (" << calc_percentage(aggregate.hash_computation_ms) << "%)" << std::endl;
        std::cout << "Atomic operations: " << aggregate.atomic_operations_ms 
                  << " ms (" << calc_percentage(aggregate.atomic_operations_ms) << "%)" << std::endl;
        
        std::cout << "\n--- Conflict Analysis ---" << std::endl;
        std::cout << "Hash conflicts: " << aggregate.hash_conflicts << std::endl;
        std::cout << "Atomic retries: " << aggregate.atomic_retries << std::endl;
        std::cout << "Memory stalls: " << aggregate.memory_stalls << std::endl;
        
        if (aggregate.operations_completed > 0) {
            std::cout << "Avg conflicts per op: " 
                      << (double)aggregate.hash_conflicts / aggregate.operations_completed << std::endl;
            std::cout << "Avg atomic retries per op: " 
                      << (double)aggregate.atomic_retries / aggregate.operations_completed << std::endl;
        }
        
        std::cout << "\n--- Throughput Metrics ---" << std::endl;
        double throughput_ops = aggregate.operations_completed / (aggregate.total_time_ms / 1000.0);
        std::cout << "Operations/sec: " << std::fixed << std::setprecision(0) << throughput_ops << std::endl;
        std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(2) 
                  << aggregate.memory_bandwidth_gbps << " GB/s" << std::endl;
        
        // Detailed operation breakdown
        std::cout << "\n--- Operation Details ---" << std::endl;
        for (const auto& [name, data] : profile_data_) {
            if (data.operations_completed > 0) {
                std::cout << name << ":" << std::endl;
                std::cout << "  Operations: " << data.operations_completed << std::endl;
                std::cout << "  Avg time: " << data.total_time_ms / data.operations_completed << " ms" << std::endl;
                std::cout << "  Bandwidth: " << data.memory_bandwidth_gbps << " GB/s" << std::endl;
            }
        }
    }
    
    // Reset profiling data
    void reset() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        profile_data_.clear();
        total_operations_ = 0;
    }
    
    // Enable/disable profiling
    void setEnabled(bool enabled) {
        profiling_enabled_ = enabled;
    }
};

} // namespace predis