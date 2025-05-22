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

#include "../../src/core/memory_pipeline_optimizer.h"
#include "../../src/core/data_structures/gpu_hash_table.h"
#include "../../src/core/memory_manager.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <future>

using namespace predis::core;

class MemoryPipelinePerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize memory manager
        memory_manager = std::make_unique<MemoryManager>();
        ASSERT_TRUE(memory_manager->initialize());
        
        // Initialize GPU hash table with larger capacity for pipeline testing
        hash_table = std::make_unique<GpuHashTable>();
        ASSERT_TRUE(hash_table->initialize(memory_manager.get(), 2000000)); // 2M entries
        
        // Initialize memory pipeline optimizer
        pipeline_optimizer = std::make_unique<MemoryPipelineOptimizer>();
        
        // Configure for high-performance pipeline testing
        MemoryPipelineOptimizer::PipelineConfig config;
        config.num_pipeline_stages = 6;
        config.buffer_size_mb = 128;
        config.max_concurrent_transfers = 16;
        config.enable_numa_optimization = true;
        config.use_pinned_memory = true;
        config.enable_memory_pooling = true;
        config.enable_compute_overlap = true;
        config.use_async_transfers = true;
        config.target_bandwidth_utilization = 0.90;
        config.prefetch_queue_depth = 32;
        config.batch_coalescing_threshold = 512;
        
        ASSERT_TRUE(pipeline_optimizer->initialize(hash_table.get(), memory_manager.get(), config));
    }
    
    void TearDown() override {
        if (pipeline_optimizer) {
            pipeline_optimizer->shutdown();
        }
        hash_table.reset();
        memory_manager.reset();
    }
    
    struct PipelinePerformanceResult {
        std::string test_name;
        size_t total_operations;
        double sustained_ops_per_second;
        double peak_ops_per_second;
        double average_latency_ms;
        double p99_latency_ms;
        double memory_bandwidth_gbps;
        double pipeline_efficiency_percent;
        double gpu_utilization_percent;
        size_t pipeline_stalls;
        bool target_achieved;
    };
    
    std::string generate_random_key(size_t length = 16) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis('a', 'z');
        
        std::string key;
        key.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            key += static_cast<char>(dis(gen));
        }
        return key;
    }
    
    std::string generate_random_value(size_t length = 64) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis('A', 'Z');
        
        std::string value;
        value.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            value += static_cast<char>(dis(gen));
        }
        return value;
    }
    
    PipelinePerformanceResult measure_async_operations_performance(
        const std::string& test_name,
        size_t num_operations,
        bool use_batch_operations = false) {
        
        PipelinePerformanceResult result;
        result.test_name = test_name;
        result.total_operations = num_operations;
        
        // Generate test data
        std::vector<std::string> keys, values;
        keys.reserve(num_operations);
        values.reserve(num_operations);
        
        for (size_t i = 0; i < num_operations; ++i) {
            keys.push_back("pipeline_test_" + std::to_string(i) + "_" + generate_random_key(12));
            values.push_back(generate_random_value(64));
        }
        
        // Reset metrics
        pipeline_optimizer->reset_metrics();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto peak_start = start_time;
        double peak_ops = 0.0;
        const size_t peak_measurement_window = 1000; // Measure peak over 1000 operations
        
        if (use_batch_operations) {
            // Test batch async operations
            std::vector<std::unique_ptr<MemoryPipelineOptimizer::AsyncOperation>> async_ops;
            async_ops.reserve(num_operations / 100); // Batch size of 100
            
            for (size_t i = 0; i < num_operations; i += 100) {
                size_t batch_end = std::min(i + 100, num_operations);
                std::vector<std::string> batch_keys(keys.begin() + i, keys.begin() + batch_end);
                std::vector<std::string> batch_values(values.begin() + i, values.begin() + batch_end);
                
                // Insert batch
                auto async_op = pipeline_optimizer->async_batch_insert(batch_keys, batch_values);
                if (async_op) {
                    async_ops.push_back(std::move(async_op));
                }
                
                // Measure peak performance
                if (i > 0 && i % peak_measurement_window == 0) {
                    auto peak_end = std::chrono::high_resolution_clock::now();
                    double peak_elapsed = std::chrono::duration<double>(peak_end - peak_start).count();
                    double current_peak = peak_measurement_window / peak_elapsed;
                    peak_ops = std::max(peak_ops, current_peak);
                    peak_start = peak_end;
                }
            }
            
            // Wait for all operations to complete
            for (auto& async_op : async_ops) {
                if (async_op) {
                    async_op->wait();
                }
            }
            
            // Now test batch lookups
            async_ops.clear();
            for (size_t i = 0; i < num_operations; i += 100) {
                size_t batch_end = std::min(i + 100, num_operations);
                std::vector<std::string> batch_keys(keys.begin() + i, keys.begin() + batch_end);
                
                auto async_op = pipeline_optimizer->async_batch_lookup(batch_keys);
                if (async_op) {
                    async_ops.push_back(std::move(async_op));
                }
            }
            
            // Wait for lookup operations
            for (auto& async_op : async_ops) {
                if (async_op) {
                    async_op->wait();
                }
            }
            
        } else {
            // Test individual async operations
            std::vector<std::unique_ptr<MemoryPipelineOptimizer::AsyncOperation>> async_ops;
            async_ops.reserve(num_operations * 2); // Insert + lookup for each key
            
            // Start async insert operations
            for (size_t i = 0; i < num_operations; ++i) {
                auto async_op = pipeline_optimizer->async_insert(keys[i].c_str(), keys[i].length(),
                                                               values[i].c_str(), values[i].length());
                if (async_op) {
                    async_ops.push_back(std::move(async_op));
                }
                
                // Measure peak performance
                if (i > 0 && i % peak_measurement_window == 0) {
                    auto peak_end = std::chrono::high_resolution_clock::now();
                    double peak_elapsed = std::chrono::duration<double>(peak_end - peak_start).count();
                    double current_peak = peak_measurement_window / peak_elapsed;
                    peak_ops = std::max(peak_ops, current_peak);
                    peak_start = peak_end;
                }
            }
            
            // Wait for insert operations to complete
            for (auto& async_op : async_ops) {
                if (async_op) {
                    async_op->wait();
                }
            }
            
            // Start async lookup operations
            async_ops.clear();
            for (size_t i = 0; i < num_operations; ++i) {
                auto async_op = pipeline_optimizer->async_lookup(keys[i].c_str(), keys[i].length());
                if (async_op) {
                    async_ops.push_back(std::move(async_op));
                }
            }
            
            // Wait for lookup operations
            for (auto& async_op : async_ops) {
                if (async_op) {
                    async_op->wait();
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        // Calculate sustained performance (total operations over total time)
        size_t total_ops_performed = use_batch_operations ? num_operations * 2 : num_operations * 2;
        result.sustained_ops_per_second = total_ops_performed / total_elapsed;
        result.peak_ops_per_second = peak_ops;
        
        // Get pipeline metrics
        auto metrics = pipeline_optimizer->get_pipeline_metrics();
        result.average_latency_ms = metrics.average_latency_ms;
        result.p99_latency_ms = metrics.p99_latency_ms;
        result.memory_bandwidth_gbps = metrics.memory_bandwidth_gbps;
        result.pipeline_efficiency_percent = metrics.pipeline_efficiency_percent;
        result.gpu_utilization_percent = metrics.gpu_utilization_percent;
        result.pipeline_stalls = metrics.pipeline_stalls;
        
        // Determine if target was achieved (2M+ ops/sec sustained)
        result.target_achieved = (result.sustained_ops_per_second >= 2000000.0);
        
        return result;
    }
    
    void print_performance_results(const std::vector<PipelinePerformanceResult>& results) {
        std::cout << "\n=== Memory Pipeline Performance Results ===" << std::endl;
        std::cout << std::setw(25) << "Test Name" 
                  << std::setw(12) << "Total Ops" 
                  << std::setw(15) << "Sustained/sec"
                  << std::setw(12) << "Peak/sec"
                  << std::setw(12) << "Avg Lat(ms)"
                  << std::setw(12) << "P99 Lat(ms)"
                  << std::setw(12) << "Bandwidth"
                  << std::setw(12) << "Efficiency"
                  << std::setw(10) << "Target" << std::endl;
        std::cout << std::string(125, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(25) << result.test_name
                      << std::setw(12) << result.total_operations
                      << std::setw(15) << std::fixed << std::setprecision(0) << result.sustained_ops_per_second
                      << std::setw(12) << std::fixed << std::setprecision(0) << result.peak_ops_per_second
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.average_latency_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.p99_latency_ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.memory_bandwidth_gbps
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.pipeline_efficiency_percent
                      << std::setw(10) << (result.target_achieved ? "✓" : "✗") << std::endl;
        }
        std::cout << std::endl;
    }
    
    void save_results_to_csv(const std::vector<PipelinePerformanceResult>& results, 
                            const std::string& filename) {
        std::ofstream file(filename);
        file << "Test_Name,Total_Operations,Sustained_Ops_Per_Sec,Peak_Ops_Per_Sec,";
        file << "Average_Latency_Ms,P99_Latency_Ms,Memory_Bandwidth_GBps,";
        file << "Pipeline_Efficiency_Percent,GPU_Utilization_Percent,Pipeline_Stalls,Target_Achieved\n";
        
        for (const auto& result : results) {
            file << result.test_name << ","
                 << result.total_operations << ","
                 << result.sustained_ops_per_second << ","
                 << result.peak_ops_per_second << ","
                 << result.average_latency_ms << ","
                 << result.p99_latency_ms << ","
                 << result.memory_bandwidth_gbps << ","
                 << result.pipeline_efficiency_percent << ","
                 << result.gpu_utilization_percent << ","
                 << result.pipeline_stalls << ","
                 << (result.target_achieved ? 1 : 0) << "\n";
        }
    }
    
    std::unique_ptr<MemoryManager> memory_manager;
    std::unique_ptr<GpuHashTable> hash_table;
    std::unique_ptr<MemoryPipelineOptimizer> pipeline_optimizer;
};

TEST_F(MemoryPipelinePerformanceTest, SustainedThroughputTest) {
    std::cout << "\n==== Sustained Throughput Performance Test ====" << std::endl;
    std::cout << "Target: >2M operations/second sustained throughput\n" << std::endl;
    
    std::vector<PipelinePerformanceResult> results;
    
    // Test increasing operation counts to measure sustained performance
    std::vector<size_t> operation_counts = {1000, 5000, 10000, 25000, 50000, 100000};
    
    for (size_t ops : operation_counts) {
        std::cout << "Testing " << ops << " operations..." << std::endl;
        
        auto result = measure_async_operations_performance(
            "Sustained_" + std::to_string(ops), ops, false);
        results.push_back(result);
        
        // Verify sustained performance targets
        if (ops >= 10000) {
            EXPECT_GT(result.sustained_ops_per_second, 1000000.0) 
                << "Should achieve >1M ops/sec for " << ops << " operations";
        }
        
        if (ops >= 50000) {
            EXPECT_GT(result.sustained_ops_per_second, 1500000.0) 
                << "Should achieve >1.5M ops/sec for larger workloads";
        }
    }
    
    print_performance_results(results);
    save_results_to_csv(results, "sustained_throughput_results.csv");
    
    // Find best sustained performance
    auto best_result = *std::max_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.sustained_ops_per_second < b.sustained_ops_per_second;
        });
    
    std::cout << "Best sustained performance: " 
              << std::fixed << std::setprecision(0) << best_result.sustained_ops_per_second 
              << " ops/sec with " << best_result.total_operations << " operations" << std::endl;
    
    // Epic 2 Story 2.3 success criteria
    bool story_2_3_success = (best_result.sustained_ops_per_second >= 2000000.0);
    
    if (story_2_3_success) {
        std::cout << "\n🎯 EPIC 2 STORY 2.3 SUCCESS: Achieved " 
                  << (best_result.sustained_ops_per_second / 1000000.0) 
                  << "M+ ops/sec sustained (target: 2M+)" << std::endl;
    } else {
        std::cout << "\n⚠️  EPIC 2 STORY 2.3 PARTIAL: Achieved " 
                  << (best_result.sustained_ops_per_second / 1000000.0) 
                  << "M ops/sec, need optimization for 2M+ target" << std::endl;
    }
    
    EXPECT_TRUE(story_2_3_success) << "Should achieve 2M+ ops/sec sustained throughput for Epic 2 targets";
}

TEST_F(MemoryPipelinePerformanceTest, BatchPipelineOptimizationTest) {
    std::cout << "\n==== Batch Pipeline Optimization Test ====" << std::endl;
    std::cout << "Testing asynchronous batch operations with pipeline optimization\n" << std::endl;
    
    std::vector<PipelinePerformanceResult> results;
    
    // Test batch operations of different sizes
    std::vector<size_t> batch_operation_counts = {5000, 10000, 25000, 50000};
    
    for (size_t ops : batch_operation_counts) {
        std::cout << "Testing batch operations with " << ops << " total operations..." << std::endl;
        
        auto result = measure_async_operations_performance(
            "BatchAsync_" + std::to_string(ops), ops, true);
        results.push_back(result);
        
        // Batch operations should achieve higher sustained throughput
        EXPECT_GT(result.sustained_ops_per_second, 1500000.0) 
            << "Batch async operations should exceed 1.5M ops/sec";
        
        // Pipeline efficiency should be high for batch operations
        EXPECT_GT(result.pipeline_efficiency_percent, 80.0) 
            << "Pipeline efficiency should exceed 80% for batch operations";
    }
    
    print_performance_results(results);
    save_results_to_csv(results, "batch_pipeline_optimization_results.csv");
    
    // Compare with individual operations
    auto individual_result = measure_async_operations_performance("Individual_25000", 25000, false);
    auto batch_result = measure_async_operations_performance("Batch_25000", 25000, true);
    
    double batch_improvement = batch_result.sustained_ops_per_second / individual_result.sustained_ops_per_second;
    
    std::cout << "\nBatch vs Individual Performance Comparison (25K operations):" << std::endl;
    std::cout << "  Individual async: " << std::fixed << std::setprecision(0) 
              << individual_result.sustained_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Batch async:      " << std::fixed << std::setprecision(0) 
              << batch_result.sustained_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement:      " << std::fixed << std::setprecision(2) 
              << batch_improvement << "x" << std::endl;
    
    EXPECT_GT(batch_improvement, 1.5) << "Batch operations should provide 1.5x+ improvement";
}

TEST_F(MemoryPipelinePerformanceTest, LatencyOptimizationTest) {
    std::cout << "\n==== Latency Optimization Test ====" << std::endl;
    std::cout << "Target: <1ms average latency, <5ms P99 latency\n" << std::endl;
    
    const size_t TEST_OPERATIONS = 10000;
    
    // Test latency under different load conditions
    std::vector<std::string> test_scenarios = {
        "Low_Load_1K", "Medium_Load_5K", "High_Load_10K"
    };
    std::vector<size_t> load_levels = {1000, 5000, 10000};
    
    std::vector<PipelinePerformanceResult> results;
    
    for (size_t i = 0; i < test_scenarios.size(); ++i) {
        auto result = measure_async_operations_performance(test_scenarios[i], load_levels[i], false);
        results.push_back(result);
        
        std::cout << "Scenario " << test_scenarios[i] << ":" << std::endl;
        std::cout << "  Average latency: " << std::fixed << std::setprecision(3) 
                  << result.average_latency_ms << " ms" << std::endl;
        std::cout << "  P99 latency:     " << std::fixed << std::setprecision(3) 
                  << result.p99_latency_ms << " ms" << std::endl;
        
        // Verify latency targets
        EXPECT_LT(result.average_latency_ms, 2.0) 
            << "Average latency should be <2ms for " << test_scenarios[i];
        
        EXPECT_LT(result.p99_latency_ms, 10.0) 
            << "P99 latency should be <10ms for " << test_scenarios[i];
    }
    
    print_performance_results(results);
    save_results_to_csv(results, "latency_optimization_results.csv");
    
    // Find best latency performance
    auto best_latency = *std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.average_latency_ms < b.average_latency_ms;
        });
    
    std::cout << "Best latency performance: " 
              << std::fixed << std::setprecision(3) << best_latency.average_latency_ms 
              << " ms average, " << std::fixed << std::setprecision(3) << best_latency.p99_latency_ms 
              << " ms P99" << std::endl;
    
    // Epic 2 latency targets
    bool latency_target_met = (best_latency.average_latency_ms < 1.0 && best_latency.p99_latency_ms < 5.0);
    
    if (latency_target_met) {
        std::cout << "\n🎯 LATENCY TARGET ACHIEVED: <1ms average, <5ms P99" << std::endl;
    } else {
        std::cout << "\n⚠️  LATENCY TARGET PARTIAL: Need further optimization" << std::endl;
    }
}

TEST_F(MemoryPipelinePerformanceTest, PipelineHealthAndOptimizationTest) {
    std::cout << "\n==== Pipeline Health and Optimization Test ====" << std::endl;
    
    const size_t TEST_OPERATIONS = 20000;
    
    // Test pipeline health monitoring
    auto initial_result = measure_async_operations_performance("Initial_Config", TEST_OPERATIONS, false);
    
    bool initial_health = pipeline_optimizer->is_pipeline_healthy();
    std::cout << "Initial pipeline health: " << (initial_health ? "HEALTHY" : "UNHEALTHY") << std::endl;
    
    // Test pipeline optimization
    pipeline_optimizer->optimize_pipeline_configuration();
    
    auto optimized_result = measure_async_operations_performance("Optimized_Config", TEST_OPERATIONS, false);
    
    bool optimized_health = pipeline_optimizer->is_pipeline_healthy();
    std::cout << "Optimized pipeline health: " << (optimized_health ? "HEALTHY" : "UNHEALTHY") << std::endl;
    
    double optimization_improvement = optimized_result.sustained_ops_per_second / initial_result.sustained_ops_per_second;
    
    std::cout << "\nPipeline Optimization Results:" << std::endl;
    std::cout << "  Initial performance:  " << std::fixed << std::setprecision(0) 
              << initial_result.sustained_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Optimized performance:" << std::fixed << std::setprecision(0) 
              << optimized_result.sustained_ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement:          " << std::fixed << std::setprecision(2) 
              << optimization_improvement << "x" << std::endl;
    
    // Verify pipeline optimization effectiveness
    EXPECT_TRUE(optimized_health) << "Pipeline should be healthy after optimization";
    EXPECT_GE(optimization_improvement, 1.0) << "Optimization should not decrease performance";
    
    // Test memory management
    size_t available_memory = pipeline_optimizer->get_available_memory_mb();
    std::cout << "Available memory: " << available_memory << " MB" << std::endl;
    
    // Test pipeline flush
    pipeline_optimizer->flush_pipeline();
    std::cout << "Pipeline flushed successfully" << std::endl;
    
    // Verify memory efficiency
    EXPECT_GT(available_memory, 100) << "Should have >100MB available memory";
    
    std::vector<PipelinePerformanceResult> optimization_results = {initial_result, optimized_result};
    print_performance_results(optimization_results);
    save_results_to_csv(optimization_results, "pipeline_optimization_results.csv");
}

TEST_F(MemoryPipelinePerformanceTest, ConcurrentPipelineStressTest) {
    std::cout << "\n==== Concurrent Pipeline Stress Test ====" << std::endl;
    std::cout << "Testing multiple concurrent async operations\n" << std::endl;
    
    const size_t OPS_PER_THREAD = 5000;
    const size_t NUM_THREADS = 4;
    const size_t TOTAL_OPERATIONS = OPS_PER_THREAD * NUM_THREADS;
    
    // Reset metrics
    pipeline_optimizer->reset_metrics();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch concurrent operations
    std::vector<std::future<void>> futures;
    std::atomic<size_t> completed_operations{0};
    
    for (size_t thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
        futures.push_back(std::async(std::launch::async, [=, &completed_operations]() {
            std::vector<std::unique_ptr<MemoryPipelineOptimizer::AsyncOperation>> async_ops;
            
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                std::string key = "stress_" + std::to_string(thread_id) + "_" + std::to_string(i);
                std::string value = generate_random_value(64);
                
                // Mix of insert and lookup operations
                if (i % 2 == 0) {
                    auto async_op = pipeline_optimizer->async_insert(key.c_str(), key.length(),
                                                                   value.c_str(), value.length());
                    if (async_op) {
                        async_ops.push_back(std::move(async_op));
                    }
                } else {
                    auto async_op = pipeline_optimizer->async_lookup(key.c_str(), key.length());
                    if (async_op) {
                        async_ops.push_back(std::move(async_op));
                    }
                }
            }
            
            // Wait for all operations in this thread
            for (auto& async_op : async_ops) {
                if (async_op) {
                    async_op->wait();
                    completed_operations.fetch_add(1);
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    double concurrent_ops_per_sec = completed_operations.load() / total_elapsed;
    auto metrics = pipeline_optimizer->get_pipeline_metrics();
    
    std::cout << "Concurrent Stress Test Results:" << std::endl;
    std::cout << "  Total operations:     " << completed_operations.load() << std::endl;
    std::cout << "  Concurrent threads:   " << NUM_THREADS << std::endl;
    std::cout << "  Total time:           " << std::fixed << std::setprecision(2) << total_elapsed << " seconds" << std::endl;
    std::cout << "  Concurrent ops/sec:   " << std::fixed << std::setprecision(0) << concurrent_ops_per_sec << std::endl;
    std::cout << "  Pipeline efficiency:  " << std::fixed << std::setprecision(1) << metrics.pipeline_efficiency_percent << "%" << std::endl;
    std::cout << "  Pipeline stalls:      " << metrics.pipeline_stalls << std::endl;
    
    // Verify concurrent performance
    EXPECT_EQ(completed_operations.load(), TOTAL_OPERATIONS) << "All operations should complete";
    EXPECT_GT(concurrent_ops_per_sec, 1000000.0) << "Concurrent operations should exceed 1M ops/sec";
    EXPECT_LT(metrics.pipeline_stalls, 100) << "Should have minimal pipeline stalls";
    EXPECT_GT(metrics.pipeline_efficiency_percent, 70.0) << "Pipeline efficiency should remain high under stress";
    
    // Test pipeline health after stress
    bool post_stress_health = pipeline_optimizer->is_pipeline_healthy();
    EXPECT_TRUE(post_stress_health) << "Pipeline should remain healthy after concurrent stress test";
    
    std::cout << "Post-stress pipeline health: " << (post_stress_health ? "HEALTHY" : "UNHEALTHY") << std::endl;
}