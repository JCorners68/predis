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

#include "../../src/core/advanced_batch_processor.h"
#include "../../src/core/gpu_hash_table.h"
#include "../../src/core/memory_manager.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace predis::core;

class AdvancedBatchPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        memory_manager = std::make_unique<MemoryManager>();
        ASSERT_TRUE(memory_manager->initialize());
        
        hash_table = std::make_unique<GpuHashTable>();
        ASSERT_TRUE(hash_table->initialize(memory_manager.get(), 1000000)); // 1M entries
        
        batch_processor = std::make_unique<AdvancedBatchProcessor>();
        
        BatchConfig config;
        config.max_batch_size = 10000;
        config.preferred_batch_size = 1024;
        config.max_concurrent_batches = 4;
        config.enable_auto_tuning = true;
        config.memory_pool_size_mb = 64;
        
        ASSERT_TRUE(batch_processor->initialize(hash_table.get(), memory_manager.get(), config));
    }
    
    void TearDown() override {
        if (batch_processor) {
            batch_processor->shutdown();
        }
        hash_table.reset();
        memory_manager.reset();
    }
    
    std::vector<std::string> generate_random_keys(size_t count, size_t key_length = 16) {
        std::vector<std::string> keys;
        keys.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis('a', 'z');
        
        for (size_t i = 0; i < count; ++i) {
            std::string key;
            key.reserve(key_length);
            for (size_t j = 0; j < key_length; ++j) {
                key += static_cast<char>(dis(gen));
            }
            keys.push_back(std::move(key));
        }
        
        return keys;
    }
    
    std::vector<std::string> generate_random_values(size_t count, size_t value_length = 64) {
        std::vector<std::string> values;
        values.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis('A', 'Z');
        
        for (size_t i = 0; i < count; ++i) {
            std::string value;
            value.reserve(value_length);
            for (size_t j = 0; j < value_length; ++j) {
                value += static_cast<char>(dis(gen));
            }
            values.push_back(std::move(value));
        }
        
        return values;
    }
    
    struct PerformanceResult {
        size_t batch_size;
        double operations_per_second;
        double latency_ms;
        double efficiency_percent;
        size_t successful_operations;
        size_t failed_operations;
    };
    
    PerformanceResult measure_batch_performance(
        const std::vector<std::string>& keys,
        const std::vector<std::string>& values,
        const std::string& operation_type) {
        
        PerformanceResult result{};
        result.batch_size = keys.size();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        AdvancedBatchProcessor::BatchResult batch_result;
        
        if (operation_type == "put") {
            batch_result = batch_processor->batch_put(keys, values);
        } else if (operation_type == "get") {
            batch_result = batch_processor->batch_get(keys);
        } else if (operation_type == "delete") {
            batch_result = batch_processor->batch_delete(keys);
        } else if (operation_type == "exists") {
            batch_result = batch_processor->batch_exists(keys);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        result.latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        result.operations_per_second = batch_result.metrics.operations_per_second;
        result.efficiency_percent = batch_result.metrics.batch_efficiency_percent;
        result.successful_operations = batch_result.successful_count;
        result.failed_operations = batch_result.failed_count;
        
        return result;
    }
    
    void print_performance_results(const std::vector<PerformanceResult>& results, 
                                 const std::string& operation_type) {
        std::cout << "\n=== " << operation_type << " Performance Results ===" << std::endl;
        std::cout << std::setw(12) << "Batch Size" 
                  << std::setw(15) << "Ops/Sec" 
                  << std::setw(12) << "Latency(ms)"
                  << std::setw(12) << "Efficiency%"
                  << std::setw(12) << "Success"
                  << std::setw(12) << "Failed" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(12) << result.batch_size
                      << std::setw(15) << std::fixed << std::setprecision(0) << result.operations_per_second
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.latency_ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.efficiency_percent
                      << std::setw(12) << result.successful_operations
                      << std::setw(12) << result.failed_operations << std::endl;
        }
        std::cout << std::endl;
    }
    
    void save_results_to_csv(const std::vector<PerformanceResult>& results,
                            const std::string& operation_type,
                            const std::string& filename) {
        std::ofstream file(filename);
        file << "Operation,Batch_Size,Ops_Per_Sec,Latency_Ms,Efficiency_Percent,Successful,Failed\n";
        
        for (const auto& result : results) {
            file << operation_type << ","
                 << result.batch_size << ","
                 << result.operations_per_second << ","
                 << result.latency_ms << ","
                 << result.efficiency_percent << ","
                 << result.successful_operations << ","
                 << result.failed_operations << "\n";
        }
    }
    
    std::unique_ptr<MemoryManager> memory_manager;
    std::unique_ptr<GpuHashTable> hash_table;
    std::unique_ptr<AdvancedBatchProcessor> batch_processor;
};

TEST_F(AdvancedBatchPerformanceTest, BatchSizeScalingTest) {
    std::vector<size_t> batch_sizes = {10, 50, 100, 500, 1000, 2000, 5000, 10000};
    std::vector<PerformanceResult> put_results, get_results, delete_results, exists_results;
    
    std::cout << "\n==== Advanced Batch Processor Scaling Test ====" << std::endl;
    std::cout << "Testing batch sizes: ";
    for (size_t size : batch_sizes) {
        std::cout << size << " ";
    }
    std::cout << "\n" << std::endl;
    
    // Test PUT operations scaling
    for (size_t batch_size : batch_sizes) {
        auto keys = generate_random_keys(batch_size);
        auto values = generate_random_values(batch_size);
        
        auto result = measure_batch_performance(keys, values, "put");
        put_results.push_back(result);
        
        // Verify performance expectations
        if (batch_size >= 1000) {
            EXPECT_GT(result.operations_per_second, 100000.0) 
                << "PUT operations should exceed 100K ops/sec for batch size " << batch_size;
        }
        EXPECT_GT(result.efficiency_percent, 90.0) 
            << "PUT efficiency should exceed 90% for batch size " << batch_size;
    }
    
    // Test GET operations scaling
    for (size_t batch_size : batch_sizes) {
        auto keys = generate_random_keys(batch_size);
        auto values = generate_random_values(batch_size);
        
        // First insert the data
        batch_processor->batch_put(keys, values);
        
        // Then test retrieval performance
        auto result = measure_batch_performance(keys, values, "get");
        get_results.push_back(result);
        
        // Verify performance expectations
        if (batch_size >= 1000) {
            EXPECT_GT(result.operations_per_second, 150000.0) 
                << "GET operations should exceed 150K ops/sec for batch size " << batch_size;
        }
        EXPECT_GT(result.efficiency_percent, 95.0) 
            << "GET efficiency should exceed 95% for batch size " << batch_size;
    }
    
    // Test EXISTS operations scaling
    for (size_t batch_size : batch_sizes) {
        auto keys = generate_random_keys(batch_size);
        
        auto result = measure_batch_performance(keys, {}, "exists");
        exists_results.push_back(result);
        
        // EXISTS operations should be fastest (read-only, no value transfer)
        if (batch_size >= 1000) {
            EXPECT_GT(result.operations_per_second, 200000.0) 
                << "EXISTS operations should exceed 200K ops/sec for batch size " << batch_size;
        }
    }
    
    // Test DELETE operations scaling  
    for (size_t batch_size : batch_sizes) {
        auto keys = generate_random_keys(batch_size);
        
        auto result = measure_batch_performance(keys, {}, "delete");
        delete_results.push_back(result);
        
        if (batch_size >= 1000) {
            EXPECT_GT(result.operations_per_second, 80000.0) 
                << "DELETE operations should exceed 80K ops/sec for batch size " << batch_size;
        }
    }
    
    // Print detailed results
    print_performance_results(put_results, "PUT");
    print_performance_results(get_results, "GET");
    print_performance_results(exists_results, "EXISTS");
    print_performance_results(delete_results, "DELETE");
    
    // Save results to CSV for further analysis
    save_results_to_csv(put_results, "PUT", "batch_put_scaling.csv");
    save_results_to_csv(get_results, "GET", "batch_get_scaling.csv");
    save_results_to_csv(exists_results, "EXISTS", "batch_exists_scaling.csv");
    save_results_to_csv(delete_results, "DELETE", "batch_delete_scaling.csv");
}

TEST_F(AdvancedBatchPerformanceTest, RedisComparisonBenchmark) {
    const size_t BENCHMARK_BATCH_SIZE = 5000;
    const size_t NUM_ITERATIONS = 10;
    
    std::cout << "\n==== Redis vs Predis Batch Performance Comparison ====" << std::endl;
    std::cout << "Batch size: " << BENCHMARK_BATCH_SIZE << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    
    std::vector<double> put_performances, get_performances;
    
    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        auto keys = generate_random_keys(BENCHMARK_BATCH_SIZE);
        auto values = generate_random_values(BENCHMARK_BATCH_SIZE);
        
        // Measure PUT performance
        auto put_result = measure_batch_performance(keys, values, "put");
        put_performances.push_back(put_result.operations_per_second);
        
        // Measure GET performance
        auto get_result = measure_batch_performance(keys, values, "get");
        get_performances.push_back(get_result.operations_per_second);
    }
    
    // Calculate average performance
    double avg_put_ops = 0.0, avg_get_ops = 0.0;
    for (double ops : put_performances) avg_put_ops += ops;
    for (double ops : get_performances) avg_get_ops += ops;
    avg_put_ops /= NUM_ITERATIONS;
    avg_get_ops /= NUM_ITERATIONS;
    
    // Redis baseline (estimated from Epic 1 measurements)
    const double REDIS_PUT_OPS = 273000.0; // Redis batch PUT ops/sec
    const double REDIS_GET_OPS = 404000.0; // Redis batch GET ops/sec
    
    double put_improvement = avg_put_ops / REDIS_PUT_OPS;
    double get_improvement = avg_get_ops / REDIS_GET_OPS;
    
    std::cout << "\nPerformance Comparison Results:" << std::endl;
    std::cout << "PUT Operations:" << std::endl;
    std::cout << "  Predis:      " << std::fixed << std::setprecision(0) << avg_put_ops << " ops/sec" << std::endl;
    std::cout << "  Redis:       " << std::fixed << std::setprecision(0) << REDIS_PUT_OPS << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << put_improvement << "x" << std::endl;
    
    std::cout << "\nGET Operations:" << std::endl;
    std::cout << "  Predis:      " << std::fixed << std::setprecision(0) << avg_get_ops << " ops/sec" << std::endl;
    std::cout << "  Redis:       " << std::fixed << std::setprecision(0) << REDIS_GET_OPS << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << get_improvement << "x" << std::endl;
    
    // Verify we achieve target performance improvements for Epic 2
    EXPECT_GT(put_improvement, 5.0) << "PUT batch operations should achieve 5x+ improvement over Redis";
    EXPECT_GT(get_improvement, 3.0) << "GET batch operations should achieve 3x+ improvement over Redis";
    
    // For investor demo, we want to show significant batch performance gains
    if (put_improvement >= 10.0) {
        std::cout << "\n🎯 EXCELLENT: Achieved " << put_improvement << "x PUT improvement (target: 10x+)" << std::endl;
    } else if (put_improvement >= 5.0) {
        std::cout << "\n✅ GOOD: Achieved " << put_improvement << "x PUT improvement (target: 5x+)" << std::endl;
    }
    
    if (get_improvement >= 25.0) {
        std::cout << "🎯 EXCELLENT: Achieved " << get_improvement << "x GET improvement (target: 25x+)" << std::endl;
    } else if (get_improvement >= 10.0) {
        std::cout << "✅ GOOD: Achieved " << get_improvement << "x GET improvement (target: 10x+)" << std::endl;
    }
}

TEST_F(AdvancedBatchPerformanceTest, AutoTuningValidation) {
    std::cout << "\n==== Auto-Tuning Performance Validation ====" << std::endl;
    
    // Start with suboptimal batch size
    BatchConfig config;
    config.max_batch_size = 10000;
    config.preferred_batch_size = 100; // Start small
    config.enable_auto_tuning = true;
    batch_processor->configure(config);
    
    size_t initial_optimal = batch_processor->get_optimal_batch_size();
    std::cout << "Initial optimal batch size: " << initial_optimal << std::endl;
    
    // Perform several operations to trigger auto-tuning
    std::vector<double> performance_history;
    
    for (int i = 0; i < 20; ++i) {
        size_t current_batch_size = batch_processor->get_optimal_batch_size();
        auto keys = generate_random_keys(current_batch_size);
        auto values = generate_random_values(current_batch_size);
        
        auto result = batch_processor->batch_put(keys, values);
        performance_history.push_back(result.metrics.operations_per_second);
        
        // Trigger auto-tuning with the recent metrics
        batch_processor->tune_batch_size(result.metrics);
        
        if (i % 5 == 0) {
            std::cout << "Iteration " << i << ": batch_size=" << current_batch_size 
                      << ", ops/sec=" << std::fixed << std::setprecision(0) << result.metrics.operations_per_second
                      << ", efficiency=" << std::fixed << std::setprecision(1) << result.metrics.batch_efficiency_percent << "%" << std::endl;
        }
    }
    
    size_t final_optimal = batch_processor->get_optimal_batch_size();
    std::cout << "Final optimal batch size: " << final_optimal << std::endl;
    
    // Verify auto-tuning improved performance
    double initial_perf = performance_history.front();
    double final_perf = performance_history.back();
    double improvement = final_perf / initial_perf;
    
    std::cout << "Performance improvement: " << std::fixed << std::setprecision(2) << improvement << "x" << std::endl;
    
    EXPECT_GT(final_optimal, initial_optimal) << "Auto-tuning should increase batch size for better performance";
    EXPECT_GT(improvement, 1.1) << "Auto-tuning should provide at least 10% performance improvement";
    
    // Verify cumulative metrics tracking
    auto cumulative = batch_processor->get_cumulative_metrics();
    EXPECT_GT(cumulative.successful_operations, 0) << "Should track successful operations";
    EXPECT_GT(cumulative.operations_per_second, 0) << "Should track average performance";
    
    std::cout << "Cumulative metrics:" << std::endl;
    std::cout << "  Total successful ops: " << cumulative.successful_operations << std::endl;
    std::cout << "  Average ops/sec: " << std::fixed << std::setprecision(0) << cumulative.operations_per_second << std::endl;
    std::cout << "  GPU bandwidth util: " << std::fixed << std::setprecision(1) << cumulative.gpu_bandwidth_utilization_percent << "%" << std::endl;
}

TEST_F(AdvancedBatchPerformanceTest, MemoryEfficiencyTest) {
    std::cout << "\n==== Memory Efficiency Test ====" << std::endl;
    
    const size_t LARGE_BATCH_SIZE = 8000;
    const size_t NUM_ITERATIONS = 5;
    
    auto initial_stats = memory_manager->get_stats();
    std::cout << "Initial memory usage: " << initial_stats.used_bytes / (1024*1024) << " MB" << std::endl;
    
    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        auto keys = generate_random_keys(LARGE_BATCH_SIZE, 32);    // Larger keys
        auto values = generate_random_values(LARGE_BATCH_SIZE, 256); // Larger values
        
        // Test large batch operations
        auto put_result = batch_processor->batch_put(keys, values);
        auto get_result = batch_processor->batch_get(keys);
        
        auto current_stats = memory_manager->get_stats();
        std::cout << "After iteration " << i << ": " << current_stats.used_bytes / (1024*1024) << " MB used" << std::endl;
        
        // Verify no memory leaks (memory usage should be stable)
        if (i > 0) {
            EXPECT_LT(current_stats.used_bytes, initial_stats.used_bytes * 1.5) 
                << "Memory usage should not grow excessively during batch operations";
        }
        
        // Verify batch operations still perform well with large data
        EXPECT_GT(put_result.metrics.operations_per_second, 50000.0) 
            << "Large batch PUT should maintain good performance";
        EXPECT_GT(get_result.metrics.operations_per_second, 75000.0) 
            << "Large batch GET should maintain good performance";
    }
    
    auto final_stats = memory_manager->get_stats();
    std::cout << "Final memory usage: " << final_stats.used_bytes / (1024*1024) << " MB" << std::endl;
    
    // Memory usage should be reasonable even after large operations
    EXPECT_LT(final_stats.used_bytes, initial_stats.used_bytes * 2.0) 
        << "Memory usage should not double during test";
}