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

#include "../../src/core/optimized_gpu_kernels.h"
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

using namespace predis::core;

class OptimizedKernelPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize memory manager
        memory_manager = std::make_unique<MemoryManager>();
        ASSERT_TRUE(memory_manager->initialize());
        
        // Initialize GPU hash table
        hash_table = std::make_unique<GpuHashTable>();
        ASSERT_TRUE(hash_table->initialize(memory_manager.get(), 1000000)); // 1M entries
        
        // Initialize optimized kernels
        optimized_kernels = std::make_unique<OptimizedGpuKernels>();
        ASSERT_TRUE(optimized_kernels->initialize(hash_table.get(), memory_manager.get()));
        
        // Configure for maximum optimization
        optimized_kernels->configure_optimization(true, false, true); // No tensor cores yet
    }
    
    void TearDown() override {
        if (optimized_kernels) {
            optimized_kernels->shutdown();
        }
        hash_table.reset();
        memory_manager.reset();
    }
    
    struct BenchmarkResult {
        std::string operation;
        std::string kernel_type;
        double ops_per_second;
        double latency_microseconds;
        double memory_bandwidth_gbps;
        double gpu_occupancy_percent;
        size_t operations_count;
        bool success;
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
    
    BenchmarkResult benchmark_optimized_insert(const std::string& key, const std::string& value, 
                                              size_t iterations = 1000) {
        BenchmarkResult result;
        result.operation = "INSERT";
        result.kernel_type = "OPTIMIZED";
        result.operations_count = iterations;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t successful_ops = 0;
        for (size_t i = 0; i < iterations; ++i) {
            std::string test_key = key + std::to_string(i);
            bool success = optimized_kernels->optimized_insert(test_key.c_str(), test_key.length(),
                                                              value.c_str(), value.length());
            if (success) successful_ops++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        
        result.ops_per_second = (successful_ops * 1000000.0) / elapsed_us;
        result.latency_microseconds = elapsed_us / iterations;
        result.success = (successful_ops == iterations);
        
        // Get metrics from last kernel execution
        auto metrics = optimized_kernels->get_last_metrics();
        result.memory_bandwidth_gbps = metrics.memory_bandwidth_gbps;
        result.gpu_occupancy_percent = metrics.gpu_occupancy_percent;
        
        return result;
    }
    
    BenchmarkResult benchmark_optimized_lookup(const std::vector<std::string>& keys, 
                                              size_t iterations = 1000) {
        BenchmarkResult result;
        result.operation = "LOOKUP";
        result.kernel_type = "OPTIMIZED";
        result.operations_count = iterations;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t successful_ops = 0;
        char value_buffer[4096];
        for (size_t i = 0; i < iterations && i < keys.size(); ++i) {
            size_t value_len = 0;
            bool success = optimized_kernels->optimized_lookup(keys[i].c_str(), keys[i].length(),
                                                              value_buffer, &value_len);
            if (success) successful_ops++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        
        result.ops_per_second = (successful_ops * 1000000.0) / elapsed_us;
        result.latency_microseconds = elapsed_us / iterations;
        result.success = (successful_ops > 0);
        
        // Get metrics from last kernel execution
        auto metrics = optimized_kernels->get_last_metrics();
        result.memory_bandwidth_gbps = metrics.memory_bandwidth_gbps;
        result.gpu_occupancy_percent = metrics.gpu_occupancy_percent;
        
        return result;
    }
    
    BenchmarkResult benchmark_baseline_operations(const std::string& operation, 
                                                 const std::vector<std::string>& keys,
                                                 const std::vector<std::string>& values,
                                                 size_t iterations = 1000) {
        BenchmarkResult result;
        result.operation = operation;
        result.kernel_type = "BASELINE";
        result.operations_count = iterations;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t successful_ops = 0;
        
        if (operation == "INSERT") {
            for (size_t i = 0; i < iterations && i < keys.size() && i < values.size(); ++i) {
                bool success = hash_table->insert(keys[i].c_str(), keys[i].length(),
                                                 values[i].c_str(), values[i].length());
                if (success) successful_ops++;
            }
        } else if (operation == "LOOKUP") {
            char value_buffer[4096];
            for (size_t i = 0; i < iterations && i < keys.size(); ++i) {
                uint32_t value_len = 0;
                bool success = hash_table->lookup(keys[i].c_str(), keys[i].length(),
                                                 value_buffer, &value_len);
                if (success) successful_ops++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        
        result.ops_per_second = (successful_ops * 1000000.0) / elapsed_us;
        result.latency_microseconds = elapsed_us / iterations;
        result.success = (successful_ops > 0);
        
        return result;
    }
    
    void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== GPU Kernel Performance Benchmark Results ===" << std::endl;
        std::cout << std::setw(10) << "Operation" 
                  << std::setw(12) << "Kernel" 
                  << std::setw(15) << "Ops/Sec" 
                  << std::setw(12) << "Latency(μs)"
                  << std::setw(12) << "Bandwidth"
                  << std::setw(12) << "Occupancy"
                  << std::setw(10) << "Success" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(10) << result.operation
                      << std::setw(12) << result.kernel_type
                      << std::setw(15) << std::fixed << std::setprecision(0) << result.ops_per_second
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.latency_microseconds
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.memory_bandwidth_gbps
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.gpu_occupancy_percent
                      << std::setw(10) << (result.success ? "✓" : "✗") << std::endl;
        }
        std::cout << std::endl;
    }
    
    void save_benchmark_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        file << "Operation,Kernel_Type,Ops_Per_Sec,Latency_Microseconds,Memory_Bandwidth_GBps,GPU_Occupancy_Percent,Operations_Count,Success\n";
        
        for (const auto& result : results) {
            file << result.operation << ","
                 << result.kernel_type << ","
                 << result.ops_per_second << ","
                 << result.latency_microseconds << ","
                 << result.memory_bandwidth_gbps << ","
                 << result.gpu_occupancy_percent << ","
                 << result.operations_count << ","
                 << (result.success ? 1 : 0) << "\n";
        }
    }
    
    std::unique_ptr<MemoryManager> memory_manager;
    std::unique_ptr<GpuHashTable> hash_table;
    std::unique_ptr<OptimizedGpuKernels> optimized_kernels;
};

TEST_F(OptimizedKernelPerformanceTest, SingleOperationOptimizationTest) {
    std::cout << "\n==== Single Operation Optimization Test ====" << std::endl;
    std::cout << "Target: 10x+ improvement over baseline kernels\n" << std::endl;
    
    const size_t NUM_OPERATIONS = 1000;
    const std::string test_key_base = "optimization_test_key_";
    const std::string test_value = generate_random_value(64);
    
    std::vector<BenchmarkResult> results;
    
    // Benchmark optimized INSERT operations
    auto opt_insert_result = benchmark_optimized_insert(test_key_base, test_value, NUM_OPERATIONS);
    results.push_back(opt_insert_result);
    
    // Prepare data for baseline comparison
    std::vector<std::string> keys, values;
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        keys.push_back(test_key_base + std::to_string(i + 10000)); // Different keys for baseline
        values.push_back(test_value);
    }
    
    // Benchmark baseline INSERT operations
    auto baseline_insert_result = benchmark_baseline_operations("INSERT", keys, values, NUM_OPERATIONS);
    results.push_back(baseline_insert_result);
    
    // Calculate improvement
    double insert_improvement = opt_insert_result.ops_per_second / baseline_insert_result.ops_per_second;
    
    std::cout << "INSERT Operation Results:" << std::endl;
    std::cout << "  Optimized:   " << std::fixed << std::setprecision(0) << opt_insert_result.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Baseline:    " << std::fixed << std::setprecision(0) << baseline_insert_result.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << insert_improvement << "x" << std::endl;
    
    // Now test LOOKUP operations
    std::vector<std::string> lookup_keys;
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        lookup_keys.push_back(test_key_base + std::to_string(i));
    }
    
    auto opt_lookup_result = benchmark_optimized_lookup(lookup_keys, NUM_OPERATIONS);
    results.push_back(opt_lookup_result);
    
    auto baseline_lookup_result = benchmark_baseline_operations("LOOKUP", lookup_keys, values, NUM_OPERATIONS);
    results.push_back(baseline_lookup_result);
    
    double lookup_improvement = opt_lookup_result.ops_per_second / baseline_lookup_result.ops_per_second;
    
    std::cout << "\nLOOKUP Operation Results:" << std::endl;
    std::cout << "  Optimized:   " << std::fixed << std::setprecision(0) << opt_lookup_result.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Baseline:    " << std::fixed << std::setprecision(0) << baseline_lookup_result.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << lookup_improvement << "x" << std::endl;
    
    // Print detailed results
    print_benchmark_results(results);
    
    // Save results for analysis
    save_benchmark_csv(results, "single_operation_optimization_results.csv");
    
    // Verify we achieved target improvements
    EXPECT_GT(insert_improvement, 3.0) << "INSERT optimization should achieve at least 3x improvement";
    EXPECT_GT(lookup_improvement, 5.0) << "LOOKUP optimization should achieve at least 5x improvement";
    
    // Verify overall performance targets
    EXPECT_GT(opt_insert_result.ops_per_second, 100000.0) << "Optimized INSERT should exceed 100K ops/sec";
    EXPECT_GT(opt_lookup_result.ops_per_second, 500000.0) << "Optimized LOOKUP should exceed 500K ops/sec";
    
    // For Epic 2 targets, we want significant single operation improvements
    if (insert_improvement >= 10.0) {
        std::cout << "\n🎯 EXCELLENT: Achieved " << insert_improvement << "x INSERT improvement (target: 10x+)" << std::endl;
    } else if (insert_improvement >= 5.0) {
        std::cout << "\n✅ GOOD: Achieved " << insert_improvement << "x INSERT improvement (target: 5x+)" << std::endl;
    }
    
    if (lookup_improvement >= 15.0) {
        std::cout << "🎯 EXCELLENT: Achieved " << lookup_improvement << "x LOOKUP improvement (target: 15x+)" << std::endl;
    } else if (lookup_improvement >= 10.0) {
        std::cout << "✅ GOOD: Achieved " << lookup_improvement << "x LOOKUP improvement (target: 10x+)" << std::endl;
    }
}

TEST_F(OptimizedKernelPerformanceTest, MemoryHierarchyOptimizationTest) {
    std::cout << "\n==== Memory Hierarchy Optimization Test ====" << std::endl;
    std::cout << "Testing L1/L2 cache efficiency and shared memory utilization\n" << std::endl;
    
    const size_t NUM_KEYS = 500;
    std::vector<std::string> test_keys, test_values;
    
    // Generate test data with varying key/value sizes to test memory patterns
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        size_t key_size = 8 + (i % 16); // Keys 8-24 bytes
        size_t value_size = 32 + (i % 64); // Values 32-96 bytes
        
        test_keys.push_back(generate_random_key(key_size));
        test_values.push_back(generate_random_value(value_size));
    }
    
    // Test different launch configurations to measure memory hierarchy impact
    std::vector<OptimizedGpuKernels::LaunchConfig> configs = {
        {32, 64},    // Low occupancy
        {128, 256},  // Medium occupancy  
        {256, 512},  // High occupancy
        {512, 1024}  // Maximum occupancy
    };
    
    std::vector<BenchmarkResult> results;
    
    for (size_t config_idx = 0; config_idx < configs.size(); ++config_idx) {
        const auto& config = configs[config_idx];
        
        std::cout << "Testing configuration: block_size=" << config.block_size 
                  << ", grid_size=" << config.grid_size << std::endl;
        
        // Insert test data
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t successful_inserts = 0;
        
        for (size_t i = 0; i < test_keys.size(); ++i) {
            bool success = optimized_kernels->optimized_insert(
                test_keys[i].c_str(), test_keys[i].length(),
                test_values[i].c_str(), test_values[i].length(), config);
            if (success) successful_inserts++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        
        BenchmarkResult result;
        result.operation = "INSERT_CONFIG_" + std::to_string(config_idx);
        result.kernel_type = "OPTIMIZED";
        result.ops_per_second = (successful_inserts * 1000000.0) / elapsed_us;
        result.latency_microseconds = elapsed_us / test_keys.size();
        result.operations_count = test_keys.size();
        result.success = (successful_inserts > test_keys.size() * 0.95); // 95% success rate
        
        auto metrics = optimized_kernels->get_last_metrics();
        result.memory_bandwidth_gbps = metrics.memory_bandwidth_gbps;
        result.gpu_occupancy_percent = metrics.gpu_occupancy_percent;
        
        results.push_back(result);
    }
    
    print_benchmark_results(results);
    save_benchmark_csv(results, "memory_hierarchy_optimization_results.csv");
    
    // Find best performing configuration
    auto best_result = *std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.ops_per_second < b.ops_per_second;
        });
    
    std::cout << "Best performing configuration achieved: " 
              << std::fixed << std::setprecision(0) << best_result.ops_per_second << " ops/sec" << std::endl;
    
    // Verify memory optimization effectiveness
    EXPECT_GT(best_result.ops_per_second, 200000.0) << "Memory optimized kernels should exceed 200K ops/sec";
    EXPECT_GT(best_result.gpu_occupancy_percent, 50.0) << "Should achieve reasonable GPU occupancy";
}

TEST_F(OptimizedKernelPerformanceTest, CooperativeGroupsEfficiencyTest) {
    std::cout << "\n==== Cooperative Groups Efficiency Test ====" << std::endl;
    std::cout << "Testing block-level parallelism and warp cooperation\n" << std::endl;
    
    const size_t NUM_OPERATIONS = 2000;
    
    // Test with cooperative groups enabled
    optimized_kernels->configure_optimization(true, false, true);
    
    std::vector<std::string> keys, values;
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        keys.push_back("coop_test_" + std::to_string(i));
        values.push_back(generate_random_value(48));
    }
    
    auto coop_insert_result = benchmark_optimized_insert("coop_insert_", values[0], NUM_OPERATIONS);
    auto coop_lookup_result = benchmark_optimized_lookup(keys, NUM_OPERATIONS);
    
    // Test with cooperative groups disabled
    optimized_kernels->configure_optimization(false, false, true);
    
    // Clear data for fair comparison
    hash_table.reset();
    hash_table = std::make_unique<GpuHashTable>();
    hash_table->initialize(memory_manager.get(), 1000000);
    optimized_kernels->initialize(hash_table.get(), memory_manager.get());
    
    auto no_coop_insert_result = benchmark_optimized_insert("no_coop_insert_", values[0], NUM_OPERATIONS);
    auto no_coop_lookup_result = benchmark_optimized_lookup(keys, NUM_OPERATIONS);
    
    std::vector<BenchmarkResult> results = {
        coop_insert_result, no_coop_insert_result,
        coop_lookup_result, no_coop_lookup_result
    };
    
    // Update result names for clarity
    results[0].kernel_type = "COOPERATIVE";
    results[1].kernel_type = "NO_COOP";
    results[2].kernel_type = "COOPERATIVE";
    results[3].kernel_type = "NO_COOP";
    
    print_benchmark_results(results);
    save_benchmark_csv(results, "cooperative_groups_efficiency_results.csv");
    
    double coop_insert_improvement = coop_insert_result.ops_per_second / no_coop_insert_result.ops_per_second;
    double coop_lookup_improvement = coop_lookup_result.ops_per_second / no_coop_lookup_result.ops_per_second;
    
    std::cout << "Cooperative Groups Performance Impact:" << std::endl;
    std::cout << "  INSERT improvement: " << std::fixed << std::setprecision(2) << coop_insert_improvement << "x" << std::endl;
    std::cout << "  LOOKUP improvement: " << std::fixed << std::setprecision(2) << coop_lookup_improvement << "x" << std::endl;
    
    // Cooperative groups should provide meaningful improvements
    EXPECT_GT(coop_insert_improvement, 1.2) << "Cooperative groups should improve INSERT performance by at least 20%";
    EXPECT_GT(coop_lookup_improvement, 1.5) << "Cooperative groups should improve LOOKUP performance by at least 50%";
    
    // Re-enable cooperative groups for subsequent tests
    optimized_kernels->configure_optimization(true, false, true);
}

TEST_F(OptimizedKernelPerformanceTest, OverallPerformanceTargetValidation) {
    std::cout << "\n==== Epic 2 Story 2.2 Performance Target Validation ====" << std::endl;
    std::cout << "Target: 10x+ single operation improvement over baseline\n" << std::endl;
    
    const size_t VALIDATION_OPERATIONS = 5000;
    const std::string key_prefix = "epic2_validation_";
    const std::string test_value = generate_random_value(64);
    
    // Comprehensive validation test
    auto opt_insert = benchmark_optimized_insert(key_prefix, test_value, VALIDATION_OPERATIONS);
    
    std::vector<std::string> validation_keys;
    for (size_t i = 0; i < VALIDATION_OPERATIONS; ++i) {
        validation_keys.push_back(key_prefix + std::to_string(i));
    }
    
    auto opt_lookup = benchmark_optimized_lookup(validation_keys, VALIDATION_OPERATIONS);
    
    // Baseline comparison (using different keys to avoid conflicts)
    std::vector<std::string> baseline_keys, baseline_values;
    for (size_t i = 0; i < VALIDATION_OPERATIONS; ++i) {
        baseline_keys.push_back("baseline_" + std::to_string(i));
        baseline_values.push_back(test_value);
    }
    
    auto baseline_insert = benchmark_baseline_operations("INSERT", baseline_keys, baseline_values, VALIDATION_OPERATIONS);
    auto baseline_lookup = benchmark_baseline_operations("LOOKUP", baseline_keys, baseline_values, VALIDATION_OPERATIONS);
    
    std::vector<BenchmarkResult> final_results = {opt_insert, baseline_insert, opt_lookup, baseline_lookup};
    print_benchmark_results(final_results);
    
    double insert_improvement = opt_insert.ops_per_second / baseline_insert.ops_per_second;
    double lookup_improvement = opt_lookup.ops_per_second / baseline_lookup.ops_per_second;
    
    std::cout << "\n=== EPIC 2 STORY 2.2 FINAL RESULTS ===" << std::endl;
    std::cout << "INSERT Performance:" << std::endl;
    std::cout << "  Optimized: " << std::fixed << std::setprecision(0) << opt_insert.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Baseline:  " << std::fixed << std::setprecision(0) << baseline_insert.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << insert_improvement << "x" << std::endl;
    
    std::cout << "\nLOOKUP Performance:" << std::endl;
    std::cout << "  Optimized: " << std::fixed << std::setprecision(0) << opt_lookup.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Baseline:  " << std::fixed << std::setprecision(0) << baseline_lookup.ops_per_second << " ops/sec" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << lookup_improvement << "x" << std::endl;
    
    // Epic 2 Story 2.2 success criteria
    bool epic2_story2_2_success = (insert_improvement >= 10.0 && lookup_improvement >= 10.0);
    
    if (epic2_story2_2_success) {
        std::cout << "\n🎯 EPIC 2 STORY 2.2 SUCCESS: Achieved " << std::min(insert_improvement, lookup_improvement) 
                  << "x+ improvement (target: 10x+)" << std::endl;
    } else {
        std::cout << "\n⚠️  EPIC 2 STORY 2.2 PARTIAL: Need further optimization to reach 10x target" << std::endl;
    }
    
    // Test assertions
    EXPECT_GT(insert_improvement, 10.0) << "INSERT should achieve 10x+ improvement for Epic 2 targets";
    EXPECT_GT(lookup_improvement, 10.0) << "LOOKUP should achieve 10x+ improvement for Epic 2 targets";
    
    // Additional performance requirements
    EXPECT_GT(opt_insert.ops_per_second, 500000.0) << "Optimized INSERT should exceed 500K ops/sec";
    EXPECT_GT(opt_lookup.ops_per_second, 1000000.0) << "Optimized LOOKUP should exceed 1M ops/sec";
    
    save_benchmark_csv(final_results, "epic2_story2_2_validation_results.csv");
}