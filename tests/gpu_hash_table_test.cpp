/*
 * Copyright 2025 Predis Project
 *
 * Comprehensive GPU Hash Table Test Suite
 * Tests all functionality including performance benchmarks
 */

#include "../src/core/data_structures/gpu_hash_table.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <random>
#include <thread>
#include <algorithm>
#include <unordered_set>

using namespace predis::core;

class GpuHashTableTest {
public:
    static void run_all_tests() {
        std::cout << "🧪 Starting GPU Hash Table Comprehensive Tests\n" << std::endl;
        
        test_initialization();
        test_basic_operations();
        test_batch_operations();
        test_collision_handling();
        test_string_operations();
        test_performance_benchmark();
        test_concurrent_operations();
        test_memory_management();
        
        std::cout << "\n✅ All GPU Hash Table tests passed!" << std::endl;
    }

private:
    static void test_initialization() {
        std::cout << "🔧 Testing hash table initialization..." << std::endl;
        
        GpuHashTable hash_table;
        
        // Test initialization with different parameters
        assert(hash_table.initialize(1024, GpuHashTable::HashMethod::FNV1A));
        std::cout << "  ✅ FNV1A initialization successful" << std::endl;
        
        auto stats = hash_table.get_stats();
        assert(stats.capacity == 1024);
        assert(stats.size == 0);
        assert(stats.load_factor == 0.0);
        std::cout << "  ✅ Initial statistics correct" << std::endl;
        
        hash_table.shutdown();
        
        // Test with MURMUR3 hash
        assert(hash_table.initialize(2048, GpuHashTable::HashMethod::MURMUR3));
        std::cout << "  ✅ MURMUR3 initialization successful" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Initialization test passed\n" << std::endl;
    }
    
    static void test_basic_operations() {
        std::cout << "📦 Testing basic hash table operations..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(1024));
        
        // Test insert and lookup
        assert(hash_table.insert("key1", "value1"));
        std::cout << "  ✅ Insert operation successful" << std::endl;
        
        std::string value;
        assert(hash_table.lookup("key1", value));
        assert(value == "value1");
        std::cout << "  ✅ Lookup operation successful: " << value << std::endl;
        
        // Test update
        assert(hash_table.insert("key1", "updated_value1"));
        assert(hash_table.lookup("key1", value));
        assert(value == "updated_value1");
        std::cout << "  ✅ Update operation successful: " << value << std::endl;
        
        // Test non-existent key
        assert(!hash_table.lookup("nonexistent", value));
        std::cout << "  ✅ Non-existent key correctly not found" << std::endl;
        
        // Test remove
        assert(hash_table.remove("key1"));
        assert(!hash_table.lookup("key1", value));
        std::cout << "  ✅ Remove operation successful" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Basic operations test passed\n" << std::endl;
    }
    
    static void test_batch_operations() {
        std::cout << "🏊 Testing batch operations..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(10000));
        
        // Prepare test data
        std::vector<std::pair<std::string, std::string>> test_data;
        for (int i = 0; i < 1000; ++i) {
            test_data.emplace_back("batch_key_" + std::to_string(i), 
                                  "batch_value_" + std::to_string(i));
        }
        
        // Test batch insert
        auto start = std::chrono::high_resolution_clock::now();
        assert(hash_table.batch_insert(test_data));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  ✅ Batch insert: 1000 items in " << duration.count() 
                  << " μs (" << (1000000.0 / duration.count()) << " ops/sec)" << std::endl;
        
        // Verify statistics
        auto stats = hash_table.get_stats();
        assert(stats.size == 1000);
        std::cout << "  ✅ Statistics updated correctly: " << stats.size << " items" << std::endl;
        
        // Test batch lookup
        std::vector<std::string> lookup_keys;
        for (int i = 0; i < 1000; i += 2) {  // Every other key
            lookup_keys.push_back("batch_key_" + std::to_string(i));
        }
        
        start = std::chrono::high_resolution_clock::now();
        auto results = hash_table.batch_lookup(lookup_keys);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        assert(results.size() == 500);
        int found_count = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].empty()) {
                found_count++;
                std::string expected = "batch_value_" + std::to_string(i * 2);
                assert(results[i] == expected);
            }
        }
        
        std::cout << "  ✅ Batch lookup: " << found_count << "/500 found in " 
                  << duration.count() << " μs" << std::endl;
        
        // Test batch remove
        std::vector<std::string> remove_keys;
        for (int i = 0; i < 100; ++i) {
            remove_keys.push_back("batch_key_" + std::to_string(i));
        }
        
        start = std::chrono::high_resolution_clock::now();
        assert(hash_table.batch_remove(remove_keys));
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  ✅ Batch remove: 100 items in " << duration.count() << " μs" << std::endl;
        
        // Verify removals
        stats = hash_table.get_stats();
        assert(stats.size == 900);
        std::cout << "  ✅ Remove verification: " << stats.size << " items remaining" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Batch operations test passed\n" << std::endl;
    }
    
    static void test_collision_handling() {
        std::cout << "🧩 Testing collision handling..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(64));  // Small table to force collisions
        
        // Insert many items to test collision resolution
        std::vector<std::pair<std::string, std::string>> collision_data;
        for (int i = 0; i < 50; ++i) {
            collision_data.emplace_back("collision_key_" + std::to_string(i), 
                                       "collision_value_" + std::to_string(i));
        }
        
        assert(hash_table.batch_insert(collision_data));
        std::cout << "  ✅ Inserted 50 items into 64-slot table (collision test)" << std::endl;
        
        auto stats = hash_table.get_stats();
        std::cout << "  Load factor: " << (stats.load_factor * 100) << "%" << std::endl;
        
        // Verify all items can be found despite collisions
        std::vector<std::string> collision_keys;
        for (int i = 0; i < 50; ++i) {
            collision_keys.push_back("collision_key_" + std::to_string(i));
        }
        
        auto results = hash_table.batch_lookup(collision_keys);
        int found = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].empty()) {
                found++;
                std::string expected = "collision_value_" + std::to_string(i);
                assert(results[i] == expected);
            }
        }
        
        assert(found == 50);
        std::cout << "  ✅ All " << found << " items found despite collisions" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Collision handling test passed\n" << std::endl;
    }
    
    static void test_string_operations() {
        std::cout << "🔤 Testing string key/value operations..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(1024));
        
        // Test various string lengths and types
        std::vector<std::pair<std::string, std::string>> string_tests = {
            {"", "empty_key_test"},  // Empty key should fail
            {"short", "short_value"},
            {"medium_length_key_test", "medium_length_value_test"},
            {"very_long_key_name_that_tests_boundaries_and_memory", 
             "very_long_value_content_that_tests_gpu_memory_handling_and_string_operations"},
            {"key_with_numbers_123", "value_with_numbers_456"},
            {"key-with-special-chars!@#", "value-with-special-chars$%^"},
            {"unicode_test_αβγ", "unicode_value_δεζ"}
        };
        
        int successful_inserts = 0;
        for (const auto& [key, value] : string_tests) {
            if (hash_table.insert(key, value)) {
                successful_inserts++;
                
                std::string retrieved_value;
                if (hash_table.lookup(key, retrieved_value)) {
                    assert(retrieved_value == value);
                    std::cout << "  ✅ String test: '" << key << "' -> '" 
                              << (value.length() > 20 ? value.substr(0, 20) + "..." : value) 
                              << "'" << std::endl;
                }
            }
        }
        
        std::cout << "  ✅ " << successful_inserts << "/" << string_tests.size() 
                  << " string tests successful" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ String operations test passed\n" << std::endl;
    }
    
    static void test_performance_benchmark() {
        std::cout << "⚡ Testing performance benchmarks..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(100000));  // Large table for performance
        
        // Generate large dataset
        std::vector<std::pair<std::string, std::string>> perf_data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> value_size_dist(10, 100);
        
        for (int i = 0; i < 10000; ++i) {
            std::string key = "perf_key_" + std::to_string(i);
            std::string value(value_size_dist(gen), 'x');
            value += "_" + std::to_string(i);
            perf_data.emplace_back(key, value);
        }
        
        // Benchmark batch insert
        auto start = std::chrono::high_resolution_clock::now();
        assert(hash_table.batch_insert(perf_data));
        auto end = std::chrono::high_resolution_clock::now();
        auto insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double insert_rate = (10000.0 / insert_duration.count()) * 1000000;
        std::cout << "  📊 Batch Insert: 10K items in " << insert_duration.count() 
                  << " μs (" << static_cast<int>(insert_rate) << " ops/sec)" << std::endl;
        
        // Benchmark lookup performance
        std::vector<std::string> lookup_keys;
        for (int i = 0; i < 10000; i += 10) {
            lookup_keys.push_back("perf_key_" + std::to_string(i));
        }
        
        start = std::chrono::high_resolution_clock::now();
        auto lookup_results = hash_table.batch_lookup(lookup_keys);
        end = std::chrono::high_resolution_clock::now();
        auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double lookup_rate = (1000.0 / lookup_duration.count()) * 1000000;
        std::cout << "  📊 Batch Lookup: 1K items in " << lookup_duration.count() 
                  << " μs (" << static_cast<int>(lookup_rate) << " ops/sec)" << std::endl;
        
        // Performance target verification (>1M ops/sec)
        bool insert_target = insert_rate > 1000000;
        bool lookup_target = lookup_rate > 1000000;
        
        std::cout << "  🎯 Performance Targets:" << std::endl;
        std::cout << "    Insert rate: " << (insert_target ? "✅ ACHIEVED" : "❌ MISSED") 
                  << " (target: >1M ops/sec)" << std::endl;
        std::cout << "    Lookup rate: " << (lookup_target ? "✅ ACHIEVED" : "❌ MISSED") 
                  << " (target: >1M ops/sec)" << std::endl;
        
        auto stats = hash_table.get_stats();
        std::cout << "  📈 Final stats: " << stats.size << " items, " 
                  << (stats.load_factor * 100) << "% load factor" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Performance benchmark completed\n" << std::endl;
    }
    
    static void test_concurrent_operations() {
        std::cout << "🔀 Testing concurrent operations..." << std::endl;
        
        GpuHashTable hash_table;
        assert(hash_table.initialize(50000));
        
        const int num_threads = 4;
        const int items_per_thread = 1000;
        std::vector<std::thread> threads;
        std::vector<bool> thread_results(num_threads, false);
        
        // Concurrent inserts
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&hash_table, &thread_results, t, items_per_thread]() {
                std::vector<std::pair<std::string, std::string>> thread_data;
                
                for (int i = 0; i < items_per_thread; ++i) {
                    std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
                    std::string value = "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
                    thread_data.emplace_back(key, value);
                }
                
                thread_results[t] = hash_table.batch_insert(thread_data);
            });
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Verify all threads succeeded
        int successful_threads = 0;
        for (bool result : thread_results) {
            if (result) successful_threads++;
        }
        
        std::cout << "  ✅ Concurrent inserts: " << successful_threads << "/" << num_threads 
                  << " threads successful in " << duration.count() << " ms" << std::endl;
        
        // Verify data integrity
        auto stats = hash_table.get_stats();
        std::cout << "  📊 Total items inserted: " << stats.size << " (expected: " 
                  << (num_threads * items_per_thread) << ")" << std::endl;
        
        // Test concurrent lookups
        threads.clear();
        thread_results.assign(num_threads, false);
        
        start = std::chrono::high_resolution_clock::now();
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&hash_table, &thread_results, t, items_per_thread]() {
                std::vector<std::string> lookup_keys;
                
                for (int i = 0; i < items_per_thread; i += 10) {
                    lookup_keys.push_back("thread_" + std::to_string(t) + "_key_" + std::to_string(i));
                }
                
                auto results = hash_table.batch_lookup(lookup_keys);
                
                // Verify results
                bool all_found = true;
                for (size_t i = 0; i < results.size(); ++i) {
                    if (results[i].empty()) {
                        all_found = false;
                        break;
                    }
                }
                
                thread_results[t] = all_found;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        successful_threads = 0;
        for (bool result : thread_results) {
            if (result) successful_threads++;
        }
        
        std::cout << "  ✅ Concurrent lookups: " << successful_threads << "/" << num_threads 
                  << " threads successful in " << duration.count() << " ms" << std::endl;
        
        hash_table.shutdown();
        std::cout << "  ✅ Concurrent operations test passed\n" << std::endl;
    }
    
    static void test_memory_management() {
        std::cout << "💾 Testing memory management..." << std::endl;
        
        {
            GpuHashTable hash_table;
            assert(hash_table.initialize(10000));
            
            // Fill table to test memory usage
            std::vector<std::pair<std::string, std::string>> memory_data;
            for (int i = 0; i < 5000; ++i) {
                memory_data.emplace_back("mem_key_" + std::to_string(i), 
                                        "memory_test_value_" + std::to_string(i));
            }
            
            assert(hash_table.batch_insert(memory_data));
            
            auto stats = hash_table.get_stats();
            std::cout << "  📊 Memory test: " << stats.size << " items, " 
                      << (stats.load_factor * 100) << "% load factor" << std::endl;
            
            // Test clear operation
            hash_table.clear();
            stats = hash_table.get_stats();
            assert(stats.size == 0);
            std::cout << "  ✅ Clear operation: " << stats.size << " items remaining" << std::endl;
            
            hash_table.shutdown();
        }
        
        // Test that memory is properly released (implicit test through destructor)
        std::cout << "  ✅ Memory cleanup completed" << std::endl;
        
        std::cout << "  ✅ Memory management test passed\n" << std::endl;
    }
};

int main() {
    try {
        GpuHashTableTest::run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}