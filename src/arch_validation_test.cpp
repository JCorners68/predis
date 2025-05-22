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

#include "api/predis_client.h"
#include <iostream>
#include <chrono>
#include <string>

using namespace predis::api;

/**
 * Basic architecture validation test
 * Tests end-to-end data flow from client API to cache core
 */
int main() {
    std::cout << "=== Predis Architecture Validation Test ===" << std::endl;
    
    // Create client instance
    PredisClient client;
    
    // Test 1: Connection
    std::cout << "\n1. Testing client connection..." << std::endl;
    if (!client.connect("localhost", 6379)) {
        std::cerr << "Failed to connect to Predis server" << std::endl;
        return 1;
    }
    std::cout << "✅ Client connected successfully" << std::endl;
    
    // Test 2: Basic PUT operation
    std::cout << "\n2. Testing PUT operation..." << std::endl;
    std::string test_key = "test_key_1";
    std::string test_value = "Hello, GPU Cache World!";
    
    auto start = std::chrono::high_resolution_clock::now();
    bool put_result = client.put(test_key, test_value);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!put_result) {
        std::cerr << "Failed to PUT data" << std::endl;
        return 1;
    }
    
    auto put_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✅ PUT operation completed in " << put_duration.count() << " µs" << std::endl;
    
    // Test 3: Basic GET operation
    std::cout << "\n3. Testing GET operation..." << std::endl;
    std::string retrieved_value;
    
    start = std::chrono::high_resolution_clock::now();
    bool get_result = client.get(test_key, retrieved_value);
    end = std::chrono::high_resolution_clock::now();
    
    if (!get_result) {
        std::cerr << "Failed to GET data" << std::endl;
        return 1;
    }
    
    auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✅ GET operation completed in " << get_duration.count() << " µs" << std::endl;
    
    // Test 4: Data integrity
    std::cout << "\n4. Testing data integrity..." << std::endl;
    if (retrieved_value != test_value) {
        std::cerr << "Data integrity check failed!" << std::endl;
        std::cerr << "Expected: '" << test_value << "'" << std::endl;
        std::cerr << "Got:      '" << retrieved_value << "'" << std::endl;
        return 1;
    }
    std::cout << "✅ Data integrity verified" << std::endl;
    
    // Test 5: Multiple operations
    std::cout << "\n5. Testing multiple operations..." << std::endl;
    const int num_ops = 100;
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_ops; ++i) {
        std::string key = "bulk_key_" + std::to_string(i);
        std::string value = "bulk_value_" + std::to_string(i);
        
        if (!client.put(key, value)) {
            std::cerr << "Failed bulk PUT operation " << i << std::endl;
            return 1;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto bulk_put_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double put_ops_per_sec = num_ops * 1000.0 / bulk_put_duration.count();
    std::cout << "✅ Bulk PUT: " << num_ops << " ops in " << bulk_put_duration.count() 
              << " ms (" << static_cast<int>(put_ops_per_sec) << " ops/sec)" << std::endl;
    
    // Test bulk GET
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_ops; ++i) {
        std::string key = "bulk_key_" + std::to_string(i);
        std::string value;
        
        if (!client.get(key, value)) {
            std::cerr << "Failed bulk GET operation " << i << std::endl;
            return 1;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto bulk_get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double get_ops_per_sec = num_ops * 1000.0 / bulk_get_duration.count();
    std::cout << "✅ Bulk GET: " << num_ops << " ops in " << bulk_get_duration.count() 
              << " ms (" << static_cast<int>(get_ops_per_sec) << " ops/sec)" << std::endl;
    
    // Test 6: Cache statistics
    std::cout << "\n6. Testing cache statistics..." << std::endl;
    auto stats = client.get_stats();
    std::cout << "Cache Statistics:" << std::endl;
    std::cout << "  Total keys: " << stats.total_keys << std::endl;
    std::cout << "  Hit rate: " << (stats.hit_ratio * 100) << "%" << std::endl;
    std::cout << "  Memory usage: " << stats.memory_usage_mb << " MB" << std::endl;
    
    // Test 7: Cleanup
    std::cout << "\n7. Testing cleanup..." << std::endl;
    if (!client.remove(test_key)) {
        std::cerr << "Failed to remove test key" << std::endl;
        return 1;
    }
    std::cout << "✅ Cleanup completed" << std::endl;
    
    // Disconnect
    client.disconnect();
    std::cout << "\n=== Architecture Validation Completed Successfully ===" << std::endl;
    
    return 0;
}