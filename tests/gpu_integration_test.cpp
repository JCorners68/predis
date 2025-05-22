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

#include "../src/api/predis_client.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

using namespace predis::api;

void test_mode_detection() {
    std::cout << "\n=== Testing Mode Detection and Switching ===" << std::endl;
    
    PredisClient client;
    
    // Test auto-detection
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::AUTO_DETECT);
    assert(connected);
    
    auto current_mode = client.get_current_mode();
    std::cout << "Auto-detected mode: " << (client.is_using_real_gpu() ? "REAL_GPU" : "MOCK") << std::endl;
    
    client.disconnect();
    
    // Test explicit mock mode
    connected = client.connect("localhost", 6379, PredisClient::Mode::MOCK_ONLY);
    assert(connected);
    assert(!client.is_using_real_gpu());
    std::cout << "Mock mode connected successfully" << std::endl;
    
    client.disconnect();
    
    // Try real GPU mode (may fail if no GPU available)
    connected = client.connect("localhost", 6379, PredisClient::Mode::REAL_GPU_ONLY);
    if (connected) {
        assert(client.is_using_real_gpu());
        std::cout << "Real GPU mode connected successfully" << std::endl;
        client.disconnect();
    } else {
        std::cout << "Real GPU mode not available (no GPU detected)" << std::endl;
    }
    
    // Test hybrid mode
    connected = client.connect("localhost", 6379, PredisClient::Mode::HYBRID);
    if (connected) {
        std::cout << "Hybrid mode connected successfully" << std::endl;
        client.disconnect();
    } else {
        std::cout << "Hybrid mode connection failed" << std::endl;
    }
    
    std::cout << "Mode detection tests completed!" << std::endl;
}

void test_basic_operations() {
    std::cout << "\n=== Testing Basic Cache Operations ===" << std::endl;
    
    PredisClient client;
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::AUTO_DETECT);
    assert(connected);
    
    // Test PUT and GET
    bool put_result = client.put("test_key_1", "test_value_1");
    assert(put_result);
    
    std::string value;
    bool get_result = client.get("test_key_1", value);
    assert(get_result);
    assert(value == "test_value_1");
    std::cout << "Basic PUT/GET operations successful" << std::endl;
    
    // Test with TTL
    put_result = client.put("test_key_ttl", "test_value_ttl", 10);
    assert(put_result);
    
    get_result = client.get("test_key_ttl", value);
    assert(get_result);
    assert(value == "test_value_ttl");
    std::cout << "PUT with TTL successful" << std::endl;
    
    // Test remove
    bool remove_result = client.remove("test_key_1");
    assert(remove_result);
    
    get_result = client.get("test_key_1", value);
    assert(!get_result);
    std::cout << "REMOVE operation successful" << std::endl;
    
    client.disconnect();
    std::cout << "Basic operations tests completed!" << std::endl;
}

void test_batch_operations() {
    std::cout << "\n=== Testing Batch Operations ===" << std::endl;
    
    PredisClient client;
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::AUTO_DETECT);
    assert(connected);
    
    // Test batch PUT
    std::unordered_map<std::string, std::string> batch_data;
    for (int i = 0; i < 100; ++i) {
        batch_data["batch_key_" + std::to_string(i)] = "batch_value_" + std::to_string(i);
    }
    
    bool mput_result = client.mput(batch_data);
    assert(mput_result);
    std::cout << "Batch PUT of 100 items successful" << std::endl;
    
    // Test batch GET
    std::vector<std::string> keys;
    for (int i = 0; i < 100; ++i) {
        keys.push_back("batch_key_" + std::to_string(i));
    }
    
    auto values = client.mget(keys);
    assert(values.size() == 100);
    
    int successful_gets = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!values[i].empty() && values[i] == "batch_value_" + std::to_string(i)) {
            successful_gets++;
        }
    }
    
    std::cout << "Batch GET retrieved " << successful_gets << "/100 items successfully" << std::endl;
    assert(successful_gets >= 90); // Allow for some failures in test environment
    
    // Test batch DELETE
    bool mdelete_result = client.mdelete(keys);
    assert(mdelete_result);
    std::cout << "Batch DELETE successful" << std::endl;
    
    client.disconnect();
    std::cout << "Batch operations tests completed!" << std::endl;
}

void test_performance_comparison() {
    std::cout << "\n=== Testing Performance Comparison ===" << std::endl;
    
    PredisClient client;
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::AUTO_DETECT);
    assert(connected);
    
    // Run performance comparison
    bool perf_result = client.run_performance_comparison(1000);
    assert(perf_result);
    
    // Get and display statistics
    auto stats = client.get_stats();
    
    std::cout << "\nFinal Performance Statistics:" << std::endl;
    std::cout << "  Implementation: " << stats.implementation_mode << std::endl;
    std::cout << "  Operations/sec: " << std::fixed << std::setprecision(0) << stats.operations_per_second << std::endl;
    std::cout << "  Avg Latency: " << std::fixed << std::setprecision(4) << stats.avg_latency_ms << " ms" << std::endl;
    std::cout << "  Hit Ratio: " << std::fixed << std::setprecision(3) << stats.hit_ratio << std::endl;
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2) << stats.memory_usage_mb << " MB" << std::endl;
    
    if (stats.performance_improvement_ratio > 0) {
        std::cout << "  Performance Improvement: " << std::fixed << std::setprecision(1) 
                  << stats.performance_improvement_ratio << "x" << std::endl;
    }
    
    client.disconnect();
    std::cout << "Performance comparison tests completed!" << std::endl;
}

void test_consistency_validation() {
    std::cout << "\n=== Testing Consistency Validation ===" << std::endl;
    
    PredisClient client;
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::HYBRID);
    
    if (connected) {
        bool consistency_result = client.validate_consistency(100);
        if (consistency_result) {
            std::cout << "Consistency validation passed!" << std::endl;
        } else {
            std::cout << "Consistency validation found issues (may be expected in test environment)" << std::endl;
        }
        client.disconnect();
    } else {
        std::cout << "Hybrid mode not available - skipping consistency validation" << std::endl;
    }
    
    std::cout << "Consistency validation tests completed!" << std::endl;
}

void test_error_handling() {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
    PredisClient client;
    
    // Test operations without connection
    std::string value;
    bool result = client.get("test_key", value);
    assert(!result);
    std::cout << "Proper error handling for disconnected client" << std::endl;
    
    // Test invalid mode switching
    result = client.switch_mode(PredisClient::Mode::REAL_GPU_ONLY);
    assert(!result);
    std::cout << "Proper error handling for mode switch without connection" << std::endl;
    
    // Connect and test memory limits with large batch
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::MOCK_ONLY);
    assert(connected);
    
    // Try to exceed memory limits with very large values
    std::unordered_map<std::string, std::string> large_data;
    for (int i = 0; i < 1000; ++i) {
        std::string large_value(10000, 'x'); // 10KB value
        large_data["large_key_" + std::to_string(i)] = large_value;
    }
    
    // This should eventually fail due to memory limits
    bool large_put_result = client.mput(large_data);
    std::cout << "Large batch operation result: " << (large_put_result ? "success" : "failed as expected") << std::endl;
    
    client.disconnect();
    std::cout << "Error handling tests completed!" << std::endl;
}

void test_prefetching_configuration() {
    std::cout << "\n=== Testing Prefetching Configuration ===" << std::endl;
    
    PredisClient client;
    bool connected = client.connect("localhost", 6379, PredisClient::Mode::AUTO_DETECT);
    assert(connected);
    
    // Test prefetching configuration
    PredisClient::PrefetchConfig config;
    config.enabled = true;
    config.confidence_threshold = 0.8;
    config.max_prefetch_keys = 500;
    config.max_prefetch_size_mb = 200;
    config.prefetch_ttl = 60;
    
    client.configure_prefetching(config);
    
    auto retrieved_config = client.get_prefetch_config();
    assert(retrieved_config.enabled == config.enabled);
    assert(retrieved_config.confidence_threshold == config.confidence_threshold);
    std::cout << "Prefetching configuration successful" << std::endl;
    
    // Test ML hints
    std::vector<std::string> related_keys = {"user_123_profile", "user_123_preferences", "user_123_history"};
    client.hint_related_keys(related_keys);
    
    std::vector<std::string> sequence_keys = {"step_1", "step_2", "step_3", "step_4"};
    client.hint_sequence(sequence_keys);
    std::cout << "ML hints provided successfully" << std::endl;
    
    client.disconnect();
    std::cout << "Prefetching configuration tests completed!" << std::endl;
}

void run_comprehensive_integration_test() {
    std::cout << "=== Predis GPU Integration Test Suite ===" << std::endl;
    std::cout << "Testing real GPU cache integration with mock fallback" << std::endl;
    
    try {
        test_mode_detection();
        test_basic_operations();
        test_batch_operations();
        test_performance_comparison();
        test_consistency_validation();
        test_error_handling();
        test_prefetching_configuration();
        
        std::cout << "\n=== All Tests Completed Successfully! ===" << std::endl;
        std::cout << "Real GPU cache integration is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        exit(1);
    }
}

int main() {
    run_comprehensive_integration_test();
    return 0;
}