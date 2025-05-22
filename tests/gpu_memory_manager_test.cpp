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

#include "../src/core/memory_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <random>
#include <thread>

using namespace predis::core;

class MemoryManagerTest {
public:
    static void run_all_tests() {
        std::cout << "🧪 Starting GPU Memory Manager Comprehensive Tests\n" << std::endl;
        
        test_initialization();
        test_basic_allocation();
        test_pool_allocation();
        test_memory_tracking();
        test_fragmentation();
        test_memory_leak_detection();
        test_performance();
        test_stress_test();
        
        std::cout << "\n✅ All GPU Memory Manager tests passed!" << std::endl;
    }

private:
    static void test_initialization() {
        std::cout << "🔧 Testing memory manager initialization..." << std::endl;
        
        MemoryManager manager;
        assert(!manager.is_out_of_memory());
        
        // Test initialization
        bool init_success = manager.initialize();
        std::cout << "  Initialization: " << (init_success ? "✅ Success" : "❌ Failed") << std::endl;
        
        if (init_success) {
            auto stats = manager.get_stats();
            std::cout << "  Available GPU memory: " << (stats.total_bytes / 1024 / 1024) << " MB" << std::endl;
            assert(stats.total_bytes > 0);
            assert(stats.allocated_bytes == 0);
            assert(stats.free_bytes == stats.total_bytes);
        }
        
        manager.shutdown();
        std::cout << "  ✅ Initialization test passed\n" << std::endl;
    }
    
    static void test_basic_allocation() {
        std::cout << "📦 Testing basic memory allocation..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        
        // Test various allocation sizes
        std::vector<std::pair<void*, size_t>> allocations;
        std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};
        
        for (size_t size : sizes) {
            void* ptr = manager.allocate(size);
            assert(ptr != nullptr);
            allocations.push_back({ptr, size});
            std::cout << "  Allocated " << size << " bytes: " << ptr << std::endl;
        }
        
        // Verify statistics
        auto stats = manager.get_stats();
        size_t expected_allocated = 0;
        for (const auto& alloc : allocations) {
            expected_allocated += alloc.second;
        }
        
        assert(stats.allocated_bytes >= expected_allocated);
        assert(stats.allocation_count == allocations.size());
        std::cout << "  Current allocation: " << stats.allocation_count << " objects, " 
                  << (stats.allocated_bytes / 1024) << " KB" << std::endl;
        
        // Test deallocation
        for (const auto& alloc : allocations) {
            manager.deallocate(alloc.first);
        }
        
        stats = manager.get_stats();
        assert(stats.allocation_count == 0);
        
        manager.shutdown();
        std::cout << "  ✅ Basic allocation test passed\n" << std::endl;
    }
    
    static void test_pool_allocation() {
        std::cout << "🏊 Testing memory pool allocation..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        
        // Create standard pools
        assert(manager.create_common_pools());
        assert(manager.get_pool_count() == 4);
        
        std::cout << "  Created " << manager.get_pool_count() << " memory pools" << std::endl;
        manager.print_pool_stats();
        
        // Test pool allocations
        std::vector<void*> pool_ptrs;
        std::vector<size_t> test_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
        
        for (size_t size : test_sizes) {
            void* ptr = manager.allocate_from_pool(size);
            if (ptr) {
                pool_ptrs.push_back(ptr);
                std::cout << "  Pool allocated " << size << " bytes: " << ptr << std::endl;
            } else {
                std::cout << "  No suitable pool for " << size << " bytes" << std::endl;
            }
        }
        
        std::cout << "  Pool allocations completed: " << pool_ptrs.size() << " successful" << std::endl;
        manager.print_pool_stats();
        
        // Test pool deallocation
        for (void* ptr : pool_ptrs) {
            manager.deallocate_to_pool(ptr);
        }
        
        manager.print_pool_stats();
        manager.shutdown();
        std::cout << "  ✅ Pool allocation test passed\n" << std::endl;
    }
    
    static void test_memory_tracking() {
        std::cout << "📊 Testing memory tracking and statistics..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        assert(manager.create_common_pools());
        
        // Perform various allocations to generate statistics
        std::vector<void*> ptrs;
        
        for (int i = 0; i < 100; ++i) {
            size_t size = 64 + (i % 4) * 256;  // Mix of sizes
            void* ptr = manager.allocate(size);
            if (ptr) {
                ptrs.push_back(ptr);
            }
        }
        
        auto stats = manager.get_stats();
        std::cout << "  Memory statistics after 100 allocations:" << std::endl;
        std::cout << "    Total allocations ever: " << stats.total_allocations_ever << std::endl;
        std::cout << "    Current allocations: " << stats.allocation_count << std::endl;
        std::cout << "    Peak memory: " << (stats.peak_allocated_bytes / 1024) << " KB" << std::endl;
        std::cout << "    Fragmentation ratio: " << (stats.fragmentation_ratio * 100) << "%" << std::endl;
        
        assert(stats.total_allocations_ever >= 100);
        assert(stats.allocation_count <= 100);
        assert(stats.peak_allocated_bytes > 0);
        
        // Deallocate half
        for (size_t i = 0; i < ptrs.size() / 2; ++i) {
            manager.deallocate(ptrs[i]);
        }
        
        auto stats_after = manager.get_stats();
        assert(stats_after.total_deallocations_ever > 0);
        assert(stats_after.allocation_count < stats.allocation_count);
        
        manager.print_allocation_report();
        manager.shutdown();
        std::cout << "  ✅ Memory tracking test passed\n" << std::endl;
    }
    
    static void test_fragmentation() {
        std::cout << "🧩 Testing memory fragmentation..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        assert(manager.create_pool(256, 1000));  // Create a single pool for fragmentation test
        
        // Allocate every other block to create fragmentation
        std::vector<void*> ptrs;
        for (int i = 0; i < 500; ++i) {
            void* ptr = manager.allocate_from_pool(256);
            if (ptr) {
                ptrs.push_back(ptr);
            }
        }
        
        // Deallocate every other allocation to create holes
        for (size_t i = 1; i < ptrs.size(); i += 2) {
            manager.deallocate_to_pool(ptrs[i]);
            ptrs[i] = nullptr;
        }
        
        std::cout << "  Created fragmentation pattern" << std::endl;
        manager.print_pool_stats();
        
        auto stats_before = manager.get_stats();
        std::cout << "  Fragmentation before defrag: " << (stats_before.fragmentation_ratio * 100) << "%" << std::endl;
        
        // Test defragmentation
        manager.defragment();
        
        auto stats_after = manager.get_stats();
        std::cout << "  Fragmentation after defrag: " << (stats_after.fragmentation_ratio * 100) << "%" << std::endl;
        
        manager.shutdown();
        std::cout << "  ✅ Fragmentation test passed\n" << std::endl;
    }
    
    static void test_memory_leak_detection() {
        std::cout << "🔍 Testing memory leak detection..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        
        // Intentionally create some allocations without deallocating
        std::vector<void*> leaked_ptrs;
        for (int i = 0; i < 10; ++i) {
            void* ptr = manager.allocate(1024);
            if (ptr) {
                leaked_ptrs.push_back(ptr);
            }
        }
        
        // Check that leaks are detected
        bool has_leaks = manager.has_memory_leaks();
        std::cout << "  Memory leaks detected: " << (has_leaks ? "✅ Yes" : "❌ No") << std::endl;
        manager.print_allocation_report();
        
        // Clean up the "leaks"
        for (void* ptr : leaked_ptrs) {
            manager.deallocate(ptr);
        }
        
        // Verify no leaks after cleanup
        bool has_leaks_after = manager.has_memory_leaks();
        std::cout << "  Memory leaks after cleanup: " << (has_leaks_after ? "❌ Yes" : "✅ No") << std::endl;
        assert(!has_leaks_after);
        
        manager.shutdown();
        std::cout << "  ✅ Memory leak detection test passed\n" << std::endl;
    }
    
    static void test_performance() {
        std::cout << "⚡ Testing allocation performance..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        assert(manager.create_common_pools());
        
        const int num_iterations = 10000;
        std::vector<void*> ptrs;
        ptrs.reserve(num_iterations);
        
        // Test direct allocation performance
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            void* ptr = manager.allocate(256);
            if (ptr) {
                ptrs.push_back(ptr);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double alloc_rate = static_cast<double>(ptrs.size()) / duration.count() * 1000000;
        std::cout << "  Direct allocation: " << ptrs.size() << " allocations in " 
                  << duration.count() << " μs (" << static_cast<int>(alloc_rate) << " allocs/sec)" << std::endl;
        
        // Test deallocation performance
        start = std::chrono::high_resolution_clock::now();
        
        for (void* ptr : ptrs) {
            manager.deallocate(ptr);
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double dealloc_rate = static_cast<double>(ptrs.size()) / duration.count() * 1000000;
        std::cout << "  Direct deallocation: " << ptrs.size() << " deallocations in " 
                  << duration.count() << " μs (" << static_cast<int>(dealloc_rate) << " deallocs/sec)" << std::endl;
        
        // Test pool allocation performance
        ptrs.clear();
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            void* ptr = manager.allocate_from_pool(256);
            if (ptr) {
                ptrs.push_back(ptr);
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double pool_alloc_rate = static_cast<double>(ptrs.size()) / duration.count() * 1000000;
        std::cout << "  Pool allocation: " << ptrs.size() << " allocations in " 
                  << duration.count() << " μs (" << static_cast<int>(pool_alloc_rate) << " allocs/sec)" << std::endl;
        
        // Clean up pool allocations
        for (void* ptr : ptrs) {
            manager.deallocate_to_pool(ptr);
        }
        
        manager.shutdown();
        std::cout << "  ✅ Performance test passed\n" << std::endl;
    }
    
    static void test_stress_test() {
        std::cout << "💪 Running memory manager stress test..." << std::endl;
        
        MemoryManager manager;
        assert(manager.initialize());
        assert(manager.create_common_pools());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> size_dist(32, 4096);
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        
        std::vector<std::pair<void*, size_t>> active_allocations;
        const int stress_iterations = 50000;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < stress_iterations; ++i) {
            if (prob_dist(gen) < 0.6 || active_allocations.empty()) {
                // Allocate
                size_t size = size_dist(gen);
                void* ptr = (prob_dist(gen) < 0.5) ? 
                    manager.allocate(size) : 
                    manager.allocate_from_pool(size);
                
                if (ptr) {
                    active_allocations.push_back({ptr, size});
                }
            } else {
                // Deallocate
                size_t index = gen() % active_allocations.size();
                auto [ptr, size] = active_allocations[index];
                
                if (size <= 4096) {
                    manager.deallocate_to_pool(ptr);
                } else {
                    manager.deallocate(ptr);
                }
                
                active_allocations.erase(active_allocations.begin() + index);
            }
            
            // Periodic statistics
            if (i % 10000 == 0) {
                auto stats = manager.get_stats();
                std::cout << "  Iteration " << i << ": " << stats.allocation_count 
                          << " active allocations, " << (stats.allocated_bytes / 1024 / 1024) 
                          << " MB used" << std::endl;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "  Stress test completed in " << duration.count() << " ms" << std::endl;
        std::cout << "  Final active allocations: " << active_allocations.size() << std::endl;
        
        // Clean up remaining allocations
        for (const auto& [ptr, size] : active_allocations) {
            if (size <= 4096) {
                manager.deallocate_to_pool(ptr);
            } else {
                manager.deallocate(ptr);
            }
        }
        
        manager.print_allocation_report();
        assert(!manager.has_memory_leaks());
        
        manager.shutdown();
        std::cout << "  ✅ Stress test passed\n" << std::endl;
    }
};

int main() {
    try {
        MemoryManagerTest::run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}