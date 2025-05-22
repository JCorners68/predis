/*
 * Simple GPU Memory Manager Test
 * Tests core functionality without heavy pool operations
 */

#include "../src/core/memory_manager.h"
#include <iostream>
#include <cassert>

using namespace predis::core;

int main() {
    std::cout << "🧪 Simple GPU Memory Manager Test\n" << std::endl;
    
    MemoryManager manager;
    
    // Test 1: Initialization
    std::cout << "1️⃣ Testing initialization..." << std::endl;
    bool init_success = manager.initialize(100 * 1024 * 1024);  // 100MB limit for testing
    if (!init_success) {
        std::cout << "❌ Initialization failed - this may be expected if no GPU is available" << std::endl;
        return 0;  // Exit gracefully if no GPU
    }
    std::cout << "✅ Initialization successful" << std::endl;
    
    auto stats = manager.get_stats();
    std::cout << "   Available memory: " << (stats.total_bytes / 1024 / 1024) << " MB" << std::endl;
    
    // Test 2: Basic allocation
    std::cout << "\n2️⃣ Testing basic allocation..." << std::endl;
    void* ptr1 = manager.allocate(1024);
    assert(ptr1 != nullptr);
    std::cout << "✅ Allocated 1KB: " << ptr1 << std::endl;
    
    void* ptr2 = manager.allocate(4096);
    assert(ptr2 != nullptr);
    std::cout << "✅ Allocated 4KB: " << ptr2 << std::endl;
    
    // Test 3: Statistics
    std::cout << "\n3️⃣ Testing statistics..." << std::endl;
    stats = manager.get_stats();
    std::cout << "   Allocated: " << stats.allocated_bytes << " bytes" << std::endl;
    std::cout << "   Active allocations: " << stats.allocation_count << std::endl;
    assert(stats.allocation_count == 2);
    assert(stats.allocated_bytes >= 5120);
    std::cout << "✅ Statistics correct" << std::endl;
    
    // Test 4: Memory pool creation
    std::cout << "\n4️⃣ Testing small memory pool..." << std::endl;
    bool pool_created = manager.create_pool(256, 100);  // Small pool for testing
    if (pool_created) {
        std::cout << "✅ Created memory pool (256B x 100 blocks)" << std::endl;
        
        void* pool_ptr = manager.allocate_from_pool(128);
        if (pool_ptr) {
            std::cout << "✅ Pool allocation successful: " << pool_ptr << std::endl;
            manager.deallocate_to_pool(pool_ptr);
            std::cout << "✅ Pool deallocation successful" << std::endl;
        }
    }
    
    // Test 5: Deallocation
    std::cout << "\n5️⃣ Testing deallocation..." << std::endl;
    manager.deallocate(ptr1);
    manager.deallocate(ptr2);
    
    stats = manager.get_stats();
    std::cout << "   Remaining allocations: " << stats.allocation_count << std::endl;
    assert(stats.allocation_count == 0);
    std::cout << "✅ Deallocation successful" << std::endl;
    
    // Test 6: Memory leak detection
    std::cout << "\n6️⃣ Testing memory leak detection..." << std::endl;
    bool has_leaks = manager.has_memory_leaks();
    std::cout << "   Memory leaks detected: " << (has_leaks ? "Yes" : "No") << std::endl;
    assert(!has_leaks);
    std::cout << "✅ No memory leaks detected" << std::endl;
    
    manager.shutdown();
    std::cout << "\n🎉 All tests passed!" << std::endl;
    
    return 0;
}