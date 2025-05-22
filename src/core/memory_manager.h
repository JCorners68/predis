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

#pragma once

#include <cstddef>
#include <memory>

namespace predis {
namespace core {

/**
 * @brief GPU VRAM memory manager
 * 
 * Manages allocation and deallocation of GPU memory for cache data
 */
class MemoryManager {
public:
    struct MemoryStats {
        size_t total_bytes = 0;
        size_t allocated_bytes = 0;
        size_t free_bytes = 0;
        double fragmentation_ratio = 0.0;
        size_t allocation_count = 0;
    };

public:
    MemoryManager();
    ~MemoryManager();

    // Initialization
    bool initialize(size_t max_memory_bytes = 0);  // 0 = auto-detect 80% of VRAM
    void shutdown();

    // Memory allocation
    void* allocate(size_t size);
    void deallocate(void* ptr);
    
    // Memory pool management
    bool create_pool(size_t block_size, size_t num_blocks);
    void* allocate_from_pool(size_t size);
    void deallocate_to_pool(void* ptr);

    // Statistics and monitoring
    MemoryStats get_stats() const;
    bool is_out_of_memory() const;
    void defragment();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis