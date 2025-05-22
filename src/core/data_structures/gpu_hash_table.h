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

#include <string>
#include <memory>

namespace predis {
namespace core {

/**
 * @brief GPU-optimized hash table for key-value storage
 * 
 * Uses GPU-friendly data structures and algorithms for high-performance lookups
 */
class GpuHashTable {
public:
    enum class HashMethod {
        FNV1A,
        MURMUR3,
        CUCKOO
    };

    struct HashStats {
        size_t capacity = 0;
        size_t size = 0;
        size_t collisions = 0;
        double load_factor = 0.0;
        size_t max_probe_distance = 0;
    };

public:
    GpuHashTable();
    ~GpuHashTable();

    // Initialization
    bool initialize(size_t initial_capacity = 1024 * 1024, HashMethod method = HashMethod::FNV1A);
    void shutdown();

    // Hash table operations
    bool insert(const std::string& key, const std::string& value);
    bool lookup(const std::string& key, std::string& value);
    bool remove(const std::string& key);
    
    // Batch operations (GPU-parallel)
    bool batch_insert(const std::vector<std::pair<std::string, std::string>>& pairs);
    std::vector<std::string> batch_lookup(const std::vector<std::string>& keys);
    bool batch_remove(const std::vector<std::string>& keys);

    // Table management
    void resize(size_t new_capacity);
    void clear();
    HashStats get_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis