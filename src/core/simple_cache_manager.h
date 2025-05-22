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
 * @brief Simple cache management class without GPU dependencies
 * 
 * Used for architecture validation before adding GPU functionality
 */
class SimpleCacheManager {
public:
    SimpleCacheManager();
    ~SimpleCacheManager();
    
    // Basic cache operations
    bool get(const std::string& key, std::string& value);
    bool put(const std::string& key, const std::string& value);
    bool remove(const std::string& key);
    
    // Statistics
    size_t size() const;
    size_t memory_usage() const;
    
    // Initialization and cleanup
    bool initialize();
    void shutdown();
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace predis