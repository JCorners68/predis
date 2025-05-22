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

#include "simple_cache_manager.h"
#include <iostream>
#include <unordered_map>
#include <string>
#include <mutex>

namespace predis {
namespace core {

struct SimpleCacheManager::Impl {
    bool initialized = false;
    std::unordered_map<std::string, std::string> cache_data;
    mutable std::mutex cache_mutex;
    size_t total_memory_usage = 0;
};

SimpleCacheManager::SimpleCacheManager() : pImpl(std::make_unique<Impl>()) {
}

SimpleCacheManager::~SimpleCacheManager() = default;

bool SimpleCacheManager::initialize() {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    std::cout << "SimpleCacheManager::initialize() - initializing in-memory cache" << std::endl;
    pImpl->initialized = true;
    return true;
}

void SimpleCacheManager::shutdown() {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    if (!pImpl->initialized) return;
    
    std::cout << "SimpleCacheManager::shutdown() - clearing cache data" << std::endl;
    pImpl->cache_data.clear();
    pImpl->total_memory_usage = 0;
    pImpl->initialized = false;
}

bool SimpleCacheManager::get(const std::string& key, std::string& value) {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    
    if (!pImpl->initialized) {
        std::cerr << "SimpleCacheManager not initialized" << std::endl;
        return false;
    }
    
    auto it = pImpl->cache_data.find(key);
    if (it == pImpl->cache_data.end()) {
        std::cout << "SimpleCacheManager::get(" << key << ") - key not found" << std::endl;
        return false;
    }
    
    value = it->second;
    std::cout << "SimpleCacheManager::get(" << key << ") - retrieved " << value.size() << " bytes" << std::endl;
    return true;
}

bool SimpleCacheManager::put(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    
    if (!pImpl->initialized) {
        std::cerr << "SimpleCacheManager not initialized" << std::endl;
        return false;
    }
    
    // Remove old value if exists
    auto it = pImpl->cache_data.find(key);
    if (it != pImpl->cache_data.end()) {
        pImpl->total_memory_usage -= it->second.size();
    }
    
    // Add new value
    pImpl->cache_data[key] = value;
    pImpl->total_memory_usage += value.size() + key.size();
    
    std::cout << "SimpleCacheManager::put(" << key << ") - stored " << value.size() << " bytes" << std::endl;
    return true;
}

bool SimpleCacheManager::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    
    if (!pImpl->initialized) {
        std::cerr << "SimpleCacheManager not initialized" << std::endl;
        return false;
    }
    
    auto it = pImpl->cache_data.find(key);
    if (it == pImpl->cache_data.end()) {
        std::cout << "SimpleCacheManager::remove(" << key << ") - key not found" << std::endl;
        return false;
    }
    
    pImpl->total_memory_usage -= it->second.size() + key.size();
    pImpl->cache_data.erase(it);
    
    std::cout << "SimpleCacheManager::remove(" << key << ") - removed from cache" << std::endl;
    return true;
}

size_t SimpleCacheManager::size() const {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    return pImpl->cache_data.size();
}

size_t SimpleCacheManager::memory_usage() const {
    std::lock_guard<std::mutex> lock(pImpl->cache_mutex);
    return pImpl->total_memory_usage;
}

} // namespace core
} // namespace predis