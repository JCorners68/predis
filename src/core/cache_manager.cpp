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

#include "cache_manager.h"
#include "memory_manager.h"
#include <iostream>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>

namespace predis {
namespace core {

struct CacheManager::Impl {
    bool initialized = false;
    std::unique_ptr<MemoryManager> memory_manager;
    std::unordered_map<std::string, void*> gpu_cache_map;  // key -> GPU pointer
    std::unordered_map<std::string, size_t> cache_sizes;   // key -> data size
};

CacheManager::CacheManager() : pImpl(std::make_unique<Impl>()) {
    pImpl->memory_manager = std::make_unique<MemoryManager>();
}

CacheManager::~CacheManager() = default;

bool CacheManager::initialize() {
    std::cout << "CacheManager::initialize() - initializing GPU memory and cache" << std::endl;
    
    // Initialize GPU memory manager
    if (!pImpl->memory_manager->initialize()) {
        std::cerr << "Failed to initialize GPU memory manager" << std::endl;
        return false;
    }
    
    // Create memory pool for cache entries (1MB blocks, 100 blocks = 100MB pool)
    if (!pImpl->memory_manager->create_pool(1024 * 1024, 100)) {
        std::cerr << "Failed to create GPU memory pool" << std::endl;
        return false;
    }
    
    auto stats = pImpl->memory_manager->get_stats();
    std::cout << "GPU Cache initialized with " << (stats.total_bytes / 1024 / 1024) 
              << " MB available memory" << std::endl;
    
    pImpl->initialized = true;
    return true;
}

void CacheManager::shutdown() {
    if (!pImpl->initialized) return;
    
    std::cout << "CacheManager::shutdown() - cleaning up GPU cache" << std::endl;
    
    // Free all cached data
    for (auto& [key, gpu_ptr] : pImpl->gpu_cache_map) {
        pImpl->memory_manager->deallocate(gpu_ptr);
    }
    pImpl->gpu_cache_map.clear();
    pImpl->cache_sizes.clear();
    
    // Shutdown memory manager
    pImpl->memory_manager->shutdown();
    
    pImpl->initialized = false;
}

bool CacheManager::get(const std::string& key, std::string& value) {
    if (!pImpl->initialized) {
        std::cerr << "CacheManager not initialized" << std::endl;
        return false;
    }
    
    auto it = pImpl->gpu_cache_map.find(key);
    if (it == pImpl->gpu_cache_map.end()) {
        std::cout << "CacheManager::get(" << key << ") - key not found" << std::endl;
        return false;  // Key not found
    }
    
    void* gpu_ptr = it->second;
    size_t data_size = pImpl->cache_sizes[key];
    
    // Copy data from GPU to CPU
    value.resize(data_size);
    cudaError_t error = cudaMemcpy(value.data(), gpu_ptr, data_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data from GPU: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "CacheManager::get(" << key << ") - retrieved " << data_size << " bytes from GPU" << std::endl;
    return true;
}

bool CacheManager::put(const std::string& key, const std::string& value) {
    if (!pImpl->initialized) {
        std::cerr << "CacheManager not initialized" << std::endl;
        return false;
    }
    
    // Remove existing entry if exists
    remove(key);
    
    size_t data_size = value.size();
    
    // Allocate GPU memory
    void* gpu_ptr = pImpl->memory_manager->allocate(data_size);
    if (!gpu_ptr) {
        std::cerr << "Failed to allocate GPU memory for key: " << key << std::endl;
        return false;
    }
    
    // Copy data from CPU to GPU
    cudaError_t error = cudaMemcpy(gpu_ptr, value.data(), data_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data to GPU: " << cudaGetErrorString(error) << std::endl;
        pImpl->memory_manager->deallocate(gpu_ptr);
        return false;
    }
    
    // Store in cache map
    pImpl->gpu_cache_map[key] = gpu_ptr;
    pImpl->cache_sizes[key] = data_size;
    
    std::cout << "CacheManager::put(" << key << ") - stored " << data_size << " bytes on GPU" << std::endl;
    return true;
}

bool CacheManager::remove(const std::string& key) {
    if (!pImpl->initialized) {
        std::cerr << "CacheManager not initialized" << std::endl;
        return false;
    }
    
    auto it = pImpl->gpu_cache_map.find(key);
    if (it == pImpl->gpu_cache_map.end()) {
        std::cout << "CacheManager::remove(" << key << ") - key not found" << std::endl;
        return false;  // Key not found
    }
    
    // Free GPU memory
    pImpl->memory_manager->deallocate(it->second);
    
    // Remove from maps
    pImpl->gpu_cache_map.erase(it);
    pImpl->cache_sizes.erase(key);
    
    std::cout << "CacheManager::remove(" << key << ") - removed from GPU cache" << std::endl;
    return true;
}

} // namespace core
} // namespace predis