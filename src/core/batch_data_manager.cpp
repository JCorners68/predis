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

#include "advanced_batch_processor.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace predis {
namespace core {

// BatchDataManager::PackedBatchData destructor
BatchDataManager::PackedBatchData::~PackedBatchData() {
    // Free host memory (pinned)
    if (h_keys_data) cudaFreeHost(h_keys_data);
    if (h_values_data) cudaFreeHost(h_values_data);
    if (h_key_offsets) cudaFreeHost(h_key_offsets);
    if (h_value_offsets) cudaFreeHost(h_value_offsets);
    if (h_key_lengths) cudaFreeHost(h_key_lengths);
    if (h_value_lengths) cudaFreeHost(h_value_lengths);
    
    // GPU memory will be freed by memory manager
    // (actual implementation would track and free these properly)
}

BatchDataManager::BatchDataManager(MemoryManager* memory_manager) 
    : memory_manager_(memory_manager) {
    
    // Initialize memory pools for common sizes
    constexpr size_t common_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576}; // 1KB to 1MB
    
    for (size_t size : common_sizes) {
        host_memory_pools_[size] = MemoryPool{std::vector<void*>(), size, 8}; // Max 8 buffers per size
        device_memory_pools_[size] = MemoryPool{std::vector<void*>(), size, 8};
    }
}

BatchDataManager::~BatchDataManager() {
    // Clean up all pooled memory
    for (auto& [size, pool] : host_memory_pools_) {
        for (void* ptr : pool.available_buffers) {
            cudaFreeHost(ptr);
        }
    }
    
    for (auto& [size, pool] : device_memory_pools_) {
        for (void* ptr : pool.available_buffers) {
            if (memory_manager_) {
                memory_manager_->deallocate(ptr);
            }
        }
    }
}

std::unique_ptr<BatchDataManager::PackedBatchData> 
BatchDataManager::pack_keys_for_gpu(const std::vector<std::string>& keys) {
    
    auto packed = std::make_unique<PackedBatchData>();
    packed->batch_size = keys.size();
    
    if (keys.empty()) {
        return packed;
    }
    
    // Calculate total memory requirements
    size_t total_keys_size = 0;
    size_t max_key_length = 0;
    
    for (const auto& key : keys) {
        total_keys_size += key.size();
        max_key_length = std::max(max_key_length, key.size());
    }
    
    // Add padding for alignment
    total_keys_size += keys.size() * 8; // 8-byte alignment padding
    packed->keys_data_size = total_keys_size;
    
    // Allocate host memory (pinned for fast transfer)
    cudaError_t error;
    
    error = cudaHostAlloc(&packed->h_keys_data, total_keys_size, cudaHostAllocPortable);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for keys: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    
    error = cudaHostAlloc(&packed->h_key_offsets, keys.size() * sizeof(size_t), cudaHostAllocPortable);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for key offsets: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    
    error = cudaHostAlloc(&packed->h_key_lengths, keys.size() * sizeof(uint32_t), cudaHostAllocPortable);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for key lengths: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    
    // Allocate GPU memory
    packed->d_keys_data = static_cast<char*>(memory_manager_->allocate(total_keys_size));
    packed->d_key_offsets = static_cast<size_t*>(memory_manager_->allocate(keys.size() * sizeof(size_t)));
    packed->d_key_lengths = static_cast<uint32_t*>(memory_manager_->allocate(keys.size() * sizeof(uint32_t)));
    packed->d_result_flags = static_cast<bool*>(memory_manager_->allocate(keys.size() * sizeof(bool)));
    
    if (!packed->d_keys_data || !packed->d_key_offsets || !packed->d_key_lengths || !packed->d_result_flags) {
        std::cerr << "Failed to allocate GPU memory for batch data" << std::endl;
        return nullptr;
    }
    
    // Pack key data with optimal memory layout
    char* write_ptr = packed->h_keys_data;
    size_t current_offset = 0;
    
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        
        // Store offset and length
        packed->h_key_offsets[i] = current_offset;
        packed->h_key_lengths[i] = static_cast<uint32_t>(key.size());
        
        // Copy key data
        std::memcpy(write_ptr + current_offset, key.c_str(), key.size());
        current_offset += key.size();
        
        // Add alignment padding
        size_t padding = (8 - (current_offset % 8)) % 8;
        current_offset += padding;
    }
    
    return packed;
}

std::unique_ptr<BatchDataManager::PackedBatchData> 
BatchDataManager::pack_key_values_for_gpu(
    const std::vector<std::pair<std::string, std::string>>& key_value_pairs) {
    
    auto packed = std::make_unique<PackedBatchData>();
    packed->batch_size = key_value_pairs.size();
    
    if (key_value_pairs.empty()) {
        return packed;
    }
    
    // Calculate total memory requirements
    size_t total_keys_size = 0;
    size_t total_values_size = 0;
    
    for (const auto& [key, value] : key_value_pairs) {
        total_keys_size += key.size() + 8; // 8-byte alignment
        total_values_size += value.size() + 8;
    }
    
    packed->keys_data_size = total_keys_size;
    packed->values_data_size = total_values_size;
    
    // Allocate host memory (pinned)
    cudaError_t error;
    
    error = cudaHostAlloc(&packed->h_keys_data, total_keys_size, cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    error = cudaHostAlloc(&packed->h_values_data, total_values_size, cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    error = cudaHostAlloc(&packed->h_key_offsets, key_value_pairs.size() * sizeof(size_t), cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    error = cudaHostAlloc(&packed->h_value_offsets, key_value_pairs.size() * sizeof(size_t), cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    error = cudaHostAlloc(&packed->h_key_lengths, key_value_pairs.size() * sizeof(uint32_t), cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    error = cudaHostAlloc(&packed->h_value_lengths, key_value_pairs.size() * sizeof(uint32_t), cudaHostAllocPortable);
    if (error != cudaSuccess) return nullptr;
    
    // Allocate GPU memory
    packed->d_keys_data = static_cast<char*>(memory_manager_->allocate(total_keys_size));
    packed->d_values_data = static_cast<char*>(memory_manager_->allocate(total_values_size));
    packed->d_key_offsets = static_cast<size_t*>(memory_manager_->allocate(key_value_pairs.size() * sizeof(size_t)));
    packed->d_value_offsets = static_cast<size_t*>(memory_manager_->allocate(key_value_pairs.size() * sizeof(size_t)));
    packed->d_key_lengths = static_cast<uint32_t*>(memory_manager_->allocate(key_value_pairs.size() * sizeof(uint32_t)));
    packed->d_value_lengths = static_cast<uint32_t*>(memory_manager_->allocate(key_value_pairs.size() * sizeof(uint32_t)));
    packed->d_result_flags = static_cast<bool*>(memory_manager_->allocate(key_value_pairs.size() * sizeof(bool)));
    
    if (!packed->d_keys_data || !packed->d_values_data || !packed->d_key_offsets || 
        !packed->d_value_offsets || !packed->d_key_lengths || !packed->d_value_lengths || 
        !packed->d_result_flags) {
        return nullptr;
    }
    
    // Pack data with optimal memory layout
    char* keys_write_ptr = packed->h_keys_data;
    char* values_write_ptr = packed->h_values_data;
    size_t key_offset = 0, value_offset = 0;
    
    for (size_t i = 0; i < key_value_pairs.size(); ++i) {
        const auto& [key, value] = key_value_pairs[i];
        
        // Store key
        packed->h_key_offsets[i] = key_offset;
        packed->h_key_lengths[i] = static_cast<uint32_t>(key.size());
        std::memcpy(keys_write_ptr + key_offset, key.c_str(), key.size());
        key_offset += key.size();
        key_offset += (8 - (key_offset % 8)) % 8; // Alignment
        
        // Store value
        packed->h_value_offsets[i] = value_offset;
        packed->h_value_lengths[i] = static_cast<uint32_t>(value.size());
        std::memcpy(values_write_ptr + value_offset, value.c_str(), value.size());
        value_offset += value.size();
        value_offset += (8 - (value_offset % 8)) % 8; // Alignment
    }
    
    return packed;
}

bool BatchDataManager::transfer_to_gpu(PackedBatchData& data, cudaStream_t stream) {
    cudaError_t error;
    
    // Transfer keys data
    if (data.h_keys_data && data.d_keys_data) {
        error = cudaMemcpyAsync(data.d_keys_data, data.h_keys_data, 
                               data.keys_data_size, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to transfer keys data to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    // Transfer values data (if present)
    if (data.h_values_data && data.d_values_data) {
        error = cudaMemcpyAsync(data.d_values_data, data.h_values_data, 
                               data.values_data_size, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            std::cerr << "Failed to transfer values data to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    // Transfer metadata arrays
    error = cudaMemcpyAsync(data.d_key_offsets, data.h_key_offsets, 
                           data.batch_size * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpyAsync(data.d_key_lengths, data.h_key_lengths, 
                           data.batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return false;
    
    if (data.h_value_offsets && data.d_value_offsets) {
        error = cudaMemcpyAsync(data.d_value_offsets, data.h_value_offsets, 
                               data.batch_size * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) return false;
    }
    
    if (data.h_value_lengths && data.d_value_lengths) {
        error = cudaMemcpyAsync(data.d_value_lengths, data.h_value_lengths, 
                               data.batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) return false;
    }
    
    return true;
}

bool BatchDataManager::transfer_from_gpu(PackedBatchData& data, cudaStream_t stream) {
    cudaError_t error;
    
    // Transfer results back from GPU
    if (data.d_result_flags) {
        // We need host memory for result flags
        if (!data.h_result_flags) {
            error = cudaHostAlloc(&data.h_result_flags, data.batch_size * sizeof(bool), cudaHostAllocPortable);
            if (error != cudaSuccess) return false;
        }
        
        error = cudaMemcpyAsync(data.h_result_flags, data.d_result_flags, 
                               data.batch_size * sizeof(bool), cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) return false;
    }
    
    // For lookup operations, transfer values data back
    if (data.d_values_data && data.values_data_size > 0) {
        error = cudaMemcpyAsync(data.h_values_data, data.d_values_data, 
                               data.values_data_size, cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) return false;
    }
    
    // Transfer value lengths (updated by GPU kernels)
    if (data.d_value_lengths && data.h_value_lengths) {
        error = cudaMemcpyAsync(data.h_value_lengths, data.d_value_lengths, 
                               data.batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) return false;
    }
    
    // Wait for transfer completion
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to synchronize stream after GPU transfer: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

std::vector<std::optional<std::string>> 
BatchDataManager::unpack_lookup_results(const PackedBatchData& data, 
                                       const std::vector<std::string>& original_keys) {
    
    std::vector<std::optional<std::string>> results;
    results.reserve(data.batch_size);
    
    if (!data.h_result_flags || !data.h_values_data || !data.h_value_lengths || !data.h_value_offsets) {
        // Return empty results if data is missing
        results.resize(data.batch_size);
        return results;
    }
    
    for (size_t i = 0; i < data.batch_size; ++i) {
        if (data.h_result_flags[i] && data.h_value_lengths[i] > 0) {
            // Extract value from packed data
            const char* value_ptr = data.h_values_data + data.h_value_offsets[i];
            uint32_t value_len = data.h_value_lengths[i];
            
            results.emplace_back(std::string(value_ptr, value_len));
        } else {
            results.emplace_back(std::nullopt);
        }
    }
    
    return results;
}

std::vector<bool> BatchDataManager::unpack_operation_results(const PackedBatchData& data) {
    std::vector<bool> results;
    results.reserve(data.batch_size);
    
    if (!data.h_result_flags) {
        results.resize(data.batch_size, false);
        return results;
    }
    
    for (size_t i = 0; i < data.batch_size; ++i) {
        results.push_back(data.h_result_flags[i]);
    }
    
    return results;
}

void* BatchDataManager::allocate_from_pool(std::map<size_t, MemoryPool>& pools, 
                                          size_t size, bool is_device) {
    // Find the smallest pool that can accommodate this size
    auto it = pools.lower_bound(size);
    if (it == pools.end()) {
        // Size too large for any pool, allocate directly
        void* ptr = nullptr;
        if (is_device) {
            ptr = memory_manager_->allocate(size);
        } else {
            cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
        }
        return ptr;
    }
    
    MemoryPool& pool = it->second;
    
    if (!pool.available_buffers.empty()) {
        // Reuse existing buffer
        void* ptr = pool.available_buffers.back();
        pool.available_buffers.pop_back();
        return ptr;
    }
    
    if (pool.available_buffers.size() < pool.max_buffers) {
        // Create new buffer
        void* ptr = nullptr;
        if (is_device) {
            ptr = memory_manager_->allocate(pool.buffer_size);
        } else {
            cudaHostAlloc(&ptr, pool.buffer_size, cudaHostAllocPortable);
        }
        return ptr;
    }
    
    // Pool full, allocate directly
    void* ptr = nullptr;
    if (is_device) {
        ptr = memory_manager_->allocate(size);
    } else {
        cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
    }
    return ptr;
}

void BatchDataManager::return_to_pool(std::map<size_t, MemoryPool>& pools, 
                                     void* ptr, size_t size) {
    auto it = pools.find(size);
    if (it != pools.end() && it->second.available_buffers.size() < it->second.max_buffers) {
        it->second.available_buffers.push_back(ptr);
    } else {
        // Free directly if not pooled
        if (pools == device_memory_pools_) {
            memory_manager_->deallocate(ptr);
        } else {
            cudaFreeHost(ptr);
        }
    }
}

} // namespace core
} // namespace predis