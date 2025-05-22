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

#include "gpu_hash_table.h"
#include "../memory_manager.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <cstring>

namespace predis {
namespace core {

// Constants for hash table implementation
constexpr size_t MAX_KEY_LENGTH = 256;
constexpr size_t MAX_VALUE_LENGTH = 4096;
constexpr float DEFAULT_LOAD_FACTOR = 0.75f;
constexpr size_t WARP_SIZE = 32;

// GPU hash table entry structure
struct __align__(16) HashEntry {
    char key[MAX_KEY_LENGTH];
    char value[MAX_VALUE_LENGTH];
    uint32_t key_len;
    uint32_t value_len;
    uint32_t hash;
    int32_t next_probe;  // For linear probing
    int32_t lock;  // 0=free, 1=locked, 2=deleted
    uint32_t padding[3];  // Align to 16 bytes for coalesced access
    
    __device__ __host__ HashEntry() : key_len(0), value_len(0), hash(0), next_probe(-1), lock(0) {
        memset(key, 0, MAX_KEY_LENGTH);
        memset(value, 0, MAX_VALUE_LENGTH);
        memset(padding, 0, sizeof(padding));
    }
};

// Hash functions
__device__ __host__ uint32_t fnv1a_hash(const char* key, size_t len) {
    const uint32_t FNV_PRIME = 0x01000193;
    const uint32_t FNV_OFFSET = 0x811c9dc5;
    
    uint32_t hash = FNV_OFFSET;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(key[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

__device__ __host__ uint32_t murmur3_hash(const char* key, size_t len) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint32_t r1 = 15;
    const uint32_t r2 = 13;
    const uint32_t m = 5;
    const uint32_t n = 0xe6546b64;
    
    uint32_t hash = 0;
    const int nblocks = len / 4;
    const uint32_t* blocks = reinterpret_cast<const uint32_t*>(key);
    
    for (int i = 0; i < nblocks; i++) {
        uint32_t k = blocks[i];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;
        
        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }
    
    const uint8_t* tail = reinterpret_cast<const uint8_t*>(key + nblocks * 4);
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = (k1 << r1) | (k1 >> (32 - r1));
                k1 *= c2;
                hash ^= k1;
    }
    
    hash ^= len;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);
    
    return hash;
}

// GPU kernels for hash table operations
__global__ void gpu_hash_insert_kernel(
    HashEntry* table,
    size_t capacity,
    const char* keys,
    const char* values,
    const uint32_t* key_lengths,
    const uint32_t* value_lengths,
    bool* results,
    size_t num_items,
    int hash_method
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;
    
    const char* key = keys + idx * MAX_KEY_LENGTH;
    const char* value = values + idx * MAX_VALUE_LENGTH;
    uint32_t key_len = key_lengths[idx];
    uint32_t value_len = value_lengths[idx];
    
    if (key_len == 0 || key_len >= MAX_KEY_LENGTH || 
        value_len == 0 || value_len >= MAX_VALUE_LENGTH) {
        results[idx] = false;
        return;
    }
    
    // Calculate hash
    uint32_t hash;
    if (hash_method == 0) { // FNV1A
        hash = fnv1a_hash(key, key_len);
    } else { // MURMUR3
        hash = murmur3_hash(key, key_len);
    }
    
    uint32_t start_slot = hash % capacity;
    uint32_t slot = start_slot;
    uint32_t probe_count = 0;
    
    // Linear probing with atomic operations
    while (probe_count < capacity) {
        HashEntry* entry = &table[slot];
        
        // Try to acquire lock
        int expected = 0;
        if (atomicCAS(&entry->lock, expected, 1) == 0) {
            // Successfully acquired lock
            
            // Check if slot is empty or marked for deletion
            if (entry->key_len == 0 || entry->lock == 2) {
                // Insert new entry
                entry->hash = hash;
                entry->key_len = key_len;
                entry->value_len = value_len;
                entry->next_probe = -1;
                
                // Copy key and value
                for (uint32_t i = 0; i < key_len; ++i) {
                    entry->key[i] = key[i];
                }
                entry->key[key_len] = '\0';
                
                for (uint32_t i = 0; i < value_len; ++i) {
                    entry->value[i] = value[i];
                }
                entry->value[value_len] = '\0';
                
                // Release lock
                __threadfence();
                entry->lock = 0;
                results[idx] = true;
                return;
            }
            
            // Check if key already exists (update case)
            if (entry->key_len == key_len && entry->hash == hash) {
                bool key_match = true;
                for (uint32_t i = 0; i < key_len; ++i) {
                    if (entry->key[i] != key[i]) {
                        key_match = false;
                        break;
                    }
                }
                
                if (key_match) {
                    // Update existing entry
                    entry->value_len = value_len;
                    for (uint32_t i = 0; i < value_len; ++i) {
                        entry->value[i] = value[i];
                    }
                    entry->value[value_len] = '\0';
                    
                    // Release lock
                    __threadfence();
                    entry->lock = 0;
                    results[idx] = true;
                    return;
                }
            }
            
            // Release lock and continue probing
            entry->lock = 0;
        }
        
        // Move to next slot (linear probing)
        slot = (slot + 1) % capacity;
        probe_count++;
        
        // Avoid infinite loops
        if (slot == start_slot) break;
    }
    
    results[idx] = false; // Table full or error
}

__global__ void gpu_hash_lookup_kernel(
    const HashEntry* table,
    size_t capacity,
    const char* keys,
    char* values,
    const uint32_t* key_lengths,
    uint32_t* value_lengths,
    bool* results,
    size_t num_items,
    int hash_method
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;
    
    const char* key = keys + idx * MAX_KEY_LENGTH;
    uint32_t key_len = key_lengths[idx];
    char* value = values + idx * MAX_VALUE_LENGTH;
    
    if (key_len == 0 || key_len >= MAX_KEY_LENGTH) {
        results[idx] = false;
        return;
    }
    
    // Calculate hash
    uint32_t hash;
    if (hash_method == 0) { // FNV1A
        hash = fnv1a_hash(key, key_len);
    } else { // MURMUR3
        hash = murmur3_hash(key, key_len);
    }
    
    uint32_t start_slot = hash % capacity;
    uint32_t slot = start_slot;
    uint32_t probe_count = 0;
    
    // Linear probing lookup
    while (probe_count < capacity) {
        const HashEntry* entry = &table[slot];
        
        // Check if we've reached an empty slot
        if (entry->key_len == 0 && entry->lock != 2) {
            break; // Key not found
        }
        
        // Skip deleted entries
        if (entry->lock == 2) {
            slot = (slot + 1) % capacity;
            probe_count++;
            continue;
        }
        
        // Check for match
        if (entry->key_len == key_len && entry->hash == hash && entry->lock != 1) {
            bool key_match = true;
            for (uint32_t i = 0; i < key_len; ++i) {
                if (entry->key[i] != key[i]) {
                    key_match = false;
                    break;
                }
            }
            
            if (key_match) {
                // Found the key, copy value
                value_lengths[idx] = entry->value_len;
                for (uint32_t i = 0; i < entry->value_len; ++i) {
                    value[i] = entry->value[i];
                }
                value[entry->value_len] = '\0';
                results[idx] = true;
                return;
            }
        }
        
        // Move to next slot
        slot = (slot + 1) % capacity;
        probe_count++;
        
        if (slot == start_slot) break;
    }
    
    results[idx] = false; // Not found
}

__global__ void gpu_hash_remove_kernel(
    HashEntry* table,
    size_t capacity,
    const char* keys,
    const uint32_t* key_lengths,
    bool* results,
    size_t num_items,
    int hash_method
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;
    
    const char* key = keys + idx * MAX_KEY_LENGTH;
    uint32_t key_len = key_lengths[idx];
    
    if (key_len == 0 || key_len >= MAX_KEY_LENGTH) {
        results[idx] = false;
        return;
    }
    
    // Calculate hash
    uint32_t hash;
    if (hash_method == 0) { // FNV1A
        hash = fnv1a_hash(key, key_len);
    } else { // MURMUR3
        hash = murmur3_hash(key, key_len);
    }
    
    uint32_t start_slot = hash % capacity;
    uint32_t slot = start_slot;
    uint32_t probe_count = 0;
    
    // Linear probing to find and remove
    while (probe_count < capacity) {
        HashEntry* entry = &table[slot];
        
        // Check if we've reached an empty slot
        if (entry->key_len == 0 && entry->lock != 2) {
            break; // Key not found
        }
        
        // Skip deleted entries
        if (entry->lock == 2) {
            slot = (slot + 1) % capacity;
            probe_count++;
            continue;
        }
        
        // Try to acquire lock for potential deletion
        int expected = 0;
        if (atomicCAS(&entry->lock, expected, 1) == 0) {
            // Check for match
            if (entry->key_len == key_len && entry->hash == hash) {
                bool key_match = true;
                for (uint32_t i = 0; i < key_len; ++i) {
                    if (entry->key[i] != key[i]) {
                        key_match = false;
                        break;
                    }
                }
                
                if (key_match) {
                    // Mark as deleted
                    entry->lock = 2;
                    entry->key_len = 0;
                    entry->value_len = 0;
                    __threadfence();
                    results[idx] = true;
                    return;
                }
            }
            
            // Release lock
            entry->lock = 0;
        }
        
        // Move to next slot
        slot = (slot + 1) % capacity;
        probe_count++;
        
        if (slot == start_slot) break;
    }
    
    results[idx] = false; // Not found
}

// Implementation structure
struct GpuHashTable::Impl {
    HashEntry* d_table = nullptr;
    HashEntry* h_table = nullptr;
    size_t capacity = 0;
    size_t size = 0;
    HashMethod hash_method = HashMethod::FNV1A;
    MemoryManager* memory_manager = nullptr;
    std::mutex table_mutex;
    
    // GPU memory buffers for batch operations
    char* d_keys_buffer = nullptr;
    char* d_values_buffer = nullptr;
    uint32_t* d_key_lengths = nullptr;
    uint32_t* d_value_lengths = nullptr;
    bool* d_results = nullptr;
    size_t buffer_capacity = 0;
    
    HashStats stats;
    bool initialized = false;
};

GpuHashTable::GpuHashTable() : pImpl(std::make_unique<Impl>()) {
}

GpuHashTable::~GpuHashTable() {
    if (pImpl->initialized) {
        shutdown();
    }
}

bool GpuHashTable::initialize(size_t initial_capacity, HashMethod method) {
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    if (pImpl->initialized) {
        std::cerr << "GpuHashTable already initialized" << std::endl;
        return false;
    }
    
    pImpl->capacity = initial_capacity;
    pImpl->hash_method = method;
    
    // Initialize memory manager
    pImpl->memory_manager = new MemoryManager();
    if (!pImpl->memory_manager->initialize()) {
        std::cerr << "Failed to initialize memory manager for hash table" << std::endl;
        delete pImpl->memory_manager;
        return false;
    }
    
    // Allocate GPU memory for hash table
    size_t table_size = pImpl->capacity * sizeof(HashEntry);
    pImpl->d_table = static_cast<HashEntry*>(pImpl->memory_manager->allocate(table_size));
    if (!pImpl->d_table) {
        std::cerr << "Failed to allocate GPU memory for hash table" << std::endl;
        return false;
    }
    
    // Initialize table entries
    cudaError_t error = cudaMemset(pImpl->d_table, 0, table_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to initialize hash table: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate host memory for synchronization
    pImpl->h_table = new HashEntry[pImpl->capacity];
    memset(pImpl->h_table, 0, table_size);
    
    // Allocate buffers for batch operations (default 10K items)
    pImpl->buffer_capacity = 10000;
    size_t keys_buffer_size = pImpl->buffer_capacity * MAX_KEY_LENGTH;
    size_t values_buffer_size = pImpl->buffer_capacity * MAX_VALUE_LENGTH;
    size_t lengths_buffer_size = pImpl->buffer_capacity * sizeof(uint32_t);
    size_t results_buffer_size = pImpl->buffer_capacity * sizeof(bool);
    
    pImpl->d_keys_buffer = static_cast<char*>(pImpl->memory_manager->allocate(keys_buffer_size));
    pImpl->d_values_buffer = static_cast<char*>(pImpl->memory_manager->allocate(values_buffer_size));
    pImpl->d_key_lengths = static_cast<uint32_t*>(pImpl->memory_manager->allocate(lengths_buffer_size));
    pImpl->d_value_lengths = static_cast<uint32_t*>(pImpl->memory_manager->allocate(lengths_buffer_size));
    pImpl->d_results = static_cast<bool*>(pImpl->memory_manager->allocate(results_buffer_size));
    
    if (!pImpl->d_keys_buffer || !pImpl->d_values_buffer || 
        !pImpl->d_key_lengths || !pImpl->d_value_lengths || !pImpl->d_results) {
        std::cerr << "Failed to allocate GPU buffers for batch operations" << std::endl;
        return false;
    }
    
    // Initialize statistics
    pImpl->stats.capacity = pImpl->capacity;
    pImpl->stats.size = 0;
    pImpl->stats.collisions = 0;
    pImpl->stats.load_factor = 0.0;
    pImpl->stats.max_probe_distance = 0;
    
    pImpl->initialized = true;
    
    std::cout << "GPU Hash Table initialized: " << pImpl->capacity << " entries, " 
              << (table_size / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

void GpuHashTable::shutdown() {
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    if (!pImpl->initialized) return;
    
    // Free GPU memory
    if (pImpl->memory_manager) {
        if (pImpl->d_table) pImpl->memory_manager->deallocate(pImpl->d_table);
        if (pImpl->d_keys_buffer) pImpl->memory_manager->deallocate(pImpl->d_keys_buffer);
        if (pImpl->d_values_buffer) pImpl->memory_manager->deallocate(pImpl->d_values_buffer);
        if (pImpl->d_key_lengths) pImpl->memory_manager->deallocate(pImpl->d_key_lengths);
        if (pImpl->d_value_lengths) pImpl->memory_manager->deallocate(pImpl->d_value_lengths);
        if (pImpl->d_results) pImpl->memory_manager->deallocate(pImpl->d_results);
        
        pImpl->memory_manager->shutdown();
        delete pImpl->memory_manager;
    }
    
    // Free host memory
    delete[] pImpl->h_table;
    
    pImpl->d_table = nullptr;
    pImpl->h_table = nullptr;
    pImpl->d_keys_buffer = nullptr;
    pImpl->d_values_buffer = nullptr;
    pImpl->d_key_lengths = nullptr;
    pImpl->d_value_lengths = nullptr;
    pImpl->d_results = nullptr;
    pImpl->memory_manager = nullptr;
    pImpl->initialized = false;
    
    std::cout << "GPU Hash Table shutdown complete" << std::endl;
}

bool GpuHashTable::insert(const std::string& key, const std::string& value) {
    if (!pImpl->initialized) return false;
    
    std::vector<std::pair<std::string, std::string>> pairs = {{key, value}};
    return batch_insert(pairs);
}

bool GpuHashTable::lookup(const std::string& key, std::string& value) {
    if (!pImpl->initialized) return false;
    
    std::vector<std::string> keys = {key};
    auto results = batch_lookup(keys);
    
    if (!results.empty() && !results[0].empty()) {
        value = results[0];
        return true;
    }
    
    return false;
}

bool GpuHashTable::remove(const std::string& key) {
    if (!pImpl->initialized) return false;
    
    std::vector<std::string> keys = {key};
    return batch_remove(keys);
}

bool GpuHashTable::batch_insert(const std::vector<std::pair<std::string, std::string>>& pairs) {
    if (!pImpl->initialized || pairs.empty()) return false;
    
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    size_t num_items = pairs.size();
    if (num_items > pImpl->buffer_capacity) {
        std::cerr << "Batch size exceeds buffer capacity" << std::endl;
        return false;
    }
    
    // Prepare host buffers
    std::vector<char> h_keys(num_items * MAX_KEY_LENGTH, 0);
    std::vector<char> h_values(num_items * MAX_VALUE_LENGTH, 0);
    std::vector<uint32_t> h_key_lengths(num_items);
    std::vector<uint32_t> h_value_lengths(num_items);
    std::vector<char> h_results(num_items, 0);
    
    // Copy data to host buffers
    for (size_t i = 0; i < num_items; ++i) {
        const auto& [key, value] = pairs[i];
        
        if (key.length() >= MAX_KEY_LENGTH || value.length() >= MAX_VALUE_LENGTH) {
            std::cerr << "Key or value too long at index " << i << std::endl;
            return false;
        }
        
        h_key_lengths[i] = static_cast<uint32_t>(key.length());
        h_value_lengths[i] = static_cast<uint32_t>(value.length());
        
        std::memcpy(&h_keys[i * MAX_KEY_LENGTH], key.c_str(), key.length());
        std::memcpy(&h_values[i * MAX_VALUE_LENGTH], value.c_str(), value.length());
    }
    
    // Copy to GPU
    cudaError_t error;
    error = cudaMemcpy(pImpl->d_keys_buffer, h_keys.data(), 
                      num_items * MAX_KEY_LENGTH, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(pImpl->d_values_buffer, h_values.data(), 
                      num_items * MAX_VALUE_LENGTH, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(pImpl->d_key_lengths, h_key_lengths.data(), 
                      num_items * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(pImpl->d_value_lengths, h_value_lengths.data(), 
                      num_items * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_items + block_size - 1) / block_size;
    int hash_method = static_cast<int>(pImpl->hash_method);
    
    gpu_hash_insert_kernel<<<grid_size, block_size>>>(
        pImpl->d_table, pImpl->capacity,
        pImpl->d_keys_buffer, pImpl->d_values_buffer,
        pImpl->d_key_lengths, pImpl->d_value_lengths,
        pImpl->d_results, num_items, hash_method
    );
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back
    error = cudaMemcpy(h_results.data(), pImpl->d_results, 
                      num_items * sizeof(bool), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    // Count successful insertions
    size_t successful = 0;
    for (char result : h_results) {
        if (result) successful++;
    }
    
    pImpl->size += successful;
    pImpl->stats.size = pImpl->size;
    pImpl->stats.load_factor = static_cast<double>(pImpl->size) / pImpl->capacity;
    
    return successful == num_items;
}

std::vector<std::string> GpuHashTable::batch_lookup(const std::vector<std::string>& keys) {
    std::vector<std::string> results;
    if (!pImpl->initialized || keys.empty()) return results;
    
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    size_t num_items = keys.size();
    if (num_items > pImpl->buffer_capacity) {
        std::cerr << "Batch size exceeds buffer capacity" << std::endl;
        return results;
    }
    
    results.resize(num_items);
    
    // Prepare host buffers
    std::vector<char> h_keys(num_items * MAX_KEY_LENGTH, 0);
    std::vector<char> h_values(num_items * MAX_VALUE_LENGTH, 0);
    std::vector<uint32_t> h_key_lengths(num_items);
    std::vector<uint32_t> h_value_lengths(num_items);
    std::vector<char> h_results(num_items, 0);
    
    // Copy keys to host buffer
    for (size_t i = 0; i < num_items; ++i) {
        if (keys[i].length() >= MAX_KEY_LENGTH) {
            std::cerr << "Key too long at index " << i << std::endl;
            return {};
        }
        
        h_key_lengths[i] = static_cast<uint32_t>(keys[i].length());
        std::memcpy(&h_keys[i * MAX_KEY_LENGTH], keys[i].c_str(), keys[i].length());
    }
    
    // Copy to GPU
    cudaError_t error;
    error = cudaMemcpy(pImpl->d_keys_buffer, h_keys.data(), 
                      num_items * MAX_KEY_LENGTH, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return {};
    
    error = cudaMemcpy(pImpl->d_key_lengths, h_key_lengths.data(), 
                      num_items * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return {};
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_items + block_size - 1) / block_size;
    int hash_method = static_cast<int>(pImpl->hash_method);
    
    gpu_hash_lookup_kernel<<<grid_size, block_size>>>(
        pImpl->d_table, pImpl->capacity,
        pImpl->d_keys_buffer, pImpl->d_values_buffer,
        pImpl->d_key_lengths, pImpl->d_value_lengths,
        pImpl->d_results, num_items, hash_method
    );
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Lookup kernel execution failed: " << cudaGetErrorString(error) << std::endl;
        return {};
    }
    
    // Copy results back
    error = cudaMemcpy(h_values.data(), pImpl->d_values_buffer, 
                      num_items * MAX_VALUE_LENGTH, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return {};
    
    error = cudaMemcpy(h_value_lengths.data(), pImpl->d_value_lengths, 
                      num_items * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return {};
    
    error = cudaMemcpy(h_results.data(), pImpl->d_results, 
                      num_items * sizeof(bool), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return {};
    
    // Extract results
    for (size_t i = 0; i < num_items; ++i) {
        if (h_results[i]) {
            const char* value_ptr = &h_values[i * MAX_VALUE_LENGTH];
            results[i] = std::string(value_ptr, h_value_lengths[i]);
        }
    }
    
    return results;
}

bool GpuHashTable::batch_remove(const std::vector<std::string>& keys) {
    if (!pImpl->initialized || keys.empty()) return false;
    
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    size_t num_items = keys.size();
    if (num_items > pImpl->buffer_capacity) {
        std::cerr << "Batch size exceeds buffer capacity" << std::endl;
        return false;
    }
    
    // Prepare host buffers
    std::vector<char> h_keys(num_items * MAX_KEY_LENGTH, 0);
    std::vector<uint32_t> h_key_lengths(num_items);
    std::vector<char> h_results(num_items, 0);
    
    // Copy keys to host buffer
    for (size_t i = 0; i < num_items; ++i) {
        if (keys[i].length() >= MAX_KEY_LENGTH) {
            std::cerr << "Key too long at index " << i << std::endl;
            return false;
        }
        
        h_key_lengths[i] = static_cast<uint32_t>(keys[i].length());
        std::memcpy(&h_keys[i * MAX_KEY_LENGTH], keys[i].c_str(), keys[i].length());
    }
    
    // Copy to GPU
    cudaError_t error;
    error = cudaMemcpy(pImpl->d_keys_buffer, h_keys.data(), 
                      num_items * MAX_KEY_LENGTH, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(pImpl->d_key_lengths, h_key_lengths.data(), 
                      num_items * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_items + block_size - 1) / block_size;
    int hash_method = static_cast<int>(pImpl->hash_method);
    
    gpu_hash_remove_kernel<<<grid_size, block_size>>>(
        pImpl->d_table, pImpl->capacity,
        pImpl->d_keys_buffer, pImpl->d_key_lengths,
        pImpl->d_results, num_items, hash_method
    );
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Remove kernel execution failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back
    error = cudaMemcpy(h_results.data(), pImpl->d_results, 
                      num_items * sizeof(bool), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    // Count successful removals
    size_t successful = 0;
    for (char result : h_results) {
        if (result) successful++;
    }
    
    pImpl->size -= successful;
    pImpl->stats.size = pImpl->size;
    pImpl->stats.load_factor = static_cast<double>(pImpl->size) / pImpl->capacity;
    
    return successful == num_items;
}

void GpuHashTable::resize(size_t new_capacity) {
    // TODO: Implement table resizing with rehashing
    std::cerr << "Hash table resizing not yet implemented" << std::endl;
}

void GpuHashTable::clear() {
    if (!pImpl->initialized) return;
    
    std::lock_guard<std::mutex> lock(pImpl->table_mutex);
    
    size_t table_size = pImpl->capacity * sizeof(HashEntry);
    cudaError_t error = cudaMemset(pImpl->d_table, 0, table_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to clear hash table: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    pImpl->size = 0;
    pImpl->stats.size = 0;
    pImpl->stats.load_factor = 0.0;
    pImpl->stats.collisions = 0;
    pImpl->stats.max_probe_distance = 0;
}

GpuHashTable::HashStats GpuHashTable::get_stats() const {
    return pImpl->stats;
}

} // namespace core
} // namespace predis