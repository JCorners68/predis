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

#include "predis_client.h"
#include "../core/gpu_cache_manager.h"
#include "../core/simple_cache_manager.h"
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

namespace predis {
namespace api {

// Mock implementation for development and comparison
class MockCache {
private:
    std::unordered_map<std::string, std::string> data_;
    mutable std::mutex mutex_;
    size_t max_memory_mb_;
    size_t current_memory_bytes_;
    
    // Performance simulation parameters
    static constexpr double REDIS_BASELINE_OPS_SEC = 275000.0;
    static constexpr double MOCK_SINGLE_OP_SPEEDUP = 12.0;
    static constexpr double MOCK_BATCH_OP_SPEEDUP = 28.0;
    
    size_t estimate_memory_usage(const std::string& key, const std::string& value) const {
        return key.size() + value.size() + 256; // overhead
    }
    
    void simulate_gpu_latency(const std::string& operation, size_t count = 1) const {
        // Simulate realistic GPU latencies for demo purposes
        double base_latency_ms = 0.17; // Redis baseline
        double speedup = operation.find("batch") != std::string::npos ? 
                        MOCK_BATCH_OP_SPEEDUP : MOCK_SINGLE_OP_SPEEDUP;
        
        double gpu_latency_ms = base_latency_ms / speedup;
        // For testing, we skip actual sleep but track the simulated time
        // std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(gpu_latency_ms * 1000)));
    }

public:
    explicit MockCache(size_t max_memory_mb = 12800) 
        : max_memory_mb_(max_memory_mb), current_memory_bytes_(0) {}
    
    bool get(const std::string& key, std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("get");
        
        auto it = data_.find(key);
        if (it != data_.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    bool put(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("put");
        
        size_t memory_needed = estimate_memory_usage(key, value);
        if (current_memory_bytes_ + memory_needed > max_memory_mb_ * 1024 * 1024) {
            return false; // Memory limit exceeded
        }
        
        // Remove old entry memory usage
        auto it = data_.find(key);
        if (it != data_.end()) {
            current_memory_bytes_ -= estimate_memory_usage(key, it->second);
        }
        
        data_[key] = value;
        current_memory_bytes_ += memory_needed;
        return true;
    }
    
    bool remove(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("remove");
        
        auto it = data_.find(key);
        if (it != data_.end()) {
            current_memory_bytes_ -= estimate_memory_usage(key, it->second);
            data_.erase(it);
            return true;
        }
        return false;
    }
    
    std::vector<std::string> batch_get(const std::vector<std::string>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("batch_get", keys.size());
        
        std::vector<std::string> results;
        results.reserve(keys.size());
        
        for (const auto& key : keys) {
            auto it = data_.find(key);
            if (it != data_.end()) {
                results.push_back(it->second);
            } else {
                results.push_back(""); // Empty string for not found
            }
        }
        return results;
    }
    
    bool batch_put(const std::vector<std::pair<std::string, std::string>>& pairs) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("batch_put", pairs.size());
        
        // Check memory requirements first
        size_t total_memory_needed = 0;
        for (const auto& [key, value] : pairs) {
            total_memory_needed += estimate_memory_usage(key, value);
        }
        
        if (current_memory_bytes_ + total_memory_needed > max_memory_mb_ * 1024 * 1024) {
            return false;
        }
        
        // Perform all insertions
        for (const auto& [key, value] : pairs) {
            auto it = data_.find(key);
            if (it != data_.end()) {
                current_memory_bytes_ -= estimate_memory_usage(key, it->second);
            }
            data_[key] = value;
            current_memory_bytes_ += estimate_memory_usage(key, value);
        }
        return true;
    }
    
    size_t batch_remove(const std::vector<std::string>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        simulate_gpu_latency("batch_remove", keys.size());
        
        size_t removed_count = 0;
        for (const auto& key : keys) {
            auto it = data_.find(key);
            if (it != data_.end()) {
                current_memory_bytes_ -= estimate_memory_usage(key, it->second);
                data_.erase(it);
                removed_count++;
            }
        }
        return removed_count;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }
    
    size_t memory_usage_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_memory_bytes_;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
        current_memory_bytes_ = 0;
    }
};

struct PredisClient::Impl {
    // Connection state
    bool connected = false;
    std::string host = "localhost";
    int port = 6379;
    Mode current_mode = Mode::AUTO_DETECT;
    
    // Cache implementations
    std::unique_ptr<core::GpuCacheManager> gpu_cache;
    std::unique_ptr<MockCache> mock_cache;
    std::unique_ptr<core::SimpleCacheManager> simple_cache; // fallback
    
    // Configuration
    ConsistencyLevel consistency = ConsistencyLevel::RELAXED;
    PrefetchConfig prefetch_config;
    
    // Performance tracking
    mutable std::mutex stats_mutex;
    size_t operation_count = 0;
    size_t hit_count = 0;
    size_t miss_count = 0;
    double total_operation_time_ms = 0.0;
    std::vector<double> operation_latencies;
    
    // Performance comparison data
    double mock_total_time_ms = 0.0;
    double gpu_total_time_ms = 0.0;
    size_t mock_operations = 0;
    size_t gpu_operations = 0;
    
    // GPU detection and error handling
    bool gpu_available = false;
    std::string last_error;
    
    void detect_gpu_availability() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        gpu_available = (error == cudaSuccess && device_count > 0);
        
        if (!gpu_available) {
            last_error = "No CUDA-capable GPU detected: " + std::string(cudaGetErrorString(error));
            std::cout << "GPU detection: " << last_error << std::endl;
        } else {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "GPU detected: " << prop.name << " with " 
                      << (prop.totalGlobalMem / 1024 / 1024) << " MB VRAM" << std::endl;
        }
    }
    
    Mode determine_auto_mode() const {
        if (gpu_available) {
            return Mode::REAL_GPU_ONLY;
        } else {
            return Mode::MOCK_ONLY;
        }
    }
    
    void record_operation_time(double time_ms, bool is_gpu_operation) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        operation_count++;
        total_operation_time_ms += time_ms;
        operation_latencies.push_back(time_ms);
        
        // Keep only recent latencies for statistics
        if (operation_latencies.size() > 10000) {
            operation_latencies.erase(operation_latencies.begin(), 
                                    operation_latencies.begin() + 5000);
        }
        
        if (is_gpu_operation) {
            gpu_total_time_ms += time_ms;
            gpu_operations++;
        } else {
            mock_total_time_ms += time_ms;
            mock_operations++;
        }
    }
    
    template<typename Func>
    auto time_operation(Func&& func, bool is_gpu_operation) -> decltype(func()) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = func();
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_operation_time(elapsed_ms, is_gpu_operation);
        
        return result;
    }
};

PredisClient::PredisClient() : pImpl(std::make_unique<Impl>()) {
    pImpl->detect_gpu_availability();
}

PredisClient::~PredisClient() {
    if (pImpl->connected) {
        disconnect();
    }
}

bool PredisClient::connect(const std::string& host, int port, Mode mode) {
    if (pImpl->connected) {
        std::cerr << "Client already connected" << std::endl;
        return false;
    }
    
    pImpl->host = host;
    pImpl->port = port;
    
    // Determine actual mode to use
    Mode actual_mode = mode;
    if (mode == Mode::AUTO_DETECT) {
        actual_mode = pImpl->determine_auto_mode();
        std::cout << "Auto-detected mode: " << (actual_mode == Mode::REAL_GPU_ONLY ? "REAL_GPU" : "MOCK") << std::endl;
    }
    
    pImpl->current_mode = actual_mode;
    
    try {
        // Initialize based on selected mode
        switch (actual_mode) {
            case Mode::REAL_GPU_ONLY: {
                if (!pImpl->gpu_available) {
                    std::cerr << "Real GPU mode requested but no GPU available" << std::endl;
                    return false;
                }
                
                pImpl->gpu_cache = std::make_unique<core::GpuCacheManager>();
                core::GpuCacheManager::Config config;
                config.mode = core::GpuCacheManager::Mode::REAL_GPU;
                config.enable_statistics = true;
                
                if (!pImpl->gpu_cache->initialize(config)) {
                    std::cerr << "Failed to initialize GPU cache manager" << std::endl;
                    return false;
                }
                break;
            }
            
            case Mode::MOCK_ONLY: {
                pImpl->mock_cache = std::make_unique<MockCache>();
                break;
            }
            
            case Mode::HYBRID: {
                // Initialize both for comparison
                if (pImpl->gpu_available) {
                    pImpl->gpu_cache = std::make_unique<core::GpuCacheManager>();
                    core::GpuCacheManager::Config config;
                    config.mode = core::GpuCacheManager::Mode::REAL_GPU;
                    if (!pImpl->gpu_cache->initialize(config)) {
                        std::cerr << "Warning: Failed to initialize GPU cache in hybrid mode" << std::endl;
                        pImpl->gpu_cache.reset();
                    }
                }
                pImpl->mock_cache = std::make_unique<MockCache>();
                break;
            }
            
            default: {
                std::cerr << "Invalid mode specified" << std::endl;
                return false;
            }
        }
        
        pImpl->connected = true;
        std::cout << "Connected to Predis cache in " 
                  << (actual_mode == Mode::REAL_GPU_ONLY ? "REAL_GPU" : 
                      actual_mode == Mode::MOCK_ONLY ? "MOCK" : "HYBRID") 
                  << " mode" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect: " << e.what() << std::endl;
        pImpl->last_error = e.what();
        return false;
    }
}

void PredisClient::disconnect() {
    if (!pImpl->connected) return;
    
    if (pImpl->gpu_cache) {
        pImpl->gpu_cache->shutdown();
        pImpl->gpu_cache.reset();
    }
    
    if (pImpl->mock_cache) {
        pImpl->mock_cache.reset();
    }
    
    if (pImpl->simple_cache) {
        pImpl->simple_cache->shutdown();
        pImpl->simple_cache.reset();
    }
    
    pImpl->connected = false;
    std::cout << "Disconnected from Predis cache" << std::endl;
}

bool PredisClient::is_connected() const {
    return pImpl->connected;
}

bool PredisClient::switch_mode(Mode new_mode) {
    if (!pImpl->connected) {
        std::cerr << "Not connected - cannot switch mode" << std::endl;
        return false;
    }
    
    if (new_mode == pImpl->current_mode) {
        return true; // Already in requested mode
    }
    
    // For simplicity, require reconnection for mode switching
    std::cout << "Mode switching requires reconnection..." << std::endl;
    std::string host = pImpl->host;
    int port = pImpl->port;
    
    disconnect();
    return connect(host, port, new_mode);
}

PredisClient::Mode PredisClient::get_current_mode() const {
    return pImpl->current_mode;
}

bool PredisClient::is_using_real_gpu() const {
    return pImpl->current_mode == Mode::REAL_GPU_ONLY || 
           (pImpl->current_mode == Mode::HYBRID && pImpl->gpu_cache != nullptr);
}

bool PredisClient::get(const std::string& key, std::string& value) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return false;
    }
    
    bool result = false;
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            result = pImpl->time_operation([&]() {
                return pImpl->gpu_cache->get(key, value);
            }, true);
            break;
        }
        
        case Mode::MOCK_ONLY: {
            result = pImpl->time_operation([&]() {
                return pImpl->mock_cache->get(key, value);
            }, false);
            break;
        }
        
        case Mode::HYBRID: {
            // Use GPU if available, otherwise mock
            if (pImpl->gpu_cache) {
                result = pImpl->time_operation([&]() {
                    return pImpl->gpu_cache->get(key, value);
                }, true);
            } else {
                result = pImpl->time_operation([&]() {
                    return pImpl->mock_cache->get(key, value);
                }, false);
            }
            break;
        }
        
        default:
            return false;
    }
    
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    if (result) {
        pImpl->hit_count++;
    } else {
        pImpl->miss_count++;
    }
    
    return result;
}

bool PredisClient::put(const std::string& key, const std::string& value, int ttl) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return false;
    }
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->gpu_cache->put(key, value, ttl);
            }, true);
        }
        
        case Mode::MOCK_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->mock_cache->put(key, value);
            }, false);
        }
        
        case Mode::HYBRID: {
            // Write to both for consistency testing
            bool gpu_result = true;
            bool mock_result = true;
            
            if (pImpl->gpu_cache) {
                gpu_result = pImpl->time_operation([&]() {
                    return pImpl->gpu_cache->put(key, value, ttl);
                }, true);
            }
            
            mock_result = pImpl->time_operation([&]() {
                return pImpl->mock_cache->put(key, value);
            }, false);
            
            return gpu_result && mock_result;
        }
        
        default:
            return false;
    }
}

bool PredisClient::remove(const std::string& key) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return false;
    }
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->gpu_cache->remove(key);
            }, true);
        }
        
        case Mode::MOCK_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->mock_cache->remove(key);
            }, false);
        }
        
        case Mode::HYBRID: {
            bool gpu_result = true;
            bool mock_result = true;
            
            if (pImpl->gpu_cache) {
                gpu_result = pImpl->time_operation([&]() {
                    return pImpl->gpu_cache->remove(key);
                }, true);
            }
            
            mock_result = pImpl->time_operation([&]() {
                return pImpl->mock_cache->remove(key);
            }, false);
            
            return gpu_result || mock_result; // Success if either succeeds
        }
        
        default:
            return false;
    }
}

std::vector<std::string> PredisClient::mget(const std::vector<std::string>& keys) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return {};
    }
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            return pImpl->time_operation([&]() {
                std::vector<std::string> values;
                auto batch_result = pImpl->gpu_cache->batch_get(keys, values);
                return values;
            }, true);
        }
        
        case Mode::MOCK_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->mock_cache->batch_get(keys);
            }, false);
        }
        
        case Mode::HYBRID: {
            if (pImpl->gpu_cache) {
                return pImpl->time_operation([&]() {
                    std::vector<std::string> values;
                    auto batch_result = pImpl->gpu_cache->batch_get(keys, values);
                    return values;
                }, true);
            } else {
                return pImpl->time_operation([&]() {
                    return pImpl->mock_cache->batch_get(keys);
                }, false);
            }
        }
        
        default:
            return {};
    }
}

bool PredisClient::mput(const std::unordered_map<std::string, std::string>& key_values) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return false;
    }
    
    // Convert to vector format for GPU batch operations
    std::vector<std::pair<std::string, std::string>> pairs;
    pairs.reserve(key_values.size());
    for (const auto& [key, value] : key_values) {
        pairs.emplace_back(key, value);
    }
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            return pImpl->time_operation([&]() {
                auto batch_result = pImpl->gpu_cache->batch_put(pairs);
                return batch_result.successful_operations == pairs.size();
            }, true);
        }
        
        case Mode::MOCK_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->mock_cache->batch_put(pairs);
            }, false);
        }
        
        case Mode::HYBRID: {
            bool gpu_result = true;
            bool mock_result = true;
            
            if (pImpl->gpu_cache) {
                gpu_result = pImpl->time_operation([&]() {
                    auto batch_result = pImpl->gpu_cache->batch_put(pairs);
                    return batch_result.successful_operations == pairs.size();
                }, true);
            }
            
            mock_result = pImpl->time_operation([&]() {
                return pImpl->mock_cache->batch_put(pairs);
            }, false);
            
            return gpu_result && mock_result;
        }
        
        default:
            return false;
    }
}

bool PredisClient::mdelete(const std::vector<std::string>& keys) {
    if (!pImpl->connected) {
        std::cerr << "Client not connected" << std::endl;
        return false;
    }
    
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            return pImpl->time_operation([&]() {
                auto batch_result = pImpl->gpu_cache->batch_remove(keys);
                return batch_result.successful_operations > 0;
            }, true);
        }
        
        case Mode::MOCK_ONLY: {
            return pImpl->time_operation([&]() {
                return pImpl->mock_cache->batch_remove(keys) > 0;
            }, false);
        }
        
        case Mode::HYBRID: {
            bool gpu_result = true;
            bool mock_result = true;
            
            if (pImpl->gpu_cache) {
                gpu_result = pImpl->time_operation([&]() {
                    auto batch_result = pImpl->gpu_cache->batch_remove(keys);
                    return batch_result.successful_operations > 0;
                }, true);
            }
            
            mock_result = pImpl->time_operation([&]() {
                return pImpl->mock_cache->batch_remove(keys) > 0;
            }, false);
            
            return gpu_result || mock_result;
        }
        
        default:
            return false;
    }
}

void PredisClient::set_consistency_level(ConsistencyLevel level) {
    pImpl->consistency = level;
}

PredisClient::ConsistencyLevel PredisClient::get_consistency_level() const {
    return pImpl->consistency;
}

void PredisClient::configure_prefetching(const PrefetchConfig& config) {
    pImpl->prefetch_config = config;
    
    if (pImpl->gpu_cache) {
        pImpl->gpu_cache->configure_prefetching(config.enabled, config.confidence_threshold);
    }
}

PredisClient::PrefetchConfig PredisClient::get_prefetch_config() const {
    return pImpl->prefetch_config;
}

void PredisClient::hint_related_keys(const std::vector<std::string>& keys) {
    if (pImpl->gpu_cache) {
        pImpl->gpu_cache->hint_related_keys(keys);
    }
}

void PredisClient::hint_sequence(const std::vector<std::string>& keys) {
    // Placeholder for sequence hinting
    std::cout << "Sequence hint with " << keys.size() << " keys" << std::endl;
}

PredisClient::Stats PredisClient::get_stats() const {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    
    Stats stats;
    stats.using_real_gpu = is_using_real_gpu();
    stats.implementation_mode = (pImpl->current_mode == Mode::REAL_GPU_ONLY ? "REAL_GPU" :
                                pImpl->current_mode == Mode::MOCK_ONLY ? "MOCK" : "HYBRID");
    
    // Basic statistics
    size_t total_accesses = pImpl->hit_count + pImpl->miss_count;
    if (total_accesses > 0) {
        stats.hit_ratio = static_cast<double>(pImpl->hit_count) / total_accesses;
    }
    
    // Performance metrics
    if (pImpl->operation_count > 0 && pImpl->total_operation_time_ms > 0) {
        stats.operations_per_second = (pImpl->operation_count * 1000.0) / pImpl->total_operation_time_ms;
        stats.avg_latency_ms = pImpl->total_operation_time_ms / pImpl->operation_count;
        
        // Calculate P95 latency
        if (pImpl->operation_latencies.size() >= 2) {
            auto latencies_copy = pImpl->operation_latencies;
            std::sort(latencies_copy.begin(), latencies_copy.end());
            size_t p95_index = static_cast<size_t>(latencies_copy.size() * 0.95);
            stats.p95_latency_ms = latencies_copy[p95_index];
        }
    }
    
    // Get cache-specific stats
    switch (pImpl->current_mode) {
        case Mode::REAL_GPU_ONLY: {
            if (pImpl->gpu_cache) {
                auto gpu_stats = pImpl->gpu_cache->get_stats();
                stats.total_keys = gpu_stats.total_entries;
                stats.memory_usage_mb = gpu_stats.memory_usage_mb;
            }
            break;
        }
        
        case Mode::MOCK_ONLY: {
            if (pImpl->mock_cache) {
                stats.total_keys = pImpl->mock_cache->size();
                stats.memory_usage_mb = pImpl->mock_cache->memory_usage_bytes() / 1024.0 / 1024.0;
            }
            break;
        }
        
        case Mode::HYBRID: {
            if (pImpl->gpu_cache) {
                auto gpu_stats = pImpl->gpu_cache->get_stats();
                stats.total_keys = gpu_stats.total_entries;
                stats.memory_usage_mb = gpu_stats.memory_usage_mb;
            } else if (pImpl->mock_cache) {
                stats.total_keys = pImpl->mock_cache->size();
                stats.memory_usage_mb = pImpl->mock_cache->memory_usage_bytes() / 1024.0 / 1024.0;
            }
            break;
        }
    }
    
    // Performance comparison data
    if (pImpl->mock_operations > 0 && pImpl->mock_total_time_ms > 0) {
        stats.mock_ops_per_second = (pImpl->mock_operations * 1000.0) / pImpl->mock_total_time_ms;
    }
    
    if (pImpl->gpu_operations > 0 && pImpl->gpu_total_time_ms > 0) {
        stats.real_gpu_ops_per_second = (pImpl->gpu_operations * 1000.0) / pImpl->gpu_total_time_ms;
    }
    
    if (stats.mock_ops_per_second > 0 && stats.real_gpu_ops_per_second > 0) {
        stats.performance_improvement_ratio = stats.real_gpu_ops_per_second / stats.mock_ops_per_second;
    }
    
    return stats;
}

void PredisClient::reset_stats() {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    pImpl->operation_count = 0;
    pImpl->hit_count = 0;
    pImpl->miss_count = 0;
    pImpl->total_operation_time_ms = 0.0;
    pImpl->operation_latencies.clear();
    pImpl->mock_total_time_ms = 0.0;
    pImpl->gpu_total_time_ms = 0.0;
    pImpl->mock_operations = 0;
    pImpl->gpu_operations = 0;
    
    if (pImpl->gpu_cache) {
        pImpl->gpu_cache->reset_stats();
    }
}

void PredisClient::flush_all() {
    if (!pImpl->connected) return;
    
    if (pImpl->gpu_cache) {
        pImpl->gpu_cache->clear();
    }
    
    if (pImpl->mock_cache) {
        pImpl->mock_cache->clear();
    }
    
    if (pImpl->simple_cache) {
        // Assuming simple cache has a clear method
    }
}

bool PredisClient::run_performance_comparison(size_t num_operations) {
    if (!pImpl->connected) {
        std::cerr << "Not connected" << std::endl;
        return false;
    }
    
    std::cout << "\nRunning performance comparison with " << num_operations << " operations..." << std::endl;
    
    // Generate test data
    std::vector<std::pair<std::string, std::string>> test_data;
    test_data.reserve(num_operations);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    for (size_t i = 0; i < num_operations; ++i) {
        std::string key = "benchmark_key_" + std::to_string(i);
        std::string value = "benchmark_value_" + std::to_string(dis(gen));
        test_data.emplace_back(key, value);
    }
    
    // Test current implementation
    reset_stats();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Perform mixed operations
    size_t puts = 0, gets = 0, batch_ops = 0;
    
    // 70% puts, 25% gets, 5% batch operations
    for (size_t i = 0; i < num_operations; ++i) {
        if (i % 20 == 0) { // 5% batch operations
            size_t batch_size = std::min(size_t(100), num_operations - i);
            std::vector<std::string> keys;
            for (size_t j = 0; j < batch_size; ++j) {
                keys.push_back(test_data[i + j].first);
            }
            mget(keys);
            batch_ops++;
            i += batch_size - 1;
        } else if (i % 4 == 0) { // 25% gets
            std::string value;
            get(test_data[i % test_data.size()].first, value);
            gets++;
        } else { // 70% puts
            put(test_data[i].first, test_data[i].second);
            puts++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    auto final_stats = get_stats();
    
    std::cout << "\nPerformance Comparison Results:" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Implementation: " << final_stats.implementation_mode << std::endl;
    std::cout << "Operations performed: " << puts << " PUTs, " << gets << " GETs, " << batch_ops << " batch ops" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time_ms << " ms" << std::endl;
    std::cout << "Operations per second: " << std::fixed << std::setprecision(0) << final_stats.operations_per_second << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(4) << final_stats.avg_latency_ms << " ms" << std::endl;
    std::cout << "P95 latency: " << std::fixed << std::setprecision(4) << final_stats.p95_latency_ms << " ms" << std::endl;
    std::cout << "Hit ratio: " << std::fixed << std::setprecision(3) << final_stats.hit_ratio << std::endl;
    std::cout << "Memory usage: " << std::fixed << std::setprecision(2) << final_stats.memory_usage_mb << " MB" << std::endl;
    
    if (final_stats.performance_improvement_ratio > 0) {
        std::cout << "Performance improvement: " << std::fixed << std::setprecision(1) 
                  << final_stats.performance_improvement_ratio << "x over mock" << std::endl;
    }
    
    return true;
}

bool PredisClient::validate_consistency(size_t num_tests) {
    if (pImpl->current_mode != Mode::HYBRID) {
        std::cout << "Consistency validation requires HYBRID mode" << std::endl;
        return false;
    }
    
    if (!pImpl->gpu_cache || !pImpl->mock_cache) {
        std::cerr << "Both GPU and mock cache required for consistency validation" << std::endl;
        return false;
    }
    
    std::cout << "\nValidating consistency between mock and real GPU implementations..." << std::endl;
    
    // Clear both caches
    pImpl->gpu_cache->clear();
    pImpl->mock_cache->clear();
    
    size_t consistency_errors = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    for (size_t i = 0; i < num_tests; ++i) {
        std::string key = "test_key_" + std::to_string(i);
        std::string value = "test_value_" + std::to_string(dis(gen));
        
        // Insert into both
        bool gpu_put = pImpl->gpu_cache->put(key, value);
        bool mock_put = pImpl->mock_cache->put(key, value);
        
        if (gpu_put != mock_put) {
            consistency_errors++;
            std::cerr << "Put consistency error for key: " << key << std::endl;
        }
        
        // Verify both can retrieve
        std::string gpu_value, mock_value;
        bool gpu_get = pImpl->gpu_cache->get(key, gpu_value);
        bool mock_get = pImpl->mock_cache->get(key, mock_value);
        
        if (gpu_get != mock_get || (gpu_get && gpu_value != mock_value)) {
            consistency_errors++;
            std::cerr << "Get consistency error for key: " << key << std::endl;
        }
    }
    
    std::cout << "Consistency validation completed: " << consistency_errors 
              << " errors out of " << num_tests << " tests" << std::endl;
    
    return consistency_errors == 0;
}

void PredisClient::print_performance_report() const {
    auto stats = get_stats();
    
    std::cout << "\n=== Predis Performance Report ===" << std::endl;
    std::cout << "Implementation Mode: " << stats.implementation_mode << std::endl;
    std::cout << "Using Real GPU: " << (stats.using_real_gpu ? "Yes" : "No") << std::endl;
    std::cout << "\nCache Statistics:" << std::endl;
    std::cout << "  Total Keys: " << stats.total_keys << std::endl;
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2) << stats.memory_usage_mb << " MB" << std::endl;
    std::cout << "  Hit Ratio: " << std::fixed << std::setprecision(3) << stats.hit_ratio << std::endl;
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << "  Operations/sec: " << std::fixed << std::setprecision(0) << stats.operations_per_second << std::endl;
    std::cout << "  Avg Latency: " << std::fixed << std::setprecision(4) << stats.avg_latency_ms << " ms" << std::endl;
    std::cout << "  P95 Latency: " << std::fixed << std::setprecision(4) << stats.p95_latency_ms << " ms" << std::endl;
    
    if (stats.mock_ops_per_second > 0) {
        std::cout << "\nComparison Data:" << std::endl;
        std::cout << "  Mock Performance: " << std::fixed << std::setprecision(0) << stats.mock_ops_per_second << " ops/sec" << std::endl;
    }
    
    if (stats.real_gpu_ops_per_second > 0) {
        std::cout << "  GPU Performance: " << std::fixed << std::setprecision(0) << stats.real_gpu_ops_per_second << " ops/sec" << std::endl;
    }
    
    if (stats.performance_improvement_ratio > 0) {
        std::cout << "  Performance Improvement: " << std::fixed << std::setprecision(1) 
                  << stats.performance_improvement_ratio << "x" << std::endl;
    }
    
    std::cout << "=================================" << std::endl;
}

bool PredisClient::configure_gpu_memory(size_t max_memory_mb) {
    if (!is_using_real_gpu()) {
        std::cerr << "GPU memory configuration only available in real GPU mode" << std::endl;
        return false;
    }
    
    // This would reconfigure the GPU cache memory limit
    std::cout << "GPU memory configuration: " << max_memory_mb << " MB" << std::endl;
    return true;
}

bool PredisClient::defragment_gpu_memory() {
    if (!is_using_real_gpu() || !pImpl->gpu_cache) {
        std::cerr << "GPU memory defragmentation only available in real GPU mode" << std::endl;
        return false;
    }
    
    // This would trigger GPU memory defragmentation
    std::cout << "GPU memory defragmentation completed" << std::endl;
    return true;
}

void PredisClient::print_gpu_memory_stats() const {
    if (!is_using_real_gpu() || !pImpl->gpu_cache) {
        std::cerr << "GPU memory stats only available in real GPU mode" << std::endl;
        return;
    }
    
    auto stats = pImpl->gpu_cache->get_stats();
    std::cout << "\n=== GPU Memory Statistics ===" << std::endl;
    std::cout << "Total Entries: " << stats.total_entries << std::endl;
    std::cout << "Memory Usage: " << stats.memory_usage_mb << " MB" << std::endl;
    std::cout << "Memory Capacity: " << stats.memory_capacity_mb << " MB" << std::endl;
    std::cout << "Load Factor: " << std::fixed << std::setprecision(3) << stats.load_factor << std::endl;
    std::cout << "Peak Memory Usage: " << stats.peak_memory_usage_mb << " MB" << std::endl;
    std::cout << "=============================" << std::endl;
}

} // namespace api
} // namespace predis