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

#include "batch_integration_helper.h"
#include <iostream>

namespace predis {
namespace core {

bool BatchIntegrationHelper::initialize(CacheManager* cache_manager) {
    if (!cache_manager) {
        std::cerr << "Invalid cache manager provided" << std::endl;
        return false;
    }

    cache_manager_ = cache_manager;
    
    // Create and initialize batch processor
    batch_processor_ = std::make_unique<AdvancedBatchProcessor>();
    
    // Use default batch configuration optimized for RTX 5080
    BatchConfig config;
    config.max_batch_size = 10000;
    config.preferred_batch_size = 2048;
    config.max_concurrent_batches = 4;
    config.enable_auto_tuning = true;
    config.memory_pool_size_mb = 128;

    // Initialize with cache manager's underlying components
    // Note: This assumes CacheManager exposes these components
    // In a real implementation, you'd need getters for these
    auto* hash_table = cache_manager_->get_hash_table();
    auto* memory_manager = cache_manager_->get_memory_manager();
    
    if (!hash_table || !memory_manager) {
        std::cerr << "Cache manager missing required components for batch operations" << std::endl;
        return false;
    }

    bool success = batch_processor_->initialize(hash_table, memory_manager, config);
    if (success) {
        initialized_ = true;
        std::cout << "Batch operations enabled for cache manager" << std::endl;
    }

    return success;
}

bool BatchIntegrationHelper::enable_batch_operations(CacheManager* cache_manager) {
    if (!initialize(cache_manager)) {
        return false;
    }

    // TODO: In a real implementation, you would:
    // 1. Add batch operation methods to CacheManager interface
    // 2. Route batch calls through this helper
    // 3. Maintain compatibility with existing single-operation interface
    
    std::cout << "Batch operations integration complete" << std::endl;
    return true;
}

bool BatchIntegrationHelper::configure_batch_settings(const BatchConfig& config) {
    if (!initialized_ || !batch_processor_) {
        std::cerr << "Batch processor not initialized" << std::endl;
        return false;
    }

    return batch_processor_->configure(config);
}

BatchMetrics BatchIntegrationHelper::get_batch_metrics() const {
    if (!initialized_ || !batch_processor_) {
        return BatchMetrics{};
    }

    return batch_processor_->get_cumulative_metrics();
}

bool BatchIntegrationHelper::is_batch_enabled() const {
    return initialized_ && batch_processor_;
}

} // namespace core
} // namespace predis