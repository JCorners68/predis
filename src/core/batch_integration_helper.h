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

#include "advanced_batch_processor.h"
#include "cache_manager.h"
#include <memory>

namespace predis {
namespace core {

/**
 * Integration helper to connect AdvancedBatchProcessor with existing cache system
 * Provides seamless batch operations through the existing cache interface
 */
class BatchIntegrationHelper {
public:
    BatchIntegrationHelper() = default;
    ~BatchIntegrationHelper() = default;

    /**
     * Initialize batch integration with existing cache manager
     */
    bool initialize(CacheManager* cache_manager);

    /**
     * Add batch capabilities to an existing cache manager
     * This enables the cache manager to use optimized batch operations
     */
    bool enable_batch_operations(CacheManager* cache_manager);

    /**
     * Configure batch processor settings through cache manager
     */
    bool configure_batch_settings(const BatchConfig& config);

    /**
     * Get batch performance metrics through cache manager interface
     */
    BatchMetrics get_batch_metrics() const;

    /**
     * Check if batch operations are available and configured
     */
    bool is_batch_enabled() const;

private:
    std::unique_ptr<AdvancedBatchProcessor> batch_processor_;
    CacheManager* cache_manager_ = nullptr;
    bool initialized_ = false;
};

} // namespace core
} // namespace predis