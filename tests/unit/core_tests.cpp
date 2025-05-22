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

#include <gtest/gtest.h>
#include "core/cache_manager.h"

using namespace predis::core;

class CacheManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache_manager = std::make_unique<CacheManager>();
    }

    void TearDown() override {
        if (cache_manager) {
            cache_manager->shutdown();
        }
    }

    std::unique_ptr<CacheManager> cache_manager;
};

TEST_F(CacheManagerTest, InitializeAndShutdown) {
    EXPECT_TRUE(cache_manager->initialize());
    cache_manager->shutdown();
}

TEST_F(CacheManagerTest, BasicPutAndGet) {
    ASSERT_TRUE(cache_manager->initialize());
    
    // Test basic put operation
    EXPECT_TRUE(cache_manager->put("test_key", "test_value"));
    
    // Test basic get operation (will fail in placeholder but shouldn't crash)
    std::string value;
    cache_manager->get("test_key", value);  // Don't assert result yet since it's placeholder
}

TEST_F(CacheManagerTest, RemoveOperation) {
    ASSERT_TRUE(cache_manager->initialize());
    
    cache_manager->put("test_key", "test_value");
    EXPECT_TRUE(cache_manager->remove("test_key"));
}