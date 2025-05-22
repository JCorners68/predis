# Copyright 2025 Predis Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)

project(Predis 
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated key-value cache with predictive prefetching"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Download and build Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Create predis_core library
add_library(predis_core STATIC
    # Core components
    src/core/simple_cache_manager.cpp
    
    # API components
    src/api/predis_client.cpp
    
    # Placeholder source files (will be implemented in Epic 1)
    # src/core/memory_manager.cu
    # src/core/data_structures/gpu_hash_table.cu
    # src/logger/access_logger.cpp
    # src/ppe/prefetch_coordinator.cpp
)

target_include_directories(predis_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(predis_core
    Threads::Threads
)

# Create main server executable
add_executable(predis_server
    src/predis_server.cpp
)

target_link_libraries(predis_server
    predis_core
)

# Architecture validation test
add_executable(arch_validation_test
    src/arch_validation_test.cpp
)

target_link_libraries(arch_validation_test
    predis_core
)

# Set output directories
set_target_properties(predis_server arch_validation_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable testing
enable_testing()

# Unit tests
add_executable(predis_unit_tests
    tests/unit/api_tests.cpp
    tests/unit/core_tests.cpp
    tests/unit/logger_tests.cpp
    tests/unit/ppe_tests.cpp
)

target_link_libraries(predis_unit_tests
    predis_core
    gtest_main
    gtest
)

target_include_directories(predis_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Integration tests
add_executable(predis_integration_tests
    tests/integration/full_system_tests.cpp
    tests/integration/performance_tests.cpp
)

target_link_libraries(predis_integration_tests
    predis_core
    gtest_main
    gtest
)

target_include_directories(predis_integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Performance benchmarks
add_executable(predis_benchmarks
    tests/performance/benchmark_suite.cpp
)

target_link_libraries(predis_benchmarks
    predis_core
)

target_include_directories(predis_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Register tests with CTest
add_test(NAME UnitTests COMMAND predis_unit_tests)
add_test(NAME IntegrationTests COMMAND predis_integration_tests)

# Custom targets for development
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build artifacts"
)

# Print configuration summary
message(STATUS "=== Predis Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Testing: ${BUILD_TESTING}")
message(STATUS "======================================")