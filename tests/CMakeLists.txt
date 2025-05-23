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

# GPU tests
add_executable(cuda_test
    gpu/cuda_test.cu
)

set_property(TARGET cuda_test PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cuda_test
    ${CUDA_LIBRARIES}
)

# Unit tests
add_executable(predis_unit_tests
    unit/api_tests.cpp
    unit/core_tests.cpp
    unit/logger_tests.cpp
    unit/ppe_tests.cpp
)

target_link_libraries(predis_unit_tests
    predis_core
    gtest_main
    gtest
    ${CUDA_LIBRARIES}
)

target_include_directories(predis_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Integration tests
add_executable(predis_integration_tests
    integration/full_system_tests.cpp
    integration/performance_tests.cpp
    integration/prefetch_integration_test.cpp
)

target_link_libraries(predis_integration_tests
    predis_core
    gtest_main
    gtest
    ${CUDA_LIBRARIES}
)

target_include_directories(predis_integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# GPU Integration Test
add_executable(gpu_integration_test
    gpu_integration_test.cpp
)

target_link_libraries(gpu_integration_test
    predis_core
    ${CUDA_LIBRARIES}
)

target_include_directories(gpu_integration_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Machine Learning tests
add_executable(predis_ml_tests
    ml/model_tests.cpp
    ml/feature_engineering_test.cpp
    ml/adaptive_learning_test.cpp
    ml/ds_metrics_test.cpp
)

target_link_libraries(predis_ml_tests
    predis_core
    gtest_main
    gtest
    ${CUDA_LIBRARIES}
)

target_include_directories(predis_ml_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Register tests with CTest
add_test(NAME CudaTest COMMAND cuda_test)
add_test(NAME UnitTests COMMAND predis_unit_tests)
add_test(NAME IntegrationTests COMMAND predis_integration_tests)
add_test(NAME GpuIntegrationTest COMMAND gpu_integration_test)
add_test(NAME MLTests COMMAND predis_ml_tests)

# Performance benchmarks
add_executable(predis_benchmarks
    performance/benchmark_suite.cpp
    performance/ml_performance_test.cpp
)

target_link_libraries(predis_benchmarks
    predis_core
    ${CUDA_LIBRARIES}
)

target_include_directories(predis_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)