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

# Core library
add_library(predis_core STATIC
    # Core cache components
    core/simple_cache_manager.cpp
    core/consistency_controller.cpp
    
    # API components
    api/predis_client.cpp
    
    # Logging components
    logger/access_logger.cpp
    logger/log_processor.cpp
    logger/log_buffer.cpp
    
    # Predictive Prefetching Engine
    ppe/prefetch_coordinator.cpp
    ppe/prefetch_monitor.cpp
    ppe/data_ingestor.cpp
    ppe/feature_generator.cpp
    ppe/prefetch_strategist.cpp
    ppe/prefetch_executor.cpp
    
    # Machine Learning components
    ml/feature_engineering.cpp
    ml/inference_engine.cpp
    ml/models/lstm_model.cpp
    ml/models/xgboost_model.cpp
    ml/models/ensemble_model.cpp
    ml/models/model_factory.cpp
    ml/adaptive_learning_system.cpp
    ml/model_performance_monitor.cpp
    
    # Benchmarks
    benchmarks/ml_performance_validator.cpp
    benchmarks/ds_metrics_suite.cpp
    benchmarks/write_performance_profiler.cpp
    
    # Dashboard
    dashboard/ml_performance_dashboard.cpp
    
    # Utilities
    utils/config_loader.cpp
    utils/error_handler.cpp
    utils/thread_pool.cpp
    utils/logger.cpp
)

target_include_directories(predis_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(predis_core
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# Set CUDA properties
set_property(TARGET predis_core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET predis_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Main server executable
add_executable(predis_server
    predis_server.cpp
)

target_link_libraries(predis_server
    predis_core
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# Architecture validation test
add_executable(arch_validation_test
    arch_validation_test.cpp
)

target_link_libraries(arch_validation_test
    predis_core
    Threads::Threads
)

# DS Metrics Benchmark executable
add_executable(ds_metrics_benchmark
    benchmarks/ds_metrics_benchmark.cpp
)

target_link_libraries(ds_metrics_benchmark
    predis_core
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# Set output directories
set_target_properties(predis_server arch_validation_test ds_metrics_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Epic 4: MLOps and Enterprise features
if(MLOPS_ENABLED)
    add_subdirectory(mlops)
    add_subdirectory(enterprise)
    
    target_link_libraries(predis_core
        mlops
        enterprise
    )
endif()