# MLOps components for Epic 4

# Data collector library
add_library(mlops_data_collector STATIC
    data_collector/production_data_collector.cpp
)

target_include_directories(mlops_data_collector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(mlops_data_collector
    pthread
    z  # for zlib compression
)

# Drift detection library
add_library(mlops_drift_detection STATIC
    drift_detection/model_drift_detector.cpp
)

target_include_directories(mlops_drift_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(mlops_drift_detection
    pthread
)

# Retraining pipeline library
add_library(mlops_retraining STATIC
    retraining/auto_retraining_pipeline.cpp
)

target_include_directories(mlops_retraining PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ml
)

target_link_libraries(mlops_retraining
    ml_models
    mlops_drift_detection
    pthread
)

# Customer adaptation library
add_library(mlops_customer_adaptation STATIC
    customer_adaptation/customer_model_manager.cpp
    customer_adaptation/federated_learning_client.cpp
)

target_include_directories(mlops_customer_adaptation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(mlops_customer_adaptation
    ml_models
    pthread
)

# Combined MLOps library
add_library(mlops STATIC)

target_link_libraries(mlops
    mlops_data_collector
    mlops_drift_detection
    mlops_retraining
    mlops_customer_adaptation
)

# MLOps tests
if(BUILD_TESTS)
    add_executable(mlops_test
        tests/data_collector_test.cpp
        tests/drift_detector_test.cpp
        tests/retraining_pipeline_test.cpp
    )
    
    target_link_libraries(mlops_test
        mlops
        gtest
        gtest_main
        pthread
    )
    
    add_test(NAME mlops_test COMMAND mlops_test)
endif()