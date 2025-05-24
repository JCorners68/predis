# Enterprise components for Epic 4

# Monitoring integrations
add_library(enterprise_monitoring STATIC
    integration/monitoring_integrations.cpp
    integration/prometheus_exporter.cpp
    integration/datadog_exporter.cpp
    integration/newrelic_exporter.cpp
    integration/cloudwatch_exporter.cpp
)

target_include_directories(enterprise_monitoring PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(enterprise_monitoring
    pthread
    curl  # for HTTP requests
)

# Deployment management
add_library(enterprise_deployment STATIC
    deployment/zero_downtime_deployment.cpp
    deployment/blue_green_controller.cpp
    deployment/canary_release_manager.cpp
)

target_include_directories(enterprise_deployment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(enterprise_deployment
    pthread
)

# Multi-cloud orchestration
add_library(enterprise_orchestration STATIC
    orchestration/multi_cloud_orchestrator.cpp
    orchestration/cost_optimizer.cpp
    orchestration/compliance_manager.cpp
)

target_include_directories(enterprise_orchestration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(enterprise_orchestration
    pthread
)

# AIOps monitoring
add_library(enterprise_aiops STATIC
    monitoring/aiops_monitoring.cpp
    monitoring/predictive_analytics.cpp
    monitoring/auto_remediation.cpp
)

target_include_directories(enterprise_aiops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ml
)

target_link_libraries(enterprise_aiops
    ml_models
    pthread
)

# Combined enterprise library
add_library(enterprise STATIC)

target_link_libraries(enterprise
    enterprise_monitoring
    enterprise_deployment
    enterprise_orchestration
    enterprise_aiops
)

# Enterprise tests
if(BUILD_TESTS)
    add_executable(enterprise_test
        tests/monitoring_integration_test.cpp
        tests/deployment_test.cpp
        tests/orchestration_test.cpp
    )
    
    target_link_libraries(enterprise_test
        enterprise
        gtest
        gtest_main
        pthread
    )
    
    add_test(NAME enterprise_test COMMAND enterprise_test)
endif()