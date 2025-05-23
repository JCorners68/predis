#include "../../src/logger/access_pattern_logger.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <iomanip>

using namespace predis::logger;
using namespace std::chrono;

class AccessLoggerOverheadTest {
private:
    static constexpr size_t NUM_OPERATIONS = 10000000;  // 10M operations
    static constexpr size_t NUM_THREADS = 8;
    static constexpr size_t NUM_KEYS = 100000;
    
    std::vector<uint64_t> key_hashes_;
    std::mt19937_64 rng_{42};
    
public:
    AccessLoggerOverheadTest() {
        // Pre-generate key hashes
        key_hashes_.reserve(NUM_KEYS);
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            key_hashes_.push_back(std::hash<std::string>{}("key_" + std::to_string(i)));
        }
    }
    
    double measure_baseline_performance() {
        std::cout << "\nMeasuring baseline performance (no logging)..." << std::endl;
        
        auto start = high_resolution_clock::now();
        
        // Simulate cache operations without logging
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            uint64_t key_hash = key_hashes_[i % NUM_KEYS];
            auto op_type = static_cast<AccessPatternLogger::OperationType>(i % 4);
            auto result = static_cast<AccessPatternLogger::CacheResult>(i % 2);
            uint32_t value_size = 64 + (i % 1024);
            
            // Simulate some work
            volatile uint64_t dummy = key_hash ^ value_size;
            (void)dummy;
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        
        double ops_per_sec = (NUM_OPERATIONS * 1000000.0) / duration;
        std::cout << "Baseline: " << std::fixed << std::setprecision(0) 
                  << ops_per_sec << " ops/sec" << std::endl;
        
        return ops_per_sec;
    }
    
    double measure_logging_performance(AccessPatternLogger& logger) {
        std::cout << "\nMeasuring performance with logging enabled..." << std::endl;
        
        auto start = high_resolution_clock::now();
        
        // Same operations but with logging
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            uint64_t key_hash = key_hashes_[i % NUM_KEYS];
            auto op_type = static_cast<AccessPatternLogger::OperationType>(i % 4);
            auto result = static_cast<AccessPatternLogger::CacheResult>(i % 2);
            uint32_t value_size = 64 + (i % 1024);
            
            // Log the access
            logger.log_access(key_hash, op_type, result, value_size);
            
            // Simulate some work
            volatile uint64_t dummy = key_hash ^ value_size;
            (void)dummy;
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        
        double ops_per_sec = (NUM_OPERATIONS * 1000000.0) / duration;
        std::cout << "With logging: " << std::fixed << std::setprecision(0) 
                  << ops_per_sec << " ops/sec" << std::endl;
        
        return ops_per_sec;
    }
    
    void measure_multi_threaded_overhead() {
        std::cout << "\nMeasuring multi-threaded performance..." << std::endl;
        
        AccessPatternLogger::Config config;
        config.buffer_capacity = 10000000;  // 10M events
        AccessPatternLogger logger(config);
        
        // Baseline multi-threaded
        auto mt_start = high_resolution_clock::now();
        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < NUM_THREADS; ++t) {
                threads.emplace_back([this, t]() {
                    size_t ops_per_thread = NUM_OPERATIONS / NUM_THREADS;
                    for (size_t i = 0; i < ops_per_thread; ++i) {
                        size_t idx = t * ops_per_thread + i;
                        uint64_t key_hash = key_hashes_[idx % NUM_KEYS];
                        volatile uint64_t dummy = key_hash ^ idx;
                        (void)dummy;
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
        auto mt_baseline_duration = duration_cast<microseconds>(
            high_resolution_clock::now() - mt_start).count();
        
        // With logging multi-threaded
        mt_start = high_resolution_clock::now();
        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < NUM_THREADS; ++t) {
                threads.emplace_back([this, &logger, t]() {
                    size_t ops_per_thread = NUM_OPERATIONS / NUM_THREADS;
                    for (size_t i = 0; i < ops_per_thread; ++i) {
                        size_t idx = t * ops_per_thread + i;
                        uint64_t key_hash = key_hashes_[idx % NUM_KEYS];
                        auto op_type = static_cast<AccessPatternLogger::OperationType>(i % 4);
                        auto result = static_cast<AccessPatternLogger::CacheResult>(i % 2);
                        uint32_t value_size = 64 + (i % 1024);
                        
                        logger.log_access(key_hash, op_type, result, value_size);
                        
                        volatile uint64_t dummy = key_hash ^ idx;
                        (void)dummy;
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
        auto mt_logging_duration = duration_cast<microseconds>(
            high_resolution_clock::now() - mt_start).count();
        
        double mt_baseline_ops = (NUM_OPERATIONS * 1000000.0) / mt_baseline_duration;
        double mt_logging_ops = (NUM_OPERATIONS * 1000000.0) / mt_logging_duration;
        double mt_overhead = ((mt_baseline_ops - mt_logging_ops) / mt_baseline_ops) * 100;
        
        std::cout << "Multi-threaded baseline: " << std::fixed << std::setprecision(0) 
                  << mt_baseline_ops << " ops/sec" << std::endl;
        std::cout << "Multi-threaded with logging: " << mt_logging_ops << " ops/sec" << std::endl;
        std::cout << "Multi-threaded overhead: " << std::fixed << std::setprecision(2) 
                  << mt_overhead << "%" << std::endl;
    }
    
    void test_pattern_analysis_performance() {
        std::cout << "\nTesting pattern analysis performance..." << std::endl;
        
        // Generate test events
        std::vector<AccessEvent> events;
        events.reserve(100000);
        
        for (size_t i = 0; i < 100000; ++i) {
            events.push_back({
                .timestamp_us = static_cast<uint64_t>(i * 1000),  // 1ms intervals
                .key_hash = key_hashes_[i % 1000],  // Use subset of keys
                .value_size = 256,
                .thread_id = static_cast<uint16_t>(i % 4),
                .operation_type = static_cast<uint8_t>(i % 2),  // GET/PUT
                .cache_result = 0  // HIT
            });
        }
        
        PatternAnalyzer analyzer;
        
        auto start = high_resolution_clock::now();
        auto patterns = analyzer.analyze_events(events);
        auto end = high_resolution_clock::now();
        
        auto analysis_time = duration_cast<milliseconds>(end - start).count();
        
        std::cout << "Analyzed " << events.size() << " events in " 
                  << analysis_time << "ms" << std::endl;
        std::cout << "Found " << patterns.size() << " patterns" << std::endl;
        std::cout << "Analysis throughput: " 
                  << (events.size() * 1000.0 / analysis_time) << " events/sec" << std::endl;
    }
    
    void run_all_tests() {
        std::cout << "=== Access Pattern Logger Performance Test ===" << std::endl;
        std::cout << "Operations: " << NUM_OPERATIONS << std::endl;
        std::cout << "Threads: " << NUM_THREADS << std::endl;
        std::cout << "Keys: " << NUM_KEYS << std::endl;
        
        // Single-threaded tests
        AccessPatternLogger::Config config;
        config.buffer_capacity = 10000000;  // 10M events
        AccessPatternLogger logger(config);
        
        double baseline_ops = measure_baseline_performance();
        double logging_ops = measure_logging_performance(logger);
        
        double overhead = ((baseline_ops - logging_ops) / baseline_ops) * 100;
        
        std::cout << "\nSingle-threaded overhead: " << std::fixed << std::setprecision(2) 
                  << overhead << "%" << std::endl;
        
        // Print statistics
        auto stats = logger.get_statistics();
        std::cout << "\nLogger Statistics:" << std::endl;
        std::cout << "  Events logged: " << stats.events_logged << std::endl;
        std::cout << "  Events dropped: " << stats.events_dropped << std::endl;
        std::cout << "  Buffer usage: " << stats.buffer_usage << std::endl;
        std::cout << "  Avg log time: " << stats.avg_log_time_ns << " ns" << std::endl;
        
        // Multi-threaded tests
        measure_multi_threaded_overhead();
        
        // Pattern analysis tests
        test_pattern_analysis_performance();
        
        // Verify overhead is less than 1%
        if (overhead < 1.0) {
            std::cout << "\n✅ SUCCESS: Overhead is less than 1% target!" << std::endl;
        } else {
            std::cout << "\n❌ FAILURE: Overhead exceeds 1% target!" << std::endl;
        }
    }
};

int main() {
    AccessLoggerOverheadTest test;
    test.run_all_tests();
    return 0;
}