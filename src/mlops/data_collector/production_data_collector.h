#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace predis {
namespace mlops {

// Represents a single cache access event for ML training
struct AccessEvent {
    uint64_t timestamp;
    std::string key;
    std::string operation;  // GET, PUT, DELETE
    bool hit;              // Cache hit or miss
    uint32_t latency_us;   // Operation latency in microseconds
    
    size_t GetSize() const {
        return sizeof(timestamp) + key.size() + operation.size() + 
               sizeof(hit) + sizeof(latency_us);
    }
};

// Configuration for production data collection
struct CollectorConfig {
    size_t buffer_capacity = 1'000'000;  // 1M events
    double sampling_rate = 0.001;        // 0.1% sampling
    size_t export_batch_size = 10'000;   // Export in 10K batches
    std::chrono::seconds export_interval{60};  // Export every minute
    std::string export_path = "/var/lib/predis/ml_data/";
    bool enable_compression = true;
};

// Lock-free circular buffer for high-performance event storage
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity);
    
    bool Push(const T& item);
    bool Pop(T& item);
    size_t Size() const;
    bool Empty() const;
    
    // Batch operations for efficiency
    size_t PushBatch(const std::vector<T>& items);
    size_t PopBatch(std::vector<T>& items, size_t max_items);
    
private:
    std::vector<T> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    const size_t capacity_;
    
    size_t GetNextIndex(size_t current) const {
        return (current + 1) % capacity_;
    }
};

// Main production data collector with minimal overhead
class ProductionDataCollector {
public:
    explicit ProductionDataCollector(const CollectorConfig& config);
    ~ProductionDataCollector();
    
    // Log access with minimal overhead (<0.1%)
    void LogAccess(const std::string& key, 
                   const std::string& operation,
                   bool hit,
                   uint32_t latency_us);
    
    // Control methods
    void Start();
    void Stop();
    void SetSamplingRate(double rate);
    
    // Metrics
    struct Metrics {
        uint64_t total_events;
        uint64_t sampled_events;
        uint64_t exported_events;
        uint64_t dropped_events;
        double avg_export_time_ms;
        size_t buffer_usage_pct;
    };
    
    Metrics GetMetrics() const;
    
private:
    CollectorConfig config_;
    std::unique_ptr<CircularBuffer<AccessEvent>> buffer_;
    std::thread export_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> total_events_{0};
    std::atomic<uint64_t> sampled_events_{0};
    std::atomic<uint64_t> exported_events_{0};
    std::atomic<uint64_t> dropped_events_{0};
    
    // Background export functionality
    void ExportLoop();
    void ExportBatch(const std::vector<AccessEvent>& events);
    std::string GenerateExportFilename() const;
    
    // Sampling decision
    bool ShouldSample() const;
    
    // Fast random number generation for sampling
    mutable std::atomic<uint64_t> rng_state_;
    uint64_t FastRandom() const;
};

// Feature extractor that runs in background thread
class RealTimeFeatureExtractor {
public:
    RealTimeFeatureExtractor();
    
    // Extract features from recent access patterns
    std::vector<float> ExtractFeatures(const std::vector<AccessEvent>& events);
    
    // Online feature statistics
    void UpdateStatistics(const AccessEvent& event);
    
private:
    // Feature extraction windows
    static constexpr size_t TEMPORAL_WINDOWS[] = {60, 300, 900, 3600};  // 1m, 5m, 15m, 1h
    
    // Online statistics
    struct FeatureStats {
        double mean;
        double variance;
        uint64_t count;
    };
    
    std::vector<FeatureStats> feature_stats_;
};

}  // namespace mlops
}  // namespace predis