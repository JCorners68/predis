#include "production_data_collector.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <zlib.h>
#include <iomanip>
#include <algorithm>

namespace predis {
namespace mlops {

// CircularBuffer implementation
template<typename T>
CircularBuffer<T>::CircularBuffer(size_t capacity) 
    : buffer_(capacity), capacity_(capacity) {
}

template<typename T>
bool CircularBuffer<T>::Push(const T& item) {
    size_t current_tail = tail_.load(std::memory_order_relaxed);
    size_t next_tail = GetNextIndex(current_tail);
    
    // Check if buffer is full
    if (next_tail == head_.load(std::memory_order_acquire)) {
        return false;  // Buffer full
    }
    
    buffer_[current_tail] = item;
    tail_.store(next_tail, std::memory_order_release);
    return true;
}

template<typename T>
bool CircularBuffer<T>::Pop(T& item) {
    size_t current_head = head_.load(std::memory_order_relaxed);
    
    // Check if buffer is empty
    if (current_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Buffer empty
    }
    
    item = buffer_[current_head];
    head_.store(GetNextIndex(current_head), std::memory_order_release);
    return true;
}

template<typename T>
size_t CircularBuffer<T>::Size() const {
    size_t head = head_.load(std::memory_order_acquire);
    size_t tail = tail_.load(std::memory_order_acquire);
    
    if (tail >= head) {
        return tail - head;
    } else {
        return capacity_ - head + tail;
    }
}

template<typename T>
bool CircularBuffer<T>::Empty() const {
    return head_.load(std::memory_order_acquire) == 
           tail_.load(std::memory_order_acquire);
}

template<typename T>
size_t CircularBuffer<T>::PushBatch(const std::vector<T>& items) {
    size_t pushed = 0;
    for (const auto& item : items) {
        if (!Push(item)) break;
        pushed++;
    }
    return pushed;
}

template<typename T>
size_t CircularBuffer<T>::PopBatch(std::vector<T>& items, size_t max_items) {
    items.clear();
    items.reserve(max_items);
    
    T item;
    size_t popped = 0;
    while (popped < max_items && Pop(item)) {
        items.push_back(std::move(item));
        popped++;
    }
    
    return popped;
}

// Explicit template instantiation
template class CircularBuffer<AccessEvent>;

// ProductionDataCollector implementation
ProductionDataCollector::ProductionDataCollector(const CollectorConfig& config)
    : config_(config), 
      buffer_(std::make_unique<CircularBuffer<AccessEvent>>(config.buffer_capacity)),
      rng_state_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

ProductionDataCollector::~ProductionDataCollector() {
    Stop();
}

void ProductionDataCollector::LogAccess(const std::string& key, 
                                       const std::string& operation,
                                       bool hit,
                                       uint32_t latency_us) {
    total_events_.fetch_add(1, std::memory_order_relaxed);
    
    // Sampling decision
    if (!ShouldSample()) {
        return;
    }
    
    AccessEvent event;
    event.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    event.key = key;
    event.operation = operation;
    event.hit = hit;
    event.latency_us = latency_us;
    
    if (!buffer_->Push(event)) {
        dropped_events_.fetch_add(1, std::memory_order_relaxed);
    } else {
        sampled_events_.fetch_add(1, std::memory_order_relaxed);
    }
}

void ProductionDataCollector::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;  // Already running
    }
    
    export_thread_ = std::thread(&ProductionDataCollector::ExportLoop, this);
}

void ProductionDataCollector::Stop() {
    running_.store(false);
    if (export_thread_.joinable()) {
        export_thread_.join();
    }
}

void ProductionDataCollector::SetSamplingRate(double rate) {
    config_.sampling_rate = std::max(0.0, std::min(1.0, rate));
}

ProductionDataCollector::Metrics ProductionDataCollector::GetMetrics() const {
    Metrics metrics;
    metrics.total_events = total_events_.load();
    metrics.sampled_events = sampled_events_.load();
    metrics.exported_events = exported_events_.load();
    metrics.dropped_events = dropped_events_.load();
    metrics.buffer_usage_pct = (buffer_->Size() * 100) / config_.buffer_capacity;
    
    // TODO: Track export timing
    metrics.avg_export_time_ms = 0.0;
    
    return metrics;
}

void ProductionDataCollector::ExportLoop() {
    std::vector<AccessEvent> batch;
    batch.reserve(config_.export_batch_size);
    
    while (running_.load()) {
        // Wait for export interval
        std::this_thread::sleep_for(config_.export_interval);
        
        // Collect batch
        size_t collected = buffer_->PopBatch(batch, config_.export_batch_size);
        
        if (collected > 0) {
            auto start = std::chrono::high_resolution_clock::now();
            ExportBatch(batch);
            auto end = std::chrono::high_resolution_clock::now();
            
            // Update metrics
            exported_events_.fetch_add(collected, std::memory_order_relaxed);
            
            // TODO: Track export timing
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            (void)duration;  // Suppress unused warning
        }
    }
    
    // Final export on shutdown
    while (!buffer_->Empty()) {
        size_t collected = buffer_->PopBatch(batch, config_.export_batch_size);
        if (collected > 0) {
            ExportBatch(batch);
            exported_events_.fetch_add(collected, std::memory_order_relaxed);
        }
    }
}

void ProductionDataCollector::ExportBatch(const std::vector<AccessEvent>& events) {
    std::string filename = GenerateExportFilename();
    
    if (config_.enable_compression) {
        // Export as compressed JSON
        std::ostringstream json_stream;
        json_stream << "[\n";
        
        for (size_t i = 0; i < events.size(); ++i) {
            const auto& event = events[i];
            json_stream << "  {\n"
                       << "    \"timestamp\": " << event.timestamp << ",\n"
                       << "    \"key\": \"" << event.key << "\",\n"
                       << "    \"operation\": \"" << event.operation << "\",\n"
                       << "    \"hit\": " << (event.hit ? "true" : "false") << ",\n"
                       << "    \"latency_us\": " << event.latency_us << "\n"
                       << "  }";
            
            if (i < events.size() - 1) {
                json_stream << ",";
            }
            json_stream << "\n";
        }
        
        json_stream << "]\n";
        
        // Compress using zlib
        std::string json_data = json_stream.str();
        uLongf compressed_size = compressBound(json_data.size());
        std::vector<uint8_t> compressed(compressed_size);
        
        if (compress(compressed.data(), &compressed_size, 
                    reinterpret_cast<const Bytef*>(json_data.data()), 
                    json_data.size()) == Z_OK) {
            
            std::ofstream file(filename + ".gz", std::ios::binary);
            if (file.is_open()) {
                file.write(reinterpret_cast<const char*>(compressed.data()), compressed_size);
                file.close();
            }
        }
    } else {
        // Export as plain CSV for simplicity
        std::ofstream file(filename + ".csv");
        if (file.is_open()) {
            file << "timestamp,key,operation,hit,latency_us\n";
            
            for (const auto& event : events) {
                file << event.timestamp << ","
                     << event.key << ","
                     << event.operation << ","
                     << (event.hit ? "1" : "0") << ","
                     << event.latency_us << "\n";
            }
            
            file.close();
        }
    }
}

std::string ProductionDataCollector::GenerateExportFilename() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream filename;
    filename << config_.export_path << "access_log_"
             << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    return filename.str();
}

bool ProductionDataCollector::ShouldSample() const {
    if (config_.sampling_rate >= 1.0) {
        return true;
    }
    
    if (config_.sampling_rate <= 0.0) {
        return false;
    }
    
    // Fast random sampling
    uint64_t random_val = FastRandom();
    double random_double = static_cast<double>(random_val % 10000) / 10000.0;
    
    return random_double < config_.sampling_rate;
}

uint64_t ProductionDataCollector::FastRandom() const {
    // xorshift64 algorithm for fast pseudo-random numbers
    uint64_t x = rng_state_.load();
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state_.store(x);
    return x;
}

// RealTimeFeatureExtractor implementation
RealTimeFeatureExtractor::RealTimeFeatureExtractor() {
    // Initialize feature statistics
    const size_t num_features = 64;  // From Epic 3
    feature_stats_.resize(num_features);
    
    for (auto& stat : feature_stats_) {
        stat.mean = 0.0;
        stat.variance = 0.0;
        stat.count = 0;
    }
}

std::vector<float> RealTimeFeatureExtractor::ExtractFeatures(const std::vector<AccessEvent>& events) {
    std::vector<float> features;
    features.reserve(64);  // 64-dimensional feature vector from Epic 3
    
    if (events.empty()) {
        features.resize(64, 0.0f);
        return features;
    }
    
    // Temporal features (32 dimensions)
    for (size_t window_idx = 0; window_idx < 4; ++window_idx) {
        uint64_t window_us = TEMPORAL_WINDOWS[window_idx] * 1000000;  // Convert to microseconds
        uint64_t cutoff = events.back().timestamp - window_us;
        
        // Count events in window
        size_t count_in_window = 0;
        size_t hits_in_window = 0;
        double total_latency = 0.0;
        
        for (const auto& event : events) {
            if (event.timestamp >= cutoff) {
                count_in_window++;
                if (event.hit) hits_in_window++;
                total_latency += event.latency_us;
            }
        }
        
        // Window features (8 per window = 32 total)
        features.push_back(static_cast<float>(count_in_window));
        features.push_back(count_in_window > 0 ? static_cast<float>(hits_in_window) / count_in_window : 0.0f);
        features.push_back(count_in_window > 0 ? static_cast<float>(total_latency / count_in_window) : 0.0f);
        features.push_back(static_cast<float>(count_in_window) / events.size());
        
        // Add 4 more statistical features per window
        features.push_back(0.0f);  // Placeholder for variance
        features.push_back(0.0f);  // Placeholder for skewness
        features.push_back(0.0f);  // Placeholder for kurtosis
        features.push_back(0.0f);  // Placeholder for entropy
    }
    
    // Frequency features (16 dimensions)
    std::unordered_map<std::string, size_t> key_counts;
    std::unordered_map<std::string, size_t> op_counts;
    
    for (const auto& event : events) {
        key_counts[event.key]++;
        op_counts[event.operation]++;
    }
    
    // Top-k frequent keys
    std::vector<std::pair<std::string, size_t>> sorted_keys(key_counts.begin(), key_counts.end());
    std::sort(sorted_keys.begin(), sorted_keys.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < 8; ++i) {
        if (i < sorted_keys.size()) {
            features.push_back(static_cast<float>(sorted_keys[i].second) / events.size());
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Operation distribution
    for (const auto& op : {"GET", "PUT", "DELETE"}) {
        auto it = op_counts.find(op);
        if (it != op_counts.end()) {
            features.push_back(static_cast<float>(it->second) / events.size());
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Pad remaining frequency features
    while (features.size() < 48) {  // 32 temporal + 16 frequency
        features.push_back(0.0f);
    }
    
    // Sequence features (8 dimensions) - simplified
    for (size_t i = 0; i < 8; ++i) {
        features.push_back(0.0f);  // Placeholder for sequence mining
    }
    
    // Relationship features (8 dimensions) - simplified
    for (size_t i = 0; i < 8; ++i) {
        features.push_back(0.0f);  // Placeholder for co-occurrence
    }
    
    return features;
}

void RealTimeFeatureExtractor::UpdateStatistics(const AccessEvent& event) {
    // Update online statistics for normalization
    // This would be called periodically to maintain feature statistics
    
    // Example: Update latency statistics (feature index 2)
    if (feature_stats_.size() > 2) {
        auto& stat = feature_stats_[2];
        stat.count++;
        
        double delta = event.latency_us - stat.mean;
        stat.mean += delta / stat.count;
        
        double delta2 = event.latency_us - stat.mean;
        stat.variance += delta * delta2;
    }
}

}  // namespace mlops
}  // namespace predis