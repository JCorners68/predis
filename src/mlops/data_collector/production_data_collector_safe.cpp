#include "production_data_collector.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <zlib.h>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <sys/statvfs.h>
#include <mutex>

namespace predis {
namespace mlops {

// Error tracking
static std::mutex error_mutex;
static std::unordered_map<std::string, size_t> error_counts;
static constexpr size_t MAX_ERROR_COUNT = 100;  // Max errors before disabling export

void LogError(const std::string& category, const std::string& message) {
    std::lock_guard<std::mutex> lock(error_mutex);
    error_counts[category]++;
    
    // Log to stderr with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::cerr << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
              << "] [ERROR] [" << category << "] " << message << std::endl;
}

// Disk space check
bool HasSufficientDiskSpace(const std::string& path, size_t required_bytes) {
    try {
        struct statvfs stat;
        if (statvfs(path.c_str(), &stat) != 0) {
            return false;
        }
        
        size_t available = stat.f_bavail * stat.f_frsize;
        return available > required_bytes * 2;  // Require 2x space for safety
        
    } catch (...) {
        return false;
    }
}

// Path validation
bool IsPathSafe(const std::string& path) {
    // Check for directory traversal attempts
    if (path.find("..") != std::string::npos) {
        return false;
    }
    
    // Check for absolute paths outside allowed directories
    if (path[0] == '/' && path.find("/var/lib/predis") != 0) {
        return false;
    }
    
    return true;
}

// CircularBuffer implementation with error handling
template<typename T>
CircularBuffer<T>::CircularBuffer(size_t capacity) 
    : capacity_(capacity) {
    
    if (capacity == 0) {
        throw std::invalid_argument("CircularBuffer capacity must be greater than 0");
    }
    
    // Ensure capacity is power of 2 for efficient modulo
    size_t power_of_2 = 1;
    while (power_of_2 < capacity) {
        power_of_2 <<= 1;
    }
    capacity_ = power_of_2;
    
    try {
        buffer_.resize(capacity_);
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Failed to allocate CircularBuffer: " + std::string(e.what()));
    }
}

template<typename T>
bool CircularBuffer<T>::Push(const T& item) {
    size_t current_tail = tail_.load(std::memory_order_relaxed);
    size_t next_tail = (current_tail + 1) & (capacity_ - 1);  // Fast modulo
    
    // Check if buffer is full
    if (next_tail == head_.load(std::memory_order_acquire)) {
        return false;  // Buffer full
    }
    
    try {
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    } catch (...) {
        // Failed to copy item
        return false;
    }
}

template<typename T>
bool CircularBuffer<T>::Pop(T& item) {
    size_t current_head = head_.load(std::memory_order_relaxed);
    
    // Check if buffer is empty
    if (current_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Buffer empty
    }
    
    try {
        item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) & (capacity_ - 1), std::memory_order_release);
        return true;
    } catch (...) {
        // Failed to move item
        return false;
    }
}

template<typename T>
size_t CircularBuffer<T>::Size() const {
    // Add cache line padding to prevent false sharing
    alignas(64) size_t head = head_.load(std::memory_order_acquire);
    alignas(64) size_t tail = tail_.load(std::memory_order_acquire);
    
    return (tail - head) & (capacity_ - 1);
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
    items.reserve(std::min(max_items, Size()));
    
    T item;
    size_t popped = 0;
    while (popped < max_items && Pop(item)) {
        try {
            items.push_back(std::move(item));
            popped++;
        } catch (const std::bad_alloc&) {
            // Out of memory, return what we have
            break;
        }
    }
    
    return popped;
}

// Explicit template instantiation
template class CircularBuffer<AccessEvent>;

// ProductionDataCollector implementation with error handling
ProductionDataCollector::ProductionDataCollector(const CollectorConfig& config)
    : config_(config), 
      rng_state_(std::chrono::steady_clock::now().time_since_epoch().count()) {
    
    // Validate configuration
    if (config.buffer_capacity < 1000) {
        throw std::invalid_argument("Buffer capacity must be at least 1000");
    }
    
    if (config.sampling_rate < 0.0 || config.sampling_rate > 1.0) {
        throw std::invalid_argument("Sampling rate must be between 0.0 and 1.0");
    }
    
    if (!IsPathSafe(config.export_path)) {
        throw std::invalid_argument("Export path contains unsafe characters");
    }
    
    // Create export directory if it doesn't exist
    try {
        std::filesystem::create_directories(config.export_path);
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to create export directory: " + std::string(e.what()));
    }
    
    // Initialize buffer
    try {
        buffer_ = std::make_unique<CircularBuffer<AccessEvent>>(config.buffer_capacity);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize buffer: " + std::string(e.what()));
    }
}

ProductionDataCollector::~ProductionDataCollector() {
    try {
        Stop();
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

void ProductionDataCollector::LogAccess(const std::string& key, 
                                       const std::string& operation,
                                       bool hit,
                                       uint32_t latency_us) {
    total_events_.fetch_add(1, std::memory_order_relaxed);
    
    // Validate input
    if (key.empty() || operation.empty()) {
        return;  // Invalid input
    }
    
    // Limit key size to prevent memory issues
    if (key.size() > 1024) {
        return;
    }
    
    // Sampling decision
    if (!ShouldSample()) {
        return;
    }
    
    try {
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
    } catch (...) {
        // Silently drop on exception to avoid impacting cache performance
        dropped_events_.fetch_add(1, std::memory_order_relaxed);
    }
}

void ProductionDataCollector::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;  // Already running
    }
    
    try {
        export_thread_ = std::thread(&ProductionDataCollector::ExportLoop, this);
    } catch (const std::system_error& e) {
        running_ = false;
        throw std::runtime_error("Failed to start export thread: " + std::string(e.what()));
    }
}

void ProductionDataCollector::Stop() {
    running_.store(false);
    
    if (export_thread_.joinable()) {
        try {
            export_thread_.join();
        } catch (...) {
            // Thread join failed, but we're stopping anyway
        }
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
    
    try {
        metrics.buffer_usage_pct = (buffer_->Size() * 100) / config_.buffer_capacity;
    } catch (...) {
        metrics.buffer_usage_pct = 0;
    }
    
    // Export timing would be tracked with atomic<double> in production
    metrics.avg_export_time_ms = 0.0;
    
    return metrics;
}

void ProductionDataCollector::ExportLoop() {
    std::vector<AccessEvent> batch;
    batch.reserve(config_.export_batch_size);
    
    size_t consecutive_errors = 0;
    
    while (running_.load()) {
        try {
            // Wait for export interval
            std::this_thread::sleep_for(config_.export_interval);
            
            // Check if too many errors
            if (consecutive_errors >= MAX_ERROR_COUNT) {
                LogError("export", "Too many consecutive errors, disabling export");
                break;
            }
            
            // Collect batch
            size_t collected = buffer_->PopBatch(batch, config_.export_batch_size);
            
            if (collected > 0) {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check disk space before export (estimate 1KB per event)
                if (!HasSufficientDiskSpace(config_.export_path, collected * 1024)) {
                    LogError("export", "Insufficient disk space");
                    consecutive_errors++;
                    
                    // Put events back in buffer if possible
                    for (const auto& event : batch) {
                        buffer_->Push(event);
                    }
                    continue;
                }
                
                ExportBatch(batch);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                // Update metrics
                exported_events_.fetch_add(collected, std::memory_order_relaxed);
                
                // Track export timing (would use atomic<double> in production)
                if (duration > 1000) {
                    LogError("export", "Export took " + std::to_string(duration) + "ms");
                }
                
                consecutive_errors = 0;  // Reset on success
            }
        } catch (const std::exception& e) {
            LogError("export_loop", e.what());
            consecutive_errors++;
        } catch (...) {
            LogError("export_loop", "Unknown error");
            consecutive_errors++;
        }
    }
    
    // Final export on shutdown with error handling
    try {
        while (!buffer_->Empty()) {
            size_t collected = buffer_->PopBatch(batch, config_.export_batch_size);
            if (collected > 0) {
                ExportBatch(batch);
                exported_events_.fetch_add(collected, std::memory_order_relaxed);
            }
        }
    } catch (const std::exception& e) {
        LogError("final_export", e.what());
    }
}

void ProductionDataCollector::ExportBatch(const std::vector<AccessEvent>& events) {
    if (events.empty()) {
        return;
    }
    
    std::string filename = GenerateExportFilename();
    
    try {
        if (config_.enable_compression) {
            // Export as compressed JSON with error handling
            std::ostringstream json_stream;
            json_stream.exceptions(std::ios::failbit | std::ios::badbit);
            
            json_stream << "[\n";
            
            for (size_t i = 0; i < events.size(); ++i) {
                const auto& event = events[i];
                
                // Escape JSON strings
                std::string escaped_key = event.key;
                size_t pos = 0;
                while ((pos = escaped_key.find('"', pos)) != std::string::npos) {
                    escaped_key.replace(pos, 1, "\\\"");
                    pos += 2;
                }
                
                json_stream << "  {\n"
                           << "    \"timestamp\": " << event.timestamp << ",\n"
                           << "    \"key\": \"" << escaped_key << "\",\n"
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
            
            if (compressed_size > 100 * 1024 * 1024) {  // 100MB limit
                throw std::runtime_error("Compressed size would exceed 100MB");
            }
            
            std::vector<uint8_t> compressed(compressed_size);
            
            int result = compress(compressed.data(), &compressed_size, 
                                reinterpret_cast<const Bytef*>(json_data.data()), 
                                json_data.size());
            
            if (result != Z_OK) {
                throw std::runtime_error("Compression failed with error: " + std::to_string(result));
            }
            
            // Write with error handling
            std::string full_path = filename + ".gz";
            std::ofstream file(full_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + full_path);
            }
            
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.write(reinterpret_cast<const char*>(compressed.data()), compressed_size);
            file.close();
            
        } else {
            // Export as plain CSV with error handling
            std::string full_path = filename + ".csv";
            std::ofstream file(full_path);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + full_path);
            }
            
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file << "timestamp,key,operation,hit,latency_us\n";
            
            for (const auto& event : events) {
                // CSV escape
                std::string escaped_key = event.key;
                if (escaped_key.find(',') != std::string::npos || 
                    escaped_key.find('"') != std::string::npos) {
                    escaped_key = "\"" + escaped_key + "\"";
                }
                
                file << event.timestamp << ","
                     << escaped_key << ","
                     << event.operation << ","
                     << (event.hit ? "1" : "0") << ","
                     << event.latency_us << "\n";
            }
            
            file.close();
        }
    } catch (const std::exception& e) {
        LogError("export_batch", "Failed to export batch: " + std::string(e.what()));
        throw;  // Re-throw to be handled by caller
    }
}

std::string ProductionDataCollector::GenerateExportFilename() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream filename;
    filename << config_.export_path;
    
    // Ensure trailing slash
    if (!config_.export_path.empty() && config_.export_path.back() != '/') {
        filename << "/";
    }
    
    filename << "access_log_"
             << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    // Add process ID to prevent conflicts
    filename << "_" << getpid();
    
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

// RealTimeFeatureExtractor implementation with error handling
RealTimeFeatureExtractor::RealTimeFeatureExtractor() {
    try {
        // Initialize feature statistics
        const size_t num_features = 64;  // From Epic 3
        feature_stats_.resize(num_features);
        
        for (auto& stat : feature_stats_) {
            stat.mean = 0.0;
            stat.variance = 0.0;
            stat.count = 0;
        }
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Failed to initialize feature extractor: " + std::string(e.what()));
    }
}

std::vector<float> RealTimeFeatureExtractor::ExtractFeatures(const std::vector<AccessEvent>& events) {
    std::vector<float> features;
    
    try {
        features.reserve(64);  // 64-dimensional feature vector from Epic 3
        
        if (events.empty()) {
            features.resize(64, 0.0f);
            return features;
        }
        
        // Temporal features (32 dimensions) with bounds checking
        for (size_t window_idx = 0; window_idx < 4; ++window_idx) {
            if (window_idx >= sizeof(TEMPORAL_WINDOWS)/sizeof(TEMPORAL_WINDOWS[0])) {
                break;
            }
            
            uint64_t window_us = TEMPORAL_WINDOWS[window_idx] * 1000000;  // Convert to microseconds
            uint64_t cutoff = events.back().timestamp > window_us ? 
                             events.back().timestamp - window_us : 0;
            
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
        
        // Frequency features (16 dimensions) with memory limits
        std::unordered_map<std::string, size_t> key_counts;
        std::unordered_map<std::string, size_t> op_counts;
        
        // Limit map sizes to prevent memory exhaustion
        const size_t MAX_UNIQUE_KEYS = 10000;
        
        for (const auto& event : events) {
            if (key_counts.size() < MAX_UNIQUE_KEYS) {
                key_counts[event.key]++;
            }
            op_counts[event.operation]++;
        }
        
        // Top-k frequent keys
        std::vector<std::pair<std::string, size_t>> sorted_keys(key_counts.begin(), key_counts.end());
        std::partial_sort(sorted_keys.begin(), 
                         sorted_keys.begin() + std::min(size_t(8), sorted_keys.size()),
                         sorted_keys.end(),
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
        
        // Ensure exactly 64 features
        features.resize(64, 0.0f);
        
    } catch (const std::exception& e) {
        LogError("feature_extraction", e.what());
        features.clear();
        features.resize(64, 0.0f);  // Return zero vector on error
    }
    
    return features;
}

void RealTimeFeatureExtractor::UpdateStatistics(const AccessEvent& event) {
    try {
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
    } catch (...) {
        // Silently ignore statistics update errors
    }
}

}  // namespace mlops
}  // namespace predis