#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

namespace predis {
namespace logger {

// Compact access event structure for minimal memory footprint
struct AccessEvent {
    uint64_t timestamp_us;    // Microsecond precision timestamp
    uint64_t key_hash;        // Hash of accessed key for privacy
    uint32_t value_size;      // Size of value in bytes
    uint16_t thread_id;       // Thread that made the access
    uint8_t operation_type;   // GET=0, PUT=1, DELETE=2, PREFETCH=3
    uint8_t cache_result;     // HIT=0, MISS=1, PREFETCH_HIT=2, EVICTED=3
} __attribute__((packed));  // 24 bytes per event

static_assert(sizeof(AccessEvent) == 24, "AccessEvent must be 24 bytes");

// Lock-free circular buffer for high-performance logging
class LockFreeCircularBuffer {
private:
    std::unique_ptr<AccessEvent[]> buffer_;
    const size_t capacity_;
    alignas(64) std::atomic<size_t> write_pos_{0};  // Cache line aligned
    alignas(64) std::atomic<size_t> read_pos_{0};   // Cache line aligned
    
public:
    explicit LockFreeCircularBuffer(size_t capacity)
        : capacity_(capacity), buffer_(std::make_unique<AccessEvent[]>(capacity)) {
        // Pre-fault pages for consistent performance
        std::memset(buffer_.get(), 0, capacity * sizeof(AccessEvent));
    }
    
    bool try_push(const AccessEvent& event) {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % capacity_;
        
        // Check if buffer is full
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false;  // Buffer full, drop event
        }
        
        // Write event
        buffer_[current_write] = event;
        
        // Update write position
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool try_pop(AccessEvent& event) {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        
        // Check if buffer is empty
        if (current_read == write_pos_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Read event
        event = buffer_[current_read];
        
        // Update read position
        size_t next_read = (current_read + 1) % capacity_;
        read_pos_.store(next_read, std::memory_order_release);
        return true;
    }
    
    size_t size() const {
        size_t write = write_pos_.load(std::memory_order_acquire);
        size_t read = read_pos_.load(std::memory_order_acquire);
        return (write >= read) ? (write - read) : (capacity_ - read + write);
    }
    
    bool empty() const {
        return write_pos_.load(std::memory_order_acquire) == 
               read_pos_.load(std::memory_order_acquire);
    }
};

// High-performance access pattern logger
class AccessPatternLogger {
public:
    enum class OperationType : uint8_t {
        GET = 0,
        PUT = 1,
        DELETE = 2,
        PREFETCH = 3
    };
    
    enum class CacheResult : uint8_t {
        HIT = 0,
        MISS = 1,
        PREFETCH_HIT = 2,
        EVICTED = 3
    };
    
    struct Config {
        size_t buffer_capacity = 1000000;        // 1M events (~24MB)
        bool enable_compression = false;         // Future: compress old events
        bool enable_persistence = false;         // Future: persist to disk
        size_t batch_export_size = 10000;        // Export batch size
        std::chrono::seconds export_interval{60}; // Export every minute
    };
    
    AccessPatternLogger();
    explicit AccessPatternLogger(const Config& config);
    ~AccessPatternLogger();
    
    // Main logging interface - designed for minimal overhead
    inline void log_access(uint64_t key_hash, OperationType op_type, 
                          CacheResult result, uint32_t value_size) {
        if (!enabled_.load(std::memory_order_relaxed)) return;
        
        AccessEvent event = {
            .timestamp_us = get_microsecond_timestamp(),
            .key_hash = key_hash,
            .value_size = value_size,
            .thread_id = static_cast<uint16_t>(get_thread_id()),
            .operation_type = static_cast<uint8_t>(op_type),
            .cache_result = static_cast<uint8_t>(result)
        };
        
        if (buffer_.try_push(event)) {
            events_logged_.fetch_add(1, std::memory_order_relaxed);
        } else {
            events_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // Control methods
    void enable() { enabled_.store(true, std::memory_order_release); }
    void disable() { enabled_.store(false, std::memory_order_release); }
    bool is_enabled() const { return enabled_.load(std::memory_order_acquire); }
    
    // Statistics
    struct Statistics {
        uint64_t events_logged;
        uint64_t events_dropped;
        uint64_t events_exported;
        size_t buffer_usage;
        double avg_log_time_ns;
    };
    
    Statistics get_statistics() const;
    
    // Export interface for ML training
    std::vector<AccessEvent> export_events(size_t max_events = 0);
    void start_background_export(std::function<void(const std::vector<AccessEvent>&)> callback);
    void stop_background_export();
    
private:
    Config config_;
    LockFreeCircularBuffer buffer_;
    std::atomic<bool> enabled_{true};
    
    // Statistics
    alignas(64) std::atomic<uint64_t> events_logged_{0};
    alignas(64) std::atomic<uint64_t> events_dropped_{0};
    alignas(64) std::atomic<uint64_t> events_exported_{0};
    
    // Background export
    std::unique_ptr<std::thread> export_thread_;
    std::atomic<bool> export_running_{false};
    std::function<void(const std::vector<AccessEvent>&)> export_callback_;
    
    // Helper methods
    static inline uint64_t get_microsecond_timestamp() {
        using namespace std::chrono;
        return duration_cast<microseconds>(
            high_resolution_clock::now().time_since_epoch()
        ).count();
    }
    
    static inline uint32_t get_thread_id() {
        static thread_local uint32_t thread_id = std::hash<std::thread::id>{}(
            std::this_thread::get_id()
        ) & 0xFFFF;
        return thread_id;
    }
    
    void export_worker();
};

// Pattern analysis results
struct AccessPattern {
    enum class PatternType {
        SEQUENTIAL,      // Keys accessed in sequence
        TEMPORAL,        // Time-based patterns
        COOCCURRENCE,    // Keys accessed together
        PERIODIC         // Regular intervals
    };
    
    PatternType type;
    std::vector<uint64_t> key_sequence;
    double confidence;
    uint32_t frequency;
    std::chrono::microseconds period;  // For periodic patterns
};

// Real-time pattern analyzer
class PatternAnalyzer {
public:
    struct Config {
        size_t min_sequence_length = 3;
        size_t max_sequence_length = 10;
        double min_confidence = 0.7;
        uint32_t min_frequency = 3;
        std::chrono::seconds analysis_window{300};  // 5 minutes
    };
    
    PatternAnalyzer();
    explicit PatternAnalyzer(const Config& config);
    
    // Analyze a batch of events
    std::vector<AccessPattern> analyze_events(const std::vector<AccessEvent>& events);
    
    // Real-time pattern detection
    void update_patterns(const AccessEvent& event);
    std::vector<AccessPattern> get_current_patterns() const;
    
    // Pattern queries for prefetching
    std::vector<uint64_t> predict_next_keys(uint64_t current_key, size_t max_predictions = 5);
    double get_access_probability(uint64_t key, std::chrono::microseconds time_window);
    
private:
    Config config_;
    
    // Pattern storage
    struct SequenceTracker {
        std::vector<uint64_t> sequence;
        uint32_t count;
        uint64_t last_timestamp;
    };
    
    std::unordered_map<uint64_t, std::vector<SequenceTracker>> sequence_patterns_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> cooccurrence_map_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> temporal_patterns_;
    
    void detect_sequential_patterns(const std::vector<AccessEvent>& events);
    void detect_temporal_patterns(const std::vector<AccessEvent>& events);
    void detect_cooccurrence_patterns(const std::vector<AccessEvent>& events);
    void detect_periodic_patterns(const std::vector<AccessEvent>& events);
};

} // namespace logger
} // namespace predis