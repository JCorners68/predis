#pragma once

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

namespace predis {
namespace logger {

// Ultra-low overhead access logger with sampling
class OptimizedAccessLogger {
public:
    struct Config {
        double sampling_rate = 0.01;          // Sample 1% of accesses
        size_t ring_buffer_size = 1048576;    // 1M slots (power of 2)
        bool enable_adaptive_sampling = true; // Adjust rate based on load
        size_t batch_size = 1024;            // Process in batches
    };
    
    // Minimal event structure - 16 bytes for cache efficiency
    struct CompactEvent {
        uint32_t timestamp_offset;  // Offset from base timestamp (4 bytes)
        uint32_t key_hash_low;      // Lower 32 bits of key hash (4 bytes)
        uint16_t value_size;        // Value size up to 64KB (2 bytes)
        uint8_t operation_type;     // Operation type (1 byte)
        uint8_t cache_result;       // Cache result (1 byte)
        uint32_t thread_id;         // Thread ID (4 bytes)
    } __attribute__((packed));
    
    static_assert(sizeof(CompactEvent) == 16, "CompactEvent must be 16 bytes");
    
    OptimizedAccessLogger();
    explicit OptimizedAccessLogger(const Config& config);
    ~OptimizedAccessLogger();
    
    // Ultra-fast logging with sampling
    inline void log_access(uint64_t key_hash, uint8_t op_type, 
                          uint8_t result, uint32_t value_size) {
        // Fast path: sampling check using thread-local random
        thread_local std::minstd_rand rng(std::hash<std::thread::id>{}(
            std::this_thread::get_id()));
        thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        if (dist(rng) > current_sampling_rate_.load(std::memory_order_relaxed)) {
            return;  // Skip this event
        }
        
        // Get slot in ring buffer
        size_t slot = write_index_.fetch_add(1, std::memory_order_relaxed) & mask_;
        
        // Write event (may overwrite old data)
        auto& event = events_[slot];
        event.timestamp_offset = get_timestamp_offset();
        event.key_hash_low = static_cast<uint32_t>(key_hash);
        event.value_size = static_cast<uint16_t>(std::min(value_size, 65535u));
        event.operation_type = op_type;
        event.cache_result = result;
        event.thread_id = get_thread_id();
        
        sampled_events_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Batch export for processing
    std::vector<CompactEvent> export_batch(size_t max_events = 0);
    
    // Control methods
    void set_sampling_rate(double rate);
    double get_effective_sampling_rate() const;
    void enable_adaptive_sampling(bool enable);
    
    // Statistics
    struct Stats {
        uint64_t total_accesses;      // Estimated from sampling
        uint64_t sampled_events;
        double current_sampling_rate;
        double estimated_overhead_percent;
    };
    
    Stats get_stats() const;
    
private:
    Config config_;
    
    // Ring buffer
    std::unique_ptr<CompactEvent[]> events_;
    const size_t mask_;  // For fast modulo
    
    // Atomic counters
    alignas(64) std::atomic<size_t> write_index_{0};
    alignas(64) std::atomic<size_t> read_index_{0};
    alignas(64) std::atomic<uint64_t> sampled_events_{0};
    alignas(64) std::atomic<double> current_sampling_rate_;
    
    // Base timestamp for offset calculation
    const uint64_t base_timestamp_;
    
    // Helper methods
    inline uint32_t get_timestamp_offset() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        return static_cast<uint32_t>(micros - base_timestamp_);
    }
    
    inline uint32_t get_thread_id() const {
        thread_local uint32_t tid = std::hash<std::thread::id>{}(
            std::this_thread::get_id()) & 0xFFFF;
        return tid;
    }
    
    // Adaptive sampling
    void adjust_sampling_rate();
};

// Batch processor for pattern analysis
class BatchPatternProcessor {
public:
    struct Pattern {
        std::vector<uint32_t> key_sequence;
        uint32_t frequency;
        float confidence;
    };
    
    // Process a batch of events offline
    std::vector<Pattern> process_batch(const std::vector<OptimizedAccessLogger::CompactEvent>& events);
    
    // Lightweight pattern detection
    bool detect_sequential_access(const OptimizedAccessLogger::CompactEvent* events, size_t count);
    bool detect_periodic_pattern(const OptimizedAccessLogger::CompactEvent* events, size_t count);
    
private:
    static constexpr size_t MIN_PATTERN_LENGTH = 3;
    static constexpr size_t MAX_PATTERN_LENGTH = 10;
    static constexpr uint32_t MIN_FREQUENCY = 3;
};

// Zero-copy event reader for analysis
class EventReader {
public:
    EventReader(const OptimizedAccessLogger::CompactEvent* events, size_t count)
        : events_(events), count_(count), pos_(0) {}
    
    bool has_next() const { return pos_ < count_; }
    const OptimizedAccessLogger::CompactEvent& next() { return events_[pos_++]; }
    void reset() { pos_ = 0; }
    
    // Filters
    void filter_by_operation(uint8_t op_type);
    void filter_by_time_window(uint32_t start_offset, uint32_t end_offset);
    
private:
    const OptimizedAccessLogger::CompactEvent* events_;
    size_t count_;
    size_t pos_;
};

} // namespace logger
} // namespace predis