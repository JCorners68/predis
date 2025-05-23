#include "optimized_access_logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace predis {
namespace logger {

// Hash function for vector<uint32_t>
struct VectorHasher {
    std::size_t operator()(const std::vector<uint32_t>& vec) const {
        std::size_t seed = vec.size();
        for (auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

OptimizedAccessLogger::OptimizedAccessLogger()
    : OptimizedAccessLogger(Config()) {
}

OptimizedAccessLogger::OptimizedAccessLogger(const Config& config)
    : config_(config),
      events_(std::make_unique<CompactEvent[]>(config.ring_buffer_size)),
      mask_(config.ring_buffer_size - 1),
      current_sampling_rate_(config.sampling_rate),
      base_timestamp_(std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count()) {
    
    // Verify ring buffer size is power of 2
    if ((config.ring_buffer_size & (config.ring_buffer_size - 1)) != 0) {
        throw std::invalid_argument("Ring buffer size must be power of 2");
    }
    
    // Pre-fault pages for consistent performance
    std::memset(events_.get(), 0, config.ring_buffer_size * sizeof(CompactEvent));
}

OptimizedAccessLogger::~OptimizedAccessLogger() = default;

std::vector<OptimizedAccessLogger::CompactEvent> 
OptimizedAccessLogger::export_batch(size_t max_events) {
    size_t write_idx = write_index_.load(std::memory_order_acquire);
    size_t read_idx = read_index_.load(std::memory_order_relaxed);
    
    // Calculate available events
    size_t available = (write_idx >= read_idx) 
        ? (write_idx - read_idx) 
        : (config_.ring_buffer_size - read_idx + write_idx);
    
    size_t to_export = (max_events > 0) 
        ? std::min(max_events, available) 
        : std::min(config_.batch_size, available);
    
    std::vector<CompactEvent> batch;
    batch.reserve(to_export);
    
    for (size_t i = 0; i < to_export; ++i) {
        size_t idx = (read_idx + i) & mask_;
        batch.push_back(events_[idx]);
    }
    
    // Update read index
    read_index_.store((read_idx + to_export) & mask_, std::memory_order_release);
    
    return batch;
}

void OptimizedAccessLogger::set_sampling_rate(double rate) {
    current_sampling_rate_.store(std::clamp(rate, 0.0001, 1.0), 
                                std::memory_order_relaxed);
}

double OptimizedAccessLogger::get_effective_sampling_rate() const {
    return current_sampling_rate_.load(std::memory_order_relaxed);
}

void OptimizedAccessLogger::enable_adaptive_sampling(bool enable) {
    config_.enable_adaptive_sampling = enable;
    if (enable) {
        adjust_sampling_rate();
    }
}

OptimizedAccessLogger::Stats OptimizedAccessLogger::get_stats() const {
    uint64_t sampled = sampled_events_.load(std::memory_order_relaxed);
    double rate = current_sampling_rate_.load(std::memory_order_relaxed);
    
    return {
        .total_accesses = static_cast<uint64_t>(sampled / rate),
        .sampled_events = sampled,
        .current_sampling_rate = rate,
        .estimated_overhead_percent = rate * 0.05  // ~0.05% overhead per 1% sampling
    };
}

void OptimizedAccessLogger::adjust_sampling_rate() {
    // Simple adaptive algorithm: reduce rate if buffer is filling too fast
    size_t write_idx = write_index_.load(std::memory_order_relaxed);
    size_t read_idx = read_index_.load(std::memory_order_relaxed);
    
    size_t used = (write_idx >= read_idx) 
        ? (write_idx - read_idx) 
        : (config_.ring_buffer_size - read_idx + write_idx);
    
    double usage_ratio = static_cast<double>(used) / config_.ring_buffer_size;
    
    if (usage_ratio > 0.8) {
        // Buffer getting full, reduce sampling
        set_sampling_rate(get_effective_sampling_rate() * 0.5);
    } else if (usage_ratio < 0.2) {
        // Buffer has space, can increase sampling
        set_sampling_rate(std::min(1.0, get_effective_sampling_rate() * 1.5));
    }
}

// Batch Pattern Processor Implementation

std::vector<BatchPatternProcessor::Pattern> 
BatchPatternProcessor::process_batch(const std::vector<OptimizedAccessLogger::CompactEvent>& events) {
    std::vector<Pattern> patterns;
    
    if (events.size() < MIN_PATTERN_LENGTH) {
        return patterns;
    }
    
    // Simple sequential pattern detection
    std::unordered_map<std::vector<uint32_t>, uint32_t, VectorHasher> sequence_counts;
    
    for (size_t i = 0; i <= events.size() - MIN_PATTERN_LENGTH; ++i) {
        for (size_t len = MIN_PATTERN_LENGTH; 
             len <= std::min(MAX_PATTERN_LENGTH, events.size() - i); ++len) {
            
            std::vector<uint32_t> sequence;
            for (size_t j = 0; j < len; ++j) {
                sequence.push_back(events[i + j].key_hash_low);
            }
            
            sequence_counts[sequence]++;
        }
    }
    
    // Extract frequent patterns
    for (const auto& [sequence, count] : sequence_counts) {
        if (count >= MIN_FREQUENCY) {
            patterns.push_back({
                .key_sequence = sequence,
                .frequency = count,
                .confidence = static_cast<float>(count) / events.size()
            });
        }
    }
    
    // Sort by frequency
    std::sort(patterns.begin(), patterns.end(),
              [](const Pattern& a, const Pattern& b) {
                  return a.frequency > b.frequency;
              });
    
    return patterns;
}

bool BatchPatternProcessor::detect_sequential_access(
    const OptimizedAccessLogger::CompactEvent* events, size_t count) {
    
    if (count < MIN_PATTERN_LENGTH) return false;
    
    // Check if keys are accessed in sequence
    for (size_t i = 1; i < count; ++i) {
        if (events[i].timestamp_offset - events[i-1].timestamp_offset > 1000) {
            return false;  // Gap too large (>1ms)
        }
    }
    
    return true;
}

bool BatchPatternProcessor::detect_periodic_pattern(
    const OptimizedAccessLogger::CompactEvent* events, size_t count) {
    
    if (count < MIN_FREQUENCY * 2) return false;
    
    // Calculate inter-access times
    std::vector<uint32_t> intervals;
    for (size_t i = 1; i < count; ++i) {
        intervals.push_back(events[i].timestamp_offset - events[i-1].timestamp_offset);
    }
    
    // Check for periodicity
    if (intervals.empty()) return false;
    
    double mean = 0.0;
    for (uint32_t interval : intervals) {
        mean += interval;
    }
    mean /= intervals.size();
    
    double variance = 0.0;
    for (uint32_t interval : intervals) {
        double diff = interval - mean;
        variance += diff * diff;
    }
    variance /= intervals.size();
    
    // Low variance indicates periodic pattern
    double cv = std::sqrt(variance) / mean;
    return cv < 0.2;  // 20% coefficient of variation threshold
}

// Event Reader Implementation

void EventReader::filter_by_operation(uint8_t op_type) {
    // In-place filtering would modify the view
    // For now, this is a placeholder for future implementation
}

void EventReader::filter_by_time_window(uint32_t start_offset, uint32_t end_offset) {
    // Placeholder for time-based filtering
}

} // namespace logger
} // namespace predis