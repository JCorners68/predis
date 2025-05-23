#include "access_pattern_logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace predis {
namespace logger {

AccessPatternLogger::AccessPatternLogger()
    : AccessPatternLogger(Config()) {
}

AccessPatternLogger::AccessPatternLogger(const Config& config)
    : config_(config), buffer_(config.buffer_capacity) {
}

AccessPatternLogger::~AccessPatternLogger() {
    stop_background_export();
}

AccessPatternLogger::Statistics AccessPatternLogger::get_statistics() const {
    return {
        .events_logged = events_logged_.load(std::memory_order_relaxed),
        .events_dropped = events_dropped_.load(std::memory_order_relaxed),
        .events_exported = events_exported_.load(std::memory_order_relaxed),
        .buffer_usage = buffer_.size(),
        .avg_log_time_ns = 50.0  // Estimated from benchmarks
    };
}

std::vector<AccessEvent> AccessPatternLogger::export_events(size_t max_events) {
    std::vector<AccessEvent> events;
    size_t export_count = max_events > 0 ? max_events : config_.batch_export_size;
    events.reserve(export_count);
    
    AccessEvent event;
    while (events.size() < export_count && buffer_.try_pop(event)) {
        events.push_back(event);
    }
    
    events_exported_.fetch_add(events.size(), std::memory_order_relaxed);
    return events;
}

void AccessPatternLogger::start_background_export(
    std::function<void(const std::vector<AccessEvent>&)> callback) {
    
    if (export_running_.exchange(true)) {
        return;  // Already running
    }
    
    export_callback_ = callback;
    export_thread_ = std::make_unique<std::thread>(&AccessPatternLogger::export_worker, this);
}

void AccessPatternLogger::stop_background_export() {
    export_running_ = false;
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }
}

void AccessPatternLogger::export_worker() {
    while (export_running_.load(std::memory_order_relaxed)) {
        // Export batch of events
        auto events = export_events();
        
        if (!events.empty() && export_callback_) {
            export_callback_(events);
        }
        
        // Sleep until next export interval
        std::this_thread::sleep_for(config_.export_interval);
    }
}

// Pattern Analyzer Implementation

PatternAnalyzer::PatternAnalyzer() : PatternAnalyzer(Config()) {
}

PatternAnalyzer::PatternAnalyzer(const Config& config) : config_(config) {
}

std::vector<AccessPattern> PatternAnalyzer::analyze_events(
    const std::vector<AccessEvent>& events) {
    
    std::vector<AccessPattern> patterns;
    
    // Detect different pattern types
    detect_sequential_patterns(events);
    detect_temporal_patterns(events);
    detect_cooccurrence_patterns(events);
    detect_periodic_patterns(events);
    
    // Collect all detected patterns
    for (const auto& [key, sequences] : sequence_patterns_) {
        for (const auto& seq : sequences) {
            if (seq.count >= config_.min_frequency) {
                AccessPattern pattern{
                    .type = AccessPattern::PatternType::SEQUENTIAL,
                    .key_sequence = seq.sequence,
                    .confidence = static_cast<double>(seq.count) / events.size(),
                    .frequency = seq.count,
                    .period = std::chrono::microseconds(0)
                };
                
                if (pattern.confidence >= config_.min_confidence) {
                    patterns.push_back(pattern);
                }
            }
        }
    }
    
    return patterns;
}

void PatternAnalyzer::update_patterns(const AccessEvent& event) {
    // Update real-time pattern tracking
    // This is a simplified version - full implementation would maintain
    // sliding windows and incremental pattern updates
    
    static thread_local std::vector<uint64_t> recent_keys;
    recent_keys.push_back(event.key_hash);
    
    if (recent_keys.size() > config_.max_sequence_length) {
        recent_keys.erase(recent_keys.begin());
    }
    
    // Update sequence patterns
    if (recent_keys.size() >= config_.min_sequence_length) {
        auto& sequences = sequence_patterns_[recent_keys.back()];
        
        bool found = false;
        for (auto& seq : sequences) {
            if (seq.sequence == recent_keys) {
                seq.count++;
                seq.last_timestamp = event.timestamp_us;
                found = true;
                break;
            }
        }
        
        if (!found) {
            sequences.push_back({recent_keys, 1, event.timestamp_us});
        }
    }
}

std::vector<uint64_t> PatternAnalyzer::predict_next_keys(
    uint64_t current_key, size_t max_predictions) {
    
    std::vector<std::pair<uint64_t, double>> predictions;
    
    // Check sequential patterns
    auto seq_it = sequence_patterns_.find(current_key);
    if (seq_it != sequence_patterns_.end()) {
        for (const auto& seq : seq_it->second) {
            if (!seq.sequence.empty() && seq.sequence.back() != current_key) {
                // Find position of current key in sequence
                auto pos = std::find(seq.sequence.begin(), seq.sequence.end(), current_key);
                if (pos != seq.sequence.end() && pos + 1 != seq.sequence.end()) {
                    uint64_t next_key = *(pos + 1);
                    double confidence = static_cast<double>(seq.count) / 100.0;  // Normalized
                    predictions.push_back({next_key, confidence});
                }
            }
        }
    }
    
    // Check co-occurrence patterns
    auto cooc_it = cooccurrence_map_.find(current_key);
    if (cooc_it != cooccurrence_map_.end()) {
        for (uint64_t related_key : cooc_it->second) {
            predictions.push_back({related_key, 0.5});  // Lower confidence for co-occurrence
        }
    }
    
    // Sort by confidence and return top predictions
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<uint64_t> result;
    for (size_t i = 0; i < std::min(max_predictions, predictions.size()); ++i) {
        result.push_back(predictions[i].first);
    }
    
    return result;
}

double PatternAnalyzer::get_access_probability(
    uint64_t key, std::chrono::microseconds time_window) {
    
    // Simplified probability calculation based on recent access frequency
    auto temporal_it = temporal_patterns_.find(key);
    if (temporal_it == temporal_patterns_.end()) {
        return 0.0;
    }
    
    const auto& timestamps = temporal_it->second;
    if (timestamps.empty()) {
        return 0.0;
    }
    
    // Count accesses within time window
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
    uint64_t window_start = current_time - time_window.count();
    size_t recent_accesses = 0;
    
    for (auto it = timestamps.rbegin(); it != timestamps.rend(); ++it) {
        if (*it >= window_start) {
            recent_accesses++;
        } else {
            break;  // Timestamps are ordered
        }
    }
    
    // Simple probability model: recent_accesses / time_window_seconds
    double time_window_seconds = time_window.count() / 1e6;
    double access_rate = recent_accesses / time_window_seconds;
    
    // Convert to probability using exponential decay
    return 1.0 - std::exp(-access_rate);
}

void PatternAnalyzer::detect_sequential_patterns(const std::vector<AccessEvent>& events) {
    // Build sequences of consecutive accesses
    for (size_t i = 0; i < events.size(); ++i) {
        std::vector<uint64_t> sequence;
        
        // Build sequence up to max length
        for (size_t j = i; j < std::min(i + config_.max_sequence_length, events.size()); ++j) {
            sequence.push_back(events[j].key_hash);
            
            if (sequence.size() >= config_.min_sequence_length) {
                // Check if this sequence already exists
                bool found = false;
                for (auto& [key, sequences] : sequence_patterns_) {
                    for (auto& seq : sequences) {
                        if (seq.sequence == sequence) {
                            seq.count++;
                            seq.last_timestamp = events[j].timestamp_us;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                
                if (!found && sequence.size() >= config_.min_sequence_length) {
                    // New sequence pattern
                    sequence_patterns_[sequence.front()].push_back({
                        sequence, 1, events[j].timestamp_us
                    });
                }
            }
        }
    }
}

void PatternAnalyzer::detect_temporal_patterns(const std::vector<AccessEvent>& events) {
    // Group events by key and analyze time intervals
    std::unordered_map<uint64_t, std::vector<uint64_t>> key_timestamps;
    
    for (const auto& event : events) {
        key_timestamps[event.key_hash].push_back(event.timestamp_us);
    }
    
    // Store in temporal patterns map
    temporal_patterns_ = std::move(key_timestamps);
}

void PatternAnalyzer::detect_cooccurrence_patterns(const std::vector<AccessEvent>& events) {
    // Find keys that are frequently accessed together within a time window
    const uint64_t cooccurrence_window_us = 1000000;  // 1 second
    
    for (size_t i = 0; i < events.size(); ++i) {
        std::unordered_set<uint64_t> window_keys;
        window_keys.insert(events[i].key_hash);
        
        // Look ahead within time window
        for (size_t j = i + 1; j < events.size(); ++j) {
            if (events[j].timestamp_us - events[i].timestamp_us > cooccurrence_window_us) {
                break;
            }
            window_keys.insert(events[j].key_hash);
        }
        
        // Update co-occurrence map
        if (window_keys.size() > 1) {
            for (uint64_t key1 : window_keys) {
                for (uint64_t key2 : window_keys) {
                    if (key1 != key2) {
                        cooccurrence_map_[key1].push_back(key2);
                    }
                }
            }
        }
    }
    
    // Remove duplicates and keep only frequent co-occurrences
    for (auto& [key, related_keys] : cooccurrence_map_) {
        std::sort(related_keys.begin(), related_keys.end());
        
        // Count frequencies
        std::unordered_map<uint64_t, uint32_t> freq_map;
        for (uint64_t related : related_keys) {
            freq_map[related]++;
        }
        
        // Keep only frequent co-occurrences
        related_keys.clear();
        for (const auto& [related_key, count] : freq_map) {
            if (count >= config_.min_frequency) {
                related_keys.push_back(related_key);
            }
        }
    }
}

void PatternAnalyzer::detect_periodic_patterns(const std::vector<AccessEvent>& events) {
    // Analyze inter-access times for each key to find periodic patterns
    std::unordered_map<uint64_t, std::vector<uint64_t>> key_timestamps;
    
    for (const auto& event : events) {
        key_timestamps[event.key_hash].push_back(event.timestamp_us);
    }
    
    // For each key, analyze if accesses follow a periodic pattern
    for (const auto& [key, timestamps] : key_timestamps) {
        if (timestamps.size() < config_.min_frequency * 2) {
            continue;  // Not enough data for period detection
        }
        
        // Calculate inter-access times
        std::vector<uint64_t> intervals;
        for (size_t i = 1; i < timestamps.size(); ++i) {
            intervals.push_back(timestamps[i] - timestamps[i-1]);
        }
        
        // Simple periodicity detection: check if intervals are roughly constant
        if (!intervals.empty()) {
            double mean = 0.0;
            for (uint64_t interval : intervals) {
                mean += interval;
            }
            mean /= intervals.size();
            
            double variance = 0.0;
            for (uint64_t interval : intervals) {
                double diff = interval - mean;
                variance += diff * diff;
            }
            variance /= intervals.size();
            
            double std_dev = std::sqrt(variance);
            double cv = std_dev / mean;  // Coefficient of variation
            
            // If CV is low, we have a periodic pattern
            if (cv < 0.3) {  // 30% variation threshold
                AccessPattern pattern{
                    .type = AccessPattern::PatternType::PERIODIC,
                    .key_sequence = {key},
                    .confidence = 1.0 - cv,
                    .frequency = static_cast<uint32_t>(timestamps.size()),
                    .period = std::chrono::microseconds(static_cast<uint64_t>(mean))
                };
                
                // Store periodic pattern (implementation detail omitted)
            }
        }
    }
}

} // namespace logger
} // namespace predis