#include "feature_engineering.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace predis {
namespace ml {

// Temporal Feature Extractor Implementation

std::vector<float> TemporalFeatureExtractor::extract(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> features;
    features.reserve(32);  // Pre-allocate for temporal features
    
    // Extract access counts for different time windows
    for (uint32_t window_size : config_.window_sizes_seconds) {
        auto window_features = extract_access_counts(events, target_key, window_size);
        features.insert(features.end(), window_features.begin(), window_features.end());
    }
    
    // Hour of day distribution
    if (config_.include_hour_of_day) {
        auto hour_features = extract_hour_distribution(events, target_key);
        features.insert(features.end(), hour_features.begin(), hour_features.end());
    }
    
    // Day of week distribution
    if (config_.include_day_of_week) {
        auto day_features = extract_day_distribution(events, target_key);
        features.insert(features.end(), day_features.begin(), day_features.end());
    }
    
    // Pad with zeros if needed
    while (features.size() < 32) {
        features.push_back(0.0f);
    }
    
    return features;
}

std::vector<float> TemporalFeatureExtractor::extract_access_counts(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key,
    uint32_t window_seconds) const {
    
    std::vector<float> counts(3, 0.0f);  // total, recent, ratio
    
    if (events.empty()) return counts;
    
    // Get current time (last event time as proxy)
    uint32_t current_time = events.back().timestamp_offset;
    uint32_t window_start = (current_time > window_seconds * 1000000) 
        ? current_time - window_seconds * 1000000 : 0;
    
    uint32_t total_count = 0;
    uint32_t window_count = 0;
    
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    
    for (const auto& event : events) {
        if (event.key_hash_low == target_key_low) {
            total_count++;
            if (event.timestamp_offset >= window_start) {
                window_count++;
            }
        }
    }
    
    counts[0] = static_cast<float>(window_count);
    counts[1] = static_cast<float>(total_count);
    counts[2] = (total_count > 0) ? static_cast<float>(window_count) / total_count : 0.0f;
    
    return counts;
}

std::vector<float> TemporalFeatureExtractor::extract_hour_distribution(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> hour_dist(24, 0.0f);
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    uint32_t total_accesses = 0;
    
    for (const auto& event : events) {
        if (event.key_hash_low == target_key_low) {
            // Convert timestamp to hour (simplified - assumes offset in microseconds)
            uint32_t hour = (event.timestamp_offset / 3600000000) % 24;
            hour_dist[hour] += 1.0f;
            total_accesses++;
        }
    }
    
    // Normalize
    if (total_accesses > 0) {
        for (auto& count : hour_dist) {
            count /= total_accesses;
        }
    }
    
    return hour_dist;
}

std::vector<float> TemporalFeatureExtractor::extract_day_distribution(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> day_dist(7, 0.0f);  // 7 days of week
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    uint32_t total_accesses = 0;
    
    for (const auto& event : events) {
        if (event.key_hash_low == target_key_low) {
            // Simplified day calculation (would need real timestamp conversion)
            uint32_t day = (event.timestamp_offset / 86400000000) % 7;
            day_dist[day] += 1.0f;
            total_accesses++;
        }
    }
    
    // Normalize
    if (total_accesses > 0) {
        for (auto& count : day_dist) {
            count /= total_accesses;
        }
    }
    
    return day_dist;
}

// Frequency Feature Extractor Implementation

std::vector<float> FrequencyFeatureExtractor::extract(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> features;
    features.reserve(16);
    
    auto stats = compute_frequency_stats(events, target_key);
    
    // Add basic frequency features
    features.push_back(static_cast<float>(stats.total_accesses));
    features.push_back(stats.access_rate);
    features.push_back(stats.time_since_last_access);
    features.push_back(stats.mean_inter_access_time);
    features.push_back(stats.std_inter_access_time);
    
    if (config_.include_regularity_score) {
        features.push_back(stats.regularity_score);
    }
    
    if (config_.include_burst_detection) {
        features.push_back(stats.burst_score);
    }
    
    // Add inter-access interval histogram
    std::vector<uint32_t> intervals;
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    uint32_t last_timestamp = 0;
    bool first_access = true;
    
    for (const auto& event : events) {
        if (event.key_hash_low == target_key_low) {
            if (!first_access) {
                intervals.push_back(event.timestamp_offset - last_timestamp);
            }
            last_timestamp = event.timestamp_offset;
            first_access = false;
        }
    }
    
    // Create histogram of inter-access times
    if (!intervals.empty()) {
        std::sort(intervals.begin(), intervals.end());
        
        // Percentiles: 10th, 25th, 50th, 75th, 90th
        std::vector<float> percentiles = {0.1f, 0.25f, 0.5f, 0.75f, 0.9f};
        for (float p : percentiles) {
            size_t idx = static_cast<size_t>(p * (intervals.size() - 1));
            features.push_back(static_cast<float>(intervals[idx]) / 1000000.0f); // Convert to seconds
        }
    } else {
        // No intervals - pad with zeros
        for (int i = 0; i < 5; ++i) {
            features.push_back(0.0f);
        }
    }
    
    // Pad to 16 features
    while (features.size() < 16) {
        features.push_back(0.0f);
    }
    
    return features;
}

FrequencyFeatureExtractor::FrequencyStats 
FrequencyFeatureExtractor::compute_frequency_stats(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    FrequencyStats stats = {};
    
    if (events.empty()) return stats;
    
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    std::vector<uint32_t> access_times;
    
    for (const auto& event : events) {
        if (event.key_hash_low == target_key_low) {
            access_times.push_back(event.timestamp_offset);
        }
    }
    
    if (access_times.empty()) return stats;
    
    stats.total_accesses = static_cast<uint32_t>(access_times.size());
    
    // Time range
    uint32_t time_range = events.back().timestamp_offset - events.front().timestamp_offset;
    if (time_range > 0) {
        stats.access_rate = static_cast<float>(stats.total_accesses) * 1000000.0f / time_range;
    }
    
    // Time since last access
    stats.time_since_last_access = static_cast<float>(
        events.back().timestamp_offset - access_times.back()) / 1000000.0f;
    
    // Inter-access time statistics
    if (access_times.size() > 1) {
        std::vector<float> inter_times;
        for (size_t i = 1; i < access_times.size(); ++i) {
            inter_times.push_back(static_cast<float>(
                access_times[i] - access_times[i-1]) / 1000000.0f);
        }
        
        float sum = std::accumulate(inter_times.begin(), inter_times.end(), 0.0f);
        stats.mean_inter_access_time = sum / inter_times.size();
        
        float sq_sum = 0.0f;
        for (float t : inter_times) {
            float diff = t - stats.mean_inter_access_time;
            sq_sum += diff * diff;
        }
        stats.std_inter_access_time = std::sqrt(sq_sum / inter_times.size());
        
        // Regularity score: inverse of coefficient of variation
        if (stats.mean_inter_access_time > 0) {
            float cv = stats.std_inter_access_time / stats.mean_inter_access_time;
            stats.regularity_score = 1.0f / (1.0f + cv);
        }
        
        // Burst detection: ratio of min to mean inter-access time
        float min_time = *std::min_element(inter_times.begin(), inter_times.end());
        stats.burst_score = 1.0f - (min_time / stats.mean_inter_access_time);
    }
    
    return stats;
}

// Sequence Feature Extractor Implementation

std::vector<float> SequenceFeatureExtractor::extract(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> features;
    features.reserve(8);
    
    auto patterns = find_sequences_ending_with(events, target_key);
    
    // Sequence predictability
    float predictability = compute_sequence_predictability(patterns);
    features.push_back(predictability);
    
    // Top sequence frequencies
    for (size_t i = 0; i < config_.top_sequences_to_track; ++i) {
        if (i < patterns.size()) {
            features.push_back(patterns[i].probability);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Markov chain features
    if (config_.include_markov_features && !patterns.empty()) {
        // Transition probability from most common predecessor
        features.push_back(patterns[0].frequency > 0 ? patterns[0].probability : 0.0f);
        
        // Entropy of sequence distribution
        float entropy = 0.0f;
        for (const auto& pattern : patterns) {
            if (pattern.probability > 0) {
                entropy -= pattern.probability * std::log2(pattern.probability);
            }
        }
        features.push_back(entropy);
    } else {
        features.push_back(0.0f);
        features.push_back(0.0f);
    }
    
    // Pad to 8 features
    while (features.size() < 8) {
        features.push_back(0.0f);
    }
    
    return features;
}

std::vector<SequenceFeatureExtractor::SequencePattern>
SequenceFeatureExtractor::find_sequences_ending_with(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<SequencePattern> patterns;
    std::unordered_map<std::vector<uint32_t>, uint32_t, 
        std::hash<std::vector<uint32_t>>> sequence_counts;
    
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    
    // Custom hasher for vector<uint32_t>
    struct VectorHash {
        size_t operator()(const std::vector<uint32_t>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
    
    std::unordered_map<std::vector<uint32_t>, uint32_t, VectorHash> seq_counts;
    
    // Find all sequences ending with target key
    for (size_t i = config_.min_sequence_length; i < events.size(); ++i) {
        if (events[i].key_hash_low == target_key_low) {
            // Extract sequences of different lengths
            for (size_t len = config_.min_sequence_length; 
                 len <= config_.max_sequence_length && len <= i + 1; ++len) {
                
                std::vector<uint32_t> sequence;
                for (size_t j = i - len + 1; j <= i; ++j) {
                    sequence.push_back(events[j].key_hash_low);
                }
                
                seq_counts[sequence]++;
            }
        }
    }
    
    // Convert to pattern list
    uint32_t total_sequences = 0;
    for (const auto& [seq, count] : seq_counts) {
        total_sequences += count;
    }
    
    for (const auto& [seq, count] : seq_counts) {
        patterns.push_back({
            seq,
            count,
            static_cast<float>(count) / total_sequences
        });
    }
    
    // Sort by frequency
    std::sort(patterns.begin(), patterns.end(),
              [](const SequencePattern& a, const SequencePattern& b) {
                  return a.frequency > b.frequency;
              });
    
    return patterns;
}

float SequenceFeatureExtractor::compute_sequence_predictability(
    const std::vector<SequencePattern>& patterns) const {
    
    if (patterns.empty()) return 0.0f;
    
    // Use top pattern probability as predictability score
    float top_prob = patterns[0].probability;
    
    // Adjust based on pattern diversity
    float diversity_penalty = 0.0f;
    if (patterns.size() > 1) {
        // Calculate entropy
        float entropy = 0.0f;
        for (const auto& pattern : patterns) {
            if (pattern.probability > 0) {
                entropy -= pattern.probability * std::log2(pattern.probability);
            }
        }
        // Normalize entropy to [0,1]
        float max_entropy = std::log2(static_cast<float>(patterns.size()));
        diversity_penalty = entropy / max_entropy;
    }
    
    return top_prob * (1.0f - 0.5f * diversity_penalty);
}

// Relationship Feature Extractor Implementation

std::vector<float> RelationshipFeatureExtractor::extract(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    std::vector<float> features;
    features.reserve(8);
    
    auto cooc_stats = analyze_cooccurrence(events, target_key);
    
    // Basic co-occurrence features
    features.push_back(static_cast<float>(cooc_stats.unique_cooccurring_keys));
    features.push_back(cooc_stats.avg_cooccurrence_strength);
    
    // Graph-based features
    if (config_.include_graph_features) {
        features.push_back(cooc_stats.clustering_coefficient);
        
        // Degree centrality (normalized)
        float degree_centrality = static_cast<float>(cooc_stats.unique_cooccurring_keys) / 
                                 std::min(static_cast<size_t>(1000), events.size());
        features.push_back(degree_centrality);
    }
    
    // Top co-occurring key strengths
    std::vector<std::pair<uint32_t, uint32_t>> top_cooc;
    for (const auto& [key, count] : cooc_stats.related_keys) {
        top_cooc.push_back({key, count});
    }
    std::sort(top_cooc.begin(), top_cooc.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < 4; ++i) {
        if (i < top_cooc.size()) {
            features.push_back(static_cast<float>(top_cooc[i].second));
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Pad to 8 features
    while (features.size() < 8) {
        features.push_back(0.0f);
    }
    
    return features;
}

RelationshipFeatureExtractor::CooccurrenceStats
RelationshipFeatureExtractor::analyze_cooccurrence(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) const {
    
    CooccurrenceStats stats = {};
    uint32_t target_key_low = static_cast<uint32_t>(target_key);
    
    // Find co-occurring keys within time windows
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].key_hash_low == target_key_low) {
            uint32_t window_start = events[i].timestamp_offset;
            uint32_t window_end = window_start + config_.cooccurrence_window_us;
            
            // Look forward within window
            for (size_t j = i + 1; j < events.size() && 
                 events[j].timestamp_offset <= window_end; ++j) {
                if (events[j].key_hash_low != target_key_low) {
                    stats.related_keys[events[j].key_hash_low]++;
                }
            }
            
            // Look backward within window
            if (i > 0) {
                for (int j = i - 1; j >= 0 && 
                     events[i].timestamp_offset - events[j].timestamp_offset <= config_.cooccurrence_window_us; --j) {
                    if (events[j].key_hash_low != target_key_low) {
                        stats.related_keys[events[j].key_hash_low]++;
                    }
                }
            }
        }
    }
    
    stats.unique_cooccurring_keys = static_cast<uint32_t>(stats.related_keys.size());
    
    // Calculate average co-occurrence strength
    if (!stats.related_keys.empty()) {
        uint32_t total_cooc = 0;
        for (const auto& [key, count] : stats.related_keys) {
            total_cooc += count;
        }
        stats.avg_cooccurrence_strength = static_cast<float>(total_cooc) / stats.related_keys.size();
    }
    
    // Simple clustering coefficient (ratio of actual to possible connections)
    // This is simplified - proper implementation would need full graph
    if (stats.unique_cooccurring_keys > 1) {
        float max_possible = static_cast<float>(stats.unique_cooccurring_keys * 
                                               (stats.unique_cooccurring_keys - 1)) / 2.0f;
        float actual_connections = 0.0f;
        
        // Count connections between co-occurring keys
        std::unordered_set<uint32_t> cooc_keys;
        for (const auto& [key, count] : stats.related_keys) {
            cooc_keys.insert(key);
        }
        
        // Simplified: assume keys that co-occur with target also co-occur with each other
        actual_connections = static_cast<float>(cooc_keys.size());
        
        stats.clustering_coefficient = actual_connections / max_possible;
    }
    
    return stats;
}

// Feature Engineering Pipeline Implementation

FeatureEngineeringPipeline::FeatureEngineeringPipeline(const Config& config)
    : config_(config),
      temporal_extractor_(std::make_unique<TemporalFeatureExtractor>()),
      frequency_extractor_(std::make_unique<FrequencyFeatureExtractor>()),
      sequence_extractor_(std::make_unique<SequenceFeatureExtractor>()),
      relationship_extractor_(std::make_unique<RelationshipFeatureExtractor>()) {
}

FeatureVector FeatureEngineeringPipeline::extract_features(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    uint64_t target_key) {
    
    // Check cache
    if (config_.enable_caching) {
        auto it = feature_cache_.find(target_key);
        if (it != feature_cache_.end() && is_cache_valid(it->second)) {
            return it->second.features;
        }
    }
    
    FeatureVector features;
    features.target_key_hash = target_key;
    features.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Extract features from each extractor
    features.temporal_features = temporal_extractor_->extract(events, target_key);
    features.frequency_features = frequency_extractor_->extract(events, target_key);
    features.sequence_features = sequence_extractor_->extract(events, target_key);
    features.relationship_features = relationship_extractor_->extract(events, target_key);
    
    // Normalize if configured
    if (config_.normalize_features && !normalization_stats_.mean.empty()) {
        normalize_features(features);
    }
    
    // Apply feature selection
    if (config_.remove_low_variance_features && !normalization_stats_.low_variance_mask.empty()) {
        apply_feature_selection(features);
    }
    
    // Cache the result
    if (config_.enable_caching) {
        feature_cache_[target_key] = {features, features.timestamp};
        
        // Evict old entries if cache is full
        if (feature_cache_.size() > config_.cache_size) {
            // Simple LRU eviction
            uint64_t oldest_time = UINT64_MAX;
            uint64_t oldest_key = 0;
            for (const auto& [key, entry] : feature_cache_) {
                if (entry.timestamp < oldest_time) {
                    oldest_time = entry.timestamp;
                    oldest_key = key;
                }
            }
            feature_cache_.erase(oldest_key);
        }
    }
    
    return features;
}
std::vector<FeatureVector> FeatureEngineeringPipeline::extract_features_batch(
    const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
    const std::vector<uint64_t>& target_keys) {
    
    std::vector<FeatureVector> batch_features;
    batch_features.reserve(target_keys.size());
    
    for (uint64_t key : target_keys) {
        batch_features.push_back(extract_features(events, key));
    }
    
    return batch_features;
}

FeatureVector FeatureEngineeringPipeline::extract_features_realtime(
    const logger::OptimizedAccessLogger::CompactEvent* recent_events,
    size_t event_count,
    uint64_t target_key) {
    
    // Convert to vector for extractors
    std::vector<logger::OptimizedAccessLogger::CompactEvent> events(
        recent_events, recent_events + event_count);
    
    // Use regular extraction with caching disabled for real-time
    bool original_caching = config_.enable_caching;
    config_.enable_caching = false;
    
    auto features = extract_features(events, target_key);
    
    config_.enable_caching = original_caching;
    
    return features;
}

void FeatureEngineeringPipeline::normalize_features(FeatureVector& features) {
    auto flat = features.flatten();
    
    if (flat.size() \!= normalization_stats_.mean.size()) {
        return;  // Mismatched dimensions
    }
    
    // Apply z-score normalization
    for (size_t i = 0; i < flat.size(); ++i) {
        if (normalization_stats_.std_dev[i] > 0) {
            flat[i] = (flat[i] - normalization_stats_.mean[i]) / normalization_stats_.std_dev[i];
        }
    }
    
    // Unflatten back to feature vectors
    size_t idx = 0;
    for (size_t i = 0; i < features.temporal_features.size(); ++i) {
        features.temporal_features[i] = flat[idx++];
    }
    for (size_t i = 0; i < features.frequency_features.size(); ++i) {
        features.frequency_features[i] = flat[idx++];
    }
    for (size_t i = 0; i < features.sequence_features.size(); ++i) {
        features.sequence_features[i] = flat[idx++];
    }
    for (size_t i = 0; i < features.relationship_features.size(); ++i) {
        features.relationship_features[i] = flat[idx++];
    }
}

void FeatureEngineeringPipeline::apply_feature_selection(FeatureVector& features) {
    // Remove low variance features
    auto apply_mask = [](std::vector<float>& vec, const std::vector<bool>& mask, size_t& mask_idx) {
        std::vector<float> filtered;
        for (size_t i = 0; i < vec.size() && mask_idx < mask.size(); ++i, ++mask_idx) {
            if (\!mask[mask_idx]) {  // false means keep the feature
                filtered.push_back(vec[i]);
            }
        }
        vec = filtered;
    };
    
    size_t mask_idx = 0;
    apply_mask(features.temporal_features, normalization_stats_.low_variance_mask, mask_idx);
    apply_mask(features.frequency_features, normalization_stats_.low_variance_mask, mask_idx);
    apply_mask(features.sequence_features, normalization_stats_.low_variance_mask, mask_idx);
    apply_mask(features.relationship_features, normalization_stats_.low_variance_mask, mask_idx);
}

bool FeatureEngineeringPipeline::is_cache_valid(const CacheEntry& entry) const {
    // Cache entries are valid for 60 seconds
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    return (current_time - entry.timestamp) < 60000000;  // 60 seconds in microseconds
}

FeatureEngineeringPipeline::FeatureStats 
FeatureEngineeringPipeline::compute_feature_statistics(
    const std::vector<FeatureVector>& features) {
    
    FeatureStats stats;
    
    if (features.empty()) return stats;
    
    // Get feature dimension
    size_t dim = features[0].dimension();
    stats.mean.resize(dim, 0.0f);
    stats.std_dev.resize(dim, 0.0f);
    stats.min_val.resize(dim, std::numeric_limits<float>::max());
    stats.max_val.resize(dim, std::numeric_limits<float>::lowest());
    
    // First pass: compute mean, min, max
    for (const auto& fv : features) {
        auto flat = fv.flatten();
        for (size_t i = 0; i < dim; ++i) {
            stats.mean[i] += flat[i];
            stats.min_val[i] = std::min(stats.min_val[i], flat[i]);
            stats.max_val[i] = std::max(stats.max_val[i], flat[i]);
        }
    }
    
    for (size_t i = 0; i < dim; ++i) {
        stats.mean[i] /= features.size();
    }
    
    // Second pass: compute standard deviation
    for (const auto& fv : features) {
        auto flat = fv.flatten();
        for (size_t i = 0; i < dim; ++i) {
            float diff = flat[i] - stats.mean[i];
            stats.std_dev[i] += diff * diff;
        }
    }
    
    for (size_t i = 0; i < dim; ++i) {
        stats.std_dev[i] = std::sqrt(stats.std_dev[i] / features.size());
    }
    
    // Identify low variance features
    stats.low_variance_mask.resize(dim, false);
    for (size_t i = 0; i < dim; ++i) {
        if (stats.std_dev[i] < config_.variance_threshold) {
            stats.low_variance_mask[i] = true;
        }
    }
    
    // Store for normalization
    normalization_stats_ = stats;
    
    return stats;
}

// GPU Feature Computer stub implementation
class GPUFeatureComputer::Impl {
public:
    Impl(const Config& config) : config_(config) {}
    
    std::vector<FeatureVector> compute_features_gpu(
        const logger::OptimizedAccessLogger::CompactEvent* events,
        size_t event_count,
        const uint64_t* target_keys,
        size_t key_count) {
        
        // TODO: Implement GPU kernels for feature extraction
        // For now, fall back to CPU implementation
        std::vector<logger::OptimizedAccessLogger::CompactEvent> event_vec(
            events, events + event_count);
        
        FeatureEngineeringPipeline pipeline;
        std::vector<uint64_t> keys(target_keys, target_keys + key_count);
        
        return pipeline.extract_features_batch(event_vec, keys);
    }
    
    void compute_features_async(
        const logger::OptimizedAccessLogger::CompactEvent* events,
        size_t event_count,
        const uint64_t* target_keys,
        size_t key_count,
        FeatureVector* output) {
        
        auto results = compute_features_gpu(events, event_count, target_keys, key_count);
        std::copy(results.begin(), results.end(), output);
    }
    
    void wait_for_completion() {
        // TODO: Implement when GPU kernels are added
    }
    
    GPUFeatureComputer::PerformanceMetrics get_performance_metrics() const {
        return {
            .avg_computation_time_ms = 0.0f,
            .peak_memory_usage_mb = 0.0f,
            .total_features_computed = 0
        };
    }
    
private:
    Config config_;
};

GPUFeatureComputer::GPUFeatureComputer(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>(config)) {
}

GPUFeatureComputer::~GPUFeatureComputer() = default;

std::vector<FeatureVector> GPUFeatureComputer::compute_features_gpu(
    const logger::OptimizedAccessLogger::CompactEvent* events,
    size_t event_count,
    const uint64_t* target_keys,
    size_t key_count) {
    return impl_->compute_features_gpu(events, event_count, target_keys, key_count);
}

void GPUFeatureComputer::compute_features_async(
    const logger::OptimizedAccessLogger::CompactEvent* events,
    size_t event_count,
    const uint64_t* target_keys,
    size_t key_count,
    FeatureVector* output) {
    impl_->compute_features_async(events, event_count, target_keys, key_count, output);
}

void GPUFeatureComputer::wait_for_completion() {
    impl_->wait_for_completion();
}

GPUFeatureComputer::PerformanceMetrics GPUFeatureComputer::get_performance_metrics() const {
    return impl_->get_performance_metrics();
}

} // namespace ml
} // namespace predis
