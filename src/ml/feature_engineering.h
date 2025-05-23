#pragma once

#include "../logger/optimized_access_logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace predis {
namespace ml {

// Feature vector for ML models
struct FeatureVector {
    // Temporal features (32 dimensions)
    std::vector<float> temporal_features;
    
    // Frequency features (16 dimensions)
    std::vector<float> frequency_features;
    
    // Sequence features (8 dimensions)
    std::vector<float> sequence_features;
    
    // Relationship features (8 dimensions)
    std::vector<float> relationship_features;
    
    // Metadata
    uint64_t target_key_hash;
    uint64_t timestamp;
    
    // Flatten all features into a single vector
    std::vector<float> flatten() const {
        std::vector<float> flat;
        flat.reserve(temporal_features.size() + frequency_features.size() + 
                    sequence_features.size() + relationship_features.size());
        
        flat.insert(flat.end(), temporal_features.begin(), temporal_features.end());
        flat.insert(flat.end(), frequency_features.begin(), frequency_features.end());
        flat.insert(flat.end(), sequence_features.begin(), sequence_features.end());
        flat.insert(flat.end(), relationship_features.begin(), relationship_features.end());
        
        return flat;
    }
    
    size_t dimension() const {
        return temporal_features.size() + frequency_features.size() + 
               sequence_features.size() + relationship_features.size();
    }
};

// Base class for feature extractors
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    virtual std::vector<float> extract(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const = 0;
};

// Temporal feature extractor
class TemporalFeatureExtractor : public FeatureExtractor {
public:
    struct Config {
        std::vector<uint32_t> window_sizes_seconds = {60, 300, 3600}; // 1min, 5min, 1hr
        bool include_hour_of_day = true;
        bool include_day_of_week = true;
        bool include_seasonality = true;
    };
    
    explicit TemporalFeatureExtractor(const Config& config = {})
        : config_(config) {}
    
    std::vector<float> extract(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const override;
    
private:
    Config config_;
    
    // Helper methods
    std::vector<float> extract_access_counts(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key,
        uint32_t window_seconds) const;
    
    std::vector<float> extract_hour_distribution(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const;
    
    std::vector<float> extract_day_distribution(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const;
};

// Frequency feature extractor
class FrequencyFeatureExtractor : public FeatureExtractor {
public:
    struct Config {
        uint32_t max_inter_access_intervals = 10;
        bool include_regularity_score = true;
        bool include_burst_detection = true;
    };
    
    explicit FrequencyFeatureExtractor(const Config& config = {})
        : config_(config) {}
    
    std::vector<float> extract(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const override;
    
private:
    Config config_;
    
    struct FrequencyStats {
        uint32_t total_accesses;
        float access_rate;  // Accesses per second
        float time_since_last_access;
        float mean_inter_access_time;
        float std_inter_access_time;
        float regularity_score;
        float burst_score;
    };
    
    FrequencyStats compute_frequency_stats(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const;
};

// Sequence feature extractor
class SequenceFeatureExtractor : public FeatureExtractor {
public:
    struct Config {
        size_t min_sequence_length = 3;
        size_t max_sequence_length = 10;
        size_t top_sequences_to_track = 5;
        bool include_markov_features = true;
    };
    
    explicit SequenceFeatureExtractor(const Config& config = {})
        : config_(config) {}
    
    std::vector<float> extract(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const override;
    
private:
    Config config_;
    
    // Sequence analysis
    struct SequencePattern {
        std::vector<uint32_t> sequence;
        uint32_t frequency;
        float probability;
    };
    
    std::vector<SequencePattern> find_sequences_ending_with(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const;
    
    float compute_sequence_predictability(
        const std::vector<SequencePattern>& patterns) const;
};

// Relationship feature extractor
class RelationshipFeatureExtractor : public FeatureExtractor {
public:
    struct Config {
        uint32_t cooccurrence_window_us = 1000000; // 1 second
        size_t max_related_keys = 20;
        bool include_graph_features = true;
    };
    
    explicit RelationshipFeatureExtractor(const Config& config = {})
        : config_(config) {}
    
    std::vector<float> extract(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const override;
    
private:
    Config config_;
    
    // Co-occurrence analysis
    struct CooccurrenceStats {
        std::unordered_map<uint32_t, uint32_t> related_keys;
        float avg_cooccurrence_strength;
        uint32_t unique_cooccurring_keys;
        float clustering_coefficient;
    };
    
    CooccurrenceStats analyze_cooccurrence(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key) const;
};

// Main feature engineering pipeline
class FeatureEngineeringPipeline {
public:
    struct Config {
        bool normalize_features = true;
        bool remove_low_variance_features = true;
        float variance_threshold = 0.01f;
        bool enable_caching = true;
        size_t cache_size = 10000;
    };
    
    explicit FeatureEngineeringPipeline(const Config& config = {});
    
    // Extract features for a single key
    FeatureVector extract_features(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        uint64_t target_key);
    
    // Batch feature extraction
    std::vector<FeatureVector> extract_features_batch(
        const std::vector<logger::OptimizedAccessLogger::CompactEvent>& events,
        const std::vector<uint64_t>& target_keys);
    
    // Real-time feature extraction (optimized for latency)
    FeatureVector extract_features_realtime(
        const logger::OptimizedAccessLogger::CompactEvent* recent_events,
        size_t event_count,
        uint64_t target_key);
    
    // Feature statistics
    struct FeatureStats {
        std::vector<float> mean;
        std::vector<float> std_dev;
        std::vector<float> min_val;
        std::vector<float> max_val;
        std::vector<bool> low_variance_mask;
    };
    
    FeatureStats compute_feature_statistics(
        const std::vector<FeatureVector>& features);
    
    // Feature selection
    std::vector<size_t> select_top_features(
        const std::vector<FeatureVector>& features,
        const std::vector<float>& targets,
        size_t num_features);
    
private:
    Config config_;
    
    // Feature extractors
    std::unique_ptr<TemporalFeatureExtractor> temporal_extractor_;
    std::unique_ptr<FrequencyFeatureExtractor> frequency_extractor_;
    std::unique_ptr<SequenceFeatureExtractor> sequence_extractor_;
    std::unique_ptr<RelationshipFeatureExtractor> relationship_extractor_;
    
    // Feature cache
    struct CacheEntry {
        FeatureVector features;
        uint64_t timestamp;
    };
    std::unordered_map<uint64_t, CacheEntry> feature_cache_;
    
    // Normalization parameters
    FeatureStats normalization_stats_;
    
    // Helper methods
    void normalize_features(FeatureVector& features);
    void apply_feature_selection(FeatureVector& features);
    bool is_cache_valid(const CacheEntry& entry) const;
};

// GPU-accelerated feature computer
class GPUFeatureComputer {
public:
    struct Config {
        size_t max_events = 100000;
        size_t max_batch_size = 1000;
        bool use_shared_memory = true;
        bool enable_profiling = false;
    };
    
    explicit GPUFeatureComputer(const Config& config = {});
    ~GPUFeatureComputer();
    
    // Compute features on GPU
    std::vector<FeatureVector> compute_features_gpu(
        const logger::OptimizedAccessLogger::CompactEvent* events,
        size_t event_count,
        const uint64_t* target_keys,
        size_t key_count);
    
    // Async computation
    void compute_features_async(
        const logger::OptimizedAccessLogger::CompactEvent* events,
        size_t event_count,
        const uint64_t* target_keys,
        size_t key_count,
        FeatureVector* output);
    
    void wait_for_completion();
    
    // Performance metrics
    struct PerformanceMetrics {
        float avg_computation_time_ms;
        float peak_memory_usage_mb;
        uint64_t total_features_computed;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    
private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ml
} // namespace predis