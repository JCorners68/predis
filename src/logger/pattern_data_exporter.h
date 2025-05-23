#pragma once

#include "access_pattern_logger.h"
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace predis {
namespace logger {

// Format for exporting access pattern data
class PatternDataExporter {
public:
    enum class ExportFormat {
        CSV,        // Simple CSV format
        JSON,       // JSON format for complex patterns
        BINARY,     // Compact binary format
        PARQUET     // Apache Parquet for ML pipelines (future)
    };
    
    struct Config {
        ExportFormat format = ExportFormat::CSV;
        std::string output_directory = "./pattern_data/";
        size_t batch_size = 10000;
        bool compress = false;
        bool include_patterns = true;
        std::chrono::seconds rotation_interval{3600};  // Rotate files hourly
    };
    
    explicit PatternDataExporter(const Config& config = {});
    ~PatternDataExporter();
    
    // Export methods
    void export_events(const std::vector<AccessEvent>& events);
    void export_patterns(const std::vector<AccessPattern>& patterns);
    
    // Batch export with pattern analysis
    void export_analyzed_batch(const std::vector<AccessEvent>& events,
                              const std::vector<AccessPattern>& patterns);
    
    // File management
    std::string get_current_filename() const;
    void rotate_file();
    
    // Statistics
    struct ExportStats {
        uint64_t events_exported;
        uint64_t patterns_exported;
        uint64_t files_created;
        size_t total_bytes_written;
        double avg_export_time_ms;
    };
    
    ExportStats get_statistics() const;
    
private:
    Config config_;
    std::unique_ptr<std::ofstream> current_file_;
    std::mutex file_mutex_;
    
    // Statistics
    std::atomic<uint64_t> events_exported_{0};
    std::atomic<uint64_t> patterns_exported_{0};
    std::atomic<uint64_t> files_created_{0};
    std::atomic<size_t> total_bytes_written_{0};
    
    // File rotation
    std::chrono::system_clock::time_point last_rotation_;
    
    // Export implementations
    void export_csv(const std::vector<AccessEvent>& events);
    void export_json(const std::vector<AccessEvent>& events,
                    const std::vector<AccessPattern>& patterns);
    void export_binary(const std::vector<AccessEvent>& events);
    
    // Helper methods
    void ensure_directory_exists();
    std::string generate_filename() const;
    void write_csv_header();
};

// ML training data formatter
class MLTrainingDataFormatter {
public:
    struct FeatureConfig {
        size_t sequence_length = 10;        // Length of key sequences
        size_t time_buckets = 24;          // Hour-of-day buckets
        size_t frequency_windows = 3;       // Different time windows for frequency
        bool include_day_of_week = true;
        bool include_key_embeddings = true;
        size_t embedding_dim = 32;
    };
    
    explicit MLTrainingDataFormatter(const FeatureConfig& config = {});
    
    // Convert raw events to ML features
    struct TrainingExample {
        std::vector<float> features;      // Input features
        float target;                     // Target (e.g., will be accessed in next N seconds)
        uint64_t key_hash;               // For tracking
        uint64_t timestamp;              // When this example was created
    };
    
    std::vector<TrainingExample> create_training_data(
        const std::vector<AccessEvent>& events,
        const std::vector<AccessPattern>& patterns);
    
    // Feature extraction for real-time inference
    std::vector<float> extract_features(
        const std::vector<AccessEvent>& recent_events,
        uint64_t target_key);
    
    // Save training data in formats suitable for ML frameworks
    void save_tensorflow_format(const std::vector<TrainingExample>& examples,
                               const std::string& filename);
    void save_pytorch_format(const std::vector<TrainingExample>& examples,
                            const std::string& filename);
    
private:
    FeatureConfig config_;
    
    // Feature extraction helpers
    std::vector<float> extract_temporal_features(
        const std::vector<AccessEvent>& events, uint64_t key);
    std::vector<float> extract_frequency_features(
        const std::vector<AccessEvent>& events, uint64_t key);
    std::vector<float> extract_sequence_features(
        const std::vector<AccessEvent>& events, uint64_t key);
    std::vector<float> extract_pattern_features(
        const std::vector<AccessPattern>& patterns, uint64_t key);
    
    // Key embedding cache (learned from data)
    std::unordered_map<uint64_t, std::vector<float>> key_embeddings_;
    void update_key_embeddings(const std::vector<AccessEvent>& events);
};

// Streaming data pipeline for continuous ML training
class StreamingDataPipeline {
public:
    struct Config {
        size_t buffer_size = 100000;
        std::chrono::seconds analysis_interval{60};
        std::chrono::seconds export_interval{300};
        bool enable_online_learning = false;
    };
    
    StreamingDataPipeline(AccessPatternLogger& logger,
                         PatternAnalyzer& analyzer,
                         const Config& config = {});
    ~StreamingDataPipeline();
    
    // Start/stop the pipeline
    void start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Register callbacks for processed data
    using DataCallback = std::function<void(const std::vector<TrainingExample>&)>;
    void register_training_callback(DataCallback callback);
    
    // Online learning support
    void update_model_feedback(uint64_t key, bool was_accessed);
    
private:
    AccessPatternLogger& logger_;
    PatternAnalyzer& analyzer_;
    Config config_;
    
    // Pipeline state
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> pipeline_thread_;
    
    // Data buffers
    std::vector<AccessEvent> event_buffer_;
    std::mutex buffer_mutex_;
    
    // Callbacks
    std::vector<DataCallback> callbacks_;
    std::mutex callback_mutex_;
    
    // Pipeline worker
    void pipeline_worker();
    void process_batch();
    
    // Feedback for online learning
    struct FeedbackEntry {
        uint64_t key_hash;
        uint64_t prediction_time;
        bool was_accessed;
    };
    std::queue<FeedbackEntry> feedback_queue_;
    std::mutex feedback_mutex_;
};

} // namespace logger
} // namespace predis