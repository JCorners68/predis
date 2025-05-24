#pragma once

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace predis {
namespace mlops {

// Forward declarations
class BaseModel;
class ModelDriftDetector;

// Configuration for retraining pipeline
struct RetrainingConfig {
    size_t min_training_samples = 10000;
    size_t max_training_samples = 1000000;
    double validation_split = 0.2;
    double performance_threshold = 0.95;  // New model must be 95% as good
    size_t ab_test_duration_hours = 24;
    double ab_test_traffic_split = 0.1;  // 10% to new model
    bool enable_automatic_rollback = true;
    std::chrono::hours retraining_cooldown{6};  // Min time between retrainings
};

// Training data batch
struct TrainingBatch {
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    std::vector<uint64_t> timestamps;
    std::string data_source;
    
    size_t Size() const { return features.size(); }
    bool IsValid() const { 
        return !features.empty() && features.size() == labels.size();
    }
};

// Model performance metrics
struct ModelMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double avg_inference_time_ms;
    size_t total_predictions;
    std::chrono::system_clock::time_point evaluation_time;
    
    bool IsBetterThan(const ModelMetrics& other, double threshold = 0.95) const {
        // New model must be at least threshold% as good
        return f1_score >= other.f1_score * threshold &&
               avg_inference_time_ms <= other.avg_inference_time_ms * 1.1;  // Allow 10% slower
    }
};

// A/B test configuration and results
struct ABTestConfig {
    std::string test_id;
    double traffic_split;  // Fraction to new model
    std::chrono::system_clock::time_point start_time;
    std::chrono::hours duration;
    bool auto_promote = true;
    bool auto_rollback = true;
};

struct ABTestResults {
    std::string test_id;
    ModelMetrics control_metrics;  // Current production model
    ModelMetrics treatment_metrics;  // New model
    double statistical_significance;
    bool treatment_is_better;
    size_t control_samples;
    size_t treatment_samples;
};

// Callback types
using RetrainingCallback = std::function<void(const std::string& model_id, bool success)>;
using MetricsCallback = std::function<void(const ModelMetrics& metrics)>;

// Auto retraining pipeline
class AutoRetrainingPipeline {
public:
    explicit AutoRetrainingPipeline(const RetrainingConfig& config);
    ~AutoRetrainingPipeline();
    
    // Start/stop the pipeline
    void Start();
    void Stop();
    
    // Trigger retraining manually
    void TriggerRetraining(const std::string& reason);
    
    // Add training data
    void AddTrainingData(const TrainingBatch& batch);
    
    // Model management
    void SetCurrentModel(std::shared_ptr<BaseModel> model);
    std::shared_ptr<BaseModel> GetCurrentModel() const;
    
    // A/B testing
    std::string StartABTest(std::shared_ptr<BaseModel> new_model, 
                           const ABTestConfig& config);
    ABTestResults GetABTestResults(const std::string& test_id) const;
    void EndABTest(const std::string& test_id, bool promote_new_model);
    
    // Callbacks
    void RegisterRetrainingCallback(RetrainingCallback callback);
    void RegisterMetricsCallback(MetricsCallback callback);
    
    // Metrics
    struct PipelineMetrics {
        size_t total_retrainings;
        size_t successful_retrainings;
        size_t failed_retrainings;
        size_t models_promoted;
        size_t models_rolled_back;
        std::chrono::system_clock::time_point last_retraining;
        double avg_retraining_time_minutes;
    };
    
    PipelineMetrics GetPipelineMetrics() const;
    
private:
    RetrainingConfig config_;
    std::atomic<bool> running_{false};
    std::thread retraining_thread_;
    std::thread monitoring_thread_;
    
    // Models
    mutable std::mutex model_mutex_;
    std::shared_ptr<BaseModel> current_model_;
    std::shared_ptr<BaseModel> candidate_model_;
    
    // Training data management
    mutable std::mutex data_mutex_;
    std::vector<TrainingBatch> training_data_buffer_;
    size_t total_training_samples_ = 0;
    
    // A/B testing
    struct ActiveABTest {
        ABTestConfig config;
        std::shared_ptr<BaseModel> treatment_model;
        std::atomic<size_t> control_requests{0};
        std::atomic<size_t> treatment_requests{0};
        ModelMetrics control_metrics;
        ModelMetrics treatment_metrics;
    };
    
    mutable std::mutex ab_test_mutex_;
    std::unordered_map<std::string, ActiveABTest> active_ab_tests_;
    
    // Metrics
    std::atomic<size_t> total_retrainings_{0};
    std::atomic<size_t> successful_retrainings_{0};
    std::atomic<size_t> failed_retrainings_{0};
    std::atomic<size_t> models_promoted_{0};
    std::atomic<size_t> models_rolled_back_{0};
    std::chrono::system_clock::time_point last_retraining_;
    
    // Callbacks
    std::vector<RetrainingCallback> retraining_callbacks_;
    std::vector<MetricsCallback> metrics_callbacks_;
    
    // Internal methods
    void RetrainingLoop();
    void MonitoringLoop();
    
    std::shared_ptr<BaseModel> TrainNewModel(const std::vector<TrainingBatch>& data);
    ModelMetrics EvaluateModel(std::shared_ptr<BaseModel> model, 
                              const TrainingBatch& validation_data);
    
    bool ShouldRetrain() const;
    std::vector<TrainingBatch> PrepareTrainingData();
    
    void PromoteModel(std::shared_ptr<BaseModel> new_model);
    void RollbackModel();
    
    void NotifyRetrainingComplete(const std::string& model_id, bool success);
    void NotifyMetricsUpdate(const ModelMetrics& metrics);
    
    std::string GenerateModelId() const;
    std::string GenerateABTestId() const;
};

// Model validator for comparing models
class ModelValidator {
public:
    ModelValidator();
    
    // Validate new model against current
    bool ValidateModel(std::shared_ptr<BaseModel> new_model,
                      std::shared_ptr<BaseModel> current_model,
                      const TrainingBatch& validation_data,
                      double performance_threshold = 0.95);
    
    // Statistical significance testing
    double CalculateStatisticalSignificance(const ModelMetrics& control,
                                           const ModelMetrics& treatment,
                                           size_t control_samples,
                                           size_t treatment_samples);
    
private:
    // Helper methods for statistical tests
    double CalculateTStatistic(double mean1, double var1, size_t n1,
                              double mean2, double var2, size_t n2);
    double CalculatePValue(double t_statistic, size_t df);
};

// Model versioning and storage
class ModelVersionManager {
public:
    explicit ModelVersionManager(const std::string& storage_path);
    
    // Save model with metadata
    std::string SaveModel(std::shared_ptr<BaseModel> model,
                         const ModelMetrics& metrics,
                         const std::string& training_config);
    
    // Load model by version
    std::shared_ptr<BaseModel> LoadModel(const std::string& version_id);
    
    // List available versions
    std::vector<std::string> ListVersions() const;
    
    // Get model metadata
    struct ModelMetadata {
        std::string version_id;
        std::string model_type;
        ModelMetrics metrics;
        std::chrono::system_clock::time_point created_at;
        std::string training_config;
        size_t model_size_bytes;
    };
    
    ModelMetadata GetMetadata(const std::string& version_id) const;
    
    // Cleanup old versions
    void CleanupOldVersions(size_t keep_recent_n = 10);
    
private:
    std::string storage_path_;
    mutable std::mutex storage_mutex_;
    
    std::string GenerateVersionPath(const std::string& version_id) const;
    void SaveMetadata(const std::string& version_id, const ModelMetadata& metadata);
    ModelMetadata LoadMetadata(const std::string& version_id) const;
};

}  // namespace mlops
}  // namespace predis