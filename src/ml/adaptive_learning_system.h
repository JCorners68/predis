#ifndef PREDIS_ML_ADAPTIVE_LEARNING_SYSTEM_H_
#define PREDIS_ML_ADAPTIVE_LEARNING_SYSTEM_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

namespace predis {
namespace ml {

// Forward declarations
class BaseModel;
class InferenceEngine;
class FeatureEngineering;

namespace ppe {
    class PrefetchMonitor;
}

/**
 * @brief Adaptive learning system for continuous model improvement
 * 
 * Implements Story 3.7 requirements:
 * - Online learning capabilities
 * - Automatic model retraining
 * - Concept drift detection
 * - Safe model deployment with rollback
 */
class AdaptiveLearningSystem {
public:
    enum class LearningMode {
        OFFLINE,          // Traditional batch learning
        ONLINE,           // Incremental updates as data arrives
        HYBRID,           // Combination of online and periodic batch
        ADAPTIVE          // Automatically switch based on performance
    };
    
    enum class DriftType {
        NONE,             // No drift detected
        GRADUAL,          // Slow concept drift
        SUDDEN,           // Abrupt change in patterns
        RECURRING,        // Periodic/seasonal patterns
        ADVERSARIAL       // Potential attack or anomaly
    };
    
    struct AdaptiveConfig {
        LearningMode learning_mode = LearningMode::HYBRID;
        size_t mini_batch_size = 100;
        size_t retraining_threshold = 10000;  // Samples before retraining
        double performance_threshold = 0.15;   // Min acceptable performance
        double drift_threshold = 0.05;         // Drift detection sensitivity
        int model_history_size = 5;            // Previous models to keep
        bool auto_rollback = true;             // Auto rollback on performance drop
        std::chrono::minutes update_interval{30};  // Model update frequency
        bool enable_a_b_testing = true;        // Test new models before full deploy
    };
    
    struct ModelVersion {
        std::string version_id;
        std::shared_ptr<BaseModel> model;
        std::chrono::time_point<std::chrono::system_clock> created_time;
        double validation_accuracy;
        double production_accuracy;
        size_t samples_trained;
        bool is_active;
        std::unordered_map<std::string, double> metadata;
    };
    
    struct LearningStats {
        std::atomic<uint64_t> total_samples{0};
        std::atomic<uint64_t> updates_performed{0};
        std::atomic<uint64_t> rollbacks_triggered{0};
        std::atomic<uint64_t> drift_detections{0};
        std::atomic<double> current_accuracy{0.0};
        std::atomic<double> baseline_accuracy{0.0};
        std::chrono::time_point<std::chrono::system_clock> last_update;
        std::chrono::time_point<std::chrono::system_clock> last_drift_detection;
    };
    
    struct DriftDetectionResult {
        DriftType drift_type;
        double drift_magnitude;
        double confidence;
        std::vector<std::string> affected_features;
        std::string recommendation;
    };

public:
    explicit AdaptiveLearningSystem(const AdaptiveConfig& config = AdaptiveConfig());
    ~AdaptiveLearningSystem();
    
    // System lifecycle
    void initialize();
    void shutdown();
    bool isRunning() const { return running_; }
    
    // Model management
    void setBaseModel(std::shared_ptr<BaseModel> model);
    void setInferenceEngine(std::shared_ptr<InferenceEngine> engine);
    std::shared_ptr<BaseModel> getCurrentModel() const;
    ModelVersion getModelVersion(const std::string& version_id) const;
    std::vector<ModelVersion> getModelHistory() const;
    
    // Online learning
    void addTrainingSample(const std::vector<float>& features, float label);
    void addTrainingBatch(const std::vector<std::vector<float>>& features,
                         const std::vector<float>& labels);
    void triggerModelUpdate();
    
    // Drift detection
    DriftDetectionResult detectDrift();
    void enableDriftDetection(bool enable);
    bool isDriftDetected() const { return last_drift_type_ != DriftType::NONE; }
    
    // Model versioning and rollback
    std::string saveCurrentModel();
    bool rollbackToVersion(const std::string& version_id);
    bool deployModel(std::shared_ptr<BaseModel> new_model, 
                    const std::string& version_id = "");
    
    // Automatic retraining
    void enableAutoRetraining(bool enable);
    void setRetrainingSchedule(std::chrono::minutes interval);
    bool isRetrainingNeeded() const;
    
    // Performance monitoring
    void updatePerformanceMetrics(double accuracy, double latency);
    LearningStats getStats() const { return stats_; }
    void resetStats();
    
    // A/B testing support
    void startABTest(std::shared_ptr<BaseModel> test_model, 
                     double test_percentage = 0.1);
    void stopABTest(bool deploy_test_model = false);
    std::pair<double, double> getABTestResults() const;
    
    // Configuration
    void updateConfig(const AdaptiveConfig& config);
    AdaptiveConfig getConfig() const { return config_; }
    
private:
    AdaptiveConfig config_;
    std::atomic<bool> running_{false};
    LearningStats stats_;
    
    // Model management
    std::shared_ptr<BaseModel> current_model_;
    std::shared_ptr<InferenceEngine> inference_engine_;
    std::deque<ModelVersion> model_history_;
    std::mutex model_mutex_;
    
    // Training data buffer
    struct TrainingBuffer {
        std::vector<std::vector<float>> features;
        std::vector<float> labels;
        std::mutex mutex;
        
        void add(const std::vector<float>& feature, float label);
        void addBatch(const std::vector<std::vector<float>>& features,
                     const std::vector<float>& labels);
        size_t size() const;
        void clear();
        std::pair<std::vector<std::vector<float>>, std::vector<float>> getAndClear();
    };
    TrainingBuffer training_buffer_;
    
    // Drift detection
    DriftType last_drift_type_ = DriftType::NONE;
    std::vector<double> performance_history_;
    std::vector<double> feature_distributions_;
    std::atomic<bool> drift_detection_enabled_{true};
    
    // A/B testing
    std::shared_ptr<BaseModel> test_model_;
    std::atomic<double> ab_test_percentage_{0.0};
    std::atomic<uint64_t> ab_control_correct_{0};
    std::atomic<uint64_t> ab_control_total_{0};
    std::atomic<uint64_t> ab_test_correct_{0};
    std::atomic<uint64_t> ab_test_total_{0};
    
    // Background threads
    std::thread learning_thread_;
    std::thread monitoring_thread_;
    std::condition_variable learning_cv_;
    std::condition_variable monitoring_cv_;
    
    // Internal methods
    void learningThreadLoop();
    void monitoringThreadLoop();
    void performModelUpdate();
    bool validateModel(std::shared_ptr<BaseModel> model);
    void updateModelHistory(const ModelVersion& version);
    
    // Drift detection methods
    double calculateKLDivergence(const std::vector<double>& dist1, 
                                const std::vector<double>& dist2);
    double calculateWassersteinDistance(const std::vector<double>& dist1,
                                       const std::vector<double>& dist2);
    DriftType classifyDrift(double magnitude, double rate_of_change);
    
    // Performance tracking
    void updatePerformanceHistory(double accuracy);
    bool isPerformanceDegraded() const;
    
    // Model persistence
    std::string generateVersionId() const;
    std::string getModelPath(const std::string& version_id) const;
};

/**
 * @brief Incremental learning strategies for different model types
 */
class IncrementalLearningStrategy {
public:
    virtual ~IncrementalLearningStrategy() = default;
    
    virtual void updateModel(BaseModel* model,
                           const std::vector<std::vector<float>>& features,
                           const std::vector<float>& labels) = 0;
    
    virtual bool supportsModelType(const std::string& model_type) const = 0;
    virtual std::string getName() const = 0;
};

class SGDStrategy : public IncrementalLearningStrategy {
public:
    SGDStrategy(double learning_rate = 0.01, double momentum = 0.9);
    
    void updateModel(BaseModel* model,
                    const std::vector<std::vector<float>>& features,
                    const std::vector<float>& labels) override;
    
    bool supportsModelType(const std::string& model_type) const override;
    std::string getName() const override { return "SGD"; }
    
private:
    double learning_rate_;
    double momentum_;
    std::unordered_map<std::string, std::vector<float>> velocity_;
};

class AdaGradStrategy : public IncrementalLearningStrategy {
public:
    AdaGradStrategy(double learning_rate = 0.01, double epsilon = 1e-8);
    
    void updateModel(BaseModel* model,
                    const std::vector<std::vector<float>>& features,
                    const std::vector<float>& labels) override;
    
    bool supportsModelType(const std::string& model_type) const override;
    std::string getName() const override { return "AdaGrad"; }
    
private:
    double learning_rate_;
    double epsilon_;
    std::unordered_map<std::string, std::vector<float>> gradient_accumulator_;
};

/**
 * @brief Concept drift detector using statistical methods
 */
class ConceptDriftDetector {
public:
    enum class Method {
        DDM,          // Drift Detection Method
        EDDM,         // Early Drift Detection Method
        ADWIN,        // Adaptive Windowing
        PAGE_HINKLEY, // Page-Hinkley test
        KSWIN         // Kolmogorov-Smirnov Windowing
    };
    
    struct DriftPoint {
        size_t sample_index;
        double drift_level;
        std::chrono::time_point<std::chrono::system_clock> timestamp;
    };
    
    ConceptDriftDetector(Method method = Method::ADWIN, 
                        double sensitivity = 0.05);
    
    void addSample(double error);
    bool isDriftDetected() const { return drift_detected_; }
    double getDriftLevel() const { return drift_level_; }
    std::vector<DriftPoint> getDriftHistory() const { return drift_history_; }
    void reset();
    
private:
    Method method_;
    double sensitivity_;
    bool drift_detected_;
    double drift_level_;
    
    // Method-specific state
    std::deque<double> window_;
    size_t max_window_size_ = 1000;
    double sum_ = 0.0;
    double sum_squared_ = 0.0;
    size_t n_ = 0;
    
    std::vector<DriftPoint> drift_history_;
    
    // Detection methods
    void detectDDM(double error);
    void detectEDDM(double error);
    void detectADWIN(double error);
    void detectPageHinkley(double error);
    void detectKSWIN(double error);
};

} // namespace ml
} // namespace predis

#endif // PREDIS_ML_ADAPTIVE_LEARNING_SYSTEM_H_