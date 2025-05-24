#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include <functional>

namespace predis {
namespace mlops {

// Configuration for drift detection
struct DriftDetectorConfig {
    double ks_threshold = 0.05;           // Kolmogorov-Smirnov test threshold
    double psi_threshold = 0.25;          // Population Stability Index threshold
    size_t min_samples = 1000;            // Minimum samples for detection
    size_t window_size = 10000;           // Sliding window size
    std::chrono::minutes check_interval{30};  // How often to check for drift
    bool enable_alerts = true;
};

// Drift detection results
struct DriftResult {
    bool drift_detected;
    double confidence;
    std::string drift_type;  // "distribution", "feature", "performance"
    std::vector<std::string> affected_features;
    std::chrono::system_clock::time_point detection_time;
    
    // Statistical test results
    double ks_statistic;
    double psi_score;
    double performance_delta;
};

// Alert callback for drift events
using DriftAlertCallback = std::function<void(const DriftResult&)>;

// Model drift detector using statistical tests
class ModelDriftDetector {
public:
    explicit ModelDriftDetector(const DriftDetectorConfig& config);
    
    // Register callback for drift alerts
    void RegisterAlertCallback(DriftAlertCallback callback);
    
    // Update with new data
    void UpdateDistribution(const std::vector<std::vector<float>>& features,
                          const std::vector<float>& predictions,
                          const std::vector<float>& actuals);
    
    // Check for drift
    DriftResult CheckForDrift();
    
    // Get historical drift scores
    std::vector<DriftResult> GetDriftHistory(size_t last_n = 10) const;
    
    // Force retraining recommendation
    bool ShouldRetrain(const DriftResult& result,
                       double performance_threshold = 0.1) const;
    
private:
    DriftDetectorConfig config_;
    std::vector<DriftAlertCallback> alert_callbacks_;
    
    // Baseline distributions (from training)
    struct BaselineDistribution {
        std::vector<std::vector<float>> feature_distributions;
        std::vector<float> prediction_distribution;
        double baseline_accuracy;
        size_t sample_count;
    };
    
    std::unique_ptr<BaselineDistribution> baseline_;
    
    // Recent data window
    struct RecentWindow {
        std::vector<std::vector<float>> features;
        std::vector<float> predictions;
        std::vector<float> actuals;
        double current_accuracy;
        
        void Add(const std::vector<float>& feature,
                float prediction,
                float actual);
        void Trim(size_t max_size);
    };
    
    RecentWindow recent_window_;
    std::vector<DriftResult> drift_history_;
    
    // Statistical tests
    double KolmogorovSmirnovTest(const std::vector<float>& dist1,
                                 const std::vector<float>& dist2) const;
    
    double PopulationStabilityIndex(const std::vector<float>& expected,
                                   const std::vector<float>& actual) const;
    
    // Feature-level drift detection
    std::vector<std::string> DetectFeatureDrift() const;
    
    // Performance drift detection
    double CalculatePerformanceDrift() const;
    
    // Alert management
    void TriggerAlerts(const DriftResult& result);
};

// ADWIN (Adaptive Windowing) algorithm for concept drift
class ADWINDriftDetector {
public:
    explicit ADWINDriftDetector(double delta = 0.002);
    
    // Update with new prediction error
    void Update(double error);
    
    // Check if drift detected
    bool DriftDetected() const { return drift_detected_; }
    
    // Get adaptive window size
    size_t GetWindowSize() const { return window_.size(); }
    
    // Reset detector
    void Reset();
    
private:
    double delta_;  // Confidence parameter
    bool drift_detected_;
    
    struct Window {
        std::vector<double> data;
        double total;
        double variance;
        size_t width;
        
        void Add(double value);
        bool CheckSplit(double delta);
    };
    
    Window window_;
};

// Ensemble drift detector combining multiple methods
class EnsembleDriftDetector {
public:
    EnsembleDriftDetector();
    
    // Add individual detectors
    void AddDetector(std::unique_ptr<ModelDriftDetector> detector);
    void AddADWINDetector(std::unique_ptr<ADWINDriftDetector> detector);
    
    // Combined drift detection
    DriftResult DetectDrift(const std::vector<std::vector<float>>& features,
                          const std::vector<float>& predictions,
                          const std::vector<float>& actuals);
    
    // Voting-based decision
    bool ConsensusReached(double threshold = 0.5) const;
    
private:
    std::vector<std::unique_ptr<ModelDriftDetector>> statistical_detectors_;
    std::vector<std::unique_ptr<ADWINDriftDetector>> adwin_detectors_;
    std::vector<DriftResult> recent_results_;
};

}  // namespace mlops
}  // namespace predis