#include "model_drift_detector.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace predis {
namespace mlops {

// Helper functions for statistical calculations
namespace {

double CalculateMean(const std::vector<float>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double CalculateStdDev(const std::vector<float>& data, double mean) {
    if (data.size() <= 1) return 0.0;
    
    double sum_sq = 0.0;
    for (float val : data) {
        double diff = val - mean;
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq / (data.size() - 1));
}

}  // anonymous namespace

// ModelDriftDetector implementation
ModelDriftDetector::ModelDriftDetector(const DriftDetectorConfig& config)
    : config_(config), baseline_(std::make_unique<BaselineDistribution>()) {
}

void ModelDriftDetector::RegisterAlertCallback(DriftAlertCallback callback) {
    alert_callbacks_.push_back(callback);
}

void ModelDriftDetector::UpdateDistribution(const std::vector<std::vector<float>>& features,
                                           const std::vector<float>& predictions,
                                           const std::vector<float>& actuals) {
    // Add to recent window
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < predictions.size() && i < actuals.size()) {
            recent_window_.Add(features[i], predictions[i], actuals[i]);
        }
    }
    
    // Trim window to max size
    recent_window_.Trim(config_.window_size);
    
    // Update current accuracy
    if (!actuals.empty() && !predictions.empty()) {
        size_t correct = 0;
        for (size_t i = 0; i < std::min(predictions.size(), actuals.size()); ++i) {
            if (std::abs(predictions[i] - actuals[i]) < 0.01f) {  // Within tolerance
                correct++;
            }
        }
        recent_window_.current_accuracy = static_cast<double>(correct) / actuals.size();
    }
    
    // Set baseline if not yet established
    if (baseline_->sample_count == 0 && recent_window_.features.size() >= config_.min_samples) {
        baseline_->feature_distributions = recent_window_.features;
        baseline_->prediction_distribution = recent_window_.predictions;
        baseline_->baseline_accuracy = recent_window_.current_accuracy;
        baseline_->sample_count = recent_window_.features.size();
    }
}

DriftResult ModelDriftDetector::CheckForDrift() {
    DriftResult result;
    result.drift_detected = false;
    result.confidence = 0.0;
    result.detection_time = std::chrono::system_clock::now();
    
    // Need minimum samples
    if (recent_window_.features.size() < config_.min_samples || 
        baseline_->sample_count == 0) {
        result.drift_type = "insufficient_data";
        return result;
    }
    
    // Check distribution drift
    if (!recent_window_.predictions.empty() && !baseline_->prediction_distribution.empty()) {
        result.ks_statistic = KolmogorovSmirnovTest(
            baseline_->prediction_distribution, 
            recent_window_.predictions
        );
        
        result.psi_score = PopulationStabilityIndex(
            baseline_->prediction_distribution,
            recent_window_.predictions
        );
    }
    
    // Check performance drift
    result.performance_delta = CalculatePerformanceDrift();
    
    // Detect feature-level drift
    result.affected_features = DetectFeatureDrift();
    
    // Determine if drift detected
    bool ks_drift = result.ks_statistic > config_.ks_threshold;
    bool psi_drift = result.psi_score > config_.psi_threshold;
    bool perf_drift = std::abs(result.performance_delta) > 0.1;  // 10% performance drop
    
    if (ks_drift || psi_drift || perf_drift) {
        result.drift_detected = true;
        
        // Determine drift type
        if (perf_drift) {
            result.drift_type = "performance";
        } else if (ks_drift && psi_drift) {
            result.drift_type = "distribution";
        } else if (!result.affected_features.empty()) {
            result.drift_type = "feature";
        } else {
            result.drift_type = "unknown";
        }
        
        // Calculate confidence
        int drift_indicators = (ks_drift ? 1 : 0) + (psi_drift ? 1 : 0) + (perf_drift ? 1 : 0);
        result.confidence = drift_indicators / 3.0;
    }
    
    // Store result
    drift_history_.push_back(result);
    if (drift_history_.size() > 100) {
        drift_history_.erase(drift_history_.begin());
    }
    
    // Trigger alerts if needed
    if (result.drift_detected && config_.enable_alerts) {
        TriggerAlerts(result);
    }
    
    return result;
}

std::vector<DriftResult> ModelDriftDetector::GetDriftHistory(size_t last_n) const {
    size_t start = drift_history_.size() > last_n ? drift_history_.size() - last_n : 0;
    return std::vector<DriftResult>(drift_history_.begin() + start, drift_history_.end());
}

bool ModelDriftDetector::ShouldRetrain(const DriftResult& result,
                                      double performance_threshold) const {
    // Retrain if:
    // 1. Performance dropped significantly
    if (result.performance_delta < -performance_threshold) {
        return true;
    }
    
    // 2. Strong drift detected with high confidence
    if (result.drift_detected && result.confidence > 0.8) {
        return true;
    }
    
    // 3. Multiple consecutive drift detections
    if (drift_history_.size() >= 3) {
        int recent_drifts = 0;
        for (size_t i = drift_history_.size() - 3; i < drift_history_.size(); ++i) {
            if (drift_history_[i].drift_detected) {
                recent_drifts++;
            }
        }
        if (recent_drifts >= 2) {
            return true;
        }
    }
    
    return false;
}

double ModelDriftDetector::KolmogorovSmirnovTest(const std::vector<float>& dist1,
                                                const std::vector<float>& dist2) const {
    if (dist1.empty() || dist2.empty()) {
        return 0.0;
    }
    
    // Sort both distributions
    std::vector<float> sorted1 = dist1;
    std::vector<float> sorted2 = dist2;
    std::sort(sorted1.begin(), sorted1.end());
    std::sort(sorted2.begin(), sorted2.end());
    
    // Calculate empirical CDFs and find maximum difference
    double max_diff = 0.0;
    size_t i = 0, j = 0;
    
    while (i < sorted1.size() && j < sorted2.size()) {
        double cdf1 = static_cast<double>(i) / sorted1.size();
        double cdf2 = static_cast<double>(j) / sorted2.size();
        
        max_diff = std::max(max_diff, std::abs(cdf1 - cdf2));
        
        if (sorted1[i] < sorted2[j]) {
            i++;
        } else if (sorted1[i] > sorted2[j]) {
            j++;
        } else {
            i++;
            j++;
        }
    }
    
    // Check remaining elements
    while (i < sorted1.size()) {
        double cdf1 = static_cast<double>(i) / sorted1.size();
        double cdf2 = 1.0;
        max_diff = std::max(max_diff, std::abs(cdf1 - cdf2));
        i++;
    }
    
    while (j < sorted2.size()) {
        double cdf1 = 1.0;
        double cdf2 = static_cast<double>(j) / sorted2.size();
        max_diff = std::max(max_diff, std::abs(cdf1 - cdf2));
        j++;
    }
    
    return max_diff;
}

double ModelDriftDetector::PopulationStabilityIndex(const std::vector<float>& expected,
                                                  const std::vector<float>& actual) const {
    if (expected.empty() || actual.empty()) {
        return 0.0;
    }
    
    // Create bins
    const size_t num_bins = 10;
    float min_val = *std::min_element(expected.begin(), expected.end());
    float max_val = *std::max_element(expected.begin(), expected.end());
    
    // Extend range slightly to avoid edge cases
    float range = max_val - min_val;
    min_val -= range * 0.01f;
    max_val += range * 0.01f;
    
    float bin_width = (max_val - min_val) / num_bins;
    
    // Count elements in each bin
    std::vector<size_t> expected_counts(num_bins, 0);
    std::vector<size_t> actual_counts(num_bins, 0);
    
    for (float val : expected) {
        size_t bin = std::min(static_cast<size_t>((val - min_val) / bin_width), num_bins - 1);
        expected_counts[bin]++;
    }
    
    for (float val : actual) {
        size_t bin = std::min(static_cast<size_t>((val - min_val) / bin_width), num_bins - 1);
        actual_counts[bin]++;
    }
    
    // Calculate PSI
    double psi = 0.0;
    for (size_t i = 0; i < num_bins; ++i) {
        double expected_pct = static_cast<double>(expected_counts[i]) / expected.size();
        double actual_pct = static_cast<double>(actual_counts[i]) / actual.size();
        
        // Avoid log(0) by adding small epsilon
        expected_pct = std::max(expected_pct, 0.0001);
        actual_pct = std::max(actual_pct, 0.0001);
        
        psi += (actual_pct - expected_pct) * std::log(actual_pct / expected_pct);
    }
    
    return psi;
}

std::vector<std::string> ModelDriftDetector::DetectFeatureDrift() const {
    std::vector<std::string> drifted_features;
    
    if (baseline_->feature_distributions.empty() || recent_window_.features.empty()) {
        return drifted_features;
    }
    
    // Compare each feature dimension
    size_t num_features = baseline_->feature_distributions[0].size();
    
    for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        // Extract feature column from baseline
        std::vector<float> baseline_feature;
        for (const auto& sample : baseline_->feature_distributions) {
            if (feature_idx < sample.size()) {
                baseline_feature.push_back(sample[feature_idx]);
            }
        }
        
        // Extract feature column from recent
        std::vector<float> recent_feature;
        for (const auto& sample : recent_window_.features) {
            if (feature_idx < sample.size()) {
                recent_feature.push_back(sample[feature_idx]);
            }
        }
        
        // Test for drift
        double ks_stat = KolmogorovSmirnovTest(baseline_feature, recent_feature);
        if (ks_stat > config_.ks_threshold) {
            drifted_features.push_back("feature_" + std::to_string(feature_idx));
        }
    }
    
    return drifted_features;
}

double ModelDriftDetector::CalculatePerformanceDrift() const {
    if (baseline_->baseline_accuracy <= 0.0) {
        return 0.0;
    }
    
    return (recent_window_.current_accuracy - baseline_->baseline_accuracy) / 
           baseline_->baseline_accuracy;
}

void ModelDriftDetector::TriggerAlerts(const DriftResult& result) {
    for (const auto& callback : alert_callbacks_) {
        callback(result);
    }
}

// RecentWindow implementation
void ModelDriftDetector::RecentWindow::Add(const std::vector<float>& feature,
                                          float prediction,
                                          float actual) {
    features.push_back(feature);
    predictions.push_back(prediction);
    actuals.push_back(actual);
}

void ModelDriftDetector::RecentWindow::Trim(size_t max_size) {
    if (features.size() > max_size) {
        size_t to_remove = features.size() - max_size;
        features.erase(features.begin(), features.begin() + to_remove);
        predictions.erase(predictions.begin(), predictions.begin() + to_remove);
        actuals.erase(actuals.begin(), actuals.begin() + to_remove);
    }
}

// ADWINDriftDetector implementation
ADWINDriftDetector::ADWINDriftDetector(double delta)
    : delta_(delta), drift_detected_(false) {
}

void ADWINDriftDetector::Update(double error) {
    window_.Add(error);
    drift_detected_ = window_.CheckSplit(delta_);
}

void ADWINDriftDetector::Reset() {
    window_ = Window();
    drift_detected_ = false;
}

void ADWINDriftDetector::Window::Add(double value) {
    data.push_back(value);
    total += value;
    variance += value * value;
    width++;
}

bool ADWINDriftDetector::Window::CheckSplit(double delta) {
    if (data.size() < 2) {
        return false;
    }
    
    // ADWIN algorithm: check all possible split points
    for (size_t split_point = 1; split_point < data.size(); ++split_point) {
        // Calculate statistics for two sub-windows
        double n1 = split_point;
        double n2 = data.size() - split_point;
        
        double sum1 = 0, sum2 = 0;
        for (size_t i = 0; i < split_point; ++i) {
            sum1 += data[i];
        }
        for (size_t i = split_point; i < data.size(); ++i) {
            sum2 += data[i];
        }
        
        double mean1 = sum1 / n1;
        double mean2 = sum2 / n2;
        
        // Calculate epsilon_cut
        double m = 1.0 / (1.0 / n1 + 1.0 / n2);
        double epsilon_cut = std::sqrt(2.0 * std::log(2.0 / delta) / m);
        
        // Check if difference is significant
        if (std::abs(mean1 - mean2) > epsilon_cut) {
            // Remove old data
            data.erase(data.begin(), data.begin() + split_point);
            
            // Recalculate statistics
            total = sum2;
            variance = 0;
            for (double val : data) {
                variance += val * val;
            }
            width = data.size();
            
            return true;  // Drift detected
        }
    }
    
    return false;
}

// EnsembleDriftDetector implementation
EnsembleDriftDetector::EnsembleDriftDetector() {
}

void EnsembleDriftDetector::AddDetector(std::unique_ptr<ModelDriftDetector> detector) {
    statistical_detectors_.push_back(std::move(detector));
}

void EnsembleDriftDetector::AddADWINDetector(std::unique_ptr<ADWINDriftDetector> detector) {
    adwin_detectors_.push_back(std::move(detector));
}

DriftResult EnsembleDriftDetector::DetectDrift(const std::vector<std::vector<float>>& features,
                                              const std::vector<float>& predictions,
                                              const std::vector<float>& actuals) {
    recent_results_.clear();
    
    // Collect results from all detectors
    for (auto& detector : statistical_detectors_) {
        detector->UpdateDistribution(features, predictions, actuals);
        recent_results_.push_back(detector->CheckForDrift());
    }
    
    // Update ADWIN detectors with prediction errors
    for (size_t i = 0; i < std::min(predictions.size(), actuals.size()); ++i) {
        double error = std::abs(predictions[i] - actuals[i]);
        for (auto& adwin : adwin_detectors_) {
            adwin->Update(error);
        }
    }
    
    // Ensemble voting
    DriftResult ensemble_result;
    ensemble_result.detection_time = std::chrono::system_clock::now();
    
    int drift_votes = 0;
    double total_confidence = 0.0;
    
    for (const auto& result : recent_results_) {
        if (result.drift_detected) {
            drift_votes++;
            total_confidence += result.confidence;
        }
    }
    
    for (const auto& adwin : adwin_detectors_) {
        if (adwin->DriftDetected()) {
            drift_votes++;
            total_confidence += 1.0;  // ADWIN gives binary result
        }
    }
    
    int total_detectors = statistical_detectors_.size() + adwin_detectors_.size();
    if (total_detectors > 0) {
        double vote_ratio = static_cast<double>(drift_votes) / total_detectors;
        ensemble_result.drift_detected = vote_ratio >= 0.5;
        ensemble_result.confidence = total_confidence / total_detectors;
        
        // Aggregate other metrics
        if (!recent_results_.empty()) {
            ensemble_result.ks_statistic = recent_results_[0].ks_statistic;
            ensemble_result.psi_score = recent_results_[0].psi_score;
            ensemble_result.performance_delta = recent_results_[0].performance_delta;
            ensemble_result.affected_features = recent_results_[0].affected_features;
        }
        
        // Determine drift type based on majority
        std::unordered_map<std::string, int> type_votes;
        for (const auto& result : recent_results_) {
            if (result.drift_detected) {
                type_votes[result.drift_type]++;
            }
        }
        
        if (!type_votes.empty()) {
            auto max_type = std::max_element(type_votes.begin(), type_votes.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            ensemble_result.drift_type = max_type->first;
        }
    }
    
    return ensemble_result;
}

bool EnsembleDriftDetector::ConsensusReached(double threshold) const {
    if (recent_results_.empty()) {
        return false;
    }
    
    int drift_count = 0;
    for (const auto& result : recent_results_) {
        if (result.drift_detected) {
            drift_count++;
        }
    }
    
    double ratio = static_cast<double>(drift_count) / recent_results_.size();
    return ratio >= threshold;
}

}  // namespace mlops
}  // namespace predis