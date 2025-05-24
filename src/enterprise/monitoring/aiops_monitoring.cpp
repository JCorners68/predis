#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

namespace predis {
namespace enterprise {

class AIOpsMonitoring {
public:
    struct Anomaly {
        std::string metric_name;
        double observed_value;
        double expected_value;
        double anomaly_score;  // 0-1, higher = more anomalous
        std::string timestamp;
        std::string probable_cause;
    };
    
    struct Prediction {
        std::string metric_name;
        std::vector<double> predicted_values;
        std::vector<std::string> timestamps;
        double confidence;
    };
    
    AIOpsMonitoring() {}
    
    void RecordMetric(const std::string& name, double value) {
        metric_history_[name].push_back(value);
        
        // Keep only recent history
        if (metric_history_[name].size() > 1000) {
            metric_history_[name].pop_front();
        }
    }
    
    std::vector<Anomaly> DetectAnomalies() {
        std::vector<Anomaly> anomalies;
        
        for (const auto& [metric_name, values] : metric_history_) {
            if (values.size() < 100) continue;
            
            // Calculate statistics
            double mean = 0.0;
            for (double v : values) mean += v;
            mean /= values.size();
            
            double stddev = 0.0;
            for (double v : values) {
                stddev += (v - mean) * (v - mean);
            }
            stddev = std::sqrt(stddev / values.size());
            
            // Check recent values for anomalies
            double recent_value = values.back();
            double z_score = std::abs((recent_value - mean) / stddev);
            
            if (z_score > 3.0) {  // 3 sigma rule
                Anomaly anomaly;
                anomaly.metric_name = metric_name;
                anomaly.observed_value = recent_value;
                anomaly.expected_value = mean;
                anomaly.anomaly_score = std::min(z_score / 10.0, 1.0);
                anomaly.timestamp = "now";  // Would use real timestamp
                
                // Determine probable cause
                if (metric_name.find("latency") != std::string::npos && recent_value > mean) {
                    anomaly.probable_cause = "Performance degradation detected";
                } else if (metric_name.find("error") != std::string::npos && recent_value > mean) {
                    anomaly.probable_cause = "Error rate spike detected";
                } else if (metric_name.find("memory") != std::string::npos && recent_value > mean * 1.5) {
                    anomaly.probable_cause = "Memory leak suspected";
                } else {
                    anomaly.probable_cause = "Unusual pattern detected";
                }
                
                anomalies.push_back(anomaly);
            }
        }
        
        return anomalies;
    }
    
    std::vector<Prediction> PredictMetrics(int hours_ahead) {
        std::vector<Prediction> predictions;
        
        for (const auto& [metric_name, values] : metric_history_) {
            if (values.size() < 24) continue;  // Need at least 24 hours
            
            Prediction pred;
            pred.metric_name = metric_name;
            pred.confidence = 0.8;  // Stub confidence
            
            // Simple linear extrapolation (stub)
            double last_value = values.back();
            double trend = 0.0;
            if (values.size() > 1) {
                trend = (values.back() - values[values.size() - 2]) / values.back();
            }
            
            for (int h = 1; h <= hours_ahead; ++h) {
                double predicted = last_value * (1 + trend * h);
                pred.predicted_values.push_back(predicted);
                pred.timestamps.push_back("+" + std::to_string(h) + "h");
            }
            
            predictions.push_back(pred);
        }
        
        return predictions;
    }
    
    std::vector<std::string> GetRecommendations(const std::vector<Anomaly>& anomalies) {
        std::vector<std::string> recommendations;
        
        for (const auto& anomaly : anomalies) {
            if (anomaly.metric_name.find("gpu_utilization") != std::string::npos && 
                anomaly.observed_value > 90) {
                recommendations.push_back("Consider scaling GPU resources - utilization above 90%");
            }
            else if (anomaly.metric_name.find("cache_hit_rate") != std::string::npos && 
                     anomaly.observed_value < anomaly.expected_value * 0.8) {
                recommendations.push_back("Cache hit rate dropped 20% - check ML model performance");
            }
            else if (anomaly.metric_name.find("latency") != std::string::npos) {
                recommendations.push_back("Latency spike detected - check for resource contention");
            }
        }
        
        return recommendations;
    }
    
private:
    std::unordered_map<std::string, std::deque<double>> metric_history_;
};

}  // namespace enterprise
}  // namespace predis