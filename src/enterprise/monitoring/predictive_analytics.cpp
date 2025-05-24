#include <vector>
#include <cmath>
#include <numeric>

namespace predis {
namespace enterprise {

class PredictiveAnalytics {
public:
    struct CapacityPrediction {
        std::string resource_type;  // "gpu_memory", "cache_capacity", "bandwidth"
        double current_usage;
        double predicted_usage_7d;
        double predicted_usage_30d;
        double capacity_limit;
        int days_until_exhaustion;
        std::string recommendation;
    };
    
    struct PerformanceTrend {
        std::string metric_name;
        double current_value;
        double trend_direction;  // -1 to 1, negative = degrading
        double trend_strength;   // 0 to 1, confidence in trend
        std::string trend_description;
    };
    
    PredictiveAnalytics() {}
    
    std::vector<CapacityPrediction> PredictCapacity(
        const std::vector<std::pair<std::string, std::vector<double>>>& usage_history) {
        
        std::vector<CapacityPrediction> predictions;
        
        for (const auto& [resource, history] : usage_history) {
            if (history.size() < 7) continue;
            
            CapacityPrediction pred;
            pred.resource_type = resource;
            pred.current_usage = history.back();
            
            // Calculate growth rate (simple linear regression)
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
            int n = history.size();
            
            for (int i = 0; i < n; ++i) {
                sum_x += i;
                sum_y += history[i];
                sum_xy += i * history[i];
                sum_xx += i * i;
            }
            
            double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
            double intercept = (sum_y - slope * sum_x) / n;
            
            // Project future usage
            pred.predicted_usage_7d = intercept + slope * (n + 7);
            pred.predicted_usage_30d = intercept + slope * (n + 30);
            
            // Set capacity limits based on resource type
            if (resource == "gpu_memory") {
                pred.capacity_limit = 16.0 * 1024 * 1024 * 1024;  // 16GB
            } else if (resource == "cache_capacity") {
                pred.capacity_limit = 8.0 * 1024 * 1024 * 1024;   // 8GB
            } else {
                pred.capacity_limit = 100.0;  // Percentage based
            }
            
            // Calculate days until exhaustion
            if (slope > 0 && pred.current_usage < pred.capacity_limit) {
                pred.days_until_exhaustion = static_cast<int>(
                    (pred.capacity_limit - pred.current_usage) / slope);
            } else {
                pred.days_until_exhaustion = -1;  // Not exhausting
            }
            
            // Generate recommendation
            if (pred.days_until_exhaustion > 0 && pred.days_until_exhaustion < 30) {
                pred.recommendation = "Critical: Capacity will be exhausted in " + 
                    std::to_string(pred.days_until_exhaustion) + " days. Plan scaling now.";
            } else if (pred.predicted_usage_30d > pred.capacity_limit * 0.8) {
                pred.recommendation = "Warning: Approaching 80% capacity within 30 days.";
            } else {
                pred.recommendation = "Capacity is healthy.";
            }
            
            predictions.push_back(pred);
        }
        
        return predictions;
    }
    
    std::vector<PerformanceTrend> AnalyzePerformanceTrends(
        const std::unordered_map<std::string, std::vector<double>>& metrics_history) {
        
        std::vector<PerformanceTrend> trends;
        
        for (const auto& [metric, history] : metrics_history) {
            if (history.size() < 10) continue;
            
            PerformanceTrend trend;
            trend.metric_name = metric;
            trend.current_value = history.back();
            
            // Calculate moving averages
            int window = std::min(20, static_cast<int>(history.size()));
            double recent_avg = 0, older_avg = 0;
            
            for (int i = history.size() - window; i < history.size(); ++i) {
                recent_avg += history[i];
            }
            recent_avg /= window;
            
            for (int i = history.size() - 2*window; i < history.size() - window; ++i) {
                if (i >= 0) older_avg += history[i];
            }
            older_avg /= window;
            
            // Calculate trend
            if (older_avg != 0) {
                trend.trend_direction = (recent_avg - older_avg) / older_avg;
            } else {
                trend.trend_direction = 0;
            }
            
            // Calculate trend strength (based on consistency)
            double variance = 0;
            for (int i = history.size() - window; i < history.size(); ++i) {
                variance += std::pow(history[i] - recent_avg, 2);
            }
            variance /= window;
            
            trend.trend_strength = 1.0 / (1.0 + std::sqrt(variance) / recent_avg);
            
            // Generate description
            if (std::abs(trend.trend_direction) < 0.05) {
                trend.trend_description = "Stable";
            } else if (trend.trend_direction > 0) {
                if (metric.find("error") != std::string::npos || 
                    metric.find("latency") != std::string::npos) {
                    trend.trend_description = "Degrading (+" + 
                        std::to_string(int(trend.trend_direction * 100)) + "%)";
                } else {
                    trend.trend_description = "Improving (+" + 
                        std::to_string(int(trend.trend_direction * 100)) + "%)";
                }
            } else {
                if (metric.find("error") != std::string::npos || 
                    metric.find("latency") != std::string::npos) {
                    trend.trend_description = "Improving (" + 
                        std::to_string(int(trend.trend_direction * 100)) + "%)";
                } else {
                    trend.trend_description = "Degrading (" + 
                        std::to_string(int(trend.trend_direction * 100)) + "%)";
                }
            }
            
            trends.push_back(trend);
        }
        
        return trends;
    }
    
    double PredictPeakLoad(const std::vector<double>& hourly_load_history) {
        if (hourly_load_history.size() < 168) {  // Less than a week
            return hourly_load_history.empty() ? 0 : 
                *std::max_element(hourly_load_history.begin(), hourly_load_history.end());
        }
        
        // Find weekly pattern
        std::vector<double> hourly_averages(24, 0);
        std::vector<int> hourly_counts(24, 0);
        
        for (size_t i = 0; i < hourly_load_history.size(); ++i) {
            int hour = i % 24;
            hourly_averages[hour] += hourly_load_history[i];
            hourly_counts[hour]++;
        }
        
        for (int h = 0; h < 24; ++h) {
            if (hourly_counts[h] > 0) {
                hourly_averages[h] /= hourly_counts[h];
            }
        }
        
        // Find peak hour and add safety margin
        double peak = *std::max_element(hourly_averages.begin(), hourly_averages.end());
        return peak * 1.2;  // 20% safety margin
    }
};

}  // namespace enterprise
}  // namespace predis