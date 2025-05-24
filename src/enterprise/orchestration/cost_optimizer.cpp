#include <string>
#include <vector>
#include <algorithm>

namespace predis {
namespace enterprise {

class CostOptimizer {
public:
    struct ResourceUsage {
        std::string resource_id;
        std::string resource_type;  // "compute", "storage", "network"
        double usage_hours;
        double cost_per_hour;
        double utilization_percentage;
    };
    
    struct Recommendation {
        std::string action;  // "resize", "terminate", "reserved_instance"
        std::string resource_id;
        double potential_savings;
        std::string reasoning;
    };
    
    CostOptimizer() {}
    
    void AddResourceUsage(const ResourceUsage& usage) {
        resources_.push_back(usage);
    }
    
    std::vector<Recommendation> GenerateRecommendations() {
        std::vector<Recommendation> recommendations;
        
        for (const auto& resource : resources_) {
            // Low utilization
            if (resource.utilization_percentage < 20.0) {
                recommendations.push_back({
                    "terminate",
                    resource.resource_id,
                    resource.usage_hours * resource.cost_per_hour,
                    "Utilization below 20%"
                });
            }
            // Medium utilization - downsize
            else if (resource.utilization_percentage < 50.0) {
                recommendations.push_back({
                    "resize",
                    resource.resource_id,
                    resource.usage_hours * resource.cost_per_hour * 0.3,
                    "Utilization below 50% - consider smaller instance"
                });
            }
            // High usage - consider reserved
            else if (resource.usage_hours > 500) {
                recommendations.push_back({
                    "reserved_instance",
                    resource.resource_id,
                    resource.usage_hours * resource.cost_per_hour * 0.4,
                    "High usage - reserved instance would save 40%"
                });
            }
        }
        
        // Sort by savings potential
        std::sort(recommendations.begin(), recommendations.end(),
            [](const auto& a, const auto& b) {
                return a.potential_savings > b.potential_savings;
            });
        
        return recommendations;
    }
    
    double CalculateTotalCost() const {
        double total = 0.0;
        for (const auto& resource : resources_) {
            total += resource.usage_hours * resource.cost_per_hour;
        }
        return total;
    }
    
private:
    std::vector<ResourceUsage> resources_;
};

}  // namespace enterprise
}  // namespace predis