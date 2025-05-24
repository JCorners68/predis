#pragma once

#include <string>
#include <vector>

namespace predis {
namespace mlops {

class FederatedLearningClient {
public:
    explicit FederatedLearningClient(const std::string& client_id);
    std::vector<float> ComputeLocalGradients(const std::vector<std::vector<float>>& features,
                                            const std::vector<float>& labels);
    void AddPrivacyNoise(std::vector<float>& gradients, float epsilon = 1.0f);
    std::vector<float> FederatedAverage(const std::vector<std::vector<float>>& client_gradients);
};

}  // namespace mlops
}  // namespace predis