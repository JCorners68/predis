#pragma once

#include <string>

namespace predis {
namespace mlops {

class CustomerModelManager {
public:
    explicit CustomerModelManager(const std::string& models_path);
    bool SaveCustomerModel(const std::string& customer_id, const std::string& model_data);
    std::string LoadCustomerModel(const std::string& customer_id);
};

}  // namespace mlops
}  // namespace predis