#include "inference_engine.h"
#include "models/model_interfaces.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>

namespace predis {
namespace ml {

InferenceEngine::InferenceEngine(const InferenceEngineConfig& config)
    : config_(config) {
    
    metrics_.start_time = std::chrono::system_clock::now();
    
    if (config_.use_gpu) {
        initializeGPU();
    }
    
    batch_processor_ = std::make_unique<BatchProcessor>(
        config_.batch_size, config_.use_gpu);
}

InferenceEngine::~InferenceEngine() {
    stop();
    if (config_.use_gpu) {
        releaseGPU();
    }
}

void InferenceEngine::setModel(std::unique_ptr<BaseModel> model) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    model_ = std::move(model);
    
    if (config_.use_gpu && model_) {
        warmupGPU();
    }
}

void InferenceEngine::updateModel(std::unique_ptr<BaseModel> new_model) {
    // Atomic model update to avoid inference disruption
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto old_model = std::move(model_);
    model_ = std::move(new_model);
    
    std::cout << "Model updated successfully\n";
}

void InferenceEngine::start() {
    if (running_) return;
    
    running_ = true;
    
    // Start worker threads
    for (int i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&InferenceEngine::workerLoop, this);
    }
    
    std::cout << "Inference engine started with " << config_.num_worker_threads 
              << " worker threads\n";
}

void InferenceEngine::stop() {
    if (!running_) return;
    
    running_ = false;
    queue_cv_.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "Inference engine stopped\n";
}

InferenceResult InferenceEngine::predict(const std::vector<std::vector<float>>& features,
                                        int priority) {
    auto request_id = submitRequest(features, priority);
    
    // Wait for result
    InferenceResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        if (getResult(request_id, result)) {
            return result;
        }
        
        // Check timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        
        if (elapsed > 30000) { // 30 second timeout
            result.success = false;
            result.error_message = "Inference timeout";
            return result;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::string InferenceEngine::submitRequest(const std::vector<std::vector<float>>& features,
                                          int priority) {
    InferenceRequest request;
    request.request_id = generateRequestId();
    request.features = features;
    request.timestamp = std::chrono::high_resolution_clock::now();
    request.priority = priority;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (request_queue_.size() >= config_.max_queue_size) {
            // Queue is full, reject request
            InferenceResult result;
            result.request_id = request.request_id;
            result.success = false;
            result.error_message = "Queue full";
            
            std::lock_guard<std::mutex> cache_lock(cache_mutex_);
            result_cache_[request.request_id] = result;
            
            return request.request_id;
        }
        
        request_queue_.push(request);
    }
    
    queue_cv_.notify_one();
    return request.request_id;
}

bool InferenceEngine::getResult(const std::string& request_id, InferenceResult& result) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = result_cache_.find(request_id);
    if (it != result_cache_.end()) {
        result = it->second;
        result_cache_.erase(it);
        return true;
    }
    
    return false;
}

std::vector<InferenceResult> InferenceEngine::predictBatch(
    const std::vector<std::vector<std::vector<float>>>& feature_batches) {
    
    std::vector<InferenceResult> results;
    std::vector<std::string> request_ids;
    
    // Submit all requests
    for (const auto& features : feature_batches) {
        auto id = submitRequest(features, 1); // Higher priority for batch
        request_ids.push_back(id);
    }
    
    // Collect results
    for (const auto& id : request_ids) {
        InferenceResult result;
        
        // Wait for result with timeout
        auto start_time = std::chrono::high_resolution_clock::now();
        while (!getResult(id, result)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            
            if (elapsed > 10000) { // 10 second timeout for batch
                result.request_id = id;
                result.success = false;
                result.error_message = "Batch inference timeout";
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        results.push_back(result);
    }
    
    return results;
}

void InferenceEngine::workerLoop() {
    while (running_) {
        auto batch = collectBatch();
        
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
}

std::vector<InferenceRequest> InferenceEngine::collectBatch() {
    std::vector<InferenceRequest> batch;
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for requests or timeout
    auto timeout = std::chrono::milliseconds(static_cast<int>(config_.batch_timeout_ms));
    queue_cv_.wait_for(lock, timeout, [this] {
        return !request_queue_.empty() || !running_;
    });
    
    if (!running_) return batch;
    
    // Collect up to batch_size requests
    while (!request_queue_.empty() && batch.size() < config_.batch_size) {
        batch.push_back(request_queue_.top());
        request_queue_.pop();
    }
    
    return batch;
}

void InferenceEngine::processBatch(const std::vector<InferenceRequest>& batch) {
    if (!model_) {
        // No model loaded, return error for all requests
        for (const auto& request : batch) {
            InferenceResult result;
            result.request_id = request.request_id;
            result.success = false;
            result.error_message = "No model loaded";
            
            std::lock_guard<std::mutex> lock(cache_mutex_);
            result_cache_[request.request_id] = result;
        }
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Combine all features for batch processing
        std::vector<std::vector<float>> all_features;
        for (const auto& request : batch) {
            all_features.insert(all_features.end(), 
                              request.features.begin(), 
                              request.features.end());
        }
        
        // Optimize for GPU if enabled
        if (config_.use_gpu) {
            optimizeBatchForGPU(all_features);
        }
        
        // Run inference
        auto predictions = model_->predict(all_features);
        auto confidence_scores = model_->getConfidenceScores(all_features);
        
        // Distribute results back to requests
        size_t offset = 0;
        for (const auto& request : batch) {
            InferenceResult result;
            result.request_id = request.request_id;
            result.success = true;
            
            // Extract predictions for this request
            size_t num_samples = request.features.size();
            result.predictions.assign(
                predictions.begin() + offset,
                predictions.begin() + offset + num_samples);
            result.confidence_scores.assign(
                confidence_scores.begin() + offset,
                confidence_scores.begin() + offset + num_samples);
            
            auto request_time = std::chrono::duration_cast<std::chrono::microseconds>(
                start_time - request.timestamp).count() / 1000.0f;
            result.inference_time_ms = request_time;
            
            offset += num_samples;
            
            // Store result
            std::lock_guard<std::mutex> lock(cache_mutex_);
            result_cache_[request.request_id] = result;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        float batch_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0f;
        
        updateMetrics(batch, batch_time, true);
        
    } catch (const std::exception& e) {
        // Handle errors
        for (const auto& request : batch) {
            InferenceResult result;
            result.request_id = request.request_id;
            result.success = false;
            result.error_message = e.what();
            
            std::lock_guard<std::mutex> lock(cache_mutex_);
            result_cache_[request.request_id] = result;
        }
        
        updateMetrics(batch, 0.0f, false);
    }
}

void InferenceEngine::updateMetrics(const std::vector<InferenceRequest>& batch,
                                   float inference_time_ms,
                                   bool success) {
    metrics_.total_requests += batch.size();
    
    if (success) {
        metrics_.successful_requests += batch.size();
        metrics_.total_inference_time_ms += inference_time_ms;
        
        // Update average batch size
        float current_avg = metrics_.avg_batch_size.load();
        float new_avg = (current_avg * (metrics_.total_requests - batch.size()) + 
                        batch.size()) / metrics_.total_requests;
        metrics_.avg_batch_size = new_avg;
    } else {
        metrics_.failed_requests += batch.size();
    }
}

std::string InferenceEngine::generateRequestId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);
    
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::stringstream ss;
    ss << "req_" << now << "_" << dis(gen);
    return ss.str();
}

void InferenceEngine::optimizeBatchForGPU(std::vector<std::vector<float>>& features) {
    // Ensure features are contiguous in memory for GPU transfer
    // This is a placeholder - actual implementation would involve
    // proper memory layout optimization
}

bool InferenceEngine::initializeGPU() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        config_.use_gpu = false;
        return false;
    }
    
    if (config_.gpu_device_id >= device_count) {
        std::cerr << "Invalid GPU device ID\n";
        config_.gpu_device_id = 0;
    }
    
    cudaSetDevice(config_.gpu_device_id);
    
    // Allocate GPU workspace
    gpu_workspace_size_ = 100 * 1024 * 1024; // 100MB workspace
    cudaMalloc(&gpu_workspace_, gpu_workspace_size_);
    
    std::cout << "GPU initialized successfully on device " << config_.gpu_device_id << "\n";
    return true;
}

void InferenceEngine::releaseGPU() {
    if (gpu_workspace_) {
        cudaFree(gpu_workspace_);
        gpu_workspace_ = nullptr;
    }
}

float InferenceEngine::getGPUMemoryUsage() const {
    if (!config_.use_gpu) return 0.0f;
    
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    return (total_memory - free_memory) / (1024.0f * 1024.0f); // Return in MB
}

void InferenceEngine::warmupGPU() {
    if (!config_.use_gpu || !model_) return;
    
    std::cout << "Warming up GPU...\n";
    
    // Run a few dummy predictions to warm up the GPU
    std::vector<std::vector<float>> dummy_features(10, std::vector<float>(100, 0.0f));
    for (int i = 0; i < 5; ++i) {
        model_->predict(dummy_features);
    }
    
    std::cout << "GPU warmup complete\n";
}

void InferenceEngine::resetMetrics() {
    metrics_.total_requests = 0;
    metrics_.successful_requests = 0;
    metrics_.failed_requests = 0;
    metrics_.total_inference_time_ms = 0.0f;
    metrics_.avg_batch_size = 0.0f;
    metrics_.gpu_utilization = 0.0f;
    metrics_.start_time = std::chrono::system_clock::now();
}

void InferenceEngine::updateConfig(const InferenceEngineConfig& config) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    bool gpu_changed = config.use_gpu != config_.use_gpu ||
                      config.gpu_device_id != config_.gpu_device_id;
    
    config_ = config;
    
    if (gpu_changed) {
        if (config_.use_gpu) {
            initializeGPU();
        } else {
            releaseGPU();
        }
    }
    
    // Update batch processor
    batch_processor_ = std::make_unique<BatchProcessor>(
        config_.batch_size, config_.use_gpu);
}

// BatchProcessor implementation
BatchProcessor::BatchProcessor(int max_batch_size, bool use_gpu)
    : max_batch_size_(max_batch_size), use_gpu_(use_gpu) {}

BatchProcessor::~BatchProcessor() {
    if (use_gpu_) {
        freeGPUMemory();
    }
}

std::vector<std::vector<float>> BatchProcessor::processBatch(
    BaseModel* model,
    const std::vector<std::vector<std::vector<float>>>& feature_batches) {
    
    std::vector<std::vector<float>> all_predictions;
    
    for (const auto& batch : feature_batches) {
        auto predictions = model->predict(batch);
        all_predictions.push_back(predictions);
    }
    
    return all_predictions;
}

void BatchProcessor::allocateGPUMemory(size_t feature_dim, size_t max_sequence_length) {
    if (!use_gpu_) return;
    
    size_t feature_size = max_batch_size_ * max_sequence_length * feature_dim * sizeof(float);
    size_t prediction_size = max_batch_size_ * sizeof(float);
    
    cudaMalloc(&d_features_, feature_size);
    cudaMalloc(&d_predictions_, prediction_size);
    
    allocated_size_ = feature_size + prediction_size;
}

void BatchProcessor::freeGPUMemory() {
    if (d_features_) {
        cudaFree(d_features_);
        d_features_ = nullptr;
    }
    if (d_predictions_) {
        cudaFree(d_predictions_);
        d_predictions_ = nullptr;
    }
    allocated_size_ = 0;
}

void BatchProcessor::padFeatures(const std::vector<std::vector<float>>& features,
                                float* output,
                                size_t padded_size) {
    // Pad features to uniform size for GPU processing
    for (size_t i = 0; i < features.size(); ++i) {
        size_t copy_size = std::min(features[i].size(), padded_size);
        std::copy(features[i].begin(), features[i].begin() + copy_size,
                 output + i * padded_size);
        
        // Zero pad the rest
        std::fill(output + i * padded_size + copy_size,
                 output + (i + 1) * padded_size,
                 0.0f);
    }
}

void BatchProcessor::unpadPredictions(const float* predictions,
                                    std::vector<float>& output,
                                    size_t original_size) {
    output.resize(original_size);
    std::copy(predictions, predictions + original_size, output.begin());
}

// Factory implementations
std::unique_ptr<InferenceEngine> InferenceEngineFactory::createDefaultEngine() {
    InferenceEngineConfig config;
    config.batch_size = 64;
    config.num_worker_threads = 2;
    config.batch_timeout_ms = 5.0f;
    config.use_gpu = true;
    
    return std::make_unique<InferenceEngine>(config);
}

std::unique_ptr<InferenceEngine> InferenceEngineFactory::createHighThroughputEngine() {
    InferenceEngineConfig config;
    config.batch_size = 256;
    config.num_worker_threads = 4;
    config.batch_timeout_ms = 10.0f;
    config.max_queue_size = 5000;
    config.use_gpu = true;
    
    return std::make_unique<InferenceEngine>(config);
}

std::unique_ptr<InferenceEngine> InferenceEngineFactory::createLowLatencyEngine() {
    InferenceEngineConfig config;
    config.batch_size = 1;
    config.num_worker_threads = 8;
    config.batch_timeout_ms = 0.5f;
    config.max_queue_size = 100;
    config.use_gpu = true;
    
    return std::make_unique<InferenceEngine>(config);
}

std::unique_ptr<InferenceEngine> InferenceEngineFactory::createBatchOptimizedEngine() {
    InferenceEngineConfig config;
    config.batch_size = 512;
    config.num_worker_threads = 2;
    config.batch_timeout_ms = 20.0f;
    config.max_queue_size = 10000;
    config.use_gpu = true;
    
    return std::make_unique<InferenceEngine>(config);
}

} // namespace ml
} // namespace predis