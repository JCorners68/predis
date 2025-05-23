#ifndef PREDIS_ML_INFERENCE_ENGINE_H_
#define PREDIS_ML_INFERENCE_ENGINE_H_

#include "models/model_interfaces.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>

namespace predis {
namespace ml {

// Forward declarations
class BaseModel;
class BatchProcessor;

// Inference request structure
struct InferenceRequest {
    std::string request_id;
    std::vector<std::vector<float>> features;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    int priority = 0;  // Higher priority = process first
    
    bool operator<(const InferenceRequest& other) const {
        return priority < other.priority;  // For priority queue
    }
};

// Inference result structure
struct InferenceResult {
    std::string request_id;
    std::vector<float> predictions;
    std::vector<float> confidence_scores;
    float inference_time_ms;
    bool success = true;
    std::string error_message;
};

// Inference engine configuration
struct InferenceEngineConfig {
    int batch_size = 64;              // Max batch size for GPU processing
    int max_queue_size = 1000;        // Max pending requests
    int num_worker_threads = 2;       // Number of inference threads
    float batch_timeout_ms = 5.0f;    // Max wait time before processing partial batch
    bool use_gpu = true;              // Enable GPU acceleration
    bool enable_profiling = true;     // Track performance metrics
    int gpu_device_id = 0;            // GPU device to use
};

// Performance metrics for the inference engine
struct InferenceMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<float> total_inference_time_ms{0.0f};
    std::atomic<float> avg_batch_size{0.0f};
    std::atomic<float> gpu_utilization{0.0f};
    std::chrono::time_point<std::chrono::system_clock> start_time;
    
    float getAverageLatency() const {
        return total_requests > 0 ? total_inference_time_ms / total_requests : 0.0f;
    }
    
    float getThroughput() const {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - start_time).count();
        return duration > 0 ? static_cast<float>(total_requests) / duration : 0.0f;
    }
};

// GPU-optimized inference engine for ML models
class InferenceEngine {
public:
    explicit InferenceEngine(const InferenceEngineConfig& config);
    ~InferenceEngine();
    
    // Model management
    void setModel(std::unique_ptr<BaseModel> model);
    void updateModel(std::unique_ptr<BaseModel> new_model);
    BaseModel* getModel() const { return model_.get(); }
    
    // Synchronous inference (blocks until complete)
    InferenceResult predict(const std::vector<std::vector<float>>& features,
                           int priority = 0);
    
    // Asynchronous inference (returns immediately)
    std::string submitRequest(const std::vector<std::vector<float>>& features,
                             int priority = 0);
    bool getResult(const std::string& request_id, InferenceResult& result);
    
    // Batch inference
    std::vector<InferenceResult> predictBatch(
        const std::vector<std::vector<std::vector<float>>>& feature_batches);
    
    // Engine control
    void start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Performance monitoring
    InferenceMetrics getMetrics() const { return metrics_; }
    void resetMetrics();
    
    // Configuration
    void updateConfig(const InferenceEngineConfig& config);
    InferenceEngineConfig getConfig() const { return config_; }
    
    // GPU management
    bool initializeGPU();
    void releaseGPU();
    float getGPUMemoryUsage() const;
    
private:
    InferenceEngineConfig config_;
    std::unique_ptr<BaseModel> model_;
    std::atomic<bool> running_{false};
    
    // Request queue and processing
    std::priority_queue<InferenceRequest> request_queue_;
    std::unordered_map<std::string, InferenceResult> result_cache_;
    std::mutex queue_mutex_;
    std::mutex cache_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    
    // Performance tracking
    InferenceMetrics metrics_;
    
    // GPU resources
    void* gpu_workspace_ = nullptr;
    size_t gpu_workspace_size_ = 0;
    std::unique_ptr<BatchProcessor> batch_processor_;
    
    // Internal methods
    void workerLoop();
    std::vector<InferenceRequest> collectBatch();
    void processBatch(const std::vector<InferenceRequest>& batch);
    std::string generateRequestId();
    void updateMetrics(const std::vector<InferenceRequest>& batch,
                      float inference_time_ms,
                      bool success);
    
    // GPU optimization
    void optimizeBatchForGPU(std::vector<std::vector<float>>& features);
    void warmupGPU();
};

// Batch processor for efficient GPU utilization
class BatchProcessor {
public:
    BatchProcessor(int max_batch_size, bool use_gpu);
    ~BatchProcessor();
    
    // Process a batch of features
    std::vector<std::vector<float>> processBatch(
        BaseModel* model,
        const std::vector<std::vector<std::vector<float>>>& feature_batches);
    
    // Memory management
    void allocateGPUMemory(size_t feature_dim, size_t max_sequence_length);
    void freeGPUMemory();
    
private:
    int max_batch_size_;
    bool use_gpu_;
    
    // GPU buffers
    float* d_features_ = nullptr;
    float* d_predictions_ = nullptr;
    size_t allocated_size_ = 0;
    
    // Padding and batching helpers
    void padFeatures(const std::vector<std::vector<float>>& features,
                    float* output,
                    size_t padded_size);
    void unpadPredictions(const float* predictions,
                         std::vector<float>& output,
                         size_t original_size);
};

// Factory for creating optimized inference engines
class InferenceEngineFactory {
public:
    static std::unique_ptr<InferenceEngine> createDefaultEngine();
    static std::unique_ptr<InferenceEngine> createHighThroughputEngine();
    static std::unique_ptr<InferenceEngine> createLowLatencyEngine();
    static std::unique_ptr<InferenceEngine> createBatchOptimizedEngine();
};

} // namespace ml
} // namespace predis

#endif // PREDIS_ML_INFERENCE_ENGINE_H_