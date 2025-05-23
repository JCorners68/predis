#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "core/gpu_hash_table.h"
#include "benchmarks/write_performance_profiler.h"

using namespace predis;

// Test configuration
constexpr size_t NUM_WRITE_OPS = 100000;
constexpr size_t KEY_SIZE = 32;
constexpr size_t VALUE_SIZES[] = {64, 256, 1024, 4096};
constexpr size_t NUM_THREADS = 256;
constexpr size_t NUM_BLOCKS = 128;

// Atomic conflict counter
__device__ unsigned int g_atomic_conflicts = 0;
__device__ unsigned int g_memory_stalls = 0;

// Original write kernel (baseline)
__global__ void baseline_write_kernel(GPUHashTable* hash_table,
                                    const uint8_t* keys,
                                    const uint8_t* values,
                                    size_t num_ops,
                                    size_t key_size,
                                    size_t value_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops) return;
    
    const uint8_t* key = keys + idx * key_size;
    const uint8_t* value = values + idx * value_size;
    
    hash_table->insert(key, key_size, value, value_size);
}

// Optimized write kernel with reduced conflicts
__global__ void optimized_write_kernel(GPUHashTable* hash_table,
                                     const uint8_t* keys,
                                     const uint8_t* values,
                                     size_t num_ops,
                                     size_t key_size,
                                     size_t value_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops) return;
    
    // Shuffle threads to reduce bank conflicts
    int shuffled_idx = (idx * 17) % num_ops;  // Prime number shuffle
    
    const uint8_t* key = keys + shuffled_idx * key_size;
    const uint8_t* value = values + shuffled_idx * value_size;
    
    // Use shared memory for coalesced access
    extern __shared__ uint8_t shared_buffer[];
    
    // Copy key to shared memory
    for (int i = threadIdx.x; i < key_size; i += blockDim.x) {
        shared_buffer[i] = key[i];
    }
    __syncthreads();
    
    // Insert with reduced conflicts
    hash_table->insert(shared_buffer, key_size, value, value_size);
}

// Kernel to analyze atomic operation conflicts
__global__ void analyze_atomic_conflicts(GPUHashTable* hash_table,
                                       const uint8_t* keys,
                                       size_t num_ops,
                                       size_t key_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops) return;
    
    const uint8_t* key = keys + idx * key_size;
    
    // Compute hash and analyze conflicts
    uint32_t hash = hash_table->compute_hash(key, key_size);
    uint32_t bucket_idx = hash % hash_table->get_num_buckets();
    
    // Try to acquire bucket lock (simulated)
    unsigned int conflicts = 0;
    while (atomicCAS(&hash_table->bucket_locks[bucket_idx], 0, 1) != 0) {
        conflicts++;
        __threadfence();
    }
    
    // Record conflicts
    atomicAdd(&g_atomic_conflicts, conflicts);
    
    // Release lock
    atomicExch(&hash_table->bucket_locks[bucket_idx], 0);
}

// Memory access pattern analysis kernel
__global__ void analyze_memory_patterns(const uint8_t* data,
                                      size_t num_accesses,
                                      size_t stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_accesses) return;
    
    // Measure memory access patterns
    unsigned long long start_cycle = clock64();
    
    // Perform strided memory access
    volatile uint8_t value = data[idx * stride];
    
    unsigned long long end_cycle = clock64();
    unsigned long long cycles = end_cycle - start_cycle;
    
    // High cycle count indicates memory stall
    if (cycles > 500) {  // Threshold for memory stall detection
        atomicAdd(&g_memory_stalls, 1);
    }
}

class WriteBottleneckAnalyzer {
private:
    WritePerformanceProfiler profiler_;
    GPUHashTable* gpu_hash_table_;
    
    // Test data
    std::vector<uint8_t> h_keys_;
    std::vector<uint8_t> h_values_;
    uint8_t* d_keys_ = nullptr;
    uint8_t* d_values_ = nullptr;
    
public:
    WriteBottleneckAnalyzer() {
        // Initialize GPU hash table
        gpu_hash_table_ = new GPUHashTable(1000000);  // 1M buckets
        
        // Generate test data
        generateTestData();
        
        // Allocate device memory
        cudaMalloc(&d_keys_, h_keys_.size());
        cudaMalloc(&d_values_, h_values_.size());
        
        // Copy data to device
        cudaMemcpy(d_keys_, h_keys_.data(), h_keys_.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values_, h_values_.data(), h_values_.size(), cudaMemcpyHostToDevice);
    }
    
    ~WriteBottleneckAnalyzer() {
        delete gpu_hash_table_;
        cudaFree(d_keys_);
        cudaFree(d_values_);
    }
    
    void generateTestData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        // Generate random keys and values
        h_keys_.resize(NUM_WRITE_OPS * KEY_SIZE);
        for (auto& byte : h_keys_) {
            byte = dis(gen);
        }
        
        // Use largest value size for allocation
        h_values_.resize(NUM_WRITE_OPS * VALUE_SIZES[3]);
        for (auto& byte : h_values_) {
            byte = dis(gen);
        }
    }
    
    void runAnalysis() {
        std::cout << "=== GPU Write Performance Bottleneck Analysis ===" << std::endl;
        std::cout << "Operations: " << NUM_WRITE_OPS << std::endl;
        std::cout << "Key size: " << KEY_SIZE << " bytes" << std::endl;
        
        // Test 1: Baseline write performance
        std::cout << "\n--- Test 1: Baseline Write Performance ---" << std::endl;
        testBaselineWrites();
        
        // Test 2: Atomic operation conflicts
        std::cout << "\n--- Test 2: Atomic Operation Analysis ---" << std::endl;
        testAtomicConflicts();
        
        // Test 3: Memory access patterns
        std::cout << "\n--- Test 3: Memory Access Pattern Analysis ---" << std::endl;
        testMemoryPatterns();
        
        // Test 4: Optimized write performance
        std::cout << "\n--- Test 4: Optimized Write Performance ---" << std::endl;
        testOptimizedWrites();
        
        // Generate final report
        profiler_.generateReport();
        
        // Provide recommendations
        generateRecommendations();
    }
    
private:
    void testBaselineWrites() {
        profiler_.reset();
        
        for (size_t value_size : VALUE_SIZES) {
            std::string test_name = "baseline_" + std::to_string(value_size) + "B";
            
            // Warm up
            baseline_write_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
                gpu_hash_table_, d_keys_, d_values_, 1000, KEY_SIZE, value_size);
            cudaDeviceSynchronize();
            
            // Profile writes
            profiler_.profileWriteOperation(test_name, KEY_SIZE, value_size, [&]() {
                baseline_write_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
                    gpu_hash_table_, d_keys_, d_values_, NUM_WRITE_OPS, KEY_SIZE, value_size);
                cudaDeviceSynchronize();
            });
            
            // Check for errors
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
        }
    }
    
    void testAtomicConflicts() {
        // Reset conflict counter
        unsigned int zero = 0;
        cudaMemcpyToSymbol(g_atomic_conflicts, &zero, sizeof(unsigned int));
        
        // Run conflict analysis
        analyze_atomic_conflicts<<<NUM_BLOCKS, NUM_THREADS>>>(
            gpu_hash_table_, d_keys_, NUM_WRITE_OPS, KEY_SIZE);
        cudaDeviceSynchronize();
        
        // Read conflict count
        unsigned int conflicts;
        cudaMemcpyFromSymbol(&conflicts, g_atomic_conflicts, sizeof(unsigned int));
        
        std::cout << "Total atomic conflicts: " << conflicts << std::endl;
        std::cout << "Average conflicts per operation: " 
                  << (double)conflicts / NUM_WRITE_OPS << std::endl;
        
        // Record in profiler
        for (unsigned int i = 0; i < conflicts; i++) {
            profiler_.recordHashConflict("atomic_analysis");
        }
    }
    
    void testMemoryPatterns() {
        // Reset stall counter
        unsigned int zero = 0;
        cudaMemcpyToSymbol(g_memory_stalls, &zero, sizeof(unsigned int));
        
        // Test different stride patterns
        size_t strides[] = {1, 16, 32, 64, 128};
        
        for (size_t stride : strides) {
            analyze_memory_patterns<<<NUM_BLOCKS, NUM_THREADS>>>(
                d_values_, NUM_WRITE_OPS, stride);
            cudaDeviceSynchronize();
            
            unsigned int stalls;
            cudaMemcpyFromSymbol(&stalls, g_memory_stalls, sizeof(unsigned int));
            
            std::cout << "Stride " << stride << ": " << stalls << " memory stalls" << std::endl;
            
            // Reset for next test
            cudaMemcpyToSymbol(g_memory_stalls, &zero, sizeof(unsigned int));
        }
    }
    
    void testOptimizedWrites() {
        profiler_.reset();
        
        for (size_t value_size : VALUE_SIZES) {
            std::string test_name = "optimized_" + std::to_string(value_size) + "B";
            
            // Calculate shared memory size
            size_t shared_mem_size = KEY_SIZE * NUM_THREADS;
            
            // Profile optimized writes
            profiler_.profileWriteOperation(test_name, KEY_SIZE, value_size, [&]() {
                optimized_write_kernel<<<NUM_BLOCKS, NUM_THREADS, shared_mem_size>>>(
                    gpu_hash_table_, d_keys_, d_values_, NUM_WRITE_OPS, KEY_SIZE, value_size);
                cudaDeviceSynchronize();
            });
        }
    }
    
    void generateRecommendations() {
        std::cout << "\n=== Performance Optimization Recommendations ===" << std::endl;
        
        std::cout << "\n1. **Atomic Operation Optimization**" << std::endl;
        std::cout << "   - Implement lock-free hash table with CAS operations" << std::endl;
        std::cout << "   - Use multiple hash tables to distribute conflicts" << std::endl;
        std::cout << "   - Consider cuckoo hashing for guaranteed O(1) writes" << std::endl;
        
        std::cout << "\n2. **Memory Access Optimization**" << std::endl;
        std::cout << "   - Ensure coalesced memory access patterns" << std::endl;
        std::cout << "   - Use shared memory for key/value staging" << std::endl;
        std::cout << "   - Align data structures to cache line boundaries" << std::endl;
        
        std::cout << "\n3. **Batch Processing Enhancement**" << std::endl;
        std::cout << "   - Group writes by hash bucket to reduce conflicts" << std::endl;
        std::cout << "   - Use persistent kernels for small writes" << std::endl;
        std::cout << "   - Implement write combining for better throughput" << std::endl;
        
        std::cout << "\n4. **GPU Utilization Improvement**" << std::endl;
        std::cout << "   - Increase occupancy with optimized register usage" << std::endl;
        std::cout << "   - Use multiple streams for concurrent operations" << std::endl;
        std::cout << "   - Profile and tune kernel launch configurations" << std::endl;
    }
};

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Set device
    cudaSetDevice(0);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    
    // Start profiling
    cudaProfilerStart();
    
    // Run analysis
    WriteBottleneckAnalyzer analyzer;
    analyzer.runAnalysis();
    
    // Stop profiling
    cudaProfilerStop();
    
    return 0;
}