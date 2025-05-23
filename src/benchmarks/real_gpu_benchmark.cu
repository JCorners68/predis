// REAL GPU BENCHMARK - ACTUAL MEASURED PERFORMANCE
// This benchmark executes real GPU operations and measures actual performance
// All results are from real execution, not simulated

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <unordered_map>
#include <atomic>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Real GPU hash table implementation
struct GPUHashEntry {
    unsigned int key;
    unsigned int value;
    int occupied;  // 1 if occupied, 0 if empty
};

class GPUHashTable {
private:
    GPUHashEntry* d_table;
    size_t table_size;
    size_t max_entries;
    
public:
    GPUHashTable(size_t size) : table_size(size) {
        max_entries = size;
        CUDA_CHECK(cudaMalloc(&d_table, sizeof(GPUHashEntry) * table_size));
        CUDA_CHECK(cudaMemset(d_table, 0, sizeof(GPUHashEntry) * table_size));
    }
    
    ~GPUHashTable() {
        if (d_table) {
            cudaFree(d_table);
        }
    }
    
    GPUHashEntry* getTable() { return d_table; }
    size_t getSize() { return table_size; }
};

// Real GPU kernels for cache operations
__device__ unsigned int hash_function(unsigned int key, size_t table_size) {
    // Simple hash function
    return (key * 2654435761u) % table_size;
}

__global__ void gpu_put_kernel(GPUHashEntry* table, size_t table_size,
                               unsigned int* keys, unsigned int* values,
                               int num_operations, int* success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_operations) return;
    
    unsigned int key = keys[tid];
    unsigned int value = values[tid];
    unsigned int hash = hash_function(key, table_size);
    
    // Linear probing for collision resolution
    for (int i = 0; i < 100; i++) {  // Max 100 probes
        unsigned int idx = (hash + i) % table_size;
        
        // Try to insert
        int old = atomicCAS(&table[idx].occupied, 0, 1);
        if (old == 0) {
            // Successfully claimed this slot
            table[idx].key = key;
            table[idx].value = value;
            atomicAdd(success_count, 1);
            break;
        } else if (table[idx].key == key) {
            // Update existing key
            table[idx].value = value;
            atomicAdd(success_count, 1);
            break;
        }
    }
}

__global__ void gpu_get_kernel(GPUHashEntry* table, size_t table_size,
                               unsigned int* keys, unsigned int* values,
                               int num_operations, int* success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_operations) return;
    
    unsigned int key = keys[tid];
    unsigned int hash = hash_function(key, table_size);
    
    // Linear probing to find key
    for (int i = 0; i < 100; i++) {
        unsigned int idx = (hash + i) % table_size;
        
        if (table[idx].occupied && table[idx].key == key) {
            values[tid] = table[idx].value;
            atomicAdd(success_count, 1);
            break;
        } else if (!table[idx].occupied) {
            // Key not found
            values[tid] = 0;
            break;
        }
    }
}

__global__ void gpu_delete_kernel(GPUHashEntry* table, size_t table_size,
                                  unsigned int* keys, int num_operations,
                                  int* success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_operations) return;
    
    unsigned int key = keys[tid];
    unsigned int hash = hash_function(key, table_size);
    
    // Linear probing to find and delete key
    for (int i = 0; i < 100; i++) {
        unsigned int idx = (hash + i) % table_size;
        
        if (table[idx].occupied && table[idx].key == key) {
            table[idx].occupied = 0;
            atomicAdd(success_count, 1);
            break;
        } else if (!table[idx].occupied) {
            // Key not found
            break;
        }
    }
}

// Benchmark result structure
struct BenchmarkResult {
    std::string operation;
    int num_operations;
    double duration_ms;
    double ops_per_second;
    double latency_us;  // microseconds per operation
    int successful_ops;
    double success_rate;
};

// Real benchmark execution
class RealGPUBenchmark {
private:
    GPUHashTable* hash_table;
    int table_size;
    
public:
    RealGPUBenchmark(int size) : table_size(size) {
        hash_table = new GPUHashTable(size);
        printf("Created GPU hash table with %d entries\n", size);
    }
    
    ~RealGPUBenchmark() {
        delete hash_table;
    }
    
    BenchmarkResult benchmarkPut(int num_operations) {
        printf("\nRunning PUT benchmark with %d operations...\n", num_operations);
        
        // Prepare data
        std::vector<unsigned int> h_keys(num_operations);
        std::vector<unsigned int> h_values(num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, table_size * 2);
        
        for (int i = 0; i < num_operations; i++) {
            h_keys[i] = dist(gen);
            h_values[i] = i;
        }
        
        // Allocate GPU memory
        unsigned int *d_keys, *d_values;
        int *d_success_count;
        int h_success_count = 0;
        
        CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned int) * num_operations));
        CUDA_CHECK(cudaMalloc(&d_values, sizeof(unsigned int) * num_operations));
        CUDA_CHECK(cudaMalloc(&d_success_count, sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), sizeof(unsigned int) * num_operations, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), sizeof(unsigned int) * num_operations, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_success_count, 0, sizeof(int)));
        
        // Configure kernel
        int block_size = 256;
        int grid_size = (num_operations + block_size - 1) / block_size;
        
        // Warm up
        gpu_put_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                  d_keys, d_values, num_operations, d_success_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        gpu_put_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                  d_keys, d_values, num_operations, d_success_count);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(&h_success_count, d_success_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Clean up
        CUDA_CHECK(cudaFree(d_keys));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_success_count));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Calculate results
        BenchmarkResult result;
        result.operation = "PUT";
        result.num_operations = num_operations;
        result.duration_ms = milliseconds;
        result.ops_per_second = (num_operations / milliseconds) * 1000.0;
        result.latency_us = (milliseconds * 1000.0) / num_operations;
        result.successful_ops = h_success_count;
        result.success_rate = (double)h_success_count / num_operations * 100.0;
        
        printf("PUT completed: %d ops in %.3f ms = %.2f million ops/sec\n",
               num_operations, milliseconds, result.ops_per_second / 1e6);
        printf("Success rate: %.1f%% (%d/%d)\n", result.success_rate, h_success_count, num_operations);
        
        return result;
    }
    
    BenchmarkResult benchmarkGet(int num_operations) {
        printf("\nRunning GET benchmark with %d operations...\n", num_operations);
        
        // First populate the table
        benchmarkPut(table_size / 2);  // Fill to 50% capacity
        
        // Prepare data for GET operations
        std::vector<unsigned int> h_keys(num_operations);
        std::vector<unsigned int> h_values(num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, table_size * 2);
        
        for (int i = 0; i < num_operations; i++) {
            h_keys[i] = dist(gen);
        }
        
        // Allocate GPU memory
        unsigned int *d_keys, *d_values;
        int *d_success_count;
        int h_success_count = 0;
        
        CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned int) * num_operations));
        CUDA_CHECK(cudaMalloc(&d_values, sizeof(unsigned int) * num_operations));
        CUDA_CHECK(cudaMalloc(&d_success_count, sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), sizeof(unsigned int) * num_operations, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_success_count, 0, sizeof(int)));
        
        // Configure kernel
        int block_size = 256;
        int grid_size = (num_operations + block_size - 1) / block_size;
        
        // Warm up
        gpu_get_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                  d_keys, d_values, num_operations, d_success_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        gpu_get_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                  d_keys, d_values, num_operations, d_success_count);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(&h_success_count, d_success_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Clean up
        CUDA_CHECK(cudaFree(d_keys));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_success_count));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Calculate results
        BenchmarkResult result;
        result.operation = "GET";
        result.num_operations = num_operations;
        result.duration_ms = milliseconds;
        result.ops_per_second = (num_operations / milliseconds) * 1000.0;
        result.latency_us = (milliseconds * 1000.0) / num_operations;
        result.successful_ops = h_success_count;
        result.success_rate = (double)h_success_count / num_operations * 100.0;
        
        printf("GET completed: %d ops in %.3f ms = %.2f million ops/sec\n",
               num_operations, milliseconds, result.ops_per_second / 1e6);
        printf("Hit rate: %.1f%% (%d/%d)\n", result.success_rate, h_success_count, num_operations);
        
        return result;
    }
    
    BenchmarkResult benchmarkDelete(int num_operations) {
        printf("\nRunning DELETE benchmark with %d operations...\n", num_operations);
        
        // Prepare data
        std::vector<unsigned int> h_keys(num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, table_size * 2);
        
        for (int i = 0; i < num_operations; i++) {
            h_keys[i] = dist(gen);
        }
        
        // Allocate GPU memory
        unsigned int *d_keys;
        int *d_success_count;
        int h_success_count = 0;
        
        CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned int) * num_operations));
        CUDA_CHECK(cudaMalloc(&d_success_count, sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), sizeof(unsigned int) * num_operations, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_success_count, 0, sizeof(int)));
        
        // Configure kernel
        int block_size = 256;
        int grid_size = (num_operations + block_size - 1) / block_size;
        
        // Warm up
        gpu_delete_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                     d_keys, num_operations, d_success_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        gpu_delete_kernel<<<grid_size, block_size>>>(hash_table->getTable(), table_size,
                                                     d_keys, num_operations, d_success_count);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(&h_success_count, d_success_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Clean up
        CUDA_CHECK(cudaFree(d_keys));
        CUDA_CHECK(cudaFree(d_success_count));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Calculate results
        BenchmarkResult result;
        result.operation = "DELETE";
        result.num_operations = num_operations;
        result.duration_ms = milliseconds;
        result.ops_per_second = (num_operations / milliseconds) * 1000.0;
        result.latency_us = (milliseconds * 1000.0) / num_operations;
        result.successful_ops = h_success_count;
        result.success_rate = (double)h_success_count / num_operations * 100.0;
        
        printf("DELETE completed: %d ops in %.3f ms = %.2f million ops/sec\n",
               num_operations, milliseconds, result.ops_per_second / 1e6);
        printf("Success rate: %.1f%% (%d/%d)\n", result.success_rate, h_success_count, num_operations);
        
        return result;
    }
};

// CPU baseline implementation for comparison
class CPUHashTable {
private:
    std::unordered_map<unsigned int, unsigned int> table;
    
public:
    BenchmarkResult benchmarkPut(int num_operations) {
        printf("\nRunning CPU PUT benchmark with %d operations...\n", num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, num_operations * 2);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_operations; i++) {
            table[dist(gen)] = i;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        BenchmarkResult result;
        result.operation = "CPU_PUT";
        result.num_operations = num_operations;
        result.duration_ms = duration.count() / 1000.0;
        result.ops_per_second = (num_operations / result.duration_ms) * 1000.0;
        result.latency_us = (double)duration.count() / num_operations;
        result.successful_ops = num_operations;
        result.success_rate = 100.0;
        
        printf("CPU PUT completed: %d ops in %.3f ms = %.2f million ops/sec\n",
               num_operations, result.duration_ms, result.ops_per_second / 1e6);
        
        return result;
    }
    
    BenchmarkResult benchmarkGet(int num_operations) {
        printf("\nRunning CPU GET benchmark with %d operations...\n", num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, num_operations * 2);
        
        int hits = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_operations; i++) {
            auto it = table.find(dist(gen));
            if (it != table.end()) {
                hits++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        BenchmarkResult result;
        result.operation = "CPU_GET";
        result.num_operations = num_operations;
        result.duration_ms = duration.count() / 1000.0;
        result.ops_per_second = (num_operations / result.duration_ms) * 1000.0;
        result.latency_us = (double)duration.count() / num_operations;
        result.successful_ops = hits;
        result.success_rate = (double)hits / num_operations * 100.0;
        
        printf("CPU GET completed: %d ops in %.3f ms = %.2f million ops/sec\n",
               num_operations, result.duration_ms, result.ops_per_second / 1e6);
        printf("Hit rate: %.1f%%\n", result.success_rate);
        
        return result;
    }
};

int main() {
    printf("=== REAL GPU BENCHMARK - ACTUAL MEASURED PERFORMANCE ===\n");
    printf("All results below are from real GPU execution, not simulated\n\n");
    
    // Check GPU
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);  // Approximate
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\n");
    
    // Benchmark parameters
    const int table_size = 10 * 1024 * 1024;  // 10M entries
    const int num_operations = 1000000;       // 1M operations per test
    
    // Run GPU benchmarks
    RealGPUBenchmark gpu_bench(table_size);
    std::vector<BenchmarkResult> gpu_results;
    
    gpu_results.push_back(gpu_bench.benchmarkPut(num_operations));
    gpu_results.push_back(gpu_bench.benchmarkGet(num_operations));
    gpu_results.push_back(gpu_bench.benchmarkDelete(num_operations));
    
    // Run CPU baseline
    CPUHashTable cpu_bench;
    std::vector<BenchmarkResult> cpu_results;
    
    cpu_results.push_back(cpu_bench.benchmarkPut(num_operations));
    cpu_results.push_back(cpu_bench.benchmarkGet(num_operations));
    
    // Calculate speedups
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("Operation  | GPU (M ops/s) | CPU (M ops/s) | Speedup\n");
    printf("-----------|---------------|---------------|--------\n");
    
    double gpu_put_ops = gpu_results[0].ops_per_second / 1e6;
    double cpu_put_ops = cpu_results[0].ops_per_second / 1e6;
    printf("PUT        | %13.2f | %13.2f | %.1fx\n", 
           gpu_put_ops, cpu_put_ops, gpu_put_ops / cpu_put_ops);
    
    double gpu_get_ops = gpu_results[1].ops_per_second / 1e6;
    double cpu_get_ops = cpu_results[1].ops_per_second / 1e6;
    printf("GET        | %13.2f | %13.2f | %.1fx\n",
           gpu_get_ops, cpu_get_ops, gpu_get_ops / cpu_get_ops);
    
    // Save results to JSON
    FILE* json_file = fopen("real_benchmark_results.json", "w");
    if (json_file) {
        fprintf(json_file, "{\n");
        fprintf(json_file, "  \"timestamp\": %ld,\n", time(NULL));
        fprintf(json_file, "  \"gpu\": \"%s\",\n", prop.name);
        fprintf(json_file, "  \"table_size\": %d,\n", table_size);
        fprintf(json_file, "  \"num_operations\": %d,\n", num_operations);
        fprintf(json_file, "  \"results\": {\n");
        fprintf(json_file, "    \"gpu\": [\n");
        
        for (size_t i = 0; i < gpu_results.size(); i++) {
            const auto& r = gpu_results[i];
            fprintf(json_file, "      {\n");
            fprintf(json_file, "        \"operation\": \"%s\",\n", r.operation.c_str());
            fprintf(json_file, "        \"ops_per_second\": %.2f,\n", r.ops_per_second);
            fprintf(json_file, "        \"latency_us\": %.3f,\n", r.latency_us);
            fprintf(json_file, "        \"success_rate\": %.1f\n", r.success_rate);
            fprintf(json_file, "      }%s\n", (i < gpu_results.size() - 1) ? "," : "");
        }
        
        fprintf(json_file, "    ],\n");
        fprintf(json_file, "    \"cpu\": [\n");
        
        for (size_t i = 0; i < cpu_results.size(); i++) {
            const auto& r = cpu_results[i];
            fprintf(json_file, "      {\n");
            fprintf(json_file, "        \"operation\": \"%s\",\n", r.operation.c_str());
            fprintf(json_file, "        \"ops_per_second\": %.2f,\n", r.ops_per_second);
            fprintf(json_file, "        \"latency_us\": %.3f\n", r.latency_us);
            fprintf(json_file, "      }%s\n", (i < cpu_results.size() - 1) ? "," : "");
        }
        
        fprintf(json_file, "    ]\n");
        fprintf(json_file, "  },\n");
        fprintf(json_file, "  \"speedup\": {\n");
        fprintf(json_file, "    \"put\": %.1f,\n", gpu_put_ops / cpu_put_ops);
        fprintf(json_file, "    \"get\": %.1f\n", gpu_get_ops / cpu_get_ops);
        fprintf(json_file, "  }\n");
        fprintf(json_file, "}\n");
        
        fclose(json_file);
        printf("\nResults saved to real_benchmark_results.json\n");
    }
    
    return 0;
}