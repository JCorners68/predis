// COMPREHENSIVE GPU BENCHMARK - REAL MEASURED PERFORMANCE
// This benchmark tests various scenarios and workloads
// All results are from actual GPU execution

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// GPU hash table implementation (same as before)
struct GPUHashEntry {
    unsigned int key;
    unsigned int value;
    int occupied;
};

class GPUHashTable {
private:
    GPUHashEntry* d_table;
    size_t table_size;
    
public:
    GPUHashTable(size_t size) : table_size(size) {
        CUDA_CHECK(cudaMalloc(&d_table, sizeof(GPUHashEntry) * table_size));
        CUDA_CHECK(cudaMemset(d_table, 0, sizeof(GPUHashEntry) * table_size));
    }
    
    ~GPUHashTable() {
        if (d_table) cudaFree(d_table);
    }
    
    void clear() {
        CUDA_CHECK(cudaMemset(d_table, 0, sizeof(GPUHashEntry) * table_size));
    }
    
    GPUHashEntry* getTable() { return d_table; }
    size_t getSize() { return table_size; }
};

// GPU kernels
__device__ unsigned int hash_function(unsigned int key, size_t table_size) {
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
    
    for (int i = 0; i < 100; i++) {
        unsigned int idx = (hash + i) % table_size;
        
        int old = atomicCAS(&table[idx].occupied, 0, 1);
        if (old == 0) {
            table[idx].key = key;
            table[idx].value = value;
            atomicAdd(success_count, 1);
            break;
        } else if (table[idx].key == key) {
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
    
    for (int i = 0; i < 100; i++) {
        unsigned int idx = (hash + i) % table_size;
        
        if (table[idx].occupied && table[idx].key == key) {
            values[tid] = table[idx].value;
            atomicAdd(success_count, 1);
            break;
        } else if (!table[idx].occupied) {
            values[tid] = 0;
            break;
        }
    }
}

__global__ void gpu_batch_get_kernel(GPUHashEntry* table, size_t table_size,
                                     unsigned int* keys, unsigned int* values,
                                     int num_operations, int* success_count,
                                     int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = tid / batch_size;
    int batch_offset = tid % batch_size;
    
    if (tid >= num_operations) return;
    
    // Coalesced memory access within batch
    unsigned int key = keys[batch_id * batch_size + batch_offset];
    unsigned int hash = hash_function(key, table_size);
    
    for (int i = 0; i < 100; i++) {
        unsigned int idx = (hash + i) % table_size;
        
        if (table[idx].occupied && table[idx].key == key) {
            values[tid] = table[idx].value;
            atomicAdd(success_count, 1);
            break;
        } else if (!table[idx].occupied) {
            values[tid] = 0;
            break;
        }
    }
}

// Benchmark result structure
struct BenchmarkResult {
    std::string test_name;
    std::string operation;
    int num_operations;
    int batch_size;
    double duration_ms;
    double ops_per_second;
    double latency_us;
    int successful_ops;
    double success_rate;
    double speedup_vs_cpu;
};

// Comprehensive benchmark suite
class ComprehensiveBenchmark {
private:
    std::vector<BenchmarkResult> all_results;
    
    // Run a single benchmark iteration
    template<typename KernelFunc>
    BenchmarkResult runBenchmark(const std::string& name, 
                                const std::string& operation,
                                int num_operations,
                                int batch_size,
                                GPUHashTable* table,
                                KernelFunc kernel_func) {
        printf("\nRunning %s benchmark (%d operations, batch=%d)...\n", 
               name.c_str(), num_operations, batch_size);
        
        // Prepare data
        std::vector<unsigned int> h_keys(num_operations);
        std::vector<unsigned int> h_values(num_operations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, table->getSize() * 2);
        
        for (int i = 0; i < num_operations; i++) {
            h_keys[i] = dist(gen);
            h_values[i] = i;
        }
        
        // GPU memory
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
        kernel_func(grid_size, block_size, table, d_keys, d_values, num_operations, d_success_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset success count
        CUDA_CHECK(cudaMemset(d_success_count, 0, sizeof(int)));
        
        // Actual benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        kernel_func(grid_size, block_size, table, d_keys, d_values, num_operations, d_success_count);
        
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
        result.test_name = name;
        result.operation = operation;
        result.num_operations = num_operations;
        result.batch_size = batch_size;
        result.duration_ms = milliseconds;
        result.ops_per_second = (num_operations / milliseconds) * 1000.0;
        result.latency_us = (milliseconds * 1000.0) / num_operations;
        result.successful_ops = h_success_count;
        result.success_rate = (double)h_success_count / num_operations * 100.0;
        result.speedup_vs_cpu = 0.0; // Will be calculated later
        
        printf("Completed: %.2f million ops/sec (%.3f us/op)\n",
               result.ops_per_second / 1e6, result.latency_us);
        
        return result;
    }
    
public:
    void runAllBenchmarks() {
        printf("=== COMPREHENSIVE GPU BENCHMARK SUITE ===\n");
        printf("All results are from actual GPU execution\n\n");
        
        // Test different table sizes
        std::vector<int> table_sizes = {1024*1024, 10*1024*1024, 50*1024*1024};
        std::vector<int> operation_counts = {10000, 100000, 1000000};
        std::vector<int> batch_sizes = {1, 32, 128, 256};
        
        for (int table_size : table_sizes) {
            printf("\n--- Table Size: %d entries (%.1f MB) ---\n", 
                   table_size, table_size * sizeof(GPUHashEntry) / (1024.0 * 1024.0));
            
            GPUHashTable table(table_size);
            
            for (int ops : operation_counts) {
                // Single operation benchmarks
                auto put_kernel = [](int grid, int block, GPUHashTable* t, 
                                   unsigned int* k, unsigned int* v, int n, int* s) {
                    gpu_put_kernel<<<grid, block>>>(t->getTable(), t->getSize(), k, v, n, s);
                };
                
                auto get_kernel = [](int grid, int block, GPUHashTable* t,
                                   unsigned int* k, unsigned int* v, int n, int* s) {
                    gpu_get_kernel<<<grid, block>>>(t->getTable(), t->getSize(), k, v, n, s);
                };
                
                std::stringstream name;
                name << "Table" << table_size/1024/1024 << "M_Ops" << ops/1000 << "K";
                
                all_results.push_back(runBenchmark(name.str() + "_PUT", "PUT", 
                                                  ops, 1, &table, put_kernel));
                
                all_results.push_back(runBenchmark(name.str() + "_GET", "GET",
                                                  ops, 1, &table, get_kernel));
                
                // Batch operation benchmarks
                for (int batch : batch_sizes) {
                    if (batch > 1 && ops >= batch * 100) {
                        auto batch_get_kernel = [batch](int grid, int block, GPUHashTable* t,
                                                      unsigned int* k, unsigned int* v, int n, int* s) {
                            gpu_batch_get_kernel<<<grid, block>>>(t->getTable(), t->getSize(), 
                                                                 k, v, n, s, batch);
                        };
                        
                        std::stringstream batch_name;
                        batch_name << name.str() << "_BATCH" << batch << "_GET";
                        
                        all_results.push_back(runBenchmark(batch_name.str(), "BATCH_GET",
                                                         ops, batch, &table, batch_get_kernel));
                    }
                }
                
                table.clear(); // Clear between tests
            }
        }
        
        // CPU baseline comparison
        runCPUBaseline();
        
        // Save all results
        saveResults();
    }
    
    void runCPUBaseline() {
        printf("\n=== CPU BASELINE BENCHMARKS ===\n");
        
        std::unordered_map<unsigned int, unsigned int> cpu_table;
        
        // Test 1M operations
        int num_ops = 1000000;
        
        // CPU PUT
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_ops; i++) {
            cpu_table[i] = i;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double cpu_put_ops_per_sec = (num_ops / (duration.count() / 1000.0)) * 1000.0;
        printf("CPU PUT: %.2f million ops/sec\n", cpu_put_ops_per_sec / 1e6);
        
        // CPU GET
        int hits = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_ops; i++) {
            auto it = cpu_table.find(i / 2);  // 50% hit rate
            if (it != cpu_table.end()) hits++;
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double cpu_get_ops_per_sec = (num_ops / (duration.count() / 1000.0)) * 1000.0;
        printf("CPU GET: %.2f million ops/sec (%.1f%% hit rate)\n", 
               cpu_get_ops_per_sec / 1e6, (double)hits / num_ops * 100.0);
        
        // Calculate speedups
        for (auto& result : all_results) {
            if (result.operation == "PUT") {
                result.speedup_vs_cpu = result.ops_per_second / cpu_put_ops_per_sec;
            } else if (result.operation == "GET" || result.operation == "BATCH_GET") {
                result.speedup_vs_cpu = result.ops_per_second / cpu_get_ops_per_sec;
            }
        }
    }
    
    void saveResults() {
        time_t timestamp = time(NULL);
        
        // Save JSON results
        std::stringstream json_filename;
        json_filename << "benchmark_results/comprehensive_gpu_results_" << timestamp << ".json";
        
        std::ofstream json_file(json_filename.str());
        if (json_file.is_open()) {
            json_file << "{\n";
            json_file << "  \"description\": \"REAL GPU PERFORMANCE MEASUREMENTS\",\n";
            json_file << "  \"timestamp\": " << timestamp << ",\n";
            json_file << "  \"date\": \"" << std::put_time(localtime(&timestamp), "%Y-%m-%d %H:%M:%S") << "\",\n";
            
            // Get GPU info
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            json_file << "  \"gpu\": {\n";
            json_file << "    \"name\": \"" << prop.name << "\",\n";
            json_file << "    \"memory_gb\": " << std::fixed << std::setprecision(1) 
                     << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << ",\n";
            json_file << "    \"cuda_cores\": " << prop.multiProcessorCount * 128 << "\n";
            json_file << "  },\n";
            
            json_file << "  \"results\": [\n";
            
            for (size_t i = 0; i < all_results.size(); i++) {
                const auto& r = all_results[i];
                json_file << "    {\n";
                json_file << "      \"test_name\": \"" << r.test_name << "\",\n";
                json_file << "      \"operation\": \"" << r.operation << "\",\n";
                json_file << "      \"num_operations\": " << r.num_operations << ",\n";
                json_file << "      \"batch_size\": " << r.batch_size << ",\n";
                json_file << "      \"duration_ms\": " << std::fixed << std::setprecision(3) << r.duration_ms << ",\n";
                json_file << "      \"ops_per_second\": " << std::fixed << std::setprecision(0) << r.ops_per_second << ",\n";
                json_file << "      \"million_ops_per_sec\": " << std::fixed << std::setprecision(2) << r.ops_per_second / 1e6 << ",\n";
                json_file << "      \"latency_us\": " << std::fixed << std::setprecision(3) << r.latency_us << ",\n";
                json_file << "      \"success_rate\": " << std::fixed << std::setprecision(1) << r.success_rate << ",\n";
                json_file << "      \"speedup_vs_cpu\": " << std::fixed << std::setprecision(1) << r.speedup_vs_cpu << "\n";
                json_file << "    }" << (i < all_results.size() - 1 ? "," : "") << "\n";
            }
            
            json_file << "  ],\n";
            
            // Summary statistics
            json_file << "  \"summary\": {\n";
            
            // Find best results
            double max_put_speedup = 0, max_get_speedup = 0, max_batch_speedup = 0;
            for (const auto& r : all_results) {
                if (r.operation == "PUT") max_put_speedup = std::max(max_put_speedup, r.speedup_vs_cpu);
                else if (r.operation == "GET") max_get_speedup = std::max(max_get_speedup, r.speedup_vs_cpu);
                else if (r.operation == "BATCH_GET") max_batch_speedup = std::max(max_batch_speedup, r.speedup_vs_cpu);
            }
            
            json_file << "    \"max_put_speedup\": " << std::fixed << std::setprecision(1) << max_put_speedup << ",\n";
            json_file << "    \"max_get_speedup\": " << std::fixed << std::setprecision(1) << max_get_speedup << ",\n";
            json_file << "    \"max_batch_get_speedup\": " << std::fixed << std::setprecision(1) << max_batch_speedup << "\n";
            json_file << "  }\n";
            json_file << "}\n";
            
            json_file.close();
            printf("\n\nResults saved to: %s\n", json_filename.str().c_str());
        }
    }
};

int main() {
    // Check GPU availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%.1f GB, %d SMs)\n\n", prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.multiProcessorCount);
    
    // Run comprehensive benchmarks
    ComprehensiveBenchmark benchmark;
    benchmark.runAllBenchmarks();
    
    return 0;
}