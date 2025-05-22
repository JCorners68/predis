# Predis Coding Standards

This document defines coding standards and style guidelines for the Predis GPU-accelerated cache project.

## C++ Coding Standards

### General Style Guidelines

**Formatting:**
- Use 4 spaces for indentation (no tabs)
- Line length: 100 characters maximum
- Use clang-format with Google style as base
- Always use braces for control structures

**Example:**
```cpp
// Good
if (gpu_memory_manager->is_initialized()) {
    return allocate_cache_entry(key, value_size);
}

// Bad  
if (gpu_memory_manager->is_initialized())
    return allocate_cache_entry(key, value_size);
```

### Naming Conventions

**Classes and Structs:**
```cpp
class CacheManager;          // PascalCase
class SimpleCacheManager;    // PascalCase
struct MemoryStats;          // PascalCase
```

**Functions and Methods:**
```cpp
bool initialize();                    // snake_case
void shutdown();                     // snake_case
size_t get_memory_usage() const;     // snake_case
bool put_cache_entry(const std::string& key);  // snake_case
```

**Variables:**
```cpp
// Local variables - snake_case
size_t cache_size = 0;
bool is_initialized = false;
std::string cache_key;

// Member variables - snake_case with trailing underscore
class CacheManager {
private:
    bool initialized_;
    size_t total_memory_bytes_;
    std::unique_ptr<MemoryManager> memory_manager_;
};

// Constants - SCREAMING_SNAKE_CASE
static const size_t MAX_CACHE_SIZE = 1024 * 1024 * 1024;  // 1GB
static const int DEFAULT_BLOCK_SIZE = 4096;
```

**Namespaces:**
```cpp
namespace predis {
namespace core {
    // Implementation
}
namespace api {
    // Implementation  
}
}
```

### Memory Management

**Smart Pointers (Required):**
```cpp
// Good - Use smart pointers
std::unique_ptr<MemoryManager> memory_manager_;
std::shared_ptr<CacheEntry> cache_entry;

// Bad - Avoid raw pointers for ownership
MemoryManager* memory_manager_;  // DON'T USE
```

**RAII Pattern:**
```cpp
class CacheManager {
public:
    CacheManager() : memory_manager_(std::make_unique<MemoryManager>()) {}
    
    ~CacheManager() {
        // Automatic cleanup via smart pointers
        shutdown();
    }
    
    // No manual memory management needed
};
```

**GPU Memory Management:**
```cpp
// Always pair CUDA allocations with proper cleanup
class GPUBuffer {
private:
    void* gpu_ptr_ = nullptr;
    size_t size_ = 0;
    
public:
    bool allocate(size_t size) {
        cudaError_t error = cudaMalloc(&gpu_ptr_, size);
        if (error != cudaSuccess) {
            std::cerr << "CUDA allocation failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        size_ = size;
        return true;
    }
    
    ~GPUBuffer() {
        if (gpu_ptr_) {
            cudaFree(gpu_ptr_);
        }
    }
};
```

### Error Handling

**Return Value Checking:**
```cpp
// Good - Check all return values
bool CacheManager::initialize() {
    if (!memory_manager_->initialize()) {
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}
```

**CUDA Error Handling:**
```cpp
// Required pattern for all CUDA operations
cudaError_t error = cudaMalloc(&gpu_ptr, size);
if (error != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
    return false;
}
```

**Exception Safety:**
```cpp
// Use RAII for exception safety
void CacheManager::process_batch(const std::vector<CacheEntry>& entries) {
    std::lock_guard<std::mutex> lock(cache_mutex_);  // Automatic unlock
    
    for (const auto& entry : entries) {
        if (!process_entry(entry)) {
            throw std::runtime_error("Failed to process cache entry");
        }
    }
    // Lock automatically released
}
```

### Performance Guidelines

**Const Correctness:**
```cpp
class CacheManager {
public:
    // Const methods for read-only operations
    size_t get_cache_size() const;
    bool is_key_present(const std::string& key) const;
    
    // Non-const for modifying operations
    bool put(const std::string& key, const std::string& value);
};
```

**Pass by Reference:**
```cpp
// Good - Avoid unnecessary copies
bool put(const std::string& key, const std::string& value);
bool get(const std::string& key, std::string& value) const;

// Bad - Expensive copies
bool put(std::string key, std::string value);  // DON'T USE
```

**Move Semantics:**
```cpp
// Support move operations for large objects
class CacheEntry {
public:
    CacheEntry(std::string key, std::vector<uint8_t> data) 
        : key_(std::move(key)), data_(std::move(data)) {}
};
```

## CUDA Coding Standards

### Kernel Naming and Organization

**Kernel Functions:**
```cpp
// Use __global__ for kernels, descriptive names
__global__ void hash_lookup_kernel(const char* keys, int* results, size_t num_keys);
__global__ void memory_copy_kernel(void* dest, const void* src, size_t size);

// Device functions use __device__
__device__ uint32_t gpu_hash_function(const char* key, size_t length);
```

### Memory Access Patterns

**Coalesced Memory Access:**
```cpp
// Good - Coalesced access pattern
__global__ void process_cache_entries(CacheEntry* entries, size_t num_entries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        // Each thread accesses consecutive memory
        entries[idx].process();
    }
}
```

**Shared Memory Usage:**
```cpp
__global__ void hash_table_lookup(const char* keys, int* results) {
    __shared__ char shared_keys[256];  // Shared memory for block
    
    // Cooperative loading into shared memory
    int tid = threadIdx.x;
    if (tid < 256) {
        shared_keys[tid] = keys[blockIdx.x * 256 + tid];
    }
    __syncthreads();
    
    // Use shared memory for lookups
}
```

### Error Checking

**Always Check CUDA Errors:**
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        return false; \
    } \
} while(0)

// Usage
CUDA_CHECK(cudaMalloc(&gpu_ptr, size));
CUDA_CHECK(cudaMemcpy(gpu_ptr, host_ptr, size, cudaMemcpyHostToDevice));
```

## Python Coding Standards

### Style Guidelines

**PEP 8 Compliance:**
- Use 4 spaces for indentation
- Line length: 88 characters (Black formatter)
- Use snake_case for functions and variables
- Use PascalCase for classes

### Type Hints

**Required for All Functions:**
```python
from typing import List, Dict, Optional, Union

def create_cache_client(host: str, port: int = 6379) -> Optional[PredisClient]:
    """Create a new Predis cache client."""
    try:
        client = PredisClient()
        if client.connect(host, port):
            return client
        return None
    except Exception as e:
        logger.error(f"Failed to create cache client: {e}")
        return None

def batch_put(entries: Dict[str, str]) -> bool:
    """Store multiple key-value pairs in the cache."""
    # Implementation
```

### Machine Learning Code

**Model Training Standards:**
```python
import torch
import numpy as np
from typing import Tuple

class PrefetchModel(torch.nn.Module):
    """Neural network for cache prefetch prediction."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        lstm_out, _ = self.lstm(x)
        # Use last output for prediction
        predictions = self.classifier(lstm_out[:, -1, :])
        return torch.sigmoid(predictions)
```

## Documentation Standards

### Header Comments

**All Public APIs:**
```cpp
/**
 * @brief Manages GPU-based key-value storage with predictive prefetching
 * 
 * The CacheManager class provides the main interface for cache operations,
 * handling GPU memory allocation, data storage, and retrieval with
 * performance optimizations for batch operations.
 * 
 * @example
 * CacheManager cache;
 * cache.initialize();
 * cache.put("key1", "value1");
 * std::string value;
 * if (cache.get("key1", value)) {
 *     std::cout << "Retrieved: " << value << std::endl;
 * }
 */
class CacheManager {
public:
    /**
     * @brief Initialize the GPU cache manager
     * @return true if initialization successful, false otherwise
     */
    bool initialize();
};
```

### Inline Comments

**Complex GPU Operations:**
```cpp
void* MemoryManager::allocate(size_t size) {
    // Align size to 256-byte boundaries for optimal GPU memory access
    size_t aligned_size = (size + 255) & ~255;
    
    // Check if we have enough free memory (keep 5% buffer)
    if (allocated_bytes_ + aligned_size > max_memory_bytes_ * 0.95) {
        return nullptr;
    }
    
    void* ptr;
    // Use cudaMalloc for GPU memory allocation
    cudaError_t error = cudaMalloc(&ptr, aligned_size);
    if (error != cudaSuccess) {
        std::cerr << "GPU allocation failed: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    
    return ptr;
}
```

### Performance Documentation

**Always Include Performance Notes:**
```cpp
/**
 * @brief Batch insert operation for improved GPU throughput
 * 
 * @param key_values Map of key-value pairs to insert
 * @return true if all insertions successful
 * 
 * @performance Achieves ~10x better throughput than individual puts
 *              by minimizing GPU kernel launch overhead
 * @memory Uses temporary GPU buffer sized for largest batch
 */
bool batch_put(const std::unordered_map<std::string, std::string>& key_values);
```

## Code Quality Tools

### Automated Formatting

**.clang-format Configuration:**
```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
BreakBeforeBraces: Attach
SpaceAfterCStyleCast: true
PointerAlignment: Left
```

### Static Analysis

**Required Tools:**
- clang-tidy for C++ static analysis
- cppcheck for additional C++ checks  
- black for Python formatting
- mypy for Python type checking

### Pre-commit Hooks

**Setup (.pre-commit-config.yaml):**
```yaml
repos:
  - repo: local
    hooks:
      - id: clang-format
        name: clang-format
        entry: clang-format
        language: system
        files: \.(cpp|h|cu)$
        args: [-i]
      
      - id: black
        name: black
        entry: black
        language: system
        files: \.py$
```

These coding standards ensure consistent, maintainable, and high-performance code across all Predis components, with special attention to GPU programming best practices and performance optimization.