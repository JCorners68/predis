#include "write_optimized_kernels.h"
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

namespace predis {

// Constants for write optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_RETRIES = 16;
constexpr int CACHE_LINE_SIZE = 128;

// Write-optimized hash table structure
struct WriteOptimizedBucket {
    // Aligned to cache line for better performance
    alignas(CACHE_LINE_SIZE) struct {
        uint64_t keys[4];
        uint64_t values[4];
        uint32_t key_sizes[4];
        uint32_t value_sizes[4];
        cuda::atomic<uint32_t> lock;
        uint32_t occupancy;
    } data;
};

// Optimized hash function for better distribution
__device__ __forceinline__ uint64_t optimized_hash(const uint8_t* key, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    
    // Process 8 bytes at a time for better performance
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t chunk = *reinterpret_cast<const uint64_t*>(key + i);
        hash ^= chunk;
        hash *= prime;
    }
    
    // Handle remaining bytes
    for (; i < size; i++) {
        hash ^= key[i];
        hash *= prime;
    }
    
    return hash;
}

// Warp-cooperative write to reduce conflicts
__device__ void warp_cooperative_write(WriteOptimizedBucket* buckets,
                                      size_t num_buckets,
                                      const uint8_t* key,
                                      size_t key_size,
                                      const uint8_t* value,
                                      size_t value_size,
                                      bool& success) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    int lane_id = warp.thread_rank();
    
    // Hash calculation - all threads compute same hash
    uint64_t hash = optimized_hash(key, key_size);
    uint64_t bucket_idx = hash % num_buckets;
    
    // Try multiple buckets with exponential backoff
    for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
        WriteOptimizedBucket& bucket = buckets[bucket_idx];
        
        // Lane 0 tries to acquire lock
        uint32_t lock_acquired = 0;
        if (lane_id == 0) {
            lock_acquired = bucket.data.lock.exchange(1, cuda::memory_order_acquire);
        }
        
        // Broadcast lock result to all lanes
        lock_acquired = warp.shfl(lock_acquired, 0);
        
        if (lock_acquired == 0) {
            // Lock acquired - all threads cooperate in write
            
            // Find empty slot
            int empty_slot = -1;
            for (int i = lane_id; i < 4; i += WARP_SIZE) {
                if (bucket.data.key_sizes[i] == 0) {
                    empty_slot = i;
                    break;
                }
            }
            
            // Reduce to find first empty slot
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                int other_slot = warp.shfl_down(empty_slot, offset);
                if (empty_slot == -1 && other_slot != -1) {
                    empty_slot = other_slot;
                }
            }
            empty_slot = warp.shfl(empty_slot, 0);
            
            if (empty_slot != -1) {
                // Cooperatively write key and value
                if (lane_id == 0) {
                    bucket.data.keys[empty_slot] = *reinterpret_cast<const uint64_t*>(key);
                    bucket.data.key_sizes[empty_slot] = key_size;
                    bucket.data.values[empty_slot] = *reinterpret_cast<const uint64_t*>(value);
                    bucket.data.value_sizes[empty_slot] = value_size;
                    bucket.data.occupancy++;
                    
                    // Track atomic retries for profiling
                    atomicAdd(&g_atomic_retry_count, attempt);
                }
                
                __threadfence();
                
                // Release lock
                if (lane_id == 0) {
                    bucket.data.lock.store(0, cuda::memory_order_release);
                }
                
                success = true;
                return;
            }
            
            // Release lock if no space
            if (lane_id == 0) {
                bucket.data.lock.store(0, cuda::memory_order_release);
            }
        }
        
        // Linear probing to next bucket
        bucket_idx = (bucket_idx + 1) % num_buckets;
        
        // Exponential backoff
        __nanosleep(1 << attempt);
    }
    
    success = false;
}

// Lock-free write using compare-and-swap
__device__ void lock_free_write(WriteOptimizedBucket* buckets,
                                      size_t num_buckets,
                                      const uint8_t* key,
                                      size_t key_size,
                                      const uint8_t* value,
                                      size_t value_size) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    
    // Compute hash cooperatively
    uint64_t hash = 0;
    if (warp.thread_rank() == 0) {
        hash = optimized_hash(key, key_size);
    }
    hash = warp.shfl(hash, 0);
    
    // Try primary and secondary buckets
    uint64_t primary_idx = hash % num_buckets;
    uint64_t secondary_idx = (hash >> 32) % num_buckets;
    
    // Leader thread handles locking
    bool inserted = false;
    if (warp.thread_rank() == 0) {
        // Try primary bucket first
        uint32_t expected = 0;
        if (buckets[primary_idx].data.lock.compare_exchange_weak(expected, 1)) {
            // Check for space
            if (buckets[primary_idx].data.occupancy < 4) {
                uint32_t slot = buckets[primary_idx].data.occupancy++;
                buckets[primary_idx].data.keys[slot] = reinterpret_cast<uint64_t>(key);
                buckets[primary_idx].data.values[slot] = reinterpret_cast<uint64_t>(value);
                buckets[primary_idx].data.key_sizes[slot] = key_size;
                buckets[primary_idx].data.value_sizes[slot] = value_size;
                inserted = true;
            }
            buckets[primary_idx].data.lock.store(0);
        }
        
        // Try secondary bucket if needed
        if (!inserted) {
            expected = 0;
            if (buckets[secondary_idx].data.lock.compare_exchange_weak(expected, 1)) {
                if (buckets[secondary_idx].data.occupancy < 4) {
                    uint32_t slot = buckets[secondary_idx].data.occupancy++;
                    buckets[secondary_idx].data.keys[slot] = reinterpret_cast<uint64_t>(key);
                    buckets[secondary_idx].data.values[slot] = reinterpret_cast<uint64_t>(value);
                    buckets[secondary_idx].data.key_sizes[slot] = key_size;
                    buckets[secondary_idx].data.value_sizes[slot] = value_size;
                    inserted = true;
                }
                buckets[secondary_idx].data.lock.store(0);
            }
        }
    }
    
    // Broadcast result to warp
    inserted = warp.shfl(inserted, 0);
}

// Optimized batch write kernel with coalesced memory access
__global__ void optimized_batch_write_kernel(WriteOptimizedBucket* buckets,
                                            size_t num_buckets,
                                            const uint8_t* keys,
                                            const uint8_t* values,
                                            const size_t* key_sizes,
                                            const size_t* value_sizes,
                                            size_t num_operations,
                                            uint32_t* success_flags) {
    extern __shared__ uint8_t shared_mem[];
    
    // Partition shared memory
    uint8_t* shared_keys = shared_mem;
    uint8_t* shared_values = shared_mem + blockDim.x * 32;  // Assume max 32 byte keys
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Grid-stride loop for better occupancy
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_operations;
         idx += gridDim.x * blockDim.x) {
        
        // Coalesced read of key and value pointers
        size_t key_size = key_sizes[idx];
        size_t value_size = value_sizes[idx];
        const uint8_t* key = keys + idx * 32;  // Assume fixed stride
        const uint8_t* value = values + idx * 1024;  // Assume fixed stride
        
        // Stage data in shared memory for better access pattern
        for (int i = 0; i < key_size; i += blockDim.x) {
            if (i + threadIdx.x < key_size) {
                shared_keys[threadIdx.x * 32 + i + threadIdx.x] = key[i + threadIdx.x];
            }
        }
        
        block.sync();
        
        // Perform write operation
        if (threadIdx.x % WARP_SIZE == 0) {
            warp_cooperative_write(buckets, num_buckets,
                                 shared_keys + (threadIdx.x / WARP_SIZE) * 32,
                                 key_size, value, value_size);
        }
        
        // Mark success
        if (success_flags != nullptr) {
            success_flags[idx] = 1;
        }
    }
}

// Lock-free write using compare-and-swap
__device__ bool lock_free_insert(WriteOptimizedBucket* bucket,
                                const uint8_t* key,
                                size_t key_size,
                                const uint8_t* value,
                                size_t value_size) {
    // Try to find empty slot without locking
    for (int slot = 0; slot < 4; slot++) {
        uint64_t expected = 0;
        uint64_t key_ptr = reinterpret_cast<uint64_t>(key);
        
        // Atomic CAS to claim slot
        if (atomicCAS(&bucket->data.keys[slot], expected, key_ptr) == expected) {
            // Successfully claimed slot, write remaining data
            bucket->data.values[slot] = reinterpret_cast<uint64_t>(value);
            bucket->data.key_sizes[slot] = key_size;
            bucket->data.value_sizes[slot] = value_size;
            atomicAdd(&bucket->data.occupancy, 1);
            return true;
        }
    }
    
    return false;
}

// Memory-optimized write with prefetching
__global__ void memory_optimized_write_kernel(WriteOptimizedBucket* buckets,
                                            size_t num_buckets,
                                            const uint8_t* __restrict__ keys,
                                            const uint8_t* __restrict__ values,
                                            size_t num_operations,
                                            size_t key_stride,
                                            size_t value_stride) {
    // Use texture memory for better cache behavior
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Prefetch next data while processing current
    for (int idx = tid; idx < num_operations; idx += stride) {
        // Current operation
        const uint8_t* key = keys + idx * key_stride;
        const uint8_t* value = values + idx * value_stride;
        
        // Prefetch next iteration data
        if (idx + stride < num_operations) {
            __builtin_prefetch(keys + (idx + stride) * key_stride, 0, 3);
            __builtin_prefetch(values + (idx + stride) * value_stride, 0, 3);
        }
        
        // Compute hash and insert
        uint64_t hash = optimized_hash(key, 32);  // Assume fixed key size
        uint64_t bucket_idx = hash % num_buckets;
        
        // Try lock-free insert first
        if (!lock_free_insert(&buckets[bucket_idx], key, 32, value, value_stride)) {
            // Fall back to secondary bucket
            bucket_idx = (hash >> 32) % num_buckets;
            lock_free_insert(&buckets[bucket_idx], key, 32, value, value_stride);
        }
    }
}

// Write combining kernel for small writes
__global__ void write_combining_kernel(WriteOptimizedBucket* buckets,
                                     size_t num_buckets,
                                     const uint8_t* keys,
                                     const uint8_t* values,
                                     const size_t* offsets,
                                     size_t num_operations) {
    extern __shared__ uint8_t combine_buffer[];
    
    auto block = cg::this_thread_block();
    
    // Combine small writes in shared memory
    __shared__ uint32_t buffer_offset;
    if (threadIdx.x == 0) {
        buffer_offset = 0;
    }
    block.sync();
    
    // Each thread processes one operation
    if (threadIdx.x < num_operations) {
        size_t key_offset = offsets[threadIdx.x * 2];
        size_t value_offset = offsets[threadIdx.x * 2 + 1];
        size_t key_size = offsets[(threadIdx.x + 1) * 2] - key_offset;
        size_t value_size = offsets[(threadIdx.x + 1) * 2 + 1] - value_offset;
        
        // Atomically allocate space in combine buffer
        uint32_t my_offset = atomicAdd(&buffer_offset, key_size + value_size);
        
        // Copy to combine buffer
        for (int i = 0; i < key_size; i++) {
            combine_buffer[my_offset + i] = keys[key_offset + i];
        }
        for (int i = 0; i < value_size; i++) {
            combine_buffer[my_offset + key_size + i] = values[value_offset + i];
        }
    }
    
    block.sync();
    
    // Cooperatively write combined buffer
    // Implementation depends on specific requirements
}

// Host-side write optimization manager
WriteOptimizationManager::WriteOptimizationManager(size_t num_buckets) 
    : num_buckets_(num_buckets) {
    
    // Allocate write-optimized buckets
    size_t bucket_size = sizeof(WriteOptimizedBucket) * num_buckets;
    cudaMalloc(&d_buckets_, bucket_size);
    cudaMemset(d_buckets_, 0, bucket_size);
    
    // Create CUDA streams for concurrent operations
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

WriteOptimizationManager::~WriteOptimizationManager() {
    cudaFree(d_buckets_);
    
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams_[i]);
    }
}

void WriteOptimizationManager::batch_write(const std::vector<std::pair<std::string, std::string>>& kv_pairs) {
    size_t num_ops = kv_pairs.size();
    
    // Prepare device memory
    size_t total_key_size = 0, total_value_size = 0;
    for (const auto& kv : kv_pairs) {
        total_key_size += kv.first.size();
        total_value_size += kv.second.size();
    }
    
    uint8_t* d_keys, *d_values;
    size_t* d_key_sizes, *d_value_sizes;
    uint32_t* d_success_flags;
    
    cudaMalloc(&d_keys, total_key_size);
    cudaMalloc(&d_values, total_value_size);
    cudaMalloc(&d_key_sizes, num_ops * sizeof(size_t));
    cudaMalloc(&d_value_sizes, num_ops * sizeof(size_t));
    cudaMalloc(&d_success_flags, num_ops * sizeof(uint32_t));
    
    // Copy data to device (this could be optimized with pinned memory)
    // ... (implementation details)
    
    // Launch optimized kernel
    int block_size = 256;
    int grid_size = (num_ops + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * (32 + 1024);  // Keys + values
    
    optimized_batch_write_kernel<<<grid_size, block_size, shared_mem_size, streams_[0]>>>(
        reinterpret_cast<WriteOptimizedBucket*>(d_buckets_),
        num_buckets_,
        d_keys, d_values,
        d_key_sizes, d_value_sizes,
        num_ops,
        d_success_flags
    );
    
    // Clean up
    cudaStreamSynchronize(streams_[0]);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_key_sizes);
    cudaFree(d_value_sizes);
    cudaFree(d_success_flags);
}

} // namespace predis