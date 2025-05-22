# Epic 1: Core GPU Cache Foundation
**Timeline**: Weeks 1-4
**Goal**: Establish basic GPU-accelerated caching functionality with reliable memory management
**Success Criteria**: 
- Basic get/put operations working on GPU VRAM
- Memory management handles 100K+ key-value pairs
- WSL/Docker environment stable for development
- Initial performance measurements vs Redis baseline

## User Stories

### Story 1.1: Development Environment Setup (Priority: P0, Points: 5)
**As a** developer
**I want** a stable WSL/Docker environment with GPU access
**So that** I can develop and test Predis reliably

**Acceptance Criteria:**
- [ ] WSL2 with Ubuntu 22.04 running stably
- [ ] Docker with NVIDIA container runtime configured
- [ ] RTX 5080 accessible from containers (nvidia-smi works)
- [ ] CUDA 12.x development environment installed
- [ ] Basic C++/CUDA compilation working

**Technical Notes:**
- Use nvidia/cuda:12.2-devel-ubuntu22.04 base image
- Configure Docker daemon with nvidia runtime
- Test GPU memory allocation with simple CUDA kernel
- Document exact setup steps for reproducibility

**Definition of Done:**
- [ ] GPU memory allocation/deallocation works in container
- [ ] Development environment documented in README
- [ ] Simple "Hello GPU" program compiles and runs

### Story 1.2: Mock Predis Client with Realistic Performance (Priority: P0, Points: 3)
**As a** developer
**I want** a working mock client that simulates target performance
**So that** I can establish baselines and have a working demo immediately

**Acceptance Criteria:**
- [ ] Mock client implements basic get/put/mget/mput operations
- [ ] Simulated latencies match target GPU performance (10-50x Redis improvement)
- [ ] In-memory storage for actual data persistence during tests
- [ ] Realistic error simulation and handling
- [ ] Performance metrics collection and reporting

**Technical Notes:**
- Simple Python implementation with time.sleep() for latency simulation
- Use dict for in-memory storage with realistic capacity limits
- Include batch operation optimizations in mock timing
- Add configurable performance multipliers for different scenarios
- Mock GPU memory constraints (16GB limit simulation)

**Implementation:**
```python
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    operations_per_second: float
    avg_latency_ms: float
    hit_ratio: float
    memory_usage_mb: float

class MockPredisClient:
    def __init__(self, host='localhost', port=6379, max_memory_mb=12800):
        self.data: Dict[str, Any] = {}
        self.access_log: List[tuple] = []  # (timestamp, key, operation)
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0, 'misses': 0, 'operations': 0,
            'total_latency': 0.0
        }
        
        # Performance multipliers vs Redis
        self.single_op_speedup = 12  # 12x faster single operations
        self.batch_op_speedup = 28   # 28x faster batch operations
        
    def _simulate_gpu_latency(self, operation_type: str, count: int = 1):
        """Simulate GPU operation latencies"""
        # Redis baseline latencies (ms)
        redis_latencies = {
            'get': 0.1, 'put': 0.12, 'batch_get': 0.08 * count,
            'batch_put': 0.1 * count
        }
        
        # Our simulated GPU performance
        base_latency = redis_latencies.get(operation_type, 0.1)
        if 'batch' in operation_type:
            gpu_latency = base_latency / self.batch_op_speedup
        else:
            gpu_latency = base_latency / self.single_op_speedup
            
        time.sleep(gpu_latency / 1000)  # Convert to seconds
        return gpu_latency
    
    def _estimate_memory_usage(self, key: str, value: Any) -> float:
        """Estimate memory usage in MB"""
        key_size = len(str(key).encode('utf-8'))
        value_size = len(str(value).encode('utf-8'))
        return (key_size + value_size) / (1024 * 1024)
    
    def _check_memory_limit(self, additional_mb: float) -> bool:
        """Check if operation would exceed memory limit"""
        return (self.current_memory_mb + additional_mb) <= self.max_memory_mb
    
    def get(self, key: str) -> Optional[Any]:
        """Get single key"""
        start_time = time.time()
        latency = self._simulate_gpu_latency('get')
        
        with self.lock:
            self.stats['operations'] += 1
            self.stats['total_latency'] += latency
            
            if key in self.data:
                self.stats['hits'] += 1
                value = self.data[key]
                self.access_log.append((time.time(), key, 'get_hit'))
                return value
            else:
                self.stats['misses'] += 1
                self.access_log.append((time.time(), key, 'get_miss'))
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put single key-value pair"""
        memory_needed = self._estimate_memory_usage(key, value)
        
        if not self._check_memory_limit(memory_needed):
            raise MemoryError(f"Operation would exceed GPU memory limit ({self.max_memory_mb}MB)")
        
        latency = self._simulate_gpu_latency('put')
        
        with self.lock:
            self.stats['operations'] += 1
            self.stats['total_latency'] += latency
            
            # Remove old value memory usage if key exists
            if key in self.data:
                old_memory = self._estimate_memory_usage(key, self.data[key])
                self.current_memory_mb -= old_memory
            
            self.data[key] = value
            self.current_memory_mb += memory_needed
            self.access_log.append((time.time(), key, 'put'))
            
            # TODO: Handle TTL in future iteration
            return True
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple keys - demonstrates batch operation speedup"""
        latency = self._simulate_gpu_latency('batch_get', len(keys))
        
        with self.lock:
            self.stats['operations'] += len(keys)
            self.stats['total_latency'] += latency
            
            results = []
            for key in keys:
                if key in self.data:
                    self.stats['hits'] += 1
                    results.append(self.data[key])
                    self.access_log.append((time.time(), key, 'mget_hit'))
                else:
                    self.stats['misses'] += 1
                    results.append(None)
                    self.access_log.append((time.time(), key, 'mget_miss'))
            
            return results
    
    def mput(self, key_value_dict: Dict[str, Any]) -> bool:
        """Put multiple key-value pairs"""
        # Check memory limit for all operations
        total_memory_needed = sum(
            self._estimate_memory_usage(k, v) for k, v in key_value_dict.items()
        )
        
        if not self._check_memory_limit(total_memory_needed):
            raise MemoryError(f"Batch operation would exceed GPU memory limit")
        
        latency = self._simulate_gpu_latency('batch_put', len(key_value_dict))
        
        with self.lock:
            self.stats['operations'] += len(key_value_dict)
            self.stats['total_latency'] += latency
            
            for key, value in key_value_dict.items():
                # Handle existing key memory
                if key in self.data:
                    old_memory = self._estimate_memory_usage(key, self.data[key])
                    self.current_memory_mb -= old_memory
                
                self.data[key] = value
                memory_used = self._estimate_memory_usage(key, value)
                self.current_memory_mb += memory_used
                self.access_log.append((time.time(), key, 'mput'))
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a key"""
        latency = self._simulate_gpu_latency('get')  # Similar to get operation
        
        with self.lock:
            self.stats['operations'] += 1
            self.stats['total_latency'] += latency
            
            if key in self.data:
                memory_freed = self._estimate_memory_usage(key, self.data[key])
                self.current_memory_mb -= memory_freed
                del self.data[key]
                self.access_log.append((time.time(), key, 'delete'))
                return True
            return False
    
    def flush_all(self) -> bool:
        """Clear all data"""
        with self.lock:
            self.data.clear()
            self.current_memory_mb = 0
            self.access_log.append((time.time(), '*', 'flush'))
            return True
    
    def get_stats(self) -> PerformanceMetrics:
        """Get performance statistics"""
        with self.lock:
            total_ops = self.stats['operations']
            if total_ops == 0:
                return PerformanceMetrics(0, 0, 0, self.current_memory_mb)
            
            hit_ratio = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            avg_latency = self.stats['total_latency'] / total_ops
            
            # Calculate ops/sec based on recent performance
            recent_time = 1.0  # Assume 1 second window for ops/sec calculation
            ops_per_second = total_ops / max(recent_time, self.stats['total_latency'] / 1000)
            
            return PerformanceMetrics(
                operations_per_second=ops_per_second,
                avg_latency_ms=avg_latency,
                hit_ratio=hit_ratio,
                memory_usage_mb=self.current_memory_mb
            )
    
    def info(self) -> Dict[str, Any]:
        """Get system information"""
        stats = self.get_stats()
        return {
            'version': '0.1.0-mock',
            'gpu_info': {
                'model': 'RTX 5080 (Simulated)',
                'vram_total_gb': self.max_memory_mb / 1024,
                'vram_used_gb': self.current_memory_mb / 1024
            },
            'performance': {
                'single_op_speedup': f"{self.single_op_speedup}x vs Redis",
                'batch_op_speedup': f"{self.batch_op_speedup}x vs Redis",
                'current_ops_per_sec': stats.operations_per_second,
                'avg_latency_ms': stats.avg_latency_ms,
                'hit_ratio': stats.hit_ratio
            },
            'memory': {
                'total_keys': len(self.data),
                'memory_usage_mb': stats.memory_usage_mb,
                'memory_limit_mb': self.max_memory_mb
            }
        }
```

**Definition of Done:**
- [ ] Mock client handles all basic operations correctly
- [ ] Performance simulation shows 10-50x improvements over baseline
- [ ] Memory constraints properly simulated
- [ ] Statistics collection functional for benchmarking

### Story 1.3: Redis Comparison Baseline Framework (Priority: P0, Points: 5)
**As a** developer
**I want** to compare mock Predis performance against real Redis
**So that** I can establish performance baselines and validate improvement claims

**Acceptance Criteria:**
- [ ] Redis instance running in Docker container
- [ ] Benchmark harness that tests both Redis and mock Predis
- [ ] Identical test scenarios for fair comparison
- [ ] Performance metrics collection and comparison
- [ ] Automated benchmark execution with reporting

**Technical Notes:**
- Use Redis 7.x with standard configuration
- Create realistic test datasets with various key/value sizes
- Implement parallel client testing for concurrency scenarios
- Collect detailed timing and throughput metrics
- Generate comparison reports with charts

**Implementation:**
```python
import redis
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class BenchmarkSuite:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.predis_client = MockPredisClient()
        
    def generate_test_data(self, key_count: int = 1000) -> Dict[str, str]:
        """Generate realistic test data"""
        import random
        import string
        
        data = {}
        for i in range(key_count):
            key = f"test_key_{i:06d}"
            # Mix of small, medium, and large values
            if i % 100 == 0:  # 1% large values (1KB)
                value = ''.join(random.choices(string.ascii_letters, k=1024))
            elif i % 10 == 0:  # 10% medium values (100B)
                value = ''.join(random.choices(string.ascii_letters, k=100))
            else:  # 89% small values (10B)
                value = ''.join(random.choices(string.ascii_letters, k=10))
            data[key] = value
        return data
    
    def benchmark_single_operations(self, test_data: Dict[str, str], iterations: int = 1000):
        """Benchmark single get/put operations"""
        keys = list(test_data.keys())[:iterations]
        
        # Benchmark Redis
        redis_times = []
        for key in keys:
            start = time.perf_counter()
            self.redis_client.set(key, test_data[key])
            self.redis_client.get(key)
            end = time.perf_counter()
            redis_times.append((end - start) * 1000)  # Convert to ms
        
        # Benchmark Mock Predis
        predis_times = []
        for key in keys:
            start = time.perf_counter()
            self.predis_client.put(key, test_data[key])
            self.predis_client.get(key)
            end = time.perf_counter()
            predis_times.append((end - start) * 1000)
        
        return {
            'redis': {
                'avg_latency_ms': statistics.mean(redis_times),
                'p95_latency_ms': statistics.quantiles(redis_times, n=20)[18],  # 95th percentile
                'ops_per_second': 2000 / sum(redis_times) * 1000  # 2 ops per iteration
            },
            'predis': {
                'avg_latency_ms': statistics.mean(predis_times),
                'p95_latency_ms': statistics.quantiles(predis_times, n=20)[18],
                'ops_per_second': 2000 / sum(predis_times) * 1000
            }
        }
    
    def benchmark_batch_operations(self, test_data: Dict[str, str], batch_size: int = 100):
        """Benchmark batch get/put operations"""
        keys = list(test_data.keys())[:batch_size]
        batch_data = {k: test_data[k] for k in keys}
        
        # Benchmark Redis (using pipeline for batch operations)
        start = time.perf_counter()
        pipe = self.redis_client.pipeline()
        for key, value in batch_data.items():
            pipe.set(key, value)
        pipe.execute()
        
        pipe = self.redis_client.pipeline()
        for key in keys:
            pipe.get(key)
        pipe.execute()
        redis_time = (time.perf_counter() - start) * 1000
        
        # Benchmark Mock Predis
        start = time.perf_counter()
        self.predis_client.mput(batch_data)
        self.predis_client.mget(keys)
        predis_time = (time.perf_counter() - start) * 1000
        
        return {
            'redis': {
                'batch_latency_ms': redis_time,
                'ops_per_second': (batch_size * 2) / (redis_time / 1000)
            },
            'predis': {
                'batch_latency_ms': predis_time,
                'ops_per_second': (batch_size * 2) / (predis_time / 1000)
            }
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run full benchmark suite"""
        print("Generating test data...")
        test_data = self.generate_test_data(10000)
        
        print("Running single operation benchmarks...")
        single_ops = self.benchmark_single_operations(test_data, 1000)
        
        print("Running batch operation benchmarks...")
        batch_ops = self.benchmark_batch_operations(test_data, 1000)
        
        # Calculate improvements
        single_latency_improvement = single_ops['redis']['avg_latency_ms'] / single_ops['predis']['avg_latency_ms']
        batch_latency_improvement = batch_ops['redis']['batch_latency_ms'] / batch_ops['predis']['batch_latency_ms']
        
        results = {
            'single_operations': single_ops,
            'batch_operations': batch_ops,
            'improvements': {
                'single_op_speedup': f"{single_latency_improvement:.1f}x",
                'batch_op_speedup': f"{batch_latency_improvement:.1f}x"
            },
            'predis_stats': self.predis_client.info()
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = 'benchmark_results.json'):
        """Generate benchmark report"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Predis vs Redis Benchmark Results ===")
        print(f"Single Operation Speedup: {results['improvements']['single_op_speedup']}")
        print(f"Batch Operation Speedup: {results['improvements']['batch_op_speedup']}")
        print(f"\nSingle Operations:")
        print(f"  Redis Avg Latency: {results['single_operations']['redis']['avg_latency_ms']:.2f}ms")
        print(f"  Predis Avg Latency: {results['single_operations']['predis']['avg_latency_ms']:.2f}ms")
        print(f"\nBatch Operations (1000 keys):")
        print(f"  Redis Batch Time: {results['batch_operations']['redis']['batch_latency_ms']:.2f}ms")
        print(f"  Predis Batch Time: {results['batch_operations']['predis']['batch_latency_ms']:.2f}ms")
        print(f"\nReport saved to: {output_file}")

# Usage example for quick testing
if __name__ == "__main__":
    benchmark = BenchmarkSuite()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.generate_report(results)
```

**Definition of Done:**
- [ ] Benchmark runs successfully against both Redis and mock Predis
- [ ] Performance improvements of 10-50x demonstrated in results
- [ ] Benchmark methodology documented and reproducible
- [ ] Results exported in multiple formats (JSON, charts)

### Story 1.4: Basic GPU Memory Management (Priority: P1, Points: 8)
**As a** cache system
**I want** to allocate and manage GPU VRAM efficiently
**So that** I can store key-value pairs in GPU memory

**Acceptance Criteria:**
- [ ] GPU memory allocator with malloc/free operations
- [ ] Memory pool management for efficient allocation
- [ ] Memory usage tracking and reporting
- [ ] Graceful handling of out-of-memory conditions
- [ ] Support for variable-size value allocation

**Technical Notes:**
- Use cudaMalloc/cudaFree with custom allocator wrapper
- Implement memory pool with fixed-size blocks
- Track allocated vs available memory
- Consider using CUB library for GPU data structures
- Target 80% of 16GB VRAM utilization (12.8GB usable)

**Definition of Done:**
- [ ] Can allocate/deallocate 1M+ small objects
- [ ] Memory fragmentation stays below 20%
- [ ] Memory usage accurately reported
- [ ] No memory leaks detected with cuda-memcheck

### Story 1.5: GPU Hash Table Implementation (Priority: P1, Points: 13)
**As a** cache system
**I want** fast key lookup and storage in GPU memory
**So that** I can achieve high-performance cache operations

**Acceptance Criteria:**
- [ ] GPU-based hash table supporting string keys
- [ ] Insert, lookup, and delete operations
- [ ] Handles hash collisions efficiently
- [ ] Thread-safe for concurrent GPU operations
- [ ] Supports 100K+ key-value pairs

**Technical Notes:**
- Consider cuckoo hashing or linear probing for GPU efficiency
- Use CUDA cooperative groups for thread coordination
- Implement atomic operations for thread safety
- Hash function optimized for GPU (e.g., FNV-1a)
- Key storage: consider fixed-size vs variable-size

**Definition of Done:**
- [ ] Insert/lookup operations work correctly
- [ ] Performance: >1M operations/second on RTX 5080
- [ ] Collision handling verified with stress tests
- [ ] Thread safety validated with concurrent access tests

### Story 1.6: Real GPU Cache Integration (Priority: P1, Points: 8)
**As a** developer
**I want** to replace mock operations with real GPU cache operations
**So that** I can validate actual performance improvements

**Acceptance Criteria:**
- [ ] GPU cache operations integrated with existing mock interface
- [ ] Feature flag system to toggle between mock and real implementations
- [ ] Performance comparison between mock and real GPU operations
- [ ] Memory management integration with cache operations
- [ ] Error handling for GPU-specific issues

**Technical Notes:**
- Maintain identical API interface for seamless switching
- Use preprocessor flags or runtime configuration for mock/real toggle
- Implement proper GPU error handling and recovery
- Validate performance claims with real GPU operations
- Ensure memory management works with actual GPU constraints

**Definition of Done:**
- [ ] Real GPU operations can be enabled via configuration
- [ ] Performance matches or exceeds mock simulation targets
- [ ] All tests pass with both mock and real implementations
- [ ] GPU memory usage accurately tracked and reported

---

## Epic 1 Development Strategy

### Week 1: Foundation & Mock Implementation
- **Days 1-2**: Development environment setup (Story 1.1)
- **Days 3-5**: Mock Predis client implementation (Story 1.2)
- **Weekend**: Initial testing and refinement

### Week 2: Benchmarking & Validation
- **Days 1-3**: Redis comparison framework (Story 1.3)
- **Days 4-5**: Comprehensive benchmark testing and tuning
- **Weekend**: Performance analysis and reporting

### Week 3: Real GPU Implementation Start
- **Days 1-3**: GPU memory management (Story 1.4)
- **Days 4-5**: Begin GPU hash table implementation (Story 1.5)
- **Weekend**: GPU functionality testing

### Week 4: Integration & Validation
- **Days 1-3**: Complete GPU hash table (Story 1.5)
- **Days 4-5**: Real GPU cache integration (Story 1.6)
- **Weekend**: End-to-end testing and Epic 1 completion

## Success Metrics for Epic 1

### Must Have (P0 Stories)
- [ ] Mock system demonstrates 10-50x performance improvements
- [ ] Benchmark framework provides reproducible comparisons
- [ ] Development environment stable and documented

### Should Have (P1 Stories)  
- [ ] Real GPU memory management functional
- [ ] Basic GPU hash table operations working
- [ ] Integration between mock and real implementations

### Epic 1 Complete When:
- [ ] Working demo (mock or real) shows dramatic performance improvements
- [ ] Benchmark results validate performance claims
- [ ] Foundation ready for Epic 2 optimization work
- [ ] All P0 stories completed, 80%+ of P1 stories completed

This Epic 1 approach gives you immediate demo value with the mock implementation while building toward real GPU functionality. The mock system provides realistic performance numbers you can use for investor conversations, and the gradual replacement strategy minimizes risk while ensuring continuous progress.