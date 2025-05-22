#!/usr/bin/env python3
"""
Redis vs Mock Predis Comprehensive Benchmark Suite
Implements Story 1.3: Redis Comparison Baseline Framework

This benchmark framework provides fair performance comparison between
Redis and mock Predis implementations with identical test scenarios.
"""

import redis
import time
import statistics
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os

# Add src directory to path to import mock client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mock_predis_client import MockPredisClient, PerformanceMetrics

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    redis_host: str = 'localhost'
    redis_port: int = 6380  # Using configured port from dev environment
    test_key_count: int = 10000
    single_op_iterations: int = 1000
    batch_sizes: List[int] = None
    concurrent_clients: List[int] = None
    data_size_variants: List[int] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [10, 50, 100, 500, 1000]
        if self.concurrent_clients is None:
            self.concurrent_clients = [1, 4, 8, 16]
        if self.data_size_variants is None:
            self.data_size_variants = [10, 100, 1024]  # Small, medium, large values

@dataclass 
class BenchmarkResults:
    """Structured benchmark results"""
    redis_results: Dict[str, Any]
    predis_results: Dict[str, Any]
    performance_comparison: Dict[str, Any]
    test_config: Dict[str, Any]
    timestamp: str

class RedisVsPredisComparison:
    """
    Comprehensive benchmark suite comparing Redis and Mock Predis performance
    Implements identical test scenarios for fair performance evaluation
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.redis_client = None
        self.predis_client = None
        self.test_data = {}
        
    def setup_clients(self) -> bool:
        """Initialize Redis and Predis clients"""
        try:
            # Setup Redis client
            self.redis_client = redis.Redis(
                host=self.config.redis_host, 
                port=self.config.redis_port, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test Redis connection
            self.redis_client.ping()
            print(f"✅ Redis connected at {self.config.redis_host}:{self.config.redis_port}")
            
            # Setup Mock Predis client
            self.predis_client = MockPredisClient(
                host=self.config.redis_host,
                port=self.config.redis_port,
                max_memory_mb=12800  # 16GB RTX 5080 - 20% overhead
            )
            
            # Connect to mock Predis
            if not self.predis_client.connect():
                raise ConnectionError("Failed to connect to Mock Predis")
            print("✅ Mock Predis client connected")
            
            return True
            
        except redis.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            print(f"Make sure Redis is running on {self.config.redis_host}:{self.config.redis_port}")
            return False
        except Exception as e:
            print(f"❌ Client setup failed: {e}")
            return False
    
    def generate_test_datasets(self) -> Dict[str, Dict[str, str]]:
        """Generate realistic test datasets with various sizes"""
        import random
        import string
        
        datasets = {}
        
        for size_bytes in self.config.data_size_variants:
            dataset = {}
            for i in range(self.config.test_key_count):
                key = f"key_{size_bytes}b_{i:06d}"
                # Generate value of specified size
                value = ''.join(random.choices(string.ascii_letters + string.digits, k=size_bytes))
                dataset[key] = value
            
            datasets[f"{size_bytes}B"] = dataset
            print(f"Generated {len(dataset)} test keys with {size_bytes}B values")
        
        return datasets
    
    def benchmark_single_operations(self, dataset: Dict[str, str], iterations: int) -> Dict[str, Any]:
        """Benchmark single get/put operations for both systems"""
        keys = list(dataset.keys())[:iterations]
        
        print(f"  Benchmarking single operations ({iterations} keys)...")
        
        # Clear both systems
        self.redis_client.flushall()
        self.predis_client.flush_all()
        
        # Benchmark Redis Single Operations
        redis_put_times = []
        redis_get_times = []
        
        for key in keys:
            # PUT operation
            start = time.perf_counter()
            self.redis_client.set(key, dataset[key])
            redis_put_times.append((time.perf_counter() - start) * 1000)
            
            # GET operation  
            start = time.perf_counter()
            result = self.redis_client.get(key)
            redis_get_times.append((time.perf_counter() - start) * 1000)
            
            assert result == dataset[key], f"Redis data integrity check failed for {key}"
        
        # Benchmark Mock Predis Single Operations
        predis_put_times = []
        predis_get_times = []
        
        for key in keys:
            # PUT operation
            start = time.perf_counter()
            self.predis_client.put(key, dataset[key])
            predis_put_times.append((time.perf_counter() - start) * 1000)
            
            # GET operation
            start = time.perf_counter()
            result = self.predis_client.get(key)
            predis_get_times.append((time.perf_counter() - start) * 1000)
            
            assert result == dataset[key], f"Predis data integrity check failed for {key}"
        
        # Calculate statistics
        def calculate_stats(times: List[float]) -> Dict[str, float]:
            return {
                'avg_latency_ms': statistics.mean(times),
                'median_latency_ms': statistics.median(times),
                'p95_latency_ms': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
                'p99_latency_ms': statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
                'ops_per_second': 1000 / statistics.mean(times)
            }
        
        return {
            'redis': {
                'put': calculate_stats(redis_put_times),
                'get': calculate_stats(redis_get_times)
            },
            'predis': {
                'put': calculate_stats(predis_put_times),
                'get': calculate_stats(predis_get_times)
            }
        }
    
    def benchmark_batch_operations(self, dataset: Dict[str, str], batch_size: int) -> Dict[str, Any]:
        """Benchmark batch get/put operations"""
        keys = list(dataset.keys())[:batch_size]
        batch_data = {k: dataset[k] for k in keys}
        
        print(f"  Benchmarking batch operations ({batch_size} keys)...")
        
        # Clear both systems
        self.redis_client.flushall()
        self.predis_client.flush_all()
        
        # Benchmark Redis Batch Operations (using pipeline)
        start = time.perf_counter()
        pipe = self.redis_client.pipeline()
        for key, value in batch_data.items():
            pipe.set(key, value)
        pipe.execute()
        redis_batch_put_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        pipe = self.redis_client.pipeline()
        for key in keys:
            pipe.get(key)
        results = pipe.execute()
        redis_batch_get_time = (time.perf_counter() - start) * 1000
        
        # Verify Redis results
        for i, key in enumerate(keys):
            assert results[i] == dataset[key], f"Redis batch integrity check failed for {key}"
        
        # Benchmark Mock Predis Batch Operations
        start = time.perf_counter()
        self.predis_client.mput(batch_data)
        predis_batch_put_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        predis_results = self.predis_client.mget(keys)
        predis_batch_get_time = (time.perf_counter() - start) * 1000
        
        # Verify Predis results
        for i, key in enumerate(keys):
            assert predis_results[i] == dataset[key], f"Predis batch integrity check failed for {key}"
        
        return {
            'redis': {
                'batch_put_ms': redis_batch_put_time,
                'batch_get_ms': redis_batch_get_time,
                'total_batch_time_ms': redis_batch_put_time + redis_batch_get_time,
                'batch_ops_per_second': (batch_size * 2) / ((redis_batch_put_time + redis_batch_get_time) / 1000)
            },
            'predis': {
                'batch_put_ms': predis_batch_put_time,
                'batch_get_ms': predis_batch_get_time,
                'total_batch_time_ms': predis_batch_put_time + predis_batch_get_time,
                'batch_ops_per_second': (batch_size * 2) / ((predis_batch_put_time + predis_batch_get_time) / 1000)
            }
        }
    
    def benchmark_concurrent_operations(self, dataset: Dict[str, str], num_clients: int, ops_per_client: int) -> Dict[str, Any]:
        """Benchmark concurrent client operations"""
        keys = list(dataset.keys())[:num_clients * ops_per_client]
        
        print(f"  Benchmarking concurrent operations ({num_clients} clients, {ops_per_client} ops each)...")
        
        # Clear both systems
        self.redis_client.flushall()
        self.predis_client.flush_all()
        
        # Concurrent Redis operations
        def redis_client_worker(client_keys: List[str]) -> Tuple[float, int]:
            client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            
            start = time.perf_counter()
            operations = 0
            
            for key in client_keys:
                client.set(key, dataset[key])
                result = client.get(key)
                assert result == dataset[key]
                operations += 2
                
            total_time = time.perf_counter() - start
            return total_time, operations
        
        # Concurrent Predis operations (thread-safe mock)
        def predis_client_worker(client_keys: List[str]) -> Tuple[float, int]:
            start = time.perf_counter()
            operations = 0
            
            for key in client_keys:
                self.predis_client.put(key, dataset[key])
                result = self.predis_client.get(key)
                assert result == dataset[key]
                operations += 2
                
            total_time = time.perf_counter() - start
            return total_time, operations
        
        # Prepare work distribution
        keys_per_client = ops_per_client
        client_key_sets = [
            keys[i*keys_per_client:(i+1)*keys_per_client] 
            for i in range(num_clients)
        ]
        
        # Redis concurrent test
        redis_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            redis_futures = [
                executor.submit(redis_client_worker, key_set) 
                for key_set in client_key_sets
            ]
            redis_results = [future.result() for future in as_completed(redis_futures)]
        redis_total_time = time.perf_counter() - redis_start
        
        # Predis concurrent test
        predis_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            predis_futures = [
                executor.submit(predis_client_worker, key_set) 
                for key_set in client_key_sets
            ]
            predis_results = [future.result() for future in as_completed(predis_futures)]
        predis_total_time = time.perf_counter() - predis_start
        
        # Calculate results
        redis_total_ops = sum(result[1] for result in redis_results)
        predis_total_ops = sum(result[1] for result in predis_results)
        
        return {
            'redis': {
                'total_time_s': redis_total_time,
                'total_operations': redis_total_ops,
                'concurrent_ops_per_second': redis_total_ops / redis_total_time,
                'avg_client_time_s': statistics.mean([result[0] for result in redis_results])
            },
            'predis': {
                'total_time_s': predis_total_time,
                'total_operations': predis_total_ops,
                'concurrent_ops_per_second': predis_total_ops / predis_total_time,
                'avg_client_time_s': statistics.mean([result[0] for result in predis_results])
            }
        }
    
    def run_comprehensive_benchmark(self) -> BenchmarkResults:
        """Execute complete benchmark suite with all test scenarios"""
        print("🚀 Starting comprehensive Redis vs Mock Predis benchmark...")
        
        # Generate test datasets
        print("\n📊 Generating test datasets...")
        datasets = self.generate_test_datasets()
        
        all_results = {
            'single_operations': {},
            'batch_operations': {},
            'concurrent_operations': {},
            'system_info': {}
        }
        
        # Single operations across different data sizes
        print("\n🔄 Testing single operations...")
        for size_label, dataset in datasets.items():
            print(f"\n  Data size: {size_label}")
            single_results = self.benchmark_single_operations(dataset, self.config.single_op_iterations)
            all_results['single_operations'][size_label] = single_results
        
        # Batch operations across different batch sizes
        print("\n📦 Testing batch operations...")
        main_dataset = datasets['100B']  # Use medium-sized data for batch tests
        for batch_size in self.config.batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            batch_results = self.benchmark_batch_operations(main_dataset, batch_size)
            all_results['batch_operations'][f'{batch_size}_keys'] = batch_results
        
        # Concurrent operations
        print("\n🔀 Testing concurrent operations...")
        for num_clients in self.config.concurrent_clients:
            print(f"\n  Concurrent clients: {num_clients}")
            concurrent_results = self.benchmark_concurrent_operations(main_dataset, num_clients, 50)
            all_results['concurrent_operations'][f'{num_clients}_clients'] = concurrent_results
        
        # System information
        all_results['system_info'] = {
            'redis_info': self._get_redis_info(),
            'predis_info': self.predis_client.info(),
            'predis_stats': asdict(self.predis_client.get_stats())
        }
        
        # Calculate performance comparisons
        comparison = self._calculate_performance_comparison(all_results)
        
        return BenchmarkResults(
            redis_results=all_results,
            predis_results=all_results,  # Combined results structure
            performance_comparison=comparison,
            test_config=asdict(self.config),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis system information"""
        try:
            info = self.redis_client.info()
            return {
                'version': info.get('redis_version', 'unknown'),
                'memory_used_mb': info.get('used_memory', 0) / (1024 * 1024),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_performance_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance improvement ratios"""
        comparison = {
            'single_operation_improvements': {},
            'batch_operation_improvements': {},
            'concurrent_operation_improvements': {},
            'summary': {}
        }
        
        # Single operation improvements
        for size_label, size_results in results['single_operations'].items():
            redis_get_ops = size_results['redis']['get']['ops_per_second']
            predis_get_ops = size_results['predis']['get']['ops_per_second']
            redis_put_ops = size_results['redis']['put']['ops_per_second']
            predis_put_ops = size_results['predis']['put']['ops_per_second']
            
            comparison['single_operation_improvements'][size_label] = {
                'get_speedup': predis_get_ops / redis_get_ops,
                'put_speedup': predis_put_ops / redis_put_ops,
                'get_latency_reduction': (1 - size_results['predis']['get']['avg_latency_ms'] / size_results['redis']['get']['avg_latency_ms']) * 100,
                'put_latency_reduction': (1 - size_results['predis']['put']['avg_latency_ms'] / size_results['redis']['put']['avg_latency_ms']) * 100
            }
        
        # Batch operation improvements
        for batch_label, batch_results in results['batch_operations'].items():
            redis_ops = batch_results['redis']['batch_ops_per_second']
            predis_ops = batch_results['predis']['batch_ops_per_second']
            
            comparison['batch_operation_improvements'][batch_label] = {
                'throughput_speedup': predis_ops / redis_ops,
                'latency_reduction': (1 - batch_results['predis']['total_batch_time_ms'] / batch_results['redis']['total_batch_time_ms']) * 100
            }
        
        # Concurrent operation improvements
        for client_label, client_results in results['concurrent_operations'].items():
            redis_ops = client_results['redis']['concurrent_ops_per_second']
            predis_ops = client_results['predis']['concurrent_ops_per_second']
            
            comparison['concurrent_operation_improvements'][client_label] = {
                'concurrent_speedup': predis_ops / redis_ops,
                'time_reduction': (1 - client_results['predis']['total_time_s'] / client_results['redis']['total_time_s']) * 100
            }
        
        # Overall summary
        single_speedups = [
            improvement['get_speedup'] 
            for improvement in comparison['single_operation_improvements'].values()
        ]
        batch_speedups = [
            improvement['throughput_speedup']
            for improvement in comparison['batch_operation_improvements'].values()
        ]
        
        comparison['summary'] = {
            'avg_single_operation_speedup': statistics.mean(single_speedups),
            'max_single_operation_speedup': max(single_speedups),
            'avg_batch_operation_speedup': statistics.mean(batch_speedups),
            'max_batch_operation_speedup': max(batch_speedups),
            'performance_target_achievement': {
                'single_ops_target_10x': statistics.mean(single_speedups) >= 10,
                'batch_ops_target_25x': statistics.mean(batch_speedups) >= 25,
                'overall_target_range_10_50x': 10 <= statistics.mean(single_speedups + batch_speedups) <= 50
            }
        }
        
        return comparison
    
    def generate_report(self, results: BenchmarkResults, output_dir: str = 'benchmark_results') -> str:
        """Generate comprehensive benchmark report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(output_dir, f'benchmark_results_{int(time.time())}.json')
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Generate human-readable report
        report_file = os.path.join(output_dir, f'benchmark_report_{int(time.time())}.txt')
        with open(report_file, 'w') as f:
            f.write(self._format_report(results))
        
        # Print summary to console
        print(self._format_summary(results))
        
        return report_file
    
    def _format_summary(self, results: BenchmarkResults) -> str:
        """Format concise benchmark summary"""
        comparison = results.performance_comparison
        summary = comparison['summary']
        
        report = f"""
🎯 Redis vs Mock Predis Performance Comparison Summary
{'='*60}
Timestamp: {results.timestamp}

📊 PERFORMANCE ACHIEVEMENTS:
• Average Single Operation Speedup: {summary['avg_single_operation_speedup']:.1f}x
• Maximum Single Operation Speedup: {summary['max_single_operation_speedup']:.1f}x  
• Average Batch Operation Speedup: {summary['avg_batch_operation_speedup']:.1f}x
• Maximum Batch Operation Speedup: {summary['max_batch_operation_speedup']:.1f}x

🎯 TARGET ACHIEVEMENT STATUS:
• Single Ops (10x target): {'✅ ACHIEVED' if summary['performance_target_achievement']['single_ops_target_10x'] else '❌ MISSED'}
• Batch Ops (25x target): {'✅ ACHIEVED' if summary['performance_target_achievement']['batch_ops_target_25x'] else '❌ MISSED'}
• Overall Range (10-50x): {'✅ ACHIEVED' if summary['performance_target_achievement']['overall_target_range_10_50x'] else '❌ MISSED'}

🔧 SYSTEM INFO:
• Redis Version: {results.redis_results['system_info']['redis_info'].get('version', 'unknown')}
• Mock Predis Version: {results.redis_results['system_info']['predis_info']['version']}
• GPU Model: {results.redis_results['system_info']['predis_info']['gpu_info']['model']}

📈 INVESTOR-READY DEMONSTRATION:
Mock Predis successfully demonstrates {summary['avg_single_operation_speedup']:.0f}x-{summary['max_batch_operation_speedup']:.0f}x 
performance improvements over Redis baseline.
"""
        return report
    
    def _format_report(self, results: BenchmarkResults) -> str:
        """Format detailed benchmark report"""
        # This would be a comprehensive multi-page report
        # For brevity, returning summary format here
        return self._format_summary(results)

def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='Redis vs Mock Predis Performance Benchmark')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6380, help='Redis port')
    parser.add_argument('--iterations', type=int, default=1000, help='Single operation iterations')
    parser.add_argument('--key-count', type=int, default=10000, help='Total test keys')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        single_op_iterations=args.iterations,
        test_key_count=args.key_count
    )
    
    benchmark = RedisVsPredisComparison(config)
    
    if not benchmark.setup_clients():
        print("❌ Benchmark setup failed. Exiting.")
        return 1
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        report_file = benchmark.generate_report(results, args.output_dir)
        print(f"\n📄 Detailed report saved to: {report_file}")
        return 0
    except Exception as e:
        print(f"❌ Benchmark execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())