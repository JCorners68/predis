#!/usr/bin/env python3
"""
Test suite for Mock Predis Client

Validates performance simulation, memory constraints, and API compatibility
to ensure mock client accurately represents target GPU performance.

Copyright 2025 Predis Project  
Licensed under the Apache License, Version 2.0
"""

import unittest
import time
import sys
import os

# Add src directory to path for mock client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mock_predis_client import MockPredisClient, PrefetchConfig, PerformanceMetrics


class TestMockPredisClient(unittest.TestCase):
    """Test suite for mock Predis client functionality"""
    
    def setUp(self):
        """Set up test client for each test"""
        self.client = MockPredisClient(max_memory_mb=100)  # Small limit for testing
        self.client.connect()
    
    def tearDown(self):
        """Clean up after each test"""
        self.client.disconnect()
    
    def test_connection_management(self):
        """Test connection and disconnection functionality"""
        # Test initial connection
        self.assertTrue(self.client.is_connected())
        
        # Test disconnection
        self.client.disconnect()
        self.assertFalse(self.client.is_connected())
        
        # Test reconnection
        self.assertTrue(self.client.connect("localhost", 6379))
        self.assertTrue(self.client.is_connected())
    
    def test_basic_operations(self):
        """Test basic get/put/remove operations"""
        # Test put and get
        self.assertTrue(self.client.put("test_key", "test_value"))
        self.assertEqual(self.client.get("test_key"), "test_value")
        
        # Test missing key
        self.assertIsNone(self.client.get("missing_key"))
        
        # Test remove
        self.assertTrue(self.client.remove("test_key"))
        self.assertIsNone(self.client.get("test_key"))
        
        # Test remove missing key
        self.assertFalse(self.client.remove("missing_key"))
    
    def test_batch_operations(self):
        """Test batch mget/mput/mdelete operations"""
        # Test mput
        test_data = {f"batch_key_{i}": f"batch_value_{i}" for i in range(10)}
        self.assertTrue(self.client.mput(test_data))
        
        # Test mget
        keys = list(test_data.keys())
        values = self.client.mget(keys)
        expected_values = [test_data[key] for key in keys]
        self.assertEqual(values, expected_values)
        
        # Test mget with missing keys
        mixed_keys = keys[:5] + ["missing_1", "missing_2"]
        mixed_values = self.client.mget(mixed_keys)
        expected_mixed = expected_values[:5] + [None, None]
        self.assertEqual(mixed_values, expected_mixed)
        
        # Test mdelete
        removed_count = self.client.mdelete(keys[:5])
        self.assertEqual(removed_count, 5)
        
        # Verify deletion
        remaining_values = self.client.mget(keys)
        expected_remaining = [None] * 5 + expected_values[5:]
        self.assertEqual(remaining_values, expected_remaining)
    
    def test_ttl_functionality(self):
        """Test time-to-live functionality"""
        # Store key with TTL
        self.client.put("ttl_key", "ttl_value", ttl=1)
        self.assertEqual(self.client.get("ttl_key"), "ttl_value")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Key should be expired (cleanup happens on next operation)
        self.client.put("trigger_cleanup", "value")  # Trigger cleanup
        self.assertIsNone(self.client.get("ttl_key"))
    
    def test_memory_constraints(self):
        """Test GPU memory limit simulation"""
        # Fill up to near memory limit
        large_value = "x" * (1024 * 1024)  # 1MB value
        stored_count = 0
        
        try:
            for i in range(200):  # Try to exceed 100MB limit with 1MB values
                self.client.put(f"large_key_{i}", large_value)
                stored_count += 1
        except MemoryError:
            # Expected when hitting memory limit
            pass
        
        # Should have hit memory limit before storing all 200 keys
        self.assertLess(stored_count, 200)
        
        # Memory usage should be tracked accurately
        stats = self.client.get_stats()
        self.assertGreater(stats.memory_usage_mb, 0)
        self.assertLessEqual(stats.memory_usage_mb, 100)
    
    def test_performance_simulation(self):
        """Test that performance targets are met"""
        # Test single operation performance
        operations = 1000
        start_time = time.time()
        
        for i in range(operations):
            self.client.put(f"perf_key_{i}", f"perf_value_{i}")
        
        for i in range(0, operations, 10):  # Sample every 10th key
            self.client.get(f"perf_key_{i}")
        
        elapsed_time = time.time() - start_time
        total_ops = operations + (operations // 10)
        ops_per_second = total_ops / elapsed_time
        
        # Should achieve good performance with mock implementation  
        self.assertGreater(ops_per_second, 100_000)  # Reasonable for Python mock
        
        # Test batch operation performance
        batch_data = {f"batch_perf_{i}": f"batch_value_{i}" for i in range(1000)}
        
        start_time = time.time()
        self.client.mput(batch_data)
        self.client.mget(list(batch_data.keys()))
        elapsed_time = time.time() - start_time
        
        batch_ops_per_second = 2000 / elapsed_time  # 2000 total ops (1000 put + 1000 get)
        
        # Batch operations should be significantly faster
        self.assertGreater(batch_ops_per_second, ops_per_second)
        self.assertGreater(batch_ops_per_second, 500_000)  # Should be faster than single ops
    
    def test_statistics_collection(self):
        """Test performance statistics collection"""
        # Perform some operations
        self.client.put("stat_key_1", "value_1")
        self.client.put("stat_key_2", "value_2")
        self.client.get("stat_key_1")  # Hit
        self.client.get("nonexistent")  # Miss
        
        stats = self.client.get_stats()
        
        # Verify statistics structure
        self.assertIsInstance(stats, PerformanceMetrics)
        self.assertGreater(stats.operations_per_second, 0)
        self.assertGreater(stats.avg_latency_ms, 0)
        self.assertEqual(stats.total_keys, 2)
        self.assertEqual(stats.hit_ratio, 0.5)  # 1 hit out of 2 get operations
        self.assertGreater(stats.memory_usage_mb, 0)
    
    def test_prefetch_configuration(self):
        """Test ML prefetching configuration"""
        # Test default configuration
        default_config = self.client.get_prefetch_config()
        self.assertIsInstance(default_config, PrefetchConfig)
        self.assertTrue(default_config.enabled)
        self.assertEqual(default_config.confidence_threshold, 0.7)
        
        # Test configuration update
        new_config = PrefetchConfig(
            enabled=False,
            confidence_threshold=0.8,
            max_prefetch_keys=150
        )
        self.client.configure_prefetching(new_config)
        
        updated_config = self.client.get_prefetch_config()
        self.assertFalse(updated_config.enabled)
        self.assertEqual(updated_config.confidence_threshold, 0.8)
        self.assertEqual(updated_config.max_prefetch_keys, 150)
        
        # Test dictionary-based configuration
        dict_config = {
            "enabled": True,
            "confidence_threshold": 0.9,
            "max_prefetch_keys": 100
        }
        self.client.configure_prefetching(dict_config)
        
        final_config = self.client.get_prefetch_config()
        self.assertTrue(final_config.enabled)
        self.assertEqual(final_config.confidence_threshold, 0.9)
        self.assertEqual(final_config.max_prefetch_keys, 100)
    
    def test_ml_hints(self):
        """Test ML hint functionality"""
        # Test related keys hint
        related_keys = ["user_123_profile", "user_123_preferences", "user_123_history"]
        self.client.hint_related_keys(related_keys)
        
        # Test sequence hint
        sequence_keys = ["step_1", "step_2", "step_3", "step_4"]
        self.client.hint_sequence(sequence_keys)
        
        # Verify hints are logged (should not raise errors)
        info = self.client.info()
        self.assertGreater(info['access_patterns']['total_accesses'], 0)
    
    def test_info_output(self):
        """Test comprehensive info output"""
        # Perform some operations to generate data
        self.client.put("info_key", "info_value")
        self.client.get("info_key")
        
        info = self.client.info()
        
        # Verify info structure
        required_sections = ['version', 'server_info', 'gpu_info', 'performance', 
                           'memory', 'prefetching', 'access_patterns']
        for section in required_sections:
            self.assertIn(section, info)
        
        # Verify GPU info
        gpu_info = info['gpu_info']
        self.assertEqual(gpu_info['model'], 'RTX 5080 (Simulated)')
        self.assertEqual(gpu_info['vram_total_gb'], 100 / 1024)  # Test limit converted to GB
        
        # Verify performance info
        perf_info = info['performance']
        self.assertEqual(perf_info['single_op_speedup'], '12.0x vs Redis')
        self.assertEqual(perf_info['batch_op_speedup'], '28.0x vs Redis')
        self.assertGreater(perf_info['target_single_ops_sec'], 3_000_000)  # 12x * 275K Redis baseline
        self.assertGreater(perf_info['target_batch_ops_sec'], 7_000_000)   # 28x * 275K Redis baseline
    
    def test_flush_all(self):
        """Test cache flush functionality"""
        # Add some data
        test_data = {f"flush_key_{i}": f"flush_value_{i}" for i in range(10)}
        self.client.mput(test_data)
        
        # Verify data exists
        self.assertEqual(len(self.client.data), 10)
        
        # Flush all data
        self.assertTrue(self.client.flush_all())
        
        # Verify all data is gone
        self.assertEqual(len(self.client.data), 0)
        stats = self.client.get_stats()
        self.assertEqual(stats.total_keys, 0)
        self.assertEqual(stats.memory_usage_mb, 0)
    
    def test_error_conditions(self):
        """Test error handling for various conditions"""
        # Test operations without connection
        disconnected_client = MockPredisClient()
        
        with self.assertRaises(ConnectionError):
            disconnected_client.get("test")
        
        with self.assertRaises(ConnectionError):
            disconnected_client.put("test", "value")
        
        with self.assertRaises(ConnectionError):
            disconnected_client.mget(["test"])
        
        # Test memory limit exceeded (100MB test limit)
        huge_value = "x" * (200 * 1024 * 1024)  # 200MB value, exceeds 100MB limit
        
        with self.assertRaises(MemoryError):
            self.client.put("huge_key", huge_value)
        
        # Test batch memory limit exceeded
        large_batch = {f"key_{i}": "x" * (10 * 1024 * 1024) for i in range(20)}  # 200MB total
        
        with self.assertRaises(MemoryError):
            self.client.mput(large_batch)


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmark tests to validate simulation accuracy"""
    
    def setUp(self):
        """Set up benchmark client"""
        self.client = MockPredisClient(max_memory_mb=1000)  # 1GB for benchmarks
        self.client.connect()
    
    def tearDown(self):
        """Clean up benchmark client"""
        self.client.disconnect()
    
    def test_single_operation_benchmark(self):
        """Benchmark single operation performance"""
        operations = 10000
        
        # Benchmark PUTs
        start_time = time.time()
        for i in range(operations):
            self.client.put(f"bench_put_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        # Benchmark GETs
        start_time = time.time()
        for i in range(operations):
            self.client.get(f"bench_put_{i}")
        get_time = time.time() - start_time
        
        put_ops_per_sec = operations / put_time
        get_ops_per_sec = operations / get_time
        
        print(f"\nSingle Operation Benchmark:")
        print(f"PUT: {put_ops_per_sec:,.0f} ops/sec")
        print(f"GET: {get_ops_per_sec:,.0f} ops/sec")
        
        # Should exceed Redis baseline (275K ops/sec) significantly (mock without sleep)
        self.assertGreater(put_ops_per_sec, 500_000)  # More realistic for mock without sleep
        self.assertGreater(get_ops_per_sec, 500_000)
    
    def test_batch_operation_benchmark(self):
        """Benchmark batch operation performance"""
        batch_size = 1000
        iterations = 10
        
        total_put_time = 0
        total_get_time = 0
        
        for i in range(iterations):
            batch_data = {f"batch_{i}_{j}": f"value_{i}_{j}" for j in range(batch_size)}
            
            # Benchmark batch PUT
            start_time = time.time()
            self.client.mput(batch_data)
            total_put_time += time.time() - start_time
            
            # Benchmark batch GET
            start_time = time.time()
            self.client.mget(list(batch_data.keys()))
            total_get_time += time.time() - start_time
        
        total_operations = batch_size * iterations
        batch_put_ops_per_sec = total_operations / total_put_time
        batch_get_ops_per_sec = total_operations / total_get_time
        
        print(f"\nBatch Operation Benchmark:")
        print(f"MPUT: {batch_put_ops_per_sec:,.0f} ops/sec")
        print(f"MGET: {batch_get_ops_per_sec:,.0f} ops/sec")
        
        # Batch operations should significantly exceed single operation performance
        self.assertGreater(batch_put_ops_per_sec, 1_000_000)  # Should be faster than single ops
        self.assertGreater(batch_get_ops_per_sec, 1_000_000)
    
    def test_memory_efficiency_benchmark(self):
        """Test memory usage efficiency"""
        # Store various sizes of data
        small_data = {f"small_{i}": "x" * 10 for i in range(1000)}
        medium_data = {f"medium_{i}": "x" * 100 for i in range(1000)}
        large_data = {f"large_{i}": "x" * 1000 for i in range(100)}
        
        # Store all data
        self.client.mput(small_data)
        stats_after_small = self.client.get_stats()
        
        self.client.mput(medium_data)
        stats_after_medium = self.client.get_stats()
        
        self.client.mput(large_data)
        stats_after_large = self.client.get_stats()
        
        print(f"\nMemory Efficiency Benchmark:")
        print(f"After 1K small (10B): {stats_after_small.memory_usage_mb:.2f} MB")
        print(f"After 1K medium (100B): {stats_after_medium.memory_usage_mb:.2f} MB")
        print(f"After 100 large (1KB): {stats_after_large.memory_usage_mb:.2f} MB")
        print(f"Total keys: {stats_after_large.total_keys}")
        
        # Memory usage should increase appropriately
        self.assertGreater(stats_after_medium.memory_usage_mb, stats_after_small.memory_usage_mb)
        self.assertGreater(stats_after_large.memory_usage_mb, stats_after_medium.memory_usage_mb)
        self.assertEqual(stats_after_large.total_keys, 2100)  # 1000 + 1000 + 100


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)