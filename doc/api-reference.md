# Predis API Reference

## Overview

Predis is a GPU-accelerated key-value cache with predictive prefetching capabilities, leveraging machine learning models to anticipate access patterns and optimize data availability.

This document provides a comprehensive reference for the Predis client APIs, configuration options, and advanced features.

## Client SDKs

Predis offers client libraries in multiple languages, all built upon the core C++/Mojo implementation.

* **Core languages**: C++, Python
* **Additional language bindings**: Java, Go, Node.js

## Core API

### Initialization

> **🎯 DEMO ESSENTIAL**: Required to connect to Predis server for benchmarking

```python
# Python
from predis import PredisClient

# Connect to a local Predis server
client = PredisClient(host='localhost', port=6379)

# Connect with authentication 
client = PredisClient(host='localhost', port=6379, password='your_password')

# Connect with custom configuration
client = PredisClient(
    host='localhost', 
    port=6379,
    consistency_level='strong',
    enable_prefetching=True,
    prefetch_confidence_threshold=0.75,
    max_pool_connections=16
)
```

```cpp
// C++
#include <predis_cpp.h>

// Connect to a local Predis server
predis::PredisClient client("localhost", 6379);

// Connect with authentication
predis::PredisClient client("localhost", 6379, "your_password");

// Connect with custom configuration
predis::PredisClientConfig config;
config.consistency_level = predis::ConsistencyLevel::STRONG;
config.enable_prefetching = true;
config.prefetch_confidence_threshold = 0.75;
config.max_pool_connections = 16;

predis::PredisClient client("localhost", 6379, config);
```

### Basic Operations

> **🎯 DEMO ESSENTIAL**: These operations are absolutely required for the basic performance demo

#### Get

Retrieves a value by its key.

```python
# Python
value = client.get("mykey")
if value is not None:
    print(f"Found value: {value}")
else:
    print("Key not found")
    
# Get multiple keys at once
values = client.mget(["key1", "key2", "key3"])
```

```cpp
// C++
auto value = client.get("mykey");
if (value.has_value()) {
    std::cout << "Found value: " << value.value() << std::endl;
} else {
    std::cout << "Key not found" << std::endl;
}

// Get multiple keys at once
auto values = client.mget({"key1", "key2", "key3"});
```

#### Put

> **🎯 DEMO ESSENTIAL**: Core operation needed for load testing

Stores a key-value pair, with an optional time-to-live.

```python
# Python
# Store without expiration
client.put("mykey", "myvalue")

# Store with TTL (in seconds)
client.put("mykey", "myvalue", ttl=60)  # Expires in 60 seconds

# Store multiple key-value pairs
client.mput({
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
})
```

```cpp
// C++
// Store without expiration
client.put("mykey", "myvalue");

// Store with TTL (in seconds)
client.put("mykey", "myvalue", 60);  // Expires in 60 seconds

// Store multiple key-value pairs
std::unordered_map<std::string, std::string> items = {
    {"key1", "value1"},
    {"key2", "value2"},
    {"key3", "value3"}
};
client.mput(items);
```

#### Delete

> **🎯 DEMO USEFUL**: Helpful for resetting test state between benchmark runs

Removes a key-value pair from the cache.

```python
# Python
client.delete("mykey")

# Delete multiple keys
client.mdelete(["key1", "key2", "key3"])
```

```cpp
// C++
client.delete("mykey");

// Delete multiple keys
client.mdelete({"key1", "key2", "key3"});
```

### Consistency Control

> **🎯 DEMO PHASE 3**: For enterprise demos showing production readiness

Configure the consistency level for operations.

```python
# Python
# Set default consistency level for all operations in this client
client.set_consistency_level(client.CONSISTENCY_STRONG)

# Alternatively:
client.set_consistency_level(client.CONSISTENCY_RELAXED)

# For operation-specific consistency
with client.consistency(client.CONSISTENCY_STRONG):
    client.put("important_key", "critical_value")
    value = client.get("another_important_key")
```

```cpp
// C++
// Set default consistency level for all operations in this client
client.setConsistencyLevel(predis::ConsistencyLevel::STRONG);

// Alternatively:
client.setConsistencyLevel(predis::ConsistencyLevel::RELAXED);

// For operation-specific consistency
{
    predis::ConsistencyScope scope(client, predis::ConsistencyLevel::STRONG);
    client.put("important_key", "critical_value");
    auto value = client.get("another_important_key");
}
```

### Namespace Management

Organize keys into namespaces for better organization and control.

```python
# Python
# Create/access a namespace
users_ns = client.namespace("users")

# Operate within namespace
users_ns.put("user:1001", user_data)
user = users_ns.get("user:1001")

# Set namespace-specific consistency
users_ns.set_consistency_level(client.CONSISTENCY_STRONG)

# Set namespace-specific prefetching settings
users_ns.configure_prefetching(
    enabled=True,
    confidence_threshold=0.8,
    max_prefetch_keys=100
)
```

```cpp
// C++
// Create/access a namespace
auto users_ns = client.namespace("users");

// Operate within namespace
users_ns.put("user:1001", user_data);
auto user = users_ns.get("user:1001");

// Set namespace-specific consistency
users_ns.setConsistencyLevel(predis::ConsistencyLevel::STRONG);

// Set namespace-specific prefetching settings
predis::PrefetchConfig config;
config.enabled = true;
config.confidence_threshold = 0.8;
config.max_prefetch_keys = 100;
users_ns.configurePrefetching(config);
```

### Advanced Operations

#### Atomic Operations

```python
# Python
# Increment a counter
new_value = client.increment("counter")
new_value = client.increment("counter", amount=5)

# Decrement a counter
new_value = client.decrement("counter")
new_value = client.decrement("counter", amount=3)

# Compare and set
success = client.compare_and_set("key", expected_value="old", new_value="new")
```

```cpp
// C++
// Increment a counter
int64_t new_value = client.increment("counter");
new_value = client.increment("counter", 5);

// Decrement a counter
new_value = client.decrement("counter");
new_value = client.decrement("counter", 3);

// Compare and set
bool success = client.compareAndSet("key", "old", "new");
```

#### Bulk Operations

> **🎯 DEMO HIGH-IMPACT**: These operations will show the biggest GPU parallelism advantages over Redis

```python
# Python
# Execute multiple operations in a single call
results = client.execute_batch([
    {"op": "put", "key": "key1", "value": "value1"},
    {"op": "get", "key": "key2"},
    {"op": "delete", "key": "key3"},
    {"op": "increment", "key": "counter", "amount": 10}
])
```

```cpp
// C++
// Execute multiple operations in a single call
std::vector<predis::Operation> operations = {
    predis::Operation::put("key1", "value1"),
    predis::Operation::get("key2"),
    predis::Operation::delete_("key3"),
    predis::Operation::increment("counter", 10)
};

auto results = client.executeBatch(operations);
```

### Predictive Prefetching Configuration

> **🎯 DEMO PHASE 2**: Essential for demonstrating ML-driven advantages over traditional caching

Configure and control the predictive prefetching behavior.

```python
# Python
# Get current prefetching status
prefetch_status = client.get_prefetch_status()
print(f"Enabled: {prefetch_status['enabled']}")
print(f"Model: {prefetch_status['active_model']}")
print(f"Hit rate improvement: {prefetch_status['hit_rate_improvement']}%")

# Configure prefetching globally
client.configure_prefetching(
    enabled=True,
    confidence_threshold=0.7,  # Prefetch keys with ≥70% confidence
    max_prefetch_keys=200,     # Max keys to prefetch at once
    max_prefetch_size_mb=100,  # Max size of prefetched data
    prefetch_ttl=30            # TTL for prefetched keys (seconds)
)

# Disable prefetching
client.configure_prefetching(enabled=False)
```

```cpp
// C++
// Get current prefetching status
auto prefetch_status = client.getPrefetchStatus();
std::cout << "Enabled: " << prefetch_status.enabled << std::endl;
std::cout << "Model: " << prefetch_status.active_model << std::endl;
std::cout << "Hit rate improvement: " << prefetch_status.hit_rate_improvement << "%" << std::endl;

// Configure prefetching globally
predis::PrefetchConfig config;
config.enabled = true;
config.confidence_threshold = 0.7;  // Prefetch keys with ≥70% confidence
config.max_prefetch_keys = 200;     // Max keys to prefetch at once
config.max_prefetch_size_mb = 100;  // Max size of prefetched data 
config.prefetch_ttl = 30;           // TTL for prefetched keys (seconds)
client.configurePrefetching(config);

// Disable prefetching
predis::PrefetchConfig disable_config;
disable_config.enabled = false;
client.configurePrefetching(disable_config);
```

### Hints and Annotations

Provide hints to improve prediction accuracy.

```python
# Python
# Hint a relationship between keys
client.hint_related_keys(["user:1001", "preferences:1001", "cart:1001"])

# Hint a sequence of keys that are likely to be accessed in order
client.hint_sequence(["step1", "step2", "step3", "step4"])

# Annotate a key with metadata to influence prefetching
client.put("image:large:1", image_data, annotations={
    "high_value": True,        # Mark as high-value content
    "access_pattern": "bursty" # Describe expected access pattern
})
```

```cpp
// C++
// Hint a relationship between keys
client.hintRelatedKeys({"user:1001", "preferences:1001", "cart:1001"});

// Hint a sequence of keys that are likely to be accessed in order
client.hintSequence({"step1", "step2", "step3", "step4"});

// Annotate a key with metadata to influence prefetching
predis::Annotations annotations;
annotations["high_value"] = true;
annotations["access_pattern"] = "bursty";
client.put("image:large:1", image_data, annotations);
```

### Monitoring and Statistics

> **🎯 DEMO ESSENTIAL**: Critical for measuring and displaying performance differences vs Redis

Access cache performance metrics and statistics.

```python
# Python
# Get general cache statistics
stats = client.get_stats()
print(f"Total keys: {stats['total_keys']}")
print(f"Cache hit ratio: {stats['hit_ratio']}")
print(f"Memory usage: {stats['memory_usage_mb']} MB")
print(f"Prefetch hit ratio: {stats['prefetch_hit_ratio']}")

# Get key-specific statistics
key_stats = client.get_key_stats("mykey")
print(f"Access count: {key_stats['access_count']}")
print(f"Last accessed: {key_stats['last_accessed']}")
print(f"Was prefetched: {key_stats['was_prefetched']}")

# Subscribe to cache events
def on_eviction(key, reason):
    print(f"Key '{key}' was evicted. Reason: {reason}")

client.on("eviction", on_eviction)
```

```cpp
// C++
// Get general cache statistics
auto stats = client.getStats();
std::cout << "Total keys: " << stats.total_keys << std::endl;
std::cout << "Cache hit ratio: " << stats.hit_ratio << std::endl;
std::cout << "Memory usage: " << stats.memory_usage_mb << " MB" << std::endl;
std::cout << "Prefetch hit ratio: " << stats.prefetch_hit_ratio << std::endl;

// Get key-specific statistics
auto key_stats = client.getKeyStats("mykey");
std::cout << "Access count: " << key_stats.access_count << std::endl;
std::cout << "Last accessed: " << key_stats.last_accessed << std::endl;
std::cout << "Was prefetched: " << key_stats.was_prefetched << std::endl;

// Subscribe to cache events
client.on("eviction", [](const std::string& key, const std::string& reason) {
    std::cout << "Key '" << key << "' was evicted. Reason: " << reason << std::endl;
});
```

### Management Commands

> **🎯 DEMO USEFUL**: Helpful for demo setup and system information display

Administrative operations for the Predis server.

```python
# Python
# Flush all keys
client.flush_all()

# Flush a specific namespace
client.namespace("users").flush()

# Get information about the Predis server
info = client.info()
print(f"Predis version: {info['version']}")
print(f"GPU model: {info['gpu_info']['model']}")
print(f"VRAM capacity: {info['gpu_info']['vram_total_gb']} GB")
print(f"VRAM used: {info['gpu_info']['vram_used_gb']} GB")
print(f"Active ML model: {info['ppe']['active_model']}")
print(f"Model last trained: {info['ppe']['model_last_trained']}")

# Manually trigger ML model retraining
client.trigger_model_training()
```

```cpp
// C++
// Flush all keys
client.flushAll();

// Flush a specific namespace
client.namespace("users").flush();

// Get information about the Predis server
auto info = client.info();
std::cout << "Predis version: " << info.version << std::endl;
std::cout << "GPU model: " << info.gpu_info.model << std::endl;
std::cout << "VRAM capacity: " << info.gpu_info.vram_total_gb << " GB" << std::endl;
std::cout << "VRAM used: " << info.gpu_info.vram_used_gb << " GB" << std::endl;
std::cout << "Active ML model: " << info.ppe.active_model << std::endl;
std::cout << "Model last trained: " << info.ppe.model_last_trained << std::endl;

// Manually trigger ML model retraining
client.triggerModelTraining();
```

## Minimal Viable Demo (MVD) API Requirements

Based on the demo strategy, here's the prioritized API implementation roadmap:

### Phase 1: Core Performance Demo (MUST HAVE)
These APIs are absolutely essential for demonstrating basic GPU parallelism advantages:

1. **Client Connection**: `PredisClient()`, basic connection management
2. **Basic Operations**: `get()`, `put()`, `mget()`, `mput()`
3. **Performance Stats**: `get_stats()` returning ops/sec, hit_ratio, avg_latency
4. **System Management**: `flush_all()`, `info()`

### Phase 2: ML Advantages Demo (HIGH IMPACT) 
These APIs demonstrate the unique ML-driven features:

5. **Prefetching Control**: `configure_prefetching()`, `get_prefetch_status()`
6. **Batch Operations**: `execute_batch()` for showing parallel processing power

### Phase 3: Enterprise Readiness (NICE TO HAVE)
These APIs show production readiness:

7. **Consistency Control**: `set_consistency_level()`
8. **Namespace Management**: `namespace()`
9. **Error Handling**: Exception classes and proper error responses

---

### Client Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | string | "localhost" | Predis server hostname |
| `port` | integer | 6379 | Predis server port |
| `password` | string | null | Authentication password |
| `timeout_ms` | integer | 5000 | Operation timeout in milliseconds |
| `connect_timeout_ms` | integer | 10000 | Connection timeout in milliseconds |
| `max_pool_connections` | integer | 8 | Maximum connections in the pool |
| `consistency_level` | enum | "relaxed" | Default consistency level ("strong" or "relaxed") |
| `enable_prefetching` | boolean | true | Whether to enable predictive prefetching |
| `prefetch_confidence_threshold` | float | 0.7 | Minimum confidence score for prefetching (0.0-1.0) |
| `max_prefetch_keys` | integer | 200 | Maximum number of keys to prefetch at once |
| `max_prefetch_size_mb` | integer | 100 | Maximum size of prefetched data in MB |
| `prefetch_ttl` | integer | 30 | TTL for prefetched keys in seconds |
| `auto_reconnect` | boolean | true | Whether to automatically reconnect |
| `retry_interval_ms` | integer | 1000 | Retry interval for reconnection attempts |
| `max_retries` | integer | 3 | Maximum number of operation retries |

### Server Configuration Options

These options are configured on the Predis server side, not in client code.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu_id` | integer | 0 | ID of the GPU to use for the cache |
| `max_memory_percent` | float | 80.0 | Maximum percentage of VRAM to use |
| `bloom_filter_size_mb` | integer | 128 | Size of Bloom filter in MB |
| `bloom_filter_false_positive_rate` | float | 0.01 | Target false positive rate for Bloom filter |
| `model_training_interval_minutes` | integer | 30 | How often to retrain the ML model |
| `model_type` | string | "ngboost" | Type of ML model to use ("ngboost" or "quantile_lstm") |
| `feature_window_size` | integer | 1000 | Number of access events to use for feature generation |
| `prefetch_batch_size` | integer | 50 | Number of keys to prefetch in a batch |
| `log_level` | string | "info" | Logging level ("debug", "info", "warn", "error") |
| `num_worker_threads` | integer | 8 | Number of worker threads for processing |
| `default_ttl` | integer | 0 | Default TTL for keys (0 = no expiration) |
| `eviction_policy` | string | "ml_informed_lru" | Cache eviction policy |
| `max_clients` | integer | 10000 | Maximum number of connected clients |
| `persistence_enabled` | boolean | false | Whether to enable persistence to disk |
| `persistence_interval_seconds` | integer | 300 | How often to persist cache to disk |

## Error Handling

Predis throws exceptions or returns error codes for various error conditions:

### Exception Types

Python:
- `PredisConnectionError`: Failed to connect to server
- `PredisTimeoutError`: Operation timed out
- `PredisAuthError`: Authentication failed
- `PredisCacheError`: General cache operation error

C++:
- `predis::ConnectionError`: Failed to connect to server
- `predis::TimeoutError`: Operation timed out
- `predis::AuthError`: Authentication failed
- `predis::CacheError`: General cache operation error

### Error Handling Examples

```python
# Python
from predis import PredisClient, PredisConnectionError, PredisTimeoutError

try:
    client = PredisClient(host='localhost', port=6379)
    value = client.get("mykey")
except PredisConnectionError as e:
    print(f"Connection error: {e}")
except PredisTimeoutError as e:
    print(f"Operation timed out: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

```cpp
// C++
#include <predis_cpp.h>

try {
    predis::PredisClient client("localhost", 6379);
    auto value = client.get("mykey");
} catch (const predis::ConnectionError& e) {
    std::cerr << "Connection error: " << e.what() << std::endl;
} catch (const predis::TimeoutError& e) {
    std::cerr << "Operation timed out: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << std::endl;
}
```

## Events and Callbacks

Register callbacks for various cache events:

```python
# Python
# Event when a key is accessed
client.on("access", lambda key: print(f"Key accessed: {key}"))

# Event when a key is evicted
client.on("eviction", lambda key, reason: print(f"Key evicted: {key}, reason: {reason}"))

# Event when prefetching occurs
client.on("prefetch", lambda keys: print(f"Prefetched {len(keys)} keys"))

# Event when prediction model is retrained
client.on("model_trained", lambda metrics: print(f"Model retrained, accuracy: {metrics['accuracy']}"))

# Event when connection state changes
client.on("connection_change", lambda state: print(f"Connection state: {state}"))
```

```cpp
// C++
// Event when a key is accessed
client.on("access", [](const std::string& key) {
    std::cout << "Key accessed: " << key << std::endl;
});

// Event when a key is evicted
client.on("eviction", [](const std::string& key, const std::string& reason) {
    std::cout << "Key evicted: " << key << ", reason: " << reason << std::endl;
});

// Event when prefetching occurs
client.on("prefetch", [](const std::vector<std::string>& keys) {
    std::cout << "Prefetched " << keys.size() << " keys" << std::endl;
});

// Event when prediction model is retrained
client.on("model_trained", [](const predis::ModelMetrics& metrics) {
    std::cout << "Model retrained, accuracy: " << metrics.accuracy << std::endl;
});

// Event when connection state changes
client.on("connection_change", [](const std::string& state) {
    std::cout << "Connection state: " << state << std::endl;
});
```

## Example Patterns

### Optimizing for High-Throughput Analytics

```python
# Python
# Configure for analytics workload
client.configure_prefetching(
    enabled=True,
    confidence_threshold=0.6,  # Lower threshold for more aggressive prefetching
    max_prefetch_keys=500,     # Larger prefetch batch size
    prefetch_ttl=120           # Longer TTL for prefetched data
)

# Set relaxed consistency for better performance
client.set_consistency_level(client.CONSISTENCY_RELAXED)

# Batch load frequently accessed analytics data
data_keys = [f"metric:{metric_id}" for metric_id in range(1, 1001)]
results = client.mget(data_keys)
```

### Optimizing for Financial Transactions

```python
# Python
# Configure for financial workload
client.configure_prefetching(
    enabled=True,
    confidence_threshold=0.85,  # Higher threshold for more selective prefetching
    max_prefetch_keys=50        # Smaller prefetch batch size
)

# Use strong consistency for critical operations
with client.consistency(client.CONSISTENCY_STRONG):
    # Update account balance atomically
    new_balance = client.increment("account:balance:1001", amount=-50)
    
    # Record transaction with guaranteed visibility
    client.put(f"transaction:{txn_id}", transaction_data)
```

### Multi-Tenant Workload Isolation

```python
# Python
# Create isolated namespaces for different tenants
tenant1 = client.namespace("tenant1")
tenant2 = client.namespace("tenant2")

# Configure differently based on workload characteristics
tenant1.configure_prefetching(
    enabled=True,
    confidence_threshold=0.7,
    max_prefetch_keys=200
)

tenant2.configure_prefetching(
    enabled=True, 
    confidence_threshold=0.8,
    max_prefetch_keys=100
)

# Use appropriate consistency for each tenant
tenant1.set_consistency_level(client.CONSISTENCY_RELAXED)
tenant2.set_consistency_level(client.CONSISTENCY_STRONG)

# Operate on tenant-specific data
tenant1.put("config", tenant1_config)
tenant2.put("config", tenant2_config)
```
