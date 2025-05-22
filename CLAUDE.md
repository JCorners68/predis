# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## preferences
- do not put files in root directory
- dont forget to make the file executable
- 

## Project Overview

Predis is a GPU-accelerated key-value cache with predictive prefetching capabilities, designed to achieve 10-50x performance improvements over Redis through machine learning-driven optimization and massive GPU parallelism.

**Core Technologies:**
- Mojo (primary language for GPU-accelerated components)
- C++ with CUDA for GPU memory management
- Python for ML model development and API bindings
- Apache Flink for real-time stream processing

## Architecture

### Major Components
1. **GPU Cache Core** (`src/core/`): Manages key-value store in GPU VRAM with optimized data structures
2. **Predictive Prefetching Engine** (`src/ppe/`): Uses ML models (NGBoost, Quantile LSTM) to predict and prefetch data  
3. **Access Pattern Logger** (`src/logger/`): Monitors cache access patterns for ML training
4. **Cache API/SDK** (`src/api/`): Provides Redis-compatible interface with ML extensions

### Key Technical Features
- **GPU-First Storage**: Primary cache data stored in GPU VRAM for ultra-fast access
- **ML-Driven Prefetching**: Machine learning models predict future access patterns
- **Hybrid Consistency**: Configurable consistency levels (strong vs. relaxed)
- **Intelligent Eviction**: ML-informed cache eviction policies

## Development Environment

**Target Hardware:**
- NVIDIA RTX 5080 (16GB VRAM, 10,752 CUDA cores) for local development
- WSL2 + Docker containers for consistent development environment

**Performance Goals:**
- 10-20x faster than Redis for basic operations
- 25-50x faster for batch operations (leveraging GPU parallelism)
- 20-30% improvement in cache hit rates through predictive prefetching

## Planned Directory Structure

Based on the architecture document, the project will follow this structure:

```
src/
├── api/                    # Cache API/SDK
│   ├── predis_client.mojo  # Client-facing API
│   └── bindings/           # Language-specific bindings
├── core/                   # GPU Cache Core
│   ├── cache_manager.mojo  # Main cache logic
│   ├── data_structures/    # GPU-optimized data structures
│   ├── memory_manager.mojo # VRAM allocation management
│   └── eviction_engine.mojo # ML-informed eviction
├── logger/                 # Access Pattern Logger
│   ├── access_logger.mojo  # Log key accesses
│   └── log_processor.mojo  # Process logs for ML
├── ppe/                    # Predictive Prefetching Engine
│   ├── models/             # ML models (NGBoost, LSTM)
│   ├── feature_generator.mojo # Time series feature extraction
│   └── prefetch_strategist.mojo # Prefetching decisions
└── utils/                  # Common utilities
```

## Development Phases

**Phase 1 (Weeks 1-4)**: Basic GPU caching and memory management
**Phase 2 (Weeks 5-8)**: Performance demonstration showing 10-25x improvement  
**Phase 3 (Weeks 9-16)**: ML prefetching with measurable hit rate improvements
**Phase 4 (Weeks 17-20)**: Investor-ready demonstration and polish

## Key API Design

The system provides Redis-compatible operations with ML extensions:

```python
# Basic operations (must achieve 10-25x Redis performance)
client.get(key)
client.put(key, value, ttl=None)
client.mget(keys)  # Batch operations for GPU advantage
client.mput(key_value_dict)

# ML-specific enhancements
client.configure_prefetching(enabled=True, confidence_threshold=0.7)
client.hint_related_keys(key_list)
client.get_stats()  # Performance metrics for demo
```

## Performance Validation

All performance claims must be rigorously validated through:
- Side-by-side Redis vs Predis benchmarks
- Statistical significance testing
- Multiple workload scenarios (read-heavy, write-heavy, mixed)
- Real-time performance monitoring and visualization

## Development Notes

- Focus on ML training workload optimization as primary market differentiator
- GPU memory management is critical due to VRAM limitations (16GB RTX 5080)
- All ML models must run inference on GPU for consistent performance
- Docker containers recommended for WSL development environment stability
- Performance demonstration quality is critical for funding success

## Risk Considerations

- WSL GPU driver stability issues (use Docker containers)
- VRAM memory constraints (implement tiered storage architecture)
- Demo reliability during investor presentations (extensive failure scenario testing)