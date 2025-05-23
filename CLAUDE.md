# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## preferences
- do not put files in root directory
- dont forget to make the file executable
- do not create duplicate files when troubleshooting.  rename the file to .bak so it is not overwritten

### CRITICAL: Fabricated Data and Results Policy
- **NEVER create documents containing fabricated metrics, benchmarks, or results without clearly labeling them as SIMULATED, MOCK, or EXAMPLE data**
- When creating test data, mockups, or examples, ALWAYS include clear headers like:
  - "SIMULATED DATA - NOT ACTUAL RESULTS"
  - "MOCK METRICS FOR TESTING"
  - "EXAMPLE OUTPUT - NOT FROM REAL EXECUTION"
- If generating placeholder metrics for framework development:
  - Use obviously fake values (e.g., 99.99%, 12345)
  - Include "TODO: Replace with actual measurements" comments
  - Name files with prefixes like `mock_`, `simulated_`, or `example_`
- When presenting any numerical results, specify their source:
  - "From actual benchmark run on [date]"
  - "Simulated data for framework testing"
  - "Targets from requirements document"
- This is critical for maintaining trust and preventing decisions based on false data

### Search and Navigation Efficiency
- ALWAYS use ripgrep (rg) instead of grep for searching code - it's faster and more efficient
  - Use `rg "pattern"` for basic searches
  - Use `rg "pattern" --type cpp` to search only C++ files
  - Use `rg "pattern" -A 3 -B 3` to show context lines
  - Use `rg "pattern" --files-with-matches` to just list files containing pattern
- Use concurrent tool calls when possible (read multiple files simultaneously, run parallel operations)
- When searching for classes/functions, use Glob tool with patterns like `**/*cache*.{cpp,h}` before using Agent tool
- Check CLAUDE.md first for project context before asking questions about architecture or preferences

### Build and Development Efficiency  
- Use `make -j$(nproc)` for parallel compilation
- Clean builds: `rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)`
- Test specific targets: `make arch_validation_test && ./bin/arch_validation_test`
- Use build artifacts in /build directory, don't rebuild unnecessarily

### Code Quality and Standards
- Follow existing code patterns and naming conventions in the codebase
- Add comprehensive error handling and logging
- Use const correctness and RAII patterns
- Prefer smart pointers over raw pointers
- Use meaningful variable names that reflect GPU/cache context

### Docker and GPU Development
- Test GPU functionality in Docker environment: `./dev_scripts/start_dev.sh`
- Use Docker for CUDA development, host system for architecture validation
- Volume mount preserves build artifacts: don't rebuild in container unless necessary
- Validate GPU access with `nvidia-smi` before running GPU code

### Task Management and Documentation
- Use TodoWrite tool for multi-step tasks (3+ steps or complex workflows)
- Mark todos as completed immediately after finishing tasks
- **CRITICAL**: Update progress in doc/completed/epic*_done.md files immediately upon story completion
  - Update story points (e.g., "8/8 points" for completed stories)
  - Update epic progress percentage (e.g., "55/55 points completed (100%)")
  - Change status from "NOT STARTED" or "IN PROGRESS" to "COMPLETED"
  - Add completion date and summary of achievements
- Keep CLAUDE.md updated with new patterns and learnings
- Document architecture decisions and performance measurements

### Performance and Benchmarking
- Always measure performance with actual numbers, not just "fast" 
- Use std::chrono for microsecond-level timing measurements
- Compare against Redis baseline when evaluating cache performance
- Log memory usage, hit rates, and operations per second
- Validate performance claims with statistical significance

### Performance Testing Procedure 
1. **Benchmark Data Collection**:
   - Run benchmarks using existing tools in src/benchmarks/
   - Save results to benchmark_results/ directory with timestamp
   - Use JSON format for structured data storage
   - Collect metrics: latency (avg, median, p95, p99), throughput (ops/sec), speedup factor

2. **HTML Report Generation**:
   - Use professional report generator from Epic 2: src/benchmarks/professional_report_generator.cpp
   - Generate HTML reports that match the actual benchmark JSON data
   - Store reports in doc/results/ directory
   - Include charts for visual representation of performance gains

3. **Performance Validation Steps**:
   - Compare baseline (Redis) vs optimized (Predis) performance
   - Calculate improvement factors for each operation type
   - Verify statistical significance (p < 0.01)
   - Document any performance gaps (e.g., PUT vs GET disparities)
   - Track progress against Epic goals (10x, 25x, 50x targets)

4. **Documentation Updates**:
   - Update doc\resultswith performance results
   - Include actual numbers from benchmark JSON files
   - Reference specific improvements and optimizations made
   - Track story point completion and remaining work

### Error Handling and Debugging
- Check return values from CUDA operations and log errors with cudaGetErrorString
- Use mutex locks for thread-safe operations in cache managers
- Validate pointers before dereferencing (especially GPU memory pointers)
- Add clear error messages that help diagnose issues quickly
- Test error conditions (out of memory, GPU failures) in addition to happy path

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