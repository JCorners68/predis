# Benchmark Results Directory

This directory contains all performance benchmark results for the Predis project.

## Result Files

### GPU Performance Benchmarks
1. **real_gpu_results_1747975875.json** (2025-01-22)
   - Simple GPU benchmark with real execution
   - PUT: 2,592.29 million ops/sec (218.7x CPU)
   - GET: 8,123.21 million ops/sec (279.4x CPU)
   - Status: ✅ REAL GPU EXECUTION

2. **comprehensive_gpu_results_1747975989.json** (2025-01-22)
   - Comprehensive benchmark with multiple table sizes
   - Peak throughput: 15.24 billion ops/sec
   - Multiple batch sizes tested
   - Status: ✅ REAL GPU EXECUTION

### ML Model Benchmarks
3. **lstm_synthetic_results_20250522_231237.json** (2025-01-22)
   - LSTM model validation on synthetic cache patterns
   - Sequential: 91.7% accuracy
   - Periodic: 86.1% accuracy
   - Zipfian: 65.3% accuracy
   - Random: 16.9% accuracy
   - Inference: 0.483ms average
   - Status: ⚠️ SYNTHETIC DATA VALIDATION

4. **lstm_validation_summary_20250522_231237.md** (2025-01-22)
   - Summary report of LSTM validation
   - Pattern-specific performance analysis
   - Status: ⚠️ SYNTHETIC PATTERNS ONLY

## Data Classification

### ✅ Real Measured Results
- GPU benchmarks use actual CUDA execution
- Timing measured with CUDA events
- Real memory operations performed

### ⚠️ Synthetic Validation
- LSTM results based on generated patterns
- Not representative of production workloads
- Demonstrates framework capability only

### ❌ No Simulated Results
- All results in this directory are from actual execution
- No fabricated or estimated metrics

## Usage

To analyze results:
```python
import json

# Load GPU results
with open('real_gpu_results_1747975875.json', 'r') as f:
    gpu_data = json.load(f)

# Load LSTM results  
with open('lstm_synthetic_results_20250522_231237.json', 'r') as f:
    lstm_data = json.load(f)
```

## Important Notes

1. **GPU Results**: Actual performance on NVIDIA RTX 5080
2. **LSTM Results**: Synthetic patterns only - real workload performance will differ
3. **No Production Data**: All tests use synthetic or generated data
4. **Reproducible**: All benchmarks can be re-run with provided scripts