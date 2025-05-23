# Predis ML Components

This directory contains the machine learning components for Predis' predictive prefetching system. The ML-driven prefetching improves cache hit rates by anticipating future accesses based on observed patterns.

## Key Features

- **Multi-strategy Zero-copy Memory Interface**: Optimized for low-latency GPU access using dynamic strategy selection
- **Synthetic Data Generation**: Realistic workload simulation for web services, databases, ML training, HFT, and gaming
- **Feature Engineering Pipeline**: Comprehensive feature extraction for temporal, sequential, and real-time patterns
- **GPU Acceleration**: Hardware-accelerated feature extraction and model inference
- **Real-time Prefetching**: Sub-millisecond latency for production environments

## Directory Structure

- `data/`: Data generation and management components
  - `synthetic/`: Synthetic data generation
  - `utils/`: Data formatting and export utilities
- `features/`: Feature engineering components
  - `extractors.py`: Core feature extraction
  - `temporal.py`: Time-based pattern detection
  - `sequential.py`: Sequence-based pattern mining
  - `realtime.py`: Real-time feature extraction
  - `pipeline.py`: Integrated feature pipeline
- `examples/`: Example scripts demonstrating usage
  - `feature_engineering_demo.py`: Complete feature engineering pipeline demo

## Getting Started

### Prerequisites

The ML components require the following dependencies:

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

For GPU acceleration (optional):
```
cupy>=10.0.0
cudf>=22.12.0
cugraph>=22.12.0
```

### Basic Usage

Here's a quick example of generating synthetic data and extracting features:

```python
from ml.data.synthetic.workloads import WebServiceWorkload
from ml.features.pipeline import FeatureEngineeringPipeline

# Generate synthetic data
workload = WebServiceWorkload(num_keys=10000)
df = workload.generate(num_accesses=100000)

# Create feature pipeline
pipeline = FeatureEngineeringPipeline(use_gpu=True)

# Initialize and extract features
pipeline.initialize(df)
feature_df = pipeline.extract_features(df)

# Predict next access
predictions = pipeline.predict_next_access(top_n=5)
for key, probability in predictions:
    print(f"Key: {key}, Probability: {probability:.4f}")
```

### Zero-Copy Optimization

For high-performance environments, use the zero-copy feature pipeline:

```python
from ml.features.pipeline import ZeroCopyFeaturePipeline

# Create zero-copy pipeline
pipeline = ZeroCopyFeaturePipeline(strategy='auto')

# Extract features using optimal memory strategy
result = pipeline.extract_features(df)
feature_df = result['features']
print(f"Used strategy: {result['strategy']}")

# Extract real-time features
realtime_features = pipeline.extract_realtime_features(keys)
```

## Feature Engineering Pipeline

The feature engineering pipeline extracts comprehensive features from cache access patterns:

### Temporal Features

- Time-of-day patterns (hourly, daily cycles)
- Day-of-week patterns (weekday/weekend)
- Seasonality (monthly, quarterly patterns)
- Access frequency over various time windows

### Sequential Features

- Key transition probabilities
- N-gram pattern detection
- Sequential pattern mining
- Cycle detection in access sequences

### Real-time Features

- Low-latency feature extraction (<1ms)
- Recency and frequency scoring
- Access pattern prediction
- Dynamic prefetch candidate selection

## Advanced Usage

For a complete demonstration of the feature engineering pipeline, run the example script:

```bash
python -m ml.examples.feature_engineering_demo --workload=combined --num-keys=10000 --num-accesses=100000 --use-gpu --zero-copy
```

To customize feature extraction:

```python
from ml.features.extractors import AccessPatternFeatureExtractor
from ml.features.temporal import TemporalFeatureGenerator
from ml.features.sequential import SequenceFeatureGenerator

# Create custom extractors
access_extractor = AccessPatternFeatureExtractor()
temporal_extractor = TemporalFeatureGenerator()
sequence_extractor = SequenceFeatureGenerator(sequence_length=10)

# Extract specific feature sets
df = access_extractor.extract_features(df)
df = temporal_extractor.extract_all_temporal_features(df)
df = sequence_extractor.extract_all_sequence_features(df)
```

## Integration with Predis

The ML components integrate with Predis' core cache system through the zero-copy memory interface, which dynamically selects between three access strategies:

1. **GPU-Direct**: Lowest-latency access via PCIe BAR1 or NVLink
2. **UVM**: Optimized Unified Virtual Memory with ML-driven page placement
3. **Peer Mapping**: Custom peer mapping with explicit coherence control

This integration enables 2-5x lower latency compared to traditional copy-based approaches while providing highly accurate predictions for prefetching decisions.

## Running Tests

Unit tests for the ML components can be run with:

```bash
python -m unittest discover -s tests/ml
```

Or run specific tests:

```bash
python -m unittest tests.ml.test_feature_engineering
```

## Documentation

For more detailed information, see the following resources:

- [Epic 3 Documentation](../../doc/epic3_ds.md): Comprehensive specification for ML components
- [Completion Notes](../../doc/completed/epic3_ds_done.md): Progress and implementation details
