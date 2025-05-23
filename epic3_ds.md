# Epic 3: Data Science & ML Prefetching Implementation

## Epic Overview

**Goal**: Implement machine learning-driven predictive prefetching to improve cache hit rates by 20-30% over baseline heuristic approaches.

**Success Criteria**: 
- Demonstrable improvement in cache hit rates using ML predictions
- <10ms prediction latency for real-time prefetching decisions
- Automated model training and retraining pipeline
- A/B testing framework showing ML superiority over simple heuristics

**Technical Approach**: Start with LSTM time-series forecasting on synthetic data, then migrate to real-world data collection and advanced models.

---

## Story 3.8: Data Science Folder Structure & Repository Organization

**Priority**: P0 | **Points**: 5 | **Dependencies**: Core cache system from Epic 2

### Acceptance Criteria
- [ ] Create organized folder structure for ML/DS components within existing Predis repo
- [ ] Establish data pipeline directories and file naming conventions
- [ ] Set up development environment configuration for ML dependencies
- [ ] Create initial placeholder files and documentation structure
- [ ] Prepare for future extraction to separate ML services repo

### Folder Structure Implementation

```
predis/
├── src/
│   ├── api/                          # Existing cache API
│   ├── core/                         # Existing GPU cache core
│   ├── ml/                           # NEW: Machine Learning Components
│   │   ├── __init__.py
│   │   ├── data/                     # Data generation and management
│   │   │   ├── __init__.py
│   │   │   ├── synthetic/            # Synthetic data generation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── generators.py     # Access pattern generators
│   │   │   │   ├── workloads.py      # Workload-specific patterns
│   │   │   │   └── validation.py     # Data validation utilities
│   │   │   ├── real/                 # Real data collection
│   │   │   │   ├── __init__.py
│   │   │   │   ├── collectors.py     # Access log collectors
│   │   │   │   └── processors.py     # Real data preprocessing
│   │   │   └── utils/                # Data utilities
│   │   │       ├── __init__.py
│   │   │       ├── formatters.py     # Data format converters
│   │   │       └── exporters.py      # Data export utilities
│   │   ├── features/                 # Feature engineering
│   │   │   ├── __init__.py
│   │   │   ├── extractors.py         # Feature extraction pipelines
│   │   │   ├── temporal.py           # Time-based features
│   │   │   ├── sequential.py         # Sequence-based features
│   │   │   └── realtime.py           # Real-time feature extraction
│   │   ├── models/                   # ML model implementations
│   │   │   ├── __init__.py
│   │   │   ├── lstm/                 # LSTM implementation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── architecture.py   # Model architecture
│   │   │   │   ├── training.py       # Training pipeline
│   │   │   │   └── inference.py      # Inference engine
│   │   │   ├── baseline/             # Baseline models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── heuristic.py      # Heuristic prefetchers
│   │   │   │   └── frequency.py      # Frequency-based models
│   │   │   └── utils/                # Model utilities
│   │   │       ├── __init__.py
│   │   │       ├── metrics.py        # Evaluation metrics
│   │   │       └── persistence.py    # Model save/load
│   │   ├── training/                 # Training infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── pipelines.py          # Training pipelines
│   │   │   ├── datasets.py           # Dataset classes
│   │   │   ├── schedulers.py         # Training schedulers
│   │   │   └── retraining.py         # Automated retraining
│   │   ├── inference/                # Inference infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── engines.py            # Prediction engines
│   │   │   ├── integration.py        # Cache system integration
│   │   │   └── optimization.py       # GPU inference optimization
│   │   ├── evaluation/               # Model evaluation and testing
│   │   │   ├── __init__.py
│   │   │   ├── ab_testing.py         # A/B testing framework
│   │   │   ├── metrics.py            # Performance metrics
│   │   │   └── monitoring.py         # Performance monitoring
│   │   └── config/                   # ML configuration
│   │       ├── __init__.py
│   │       ├── model_configs.py      # Model configurations
│   │       ├── training_configs.py   # Training configurations
│   │       └── inference_configs.py  # Inference configurations
│   ├── utils/                        # Existing utilities
│   └── predis_server.mojo            # Existing main server
├── data/                             # NEW: Data storage and artifacts
│   ├── synthetic/                    # Generated synthetic datasets
│   │   ├── zipfian/                  # Zipfian distribution datasets
│   │   ├── temporal/                 # Temporal pattern datasets
│   │   ├── hft/                      # High-frequency trading patterns
│   │   ├── ml_training/              # ML training simulation datasets
│   │   └── gaming/                   # Gaming workload datasets
│   ├── real/                         # Real-world collected data
│   │   ├── raw/                      # Raw access logs
│   │   ├── processed/                # Processed training data
│   │   └── features/                 # Extracted features
│   ├── models/                       # Trained model artifacts
│   │   ├── lstm/                     # LSTM model weights
│   │   ├── baseline/                 # Baseline model artifacts
│   │   └── experiments/              # Experimental models
│   └── benchmarks/                   # Performance benchmark data
│       ├── synthetic/                # Synthetic data benchmarks
│       └── real/                     # Real data benchmarks
├── notebooks/                        # NEW: Jupyter notebooks for analysis
│   ├── data_exploration/             # Data analysis notebooks
│   │   ├── synthetic_analysis.ipynb  # Synthetic data analysis
│   │   └── real_data_analysis.ipynb  # Real data analysis
│   ├── model_development/            # Model development notebooks
│   │   ├── lstm_experiments.ipynb    # LSTM experimentation
│   │   └── baseline_comparison.ipynb # Baseline comparisons
│   └── performance_analysis/         # Performance analysis
│       ├── ab_test_results.ipynb     # A/B test analysis
│       └── monitoring_dashboard.ipynb # Performance monitoring
├── tests/                            # Existing tests + ML tests
│   ├── unit/
│   ├── integration/
│   ├── performance/
│   └── ml/                           # NEW: ML-specific tests
│       ├── test_data_generation.py   # Synthetic data tests
│       ├── test_feature_engineering.py # Feature engineering tests
│       ├── test_model_training.py    # Model training tests
│       ├── test_inference.py         # Inference tests
│       └── test_integration.py       # ML-cache integration tests
├── scripts/                          # Existing scripts + ML scripts
│   ├── build_predis.sh
│   ├── run_server.sh
│   └── ml/                           # NEW: ML-specific scripts
│       ├── generate_synthetic_data.py # Data generation s

### Acceptance Criteria
- [ ] Generate realistic cache access patterns using statistical distributions
- [ ] Multiple workload scenarios (HFT, ML training, gaming, web)
- [ ] Configurable pattern parameters (seasonality, burstiness, key relationships)
- [ ] Export to standardized format for ML training
- [ ] Validation that synthetic patterns match real-world characteristics

### Technical Implementation

#### Synthetic Data Generation Approaches

**1. Zipfian Distribution (80/20 Rule)**
```python
def generate_zipfian_access_pattern(num_keys=10000, num_accesses=1000000, alpha=1.0):
    """
    Generate cache access patterns following Zipfian distribution
    20% of keys get 80% of traffic (typical in real applications)
    """
    # Generate key popularity ranks
    ranks = np.arange(1, num_keys + 1)
    probabilities = 1.0 / (ranks ** alpha)
    probabilities = probabilities / probabilities.sum()
    
    # Generate access sequence
    access_sequence = []
    timestamps = []
    
    for i in range(num_accesses):
        # Select key based on Zipfian probability
        key_id = np.random.choice(num_keys, p=probabilities)
        timestamp = time.time() + i * 0.001  # 1ms intervals
        
        access_sequence.append({
            'timestamp': timestamp,
            'key': f"key_{key_id}",
            'operation': 'GET',
            'workload_type': 'zipfian'
        })
    
    return access_sequence
```

**2. Temporal Patterns (Daily/Weekly Cycles)**
```python
def generate_temporal_access_pattern(duration_hours=24, keys_per_hour=1000):
    """
    Generate time-based access patterns with daily cycles
    Simulates business hours, peak traffic periods, etc.
    """
    access_sequence = []
    base_time = time.time()
    
    for hour in range(duration_hours):
        # Simulate daily cycle (higher traffic during business hours)
        hour_of_day = hour % 24
        if 9 <= hour_of_day <= 17:  # Business hours
            traffic_multiplier = 3.0
        elif 18 <= hour_of_day <= 22:  # Evening peak
            traffic_multiplier = 2.0
        else:  # Night/early morning
            traffic_multiplier = 0.5
            
        num_accesses = int(keys_per_hour * traffic_multiplier)
        
        for access in range(num_accesses):
            timestamp = base_time + hour * 3600 + access * (3600 / num_accesses)
            key_id = np.random.randint(0, 10000)
            
            access_sequence.append({
                'timestamp': timestamp,
                'key': f"temporal_key_{key_id}",
                'operation': 'GET',
                'workload_type': 'temporal'
            })
    
    return access_sequence
```

**3. Burst Patterns (ML Training Simulation)**
```python
def generate_ml_training_pattern(num_epochs=10, batch_size=256, dataset_size=50000):
    """
    Simulate ML training access patterns with sequential batch loading
    """
    access_sequence = []
    base_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle dataset order each epoch
        data_indices = np.random.permutation(dataset_size)
        
        for batch_start in range(0, dataset_size, batch_size):
            batch_indices = data_indices[batch_start:batch_start + batch_size]
            
            # Sequential access within batch (predictable)
            for i, data_idx in enumerate(batch_indices):
                timestamp = base_time + epoch * 3600 + batch_start * 0.1 + i * 0.001
                
                access_sequence.append({
                    'timestamp': timestamp,
                    'key': f"data_{data_idx}",
                    'operation': 'GET',
                    'workload_type': 'ml_training',
                    'epoch': epoch,
                    'batch': batch_start // batch_size
                })
    
    return access_sequence
```

#### Workload-Specific Patterns

**High-Frequency Trading**
```python
def generate_hft_pattern(symbols=1000, trades_per_second=10000):
    """
    Simulate HFT access patterns with hot symbols and market events
    """
    # Hot symbols (top 10% get 90% of traffic)
    hot_symbols = symbols // 10
    
    access_sequence = []
    base_time = time.time()
    
    for second in range(3600):  # 1 hour simulation
        for trade in range(trades_per_second):
            timestamp = base_time + second + trade / trades_per_second
            
            # 90% probability of accessing hot symbol
            if np.random.random() < 0.9:
                symbol_id = np.random.randint(0, hot_symbols)
            else:
                symbol_id = np.random.randint(hot_symbols, symbols)
            
            access_sequence.append({
                'timestamp': timestamp,
                'key': f"price_{symbol_id}",
                'operation': 'GET',
                'workload_type': 'hft'
            })
    
    return access_sequence
```

#### Data Export and Validation

```python
def export_synthetic_data(access_patterns, output_file):
    """
    Export synthetic access patterns to standardized format for ML training
    """
    df = pd.DataFrame(access_patterns)
    
    # Add derived features
    df['hour_of_day'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
    df = df.sort_values('timestamp')
    
    # Add time-based features
    df['time_since_last_access'] = df.groupby('key')['timestamp'].diff()
    df['access_count'] = df.groupby('key').cumcount() + 1
    
    # Export to multiple formats
    df.to_csv(f"{output_file}.csv", index=False)
    df.to_parquet(f"{output_file}.parquet")
    
    return df

def validate_synthetic_patterns(df):
    """
    Validate that synthetic patterns match expected characteristics
    """
    validation_results = {}
    
    # Check Zipfian distribution
    key_counts = df['key'].value_counts()
    top_20_percent = int(len(key_counts) * 0.2)
    top_20_traffic = key_counts.head(top_20_percent).sum()
    total_traffic = key_counts.sum()
    
    validation_results['zipfian_ratio'] = top_20_traffic / total_traffic
    validation_results['expected_range'] = (0.7, 0.9)  # Should be 70-90%
    
    # Check temporal patterns
    hourly_traffic = df.groupby(df['timestamp'].apply(
        lambda x: pd.to_datetime(x, unit='s').hour
    )).size()
    
    validation_results['peak_hour_traffic'] = hourly_traffic.max()
    validation_results['off_peak_traffic'] = hourly_traffic.min()
    validation_results['peak_ratio'] = hourly_traffic.max() / hourly_traffic.min()
    
    return validation_results
```

---

## Story 3.2: Feature Engineering Pipeline

**Priority**: P0 | **Points**: 8 | **Dependencies**: Story 3.1

### Acceptance Criteria
- [ ] Time-series features extracted from access logs
- [ ] Key relationship features (co-occurrence, sequence patterns)
- [ ] Temporal features (hour, day, seasonality)
- [ ] GPU-optimized feature computation
- [ ] Real-time feature extraction for inference

### Technical Implementation

#### Core Feature Engineering

```python
class CacheAccessFeatureExtractor:
    def __init__(self, window_size=1000, gpu_enabled=True):
        self.window_size = window_size
        self.gpu_enabled = gpu_enabled
        if gpu_enabled:
            import cudf
            self.df_lib = cudf
        else:
            import pandas as pd
            self.df_lib = pd
    
    def extract_temporal_features(self, access_df):
        """
        Extract time-based features from access patterns
        """
        # Convert timestamp to datetime
        access_df['datetime'] = self.df_lib.to_datetime(access_df['timestamp'], unit='s')
        
        # Basic temporal features
        access_df['hour_of_day'] = access_df['datetime'].dt.hour
        access_df['day_of_week'] = access_df['datetime'].dt.dayofweek
        access_df['minute_of_hour'] = access_df['datetime'].dt.minute
        
        # Cyclical encoding for temporal features
        access_df['hour_sin'] = np.sin(2 * np.pi * access_df['hour_of_day'] / 24)
        access_df['hour_cos'] = np.cos(2 * np.pi * access_df['hour_of_day'] / 24)
        access_df['dow_sin'] = np.sin(2 * np.pi * access_df['day_of_week'] / 7)
        access_df['dow_cos'] = np.cos(2 * np.pi * access_df['day_of_week'] / 7)
        
        return access_df
    
    def extract_sequence_features(self, access_df):
        """
        Extract sequential access pattern features
        """
        # Sort by timestamp
        access_df = access_df.sort_values('timestamp')
        
        # Time since last access for each key
        access_df['time_since_last_access'] = access_df.groupby('key')['timestamp'].diff()
        
        # Access frequency features
        access_df['access_count'] = access_df.groupby('key').cumcount() + 1
        access_df['access_frequency'] = access_df.groupby('key')['timestamp'].transform(
            lambda x: len(x) / (x.max() - x.min() + 1)
        )
        
        # Recent access patterns (sliding window)
        access_df['recent_access_count'] = access_df.groupby('key')['timestamp'].transform(
            lambda x: self._count_recent_accesses(x, window_seconds=3600)
        )
        
        return access_df
    
    def extract_cooccurrence_features(self, access_df):
        """
        Extract key co-occurrence and relationship features
        """
        # Group accesses by time windows to find co-occurring keys
        access_df['time_bucket'] = (access_df['timestamp'] // 10).astype(int)  # 10-second buckets
        
        # Calculate key co-occurrence within time buckets
        cooccurrence_pairs = []
        for bucket, group in access_df.groupby('time_bucket'):
            keys_in_bucket = group['key'].unique()
            for i, key1 in enumerate(keys_in_bucket):
                for key2 in keys_in_bucket[i+1:]:
                    cooccurrence_pairs.append({
                        'key1': key1,
                        'key2': key2,
                        'bucket': bucket
                    })
        
        cooccurrence_df = self.df_lib.DataFrame(cooccurrence_pairs)
        cooccurrence_counts = cooccurrence_df.groupby(['key1', 'key2']).size().reset_index()
        cooccurrence_counts.columns = ['key1', 'key2', 'cooccurrence_count']
        
        return cooccurrence_counts
    
    def extract_lag_features(self, access_df, lags=[1, 5, 10, 30]):
        """
        Extract lagged features for time series prediction
        """
        # Create time-ordered sequence for each key
        key_sequences = {}
        for key, group in access_df.groupby('key'):
            timestamps = group['timestamp'].sort_values().values
            key_sequences[key] = timestamps
        
        lag_features = []
        for key, timestamps in key_sequences.items():
            for i in range(len(timestamps)):
                features = {'key': key, 'timestamp': timestamps[i]}
                
                # Add lag features
                for lag in lags:
                    if i >= lag:
                        features[f'access_lag_{lag}'] = 1  # Access occurred lag steps ago
                        features[f'time_diff_lag_{lag}'] = timestamps[i] - timestamps[i-lag]
                    else:
                        features[f'access_lag_{lag}'] = 0
                        features[f'time_diff_lag_{lag}'] = np.nan
                
                lag_features.append(features)
        
        return self.df_lib.DataFrame(lag_features)
    
    def _count_recent_accesses(self, timestamps, window_seconds=3600):
        """
        Count accesses within recent time window
        """
        result = []
        for i, ts in enumerate(timestamps):
            recent_count = sum(1 for t in timestamps[:i] if ts - t <= window_seconds)
            result.append(recent_count)
        return result
```

#### GPU-Optimized Feature Computation

```python
def gpu_optimized_feature_extraction(access_df):
    """
    GPU-accelerated feature extraction using cuDF
    """
    import cudf
    import cupy as cp
    
    # Convert to GPU DataFrame
    if not isinstance(access_df, cudf.DataFrame):
        gpu_df = cudf.from_pandas(access_df)
    else:
        gpu_df = access_df
    
    # GPU-accelerated groupby operations
    gpu_df['access_rank'] = gpu_df.groupby('key')['timestamp'].rank(method='dense')
    gpu_df['key_access_count'] = gpu_df.groupby('key')['timestamp'].transform('count')
    
    # GPU-accelerated rolling window features
    gpu_df = gpu_df.sort_values(['key', 'timestamp'])
    gpu_df['rolling_access_rate'] = gpu_df.groupby('key')['timestamp'].rolling(
        window=10, min_periods=1
    ).count().reset_index(drop=True)
    
    return gpu_df
```

#### Real-Time Feature Extraction

```python
class RealTimeFeatureExtractor:
    def __init__(self, history_window=10000):
        self.access_history = collections.deque(maxlen=history_window)
        self.key_stats = {}
        
    def extract_features_for_prediction(self, current_time, recent_keys):
        """
        Extract features for real-time prediction
        Must complete in <1ms for real-time use
        """
        features = {}
        
        # Temporal features
        dt = datetime.fromtimestamp(current_time)
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        
        # Recent access patterns
        recent_access_times = [t for t, _ in self.access_history if current_time - t < 3600]
        features['recent_access_rate'] = len(recent_access_times) / 3600
        
        # Key-specific features for recently accessed keys
        key_features = {}
        for key in recent_keys[-10:]:  # Last 10 accessed keys
            if key in self.key_stats:
                key_features[f'{key}_frequency'] = self.key_stats[key]['frequency']
                key_features[f'{key}_last_access'] = current_time - self.key_stats[key]['last_access']
        
        features.update(key_features)
        return features
    
    def update_with_access(self, timestamp, key):
        """
        Update feature extraction state with new access
        """
        self.access_history.append((timestamp, key))
        
        if key not in self.key_stats:
            self.key_stats[key] = {'count': 0, 'first_access': timestamp}
        
        self.key_stats[key]['count'] += 1
        self.key_stats[key]['last_access'] = timestamp
        self.key_stats[key]['frequency'] = self.key_stats[key]['count'] / (
            timestamp - self.key_stats[key]['first_access'] + 1
        )
```

---

## Story 3.3: LSTM Model Training Pipeline

**Priority**: P0 | **Points**: 13 | **Dependencies**: Story 3.2

### Acceptance Criteria
- [ ] LSTM model architecture for sequence prediction
- [ ] Training pipeline with synthetic data
- [ ] Model validation and performance metrics
- [ ] GPU-accelerated training and inference
- [ ] Model persistence and versioning

### Technical Implementation

#### LSTM Model Architecture

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CacheAccessLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_keys=10000, dropout=0.2):
        super(CacheAccessLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        
        # Key embedding layer
        self.key_embedding = nn.Embedding(num_keys, 64)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size + 64,  # features + key embedding
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_keys)  # Predict next key
        self.confidence = nn.Linear(hidden_size, 1)  # Prediction confidence
        
    def forward(self, x, key_ids, hidden=None):
        batch_size = x.size(0)
        
        # Embed keys
        key_emb = self.key_embedding(key_ids)
        
        # Concatenate features and key embeddings
        lstm_input = torch.cat([x, key_emb], dim=-1)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Predictions
        dropped = self.dropout(last_output)
        key_logits = self.fc(dropped)
        confidence = torch.sigmoid(self.confidence(dropped))
        
        return key_logits, confidence, hidden

class CacheAccessDataset(Dataset):
    def __init__(self, access_data, sequence_length=50, prediction_horizon=10):
        self.access_data = access_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        sequences = []
        
        # Group by session or time windows
        for i in range(len(self.access_data) - self.sequence_length - self.prediction_horizon):
            # Input sequence
            input_seq = self.access_data[i:i + self.sequence_length]
            
            # Target sequence (what to predict)
            target_seq = self.access_data[
                i + self.sequence_length:i + self.sequence_length + self.prediction_horizon
            ]
            
            sequences.append({
                'input_features': self._extract_features(input_seq),
                'input_keys': self._extract_key_ids(input_seq),
                'target_keys': self._extract_key_ids(target_seq)
            })
            
        return sequences
    
    def _extract_features(self, sequence):
        """Extract numerical features from access sequence"""
        features = []
        for access in sequence:
            feature_vector = [
                access['hour_sin'],
                access['hour_cos'],
                access['dow_sin'],
                access['dow_cos'],
                access.get('time_since_last_access', 0),
                access.get('access_frequency', 0),
                access.get('recent_access_count', 0)
            ]
            features.append(feature_vector)
        return torch.FloatTensor(features)
    
    def _extract_key_ids(self, sequence):
        """Extract key IDs from sequence"""
        # Convert string keys to integers
        key_ids = []
        for access in sequence:
            key_str = access['key']
            # Extract numeric part or use hash
            if 'key_' in key_str:
                key_id = int(key_str.split('_')[1])
            else:
                key_id = hash(key_str) % 10000  # Fallback to hash
            key_ids.append(key_id)
        return torch.LongTensor(key_ids)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
```

#### Training Pipeline

```python
class CacheLSTMTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        for batch in dataloader:
            input_features = batch['input_features'].to(self.device)
            input_keys = batch['input_keys'].to(self.device)
            target_keys = batch['target_keys'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            key_logits, confidence, _ = self.model(input_features, input_keys)
            
            # Calculate losses
            # Predict first key in target sequence
            key_loss = self.criterion(key_logits, target_keys[:, 0])
            
            # Confidence should be higher for correct predictions
            predicted_keys = torch.argmax(key_logits, dim=1)
            correct_predictions = (predicted_keys == target_keys[:, 0]).float()
            confidence_loss = self.mse_loss(confidence.squeeze(), correct_predictions)
            
            total_loss_batch = key_loss + 0.1 * confidence_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            accuracy = (predicted_keys == target_keys[:, 0]).float().mean()
            total_loss += total_loss_batch.item()
            total_accuracy += accuracy.item()
        
        return total_loss / len(dataloader), total_accuracy / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_features = batch['input_features'].to(self.device)
                input_keys = batch['input_keys'].to(self.device)
                target_keys = batch['target_keys'].to(self.device)
                
                key_logits, confidence, _ = self.model(input_features, input_keys)
                
                # Calculate accuracy
                predicted_keys = torch.argmax(key_logits, dim=1)
                accuracy = (predicted_keys == target_keys[:, 0]).float().mean()
                
                total_accuracy += accuracy.item()
                
                # Store predictions for analysis
                predictions.extend([{
                    'predicted': predicted_keys.cpu().numpy(),
                    'actual': target_keys[:, 0].cpu().numpy(),
                    'confidence': confidence.cpu().numpy()
                }])
        
        return total_accuracy / len(dataloader), predictions

def train_lstm_model(synthetic_data_path, epochs=100, batch_size=32):
    """
    Main training function
    """
    # Load synthetic data
    import pandas as pd
    df = pd.read_csv(synthetic_data_path)
    access_data = df.to_dict('records')
    
    # Create datasets
    dataset = CacheAccessDataset(access_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CacheAccessLSTM(input_size=7)  # 7 features from feature engineering
    trainer = CacheLSTMTrainer(model)
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_acc, predictions = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_cache_lstm.pth')
            
        # Early stopping
        if val_acc > 0.95:  # Good enough for demo
            break
    
    return model, best_val_accuracy
```

#### Model Inference Pipeline

```python
class CachePredictionEngine:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = CacheAccessLSTM(input_size=7)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        
        self.recent_accesses = collections.deque(maxlen=50)  # Sequence length
        
    def predict_next_keys(self, recent_features, recent_keys, top_k=10):
        """
        Predict next most likely keys to be accessed
        Must complete in <10ms for real-time use
        """
        with torch.no_grad():
            # Convert to tensors
            features_tensor = torch.FloatTensor(recent_features).unsqueeze(0).to(self.device)
            keys_tensor = torch.LongTensor(recent_keys).unsqueeze(0).to(self.device)
            
            # Model prediction
            key_logits, confidence, _ = self.model(features_tensor, keys_tensor)
            
            # Get top-k predictions
            probabilities = torch.softmax(key_logits, dim=1)
            top_probs, top_keys = torch.topk(probabilities, top_k, dim=1)
            
            predictions = []
            for i in range(top_k):
                predictions.append({
                    'key_id': top_keys[0][i].item(),
                    'probability': top_probs[0][i].item(),
                    'confidence': confidence[0][0].item()
                })
            
            return predictions
    
    def update_history(self, access_event):
        """
        Update model's access history for sequence prediction
        """
        self.recent_accesses.append(access_event)
        
        # Maintain only recent sequence for prediction
        if len(self.recent_accesses) > 50:
            self.recent_accesses.popleft()
```

---

## Story 3.4: Model Deployment & Integration Infrastructure

**Priority**: P1 | **Points**: 21 | **Dependencies**: Story 3.3, Epic 2 completion

### Acceptance Criteria
- [ ] Real-time model inference integration with cache core
- [ ] A/B testing framework comparing ML vs. heuristic prefetching
- [ ] Model performance monitoring and alerting
- [ ] Automated model retraining pipeline
- [ ] Fallback mechanisms when ML predictions fail

### Technical Implementation

#### Integration with Cache Core

```python
class MLPrefetchingEngine:
    def __init__(self, model_path, cache_interface, fallback_strategy='lru'):
        self.prediction_engine = CachePredictionEngine(model_path)
        self.cache_interface = cache_interface
        self.fallback_strategy = fallback_strategy
        self.feature_extractor = RealTimeFeatureExtractor()
        
        # Performance tracking
        self.metrics = {
            'ml_predictions': 0,
            'ml_hits': 0,
            'fallback_predictions': 0,
            'fallback_hits': 0,
            'total_requests': 0,
            'prediction_latency': []
        }
        
    def handle_cache_access(self, key, operation='GET'):
        """
        Main integration point with cache system
        Called on every cache access to update ML state and trigger prefetching
        """
        start_time = time.time()
        
        # Update feature extraction state
        self.feature_extractor.update_with_access(start_time, key)
        self.metrics['total_requests'] += 1
        
        # Check if we should trigger prefetching
        if self._should_trigger_prefetch():
            self._trigger_ml_prefetch()
        
        # Track access for future training data
        self._log_access_for_training(key, operation, start_time)
    
    def _should_trigger_prefetch(self):
        """
        Decide when to trigger ML-based prefetching
        Balance between prediction accuracy and computational cost
        """
        # Trigger every N accesses or based on cache hit rate
        return (self.metrics['total_requests'] % 10 == 0 or 
                self._get_recent_hit_rate() < 0.8)
    
    def _trigger_ml_prefetch(self):
        """
        Generate ML predictions and issue prefetch commands to cache
        """
        try:
            prediction_start = time.time()
            
            # Get recent access features and keys
            recent_features = self.feature_extractor.extract_features_for_prediction(
                time.time(), 
                [access[1] for access in list(self.feature_extractor.access_history)[-50:]]
            )
            
            recent_keys = [access[1] for access in list(self.feature_extractor.access_history)[-50:]]
            recent_key_ids = [hash(key) % 10000 for key in recent_keys]
            
            # Get ML predictions
            predictions = self.prediction_engine.predict_next_keys(
                list(recent_features.values())[-50:],  # Last 50 feature vectors
                recent_key_ids,
                top_k=20
            )
            
            prediction_latency = (time.time() - prediction_start) * 1000  # ms
            self.metrics['prediction_latency'].append(prediction_latency)
            
            # Filter high-confidence predictions
            high_confidence_predictions = [
                p for p in predictions 
                if p['probability'] > 0.1 and p['confidence'] > 0.7
            ]
            
            if high_confidence_predictions:
                # Issue prefetch commands to cache
                prefetch_keys = [f"key_{p['key_id']}" for p in high_confidence_predictions[:10]]
                self._issue_prefetch_commands(prefetch_keys, source='ml')
                self.metrics['ml_predictions'] += len(prefetch_keys)
            else:
                # Fallback to heuristic prefetching
                self._trigger_fallback_prefetch()
                
        except Exception as e:
            print(f"ML prefetch failed: {e}")
            self._trigger_fallback_prefetch()
    
    def _trigger_fallback_prefetch(self):
        """
        Fallback to simple heuristic-based prefetching when ML fails
        """
        # Simple LRU-based prefetching
        if self.fallback_strategy == 'lru':
            recent_keys = list(set([
                access[1] for access in list(self.feature_extractor.access_history)[-100:]
            ]))
            
            # Prefetch most recently accessed keys
            prefetch_keys = recent_keys[-10:]
            self._issue_prefetch_commands(prefetch_keys, source='fallback')
            self.metrics['fallback_predictions'] += len(prefetch_keys)
    
    def _issue_prefetch_commands(self, keys, source='ml'):
        """
        Issue prefetch commands to the cache system
        """
        for key in keys:
            # Check if key is already in cache
            if not self.cache_interface.exists(key):
                # Issue async prefetch command
                self.cache_interface.prefetch_async(key, source=source)
    
    def _log_access_for_training(self, key, operation, timestamp):
        """
        Log access events for future model retraining
        """
        access_event = {
            'timestamp': timestamp,
            'key': key,
            'operation': operation,
            'hour_of_day': datetime.fromtimestamp(timestamp).hour,
            'day_of_week': datetime.fromtimestamp(timestamp).weekday()
        }
        
        # Write to training data log (async)
        self._append_to_training_log(access_event)
    
    def get_performance_metrics(self):
        """
        Return current ML prefetching performance metrics
        """
        total_predictions = self.metrics['ml_predictions'] + self.metrics['fallback_predictions']
        total_hits = self.metrics['ml_hits'] + self.metrics['fallback_hits']
        
        return {
            'total_requests': self.metrics['total_requests'],
            'ml_hit_rate': self.metrics['ml_hits'] / max(self.metrics['ml_predictions'], 1),
            'fallback_hit_rate': self.metrics['fallback_hits'] / max(self.metrics['fallback_predictions'], 1),
            'overall_hit_rate': total_hits / max(total_predictions, 1),
            'avg_prediction_latency_ms': np.mean(self.metrics['prediction_latency']) if self.metrics['prediction_latency'] else 0,
            'ml_prediction_ratio': self.metrics['ml_predictions'] / max(total_predictions, 1)
        }
```

#### A/B Testing Framework

```python
class PrefetchingABTest:
    def __init__(self, cache_interface, test_ratio=0.5):
        self.cache_interface = cache_interface
        self.test_ratio = test_ratio
        
        # Initialize both strategies
        self.ml_engine = MLPrefetchingEngine('best_cache_lstm.pth', cache_interface)
        self.baseline_engine = BaselinePrefetcher(cache_interface)
        
        # Performance tracking
        self.groups = {
            'ml': {'requests': 0, 'hits': 0, 'misses': 0, 'latency': []},
            'baseline': {'requests': 0, 'hits': 0, 'misses': 0, 'latency': []}
        }
        
    def handle_request(self, key, operation='GET'):
        """
        Route requests to either ML or baseline group for A/B testing
        """
        # Deterministic assignment based on key hash
        group = 'ml' if hash(key) % 100 < (self.test_ratio * 100) else 'baseline'
        
        start_time = time.time()
        
        if group == 'ml':
            result = self.ml_engine.handle_cache_access(key, operation)
        else:
            result = self.baseline_engine.handle_cache_access(key, operation)
        
        latency = (time.time() - start_time) * 1000  # ms
        
        # Track performance
        self.groups[group]['requests'] += 1
        self.groups[group]['latency'].append(latency)
        
        # Track hit/miss (would need cache interface feedback)
        if self.cache_interface.get(key) is not None:
            self.groups[group]['hits'] += 1
        else:
            self.groups[group]['misses'] += 1
        
        return result
    
    def get_ab_test_results(self):
        """
        Return A/B test performance comparison
        """
        results = {}
        
        for group_name, metrics in self.groups.items():
            total_requests = metrics['requests']
            if total_requests > 0:
                results[group_name] = {
                    'requests': total_requests,
                    'hit_rate': metrics['hits'] / total_requests,
                    'avg_latency_ms': np.mean(metrics['latency']),
                    'p95_latency_ms': np.percentile(metrics['latency'], 95)
                }
        
        # Statistical significance test
        if len(self.groups['ml']['latency']) > 100 and len(self.groups['baseline']['latency']) > 100:
            from scipy import stats
            ml_hit_rate = self.groups['ml']['hits'] / self.groups['ml']['requests']
            baseline_hit_rate = self.groups['baseline']['hits'] / self.groups['baseline']['requests']
            
            # Chi-square test for hit rate difference
            contingency_table = [
                [self.groups['ml']['hits'], self.groups['ml']['misses']],
                [self.groups['baseline']['hits'], self.groups['baseline']['misses']]
            ]
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
            results['statistical_significance'] = {
                'p_value': p_value,
                'significant': p_value < 0.05,
                'ml_advantage': ml_hit_rate - baseline_hit_rate
            }
        
        return results

class BaselinePrefetcher:
    """
    Simple baseline prefetching strategy for comparison
    """
    def __init__(self, cache_interface):
        self.cache_interface = cache_interface
        self.access_frequency = {}
        self.recent_accesses = collections.deque(maxlen=1000)
        
    def handle_cache_access(self, key, operation='GET'):
        # Update frequency tracking
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
        self.recent_accesses.append(key)
        
        # Simple frequency-based prefetching
        if len(self.recent_accesses) > 50 and len(self.recent_accesses) % 10 == 0:
            # Prefetch most frequently accessed keys
            sorted_keys = sorted(self.access_frequency.items(), key=lambda x: x[1], reverse=True)
            prefetch_keys = [k for k, _ in sorted_keys[:5]]
            
            for key in prefetch_keys:
                if not self.cache_interface.exists(key):
                    self.cache_interface.prefetch_async(key, source='baseline')
```

#### Automated Model Retraining

```python
class ModelRetrainingPipeline:
    def __init__(self, training_log_path, model_save_path, retrain_interval_hours=24):
        self.training_log_path = training_log_path
        self.model_save_path = model_save_path
        self.retrain_interval = retrain_interval_hours * 3600  # Convert to seconds
        self.last_retrain_time = time.time()
        
    def should_retrain(self):
        """
        Check if model should be retrained based on time or data volume
        """
        time_based = (time.time() - self.last_retrain_time) > self.retrain_interval
        
        # Data volume based (retrain when enough new data)
        try:
            current_log_size = os.path.getsize(self.training_log_path)
            data_based = current_log_size > 100 * 1024 * 1024  # 100MB of new data
        except:
            data_based = False
            
        return time_based or data_based
    
    def retrain_model(self):
        """
        Retrain model with new real-world access data
        """
        print("Starting model retraining...")
        
        try:
            # Load new training data
            new_data = self._load_training_log()
            
            # Combine with synthetic data if needed
            if len(new_data) < 10000:  # Not enough real data yet
                synthetic_data = self._load_synthetic_data()
                training_data = synthetic_data + new_data
            else:
                training_data = new_data
            
            # Retrain model
            model, accuracy = train_lstm_model(training_data, epochs=50)
            
            # Validate performance improvement
            if accuracy > 0.7:  # Reasonable threshold
                # Save new model
                torch.save(model.state_dict(), f"{self.model_save_path}_v{int(time.time())}.pth")
                torch.save(model.state_dict(), self.model_save_path)  # Current version
                
                self.last_retrain_time = time.time()
                print(f"Model retrained successfully. New accuracy: {accuracy:.4f}")
                return True
            else:
                print(f"Retraining failed. Accuracy too low: {accuracy:.4f}")
                return False
                
        except Exception as e:
            print(f"Model retraining failed: {e}")
            return False
    
    def _load_training_log(self):
        """
        Load and preprocess recent access logs for training
        """
        import pandas as pd
        
        # Load raw access logs
        df = pd.read_json(self.training_log_path, lines=True)
        
        # Feature engineering on real data
        feature_extractor = CacheAccessFeatureExtractor()
        df = feature_extractor.extract_temporal_features(df)
        df = feature_extractor.extract_sequence_features(df)
        
        return df.to_dict('records')
```

---

## Story 3.5: Performance Monitoring & Optimization

**Priority**: P2 | **Points**: 8 | **Dependencies**: Story 3.4

### Acceptance Criteria
- [ ] Real-time performance dashboard
- [ ] Model drift detection and alerting
- [ ] Prediction latency optimization (<5ms target)
- [ ] Cache hit rate improvement measurement
- [ ] Automated performance reporting

### Technical Implementation

```python
class MLPerformanceMonitor:
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'hit_rate_drop': 0.05,  # Alert if hit rate drops by 5%
            'prediction_latency_ms': 10,  # Alert if prediction takes >10ms
            'model_accuracy': 0.6  # Alert if accuracy drops below 60%
        }
        
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.alerts = []
        
    def update_metrics(self, ml_engine_metrics):
        """
        Update current performance metrics
        """
        self.current_metrics = {
            'timestamp': time.time(),
            'hit_rate': ml_engine_metrics['ml_hit_rate'],
            'prediction_latency': ml_engine_metrics['avg_prediction_latency_ms'],
            'total_requests': ml_engine_metrics['total_requests'],
            'ml_usage_ratio': ml_engine_metrics['ml_prediction_ratio']
        }
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """
        Check current metrics against alert thresholds
        """
        if not self.baseline_metrics:
            return  # No baseline to compare against
        
        # Hit rate drop alert
        hit_rate_drop = self.baseline_metrics['hit_rate'] - self.current_metrics['hit_rate']
        if hit_rate_drop > self.alert_thresholds['hit_rate_drop']:
            self._create_alert('hit_rate_drop', hit_rate_drop)
        
        # Latency alert
        if self.current_metrics['prediction_latency'] > self.alert_thresholds['prediction_latency_ms']:
            self._create_alert('high_latency', self.current_metrics['prediction_latency'])
    
    def _create_alert(self, alert_type, value):
        """
        Create performance alert
        """
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'value': value,
            'threshold': self.alert_thresholds.get(alert_type),
            'severity': 'high' if alert_type == 'hit_rate_drop' else 'medium'
        }
        
        self.alerts.append(alert)
        print(f"ALERT: {alert_type} - Value: {value}, Threshold: {alert['threshold']}")
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. **Story 3.1**: Implement synthetic data generation
2. **Story 3.2**: Build feature engineering pipeline
3. **Basic validation**: Ensure synthetic data looks realistic

### Phase 2: ML Core (Week 3-4)  
1. **Story 3.3**: Train initial LSTM model on synthetic data
2. **Model validation**: Achieve >70% accuracy on synthetic patterns
3. **Performance optimization**: Get inference <10ms

### Phase 3: Integration (Week 5-6)
1. **Story 3.4**: Integrate ML with cache system
2. **A/B testing**: Compare ML vs. baseline prefetching
3. **Real data collection**: Start logging actual cache accesses

### Phase 4: Production Ready (Week 7-8)
1. **Story 3.5**: Performance monitoring and alerting
2. **Model retraining**: Automated pipeline with real data
3. **Documentation**: Complete API docs and usage guides

## Success Metrics

### Technical Metrics
- **Model Accuracy**: >70% on synthetic data, >60% on real data
- **Prediction Latency**: <10ms for real-time inference
- **Cache Hit Rate Improvement**: 15-25% over baseline heuristics
- **System Reliability**: 99.9% uptime for ML predictions

### Business Metrics  
- **Performance Demo**: Clear improvement in cache efficiency
- **Investor Appeal**: Technical differentiation story
- **Scalability Proof**: Works with realistic data volumes
- **Future Roadmap**: Clear path to advanced ML features

This epic provides the ML foundation needed to demonstrate Predis's intelligent prefetching capabilities while building toward a production-ready system that can learn from real-world usage patterns.