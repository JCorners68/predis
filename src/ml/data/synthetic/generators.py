"""
Synthetic Data Generators

This module implements various synthetic cache access pattern generators
that simulate realistic workloads for training ML prefetching models.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccessPattern:
    """Base class for access pattern representation"""
    
    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type
        self.access_sequence = []
        
    def add_access(self, timestamp: float, key: str, operation: str, 
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a single access to the sequence"""
        access = {
            'timestamp': timestamp,
            'key': key,
            'operation': operation,
            'workload_type': self.pattern_type
        }
        
        if metadata:
            access.update(metadata)
            
        self.access_sequence.append(access)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert access sequence to DataFrame"""
        return pd.DataFrame(self.access_sequence)
    
    def __len__(self):
        return len(self.access_sequence)


def generate_zipfian_access_pattern(num_keys: int = 10000, 
                                   num_accesses: int = 1000000, 
                                   alpha: float = 1.0) -> List[Dict[str, Any]]:
    """
    Generate cache access patterns following Zipfian distribution
    20% of keys get 80% of traffic (typical in real applications)
    
    Args:
        num_keys: Total number of unique keys
        num_accesses: Total number of access operations to generate
        alpha: Zipfian skew parameter (higher = more skewed)
        
    Returns:
        List of access records with timestamp, key, operation and metadata
    """
    logger.info(f"Generating Zipfian access pattern with {num_keys} keys, " 
                f"{num_accesses} accesses, alpha={alpha}")
    
    # Generate key popularity ranks
    ranks = np.arange(1, num_keys + 1)
    probabilities = 1.0 / (ranks ** alpha)
    probabilities = probabilities / probabilities.sum()
    
    # Initialize pattern collector
    pattern = AccessPattern("zipfian")
    
    # Generate access sequence
    base_time = time.time()
    
    for i in range(num_accesses):
        # Select key based on Zipfian probability
        key_id = np.random.choice(num_keys, p=probabilities)
        timestamp = base_time + i * 0.001  # 1ms intervals
        
        # Add access to the pattern
        pattern.add_access(
            timestamp=timestamp,
            key=f"key_{key_id}",
            operation='GET',
            metadata={'popularity_rank': int(ranks[key_id-1])}
        )
    
    logger.info(f"Generated {len(pattern)} Zipfian access records")
    return pattern.access_sequence


def generate_temporal_access_pattern(duration_hours: int = 24, 
                                    keys_per_hour: int = 1000,
                                    num_keys: int = 10000) -> List[Dict[str, Any]]:
    """
    Generate time-based access patterns with daily cycles
    Simulates business hours, peak traffic periods, etc.
    
    Args:
        duration_hours: Total duration to simulate in hours
        keys_per_hour: Base number of keys accessed per hour
        num_keys: Total unique keys in the system
        
    Returns:
        List of access records with timestamp, key, operation and metadata
    """
    logger.info(f"Generating temporal access pattern for {duration_hours} hours")
    
    # Initialize pattern collector
    pattern = AccessPattern("temporal")
    
    # Generate access sequence
    base_time = time.time()
    
    for hour in range(duration_hours):
        # Simulate daily cycle (higher traffic during business hours)
        hour_of_day = hour % 24
        
        # Determine traffic multiplier based on time of day
        if 9 <= hour_of_day <= 17:  # Business hours
            traffic_multiplier = 3.0
        elif 18 <= hour_of_day <= 22:  # Evening peak
            traffic_multiplier = 2.0
        else:  # Night/early morning
            traffic_multiplier = 0.5
            
        # Calculate accesses for this hour
        num_accesses = int(keys_per_hour * traffic_multiplier)
        
        # Generate accesses for this hour
        for access in range(num_accesses):
            # Calculate precise timestamp
            timestamp = base_time + hour * 3600 + access * (3600 / num_accesses)
            
            # Select a key (using uniform distribution here, but could be Zipfian)
            key_id = np.random.randint(0, num_keys)
            
            # Add access to the pattern
            pattern.add_access(
                timestamp=timestamp,
                key=f"temporal_key_{key_id}",
                operation='GET',
                metadata={
                    'hour_of_day': hour_of_day,
                    'day_of_week': (hour // 24) % 7,
                    'traffic_level': traffic_multiplier
                }
            )
    
    logger.info(f"Generated {len(pattern)} temporal access records")
    return pattern.access_sequence


def generate_ml_training_pattern(num_epochs: int = 10, 
                                batch_size: int = 256, 
                                dataset_size: int = 50000) -> List[Dict[str, Any]]:
    """
    Simulate ML training access patterns with sequential batch loading
    
    Args:
        num_epochs: Number of training epochs to simulate
        batch_size: Batch size for training
        dataset_size: Total dataset size (number of samples)
        
    Returns:
        List of access records with timestamp, key, operation and metadata
    """
    logger.info(f"Generating ML training pattern for {num_epochs} epochs, "
                f"{dataset_size} samples, batch size {batch_size}")
    
    # Initialize pattern collector
    pattern = AccessPattern("ml_training")
    
    # Generate access sequence
    base_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle dataset order each epoch (realistic ML training behavior)
        data_indices = np.random.permutation(dataset_size)
        
        # Process batches
        for batch_idx, batch_start in enumerate(range(0, dataset_size, batch_size)):
            # Get indices for this batch
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_indices = data_indices[batch_start:batch_end]
            
            # Sequential access within batch (predictable pattern)
            for i, data_idx in enumerate(batch_indices):
                # Calculate timestamp with realistic timing
                # Each batch takes ~100ms, each sample ~0.1ms within a batch
                timestamp = (base_time + 
                            epoch * (dataset_size / batch_size * 0.1) +  # epoch time
                            batch_idx * 0.1 +                           # batch time
                            i * 0.0001)                                 # sample time
                
                # Add access to the pattern
                pattern.add_access(
                    timestamp=timestamp,
                    key=f"data_{data_idx}",
                    operation='GET',
                    metadata={
                        'epoch': epoch,
                        'batch': batch_idx,
                        'position_in_batch': i,
                        'data_index': int(data_idx)
                    }
                )
    
    logger.info(f"Generated {len(pattern)} ML training access records")
    return pattern.access_sequence


def generate_hft_pattern(symbols: int = 1000, 
                        trades_per_second: int = 10000,
                        duration_seconds: int = 3600) -> List[Dict[str, Any]]:
    """
    Simulate HFT (High-Frequency Trading) access patterns 
    with hot symbols and market events
    
    Args:
        symbols: Total number of trading symbols
        trades_per_second: Number of trades per second
        duration_seconds: Simulation duration in seconds
        
    Returns:
        List of access records with timestamp, key, operation and metadata
    """
    logger.info(f"Generating HFT pattern with {symbols} symbols, "
                f"{trades_per_second} trades/sec for {duration_seconds} seconds")
    
    # Hot symbols (top 10% get 90% of traffic)
    hot_symbols = symbols // 10
    
    # Initialize pattern collector
    pattern = AccessPattern("hft")
    
    # Generate access sequence
    base_time = time.time()
    
    for second in range(duration_seconds):
        # Inject some market volatility - occasionally increase trading volume
        volatility_factor = 1.0
        if np.random.random() < 0.05:  # 5% chance of market event
            volatility_factor = np.random.choice([2.0, 3.0, 5.0])  # 2x, 3x or 5x volume
            
        # Calculate trades this second with volatility
        second_trades = int(trades_per_second * volatility_factor)
        
        for trade in range(second_trades):
            # Precise timestamp
            timestamp = base_time + second + trade / second_trades
            
            # 90% probability of accessing hot symbol
            if np.random.random() < 0.9:
                symbol_id = np.random.randint(0, hot_symbols)
                is_hot = True
            else:
                symbol_id = np.random.randint(hot_symbols, symbols)
                is_hot = False
            
            # Add access to the pattern
            pattern.add_access(
                timestamp=timestamp,
                key=f"price_{symbol_id}",
                operation='GET',
                metadata={
                    'is_hot_symbol': is_hot,
                    'volatility_factor': volatility_factor,
                    'symbol_id': int(symbol_id)
                }
            )
    
    logger.info(f"Generated {len(pattern)} HFT access records")
    return pattern.access_sequence


def generate_gaming_pattern(num_players: int = 1000, 
                           num_game_objects: int = 10000,
                           duration_minutes: int = 60) -> List[Dict[str, Any]]:
    """
    Simulate gaming workload with player focus areas and object popularity
    
    Args:
        num_players: Number of concurrent players
        num_game_objects: Total number of game objects
        duration_minutes: Simulation duration in minutes
        
    Returns:
        List of access records with timestamp, key, operation and metadata
    """
    logger.info(f"Generating gaming pattern with {num_players} players, "
                f"{num_game_objects} objects for {duration_minutes} minutes")
    
    # Initialize pattern collector
    pattern = AccessPattern("gaming")
    
    # Generate access sequence
    base_time = time.time()
    duration_seconds = duration_minutes * 60
    
    # Each player has their own focus area (subset of game objects)
    player_focus_areas = {}
    for player_id in range(num_players):
        # Each player focuses on 5-20% of objects
        focus_size = int(num_game_objects * np.random.uniform(0.05, 0.2))
        player_focus_areas[player_id] = np.random.choice(
            num_game_objects, size=focus_size, replace=False)
    
    # Simulate player interactions over time
    for second in range(duration_seconds):
        # Each player makes 1-10 object accesses per second
        for player_id in range(num_players):
            # Player online probability (some players log off)
            if np.random.random() > 0.8:  # 80% chance player is online
                continue
                
            # Number of objects this player accesses this second
            accesses = np.random.randint(1, 10)
            
            for i in range(accesses):
                # Calculate timestamp (spread within the second)
                timestamp = base_time + second + i / accesses
                
                # 90% of accesses are to player's focus area
                if np.random.random() < 0.9:
                    # Access from player's focus area
                    obj_id = np.random.choice(player_focus_areas[player_id])
                else:
                    # Random access to any object
                    obj_id = np.random.randint(0, num_game_objects)
                
                # Determine operation (mostly reads, some writes)
                operation = 'GET' if np.random.random() < 0.8 else 'PUT'
                
                # Add access to the pattern
                pattern.add_access(
                    timestamp=timestamp,
                    key=f"game_obj_{obj_id}",
                    operation=operation,
                    metadata={
                        'player_id': player_id,
                        'in_focus_area': obj_id in player_focus_areas[player_id],
                        'second': second
                    }
                )
    
    logger.info(f"Generated {len(pattern)} gaming access records")
    return pattern.access_sequence


def export_synthetic_data(access_patterns: List[Dict[str, Any]], 
                         output_file: str) -> pd.DataFrame:
    """
    Export synthetic access patterns to standardized format for ML training
    
    Args:
        access_patterns: List of access pattern dictionaries
        output_file: Base filename for output (without extension)
        
    Returns:
        DataFrame containing the exported data with added features
    """
    logger.info(f"Exporting {len(access_patterns)} access patterns to {output_file}")
    
    # Convert to DataFrame
    df = pd.DataFrame(access_patterns)
    
    # Add derived features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp')
    
    # Add time-based features
    df['time_since_last_access'] = df.groupby('key')['timestamp'].diff()
    df['access_count'] = df.groupby('key').cumcount() + 1
    
    # Export to multiple formats
    df.to_csv(f"{output_file}.csv", index=False)
    df.to_parquet(f"{output_file}.parquet")
    
    # Save a smaller sample for quick testing
    if len(df) > 10000:
        sample_size = min(10000, len(df) // 10)
        df.sample(sample_size).to_csv(f"{output_file}_sample.csv", index=False)
    
    logger.info(f"Data exported successfully to {output_file}.csv and {output_file}.parquet")
    return df


def validate_synthetic_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that synthetic patterns match expected characteristics
    
    Args:
        df: DataFrame containing access patterns
        
    Returns:
        Dictionary of validation metrics and checks
    """
    logger.info(f"Validating synthetic patterns with {len(df)} records")
    validation_results = {}
    
    # Check Zipfian distribution (if applicable)
    if 'zipfian' in df['workload_type'].unique():
        zipf_df = df[df['workload_type'] == 'zipfian']
        
        # Calculate key frequency distribution
        key_counts = zipf_df['key'].value_counts()
        top_20_percent = int(len(key_counts) * 0.2)
        top_20_traffic = key_counts.head(top_20_percent).sum()
        total_traffic = key_counts.sum()
        
        # Calculate Zipfian ratio (should be close to 80/20 rule)
        zipf_ratio = top_20_traffic / total_traffic
        validation_results['zipfian_ratio'] = zipf_ratio
        validation_results['zipfian_valid'] = 0.7 <= zipf_ratio <= 0.9
        validation_results['zipfian_expected_range'] = (0.7, 0.9)
    
    # Check temporal patterns (if applicable)
    if 'temporal' in df['workload_type'].unique():
        temp_df = df[df['workload_type'] == 'temporal']
        
        # Calculate hourly distribution
        hourly_traffic = temp_df.groupby('hour_of_day').size()
        
        # Check peak vs off-peak ratio
        validation_results['peak_hour_traffic'] = hourly_traffic.max()
        validation_results['off_peak_traffic'] = hourly_traffic.min()
        validation_results['peak_ratio'] = hourly_traffic.max() / hourly_traffic.min()
        validation_results['temporal_valid'] = validation_results['peak_ratio'] >= 2.0
    
    # Check ML training patterns (if applicable)
    if 'ml_training' in df['workload_type'].unique():
        ml_df = df[df['workload_type'] == 'ml_training']
        
        # Check for batch access patterns
        if 'batch' in ml_df.columns:
            batch_sizes = ml_df.groupby('batch').size()
            validation_results['batch_size_std'] = batch_sizes.std() / batch_sizes.mean()
            validation_results['ml_valid'] = validation_results['batch_size_std'] < 0.1
    
    # Check HFT patterns (if applicable)
    if 'hft' in df['workload_type'].unique():
        hft_df = df[df['workload_type'] == 'hft']
        
        # Check hot symbol concentration
        if 'is_hot_symbol' in hft_df.columns:
            hot_symbol_ratio = hft_df['is_hot_symbol'].mean()
            validation_results['hot_symbol_ratio'] = hot_symbol_ratio
            validation_results['hft_valid'] = 0.85 <= hot_symbol_ratio <= 0.95
    
    # Overall validation
    validation_results['patterns_validated'] = sum(
        1 for k, v in validation_results.items() if k.endswith('_valid') and v)
    validation_results['total_pattern_types'] = sum(
        1 for pattern in ['zipfian', 'temporal', 'ml_training', 'hft', 'gaming'] 
        if pattern in df['workload_type'].unique())
    
    validation_results['overall_valid'] = (
        validation_results['patterns_validated'] == validation_results['total_pattern_types'])
    
    logger.info(f"Validation complete: {validation_results['overall_valid']}")
    return validation_results
