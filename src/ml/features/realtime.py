"""
Real-time Feature Extraction for Predis ML

This module provides real-time feature extraction optimized for Predis' ML-driven prefetching,
leveraging the multi-strategy zero-copy memory interface system for ultra-low latency.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import os
from collections import deque, defaultdict
import threading
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealtimeFeatureExtractor:
    """
    Extracts features in real-time from cache access streams
    
    This class is optimized for low-latency feature extraction using
    Predis' multi-strategy zero-copy memory interface system.
    """
    
    def __init__(self, 
                window_size: int = 100, 
                update_interval: float = 0.01,
                use_gpu: bool = True):
        """
        Initialize real-time feature extractor
        
        Args:
            window_size: Number of recent accesses to keep in memory
            update_interval: Minimum time between feature updates (seconds)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.use_gpu = use_gpu
        self.has_gpu = False
        
        # Access history as a circular buffer
        self.access_history = deque(maxlen=window_size)
        
        # Feature cache to avoid redundant computation
        self.feature_cache = {}
        
        # State for tracking feature computation
        self.last_update_time = 0
        self.num_updates = 0
        self.running = False
        self.lock = threading.RLock()
        
        # Key metadata
        self.key_stats = defaultdict(lambda: {
            'access_count': 0,
            'last_access_time': 0,
            'inter_access_times': deque(maxlen=10),
            'last_prediction_time': 0,
            'prediction_success_rate': 0.0
        })
        
        # Statistics
        self.stats = {
            'feature_extraction_time': deque(maxlen=1000),
            'features_per_second': 0.0,
            'cache_hit_rate': 0.0,
            'total_access_count': 0
        }
        
        # Try to import GPU libraries if requested
        if use_gpu:
            try:
                import cupy as cp
                import cudf
                self.has_gpu = True
                logger.info("GPU libraries available, using GPU acceleration")
            except ImportError:
                logger.warning("GPU libraries not available, falling back to CPU implementation")
    
    def start(self) -> None:
        """Start real-time feature extraction"""
        with self.lock:
            if not self.running:
                self.running = True
                logger.info("Started real-time feature extraction")
    
    def stop(self) -> None:
        """Stop real-time feature extraction"""
        with self.lock:
            if self.running:
                self.running = False
                logger.info("Stopped real-time feature extraction")
    
    def add_access(self, key: str, operation: str, timestamp: Optional[float] = None) -> None:
        """
        Add a new access event to the history
        
        Args:
            key: Cache key
            operation: Operation type (GET, PUT, DEL)
            timestamp: Access timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Create access record
        access = {
            'key': key,
            'operation': operation,
            'timestamp': timestamp
        }
        
        with self.lock:
            # Add to history
            self.access_history.append(access)
            
            # Update key stats
            if key in self.key_stats:
                prev_time = self.key_stats[key]['last_access_time']
                if prev_time > 0:
                    inter_access_time = timestamp - prev_time
                    self.key_stats[key]['inter_access_times'].append(inter_access_time)
            
            self.key_stats[key]['access_count'] += 1
            self.key_stats[key]['last_access_time'] = timestamp
            
            # Update global stats
            self.stats['total_access_count'] += 1
    
    def extract_features_for_key(self, key: str) -> Dict[str, float]:
        """
        Extract features for a specific key in real-time
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary of feature name to value
        """
        current_time = time.time()
        
        # Check if we need to update features
        with self.lock:
            # If key not in history, return empty features
            if key not in self.key_stats:
                return {}
            
            # Check if we can use cached features
            if key in self.feature_cache:
                cache_time, features = self.feature_cache[key]
                if current_time - cache_time < self.update_interval:
                    self.stats['cache_hit_rate'] = (
                        0.99 * self.stats['cache_hit_rate'] + 0.01
                    )
                    return features
            
            # Need to compute new features
            start_time = time.time()
            
            # Extract features from history
            features = {}
            
            # Temporal features
            features['recency'] = current_time - self.key_stats[key]['last_access_time']
            
            # Frequency features
            features['access_count'] = self.key_stats[key]['access_count']
            
            # Get recent accesses for this key
            recent_key_accesses = [
                access for access in self.access_history
                if access['key'] == key
            ]
            
            if recent_key_accesses:
                # Calculate inter-access time statistics if we have enough data
                inter_access_times = self.key_stats[key]['inter_access_times']
                if len(inter_access_times) > 1:
                    features['mean_inter_access_time'] = np.mean(inter_access_times)
                    features['std_inter_access_time'] = np.std(inter_access_times)
                    features['min_inter_access_time'] = np.min(inter_access_times)
                    features['max_inter_access_time'] = np.max(inter_access_times)
                    features['cv_inter_access_time'] = (
                        features['std_inter_access_time'] / features['mean_inter_access_time']
                        if features['mean_inter_access_time'] > 0 else 0
                    )
                else:
                    # Default values for new keys
                    features['mean_inter_access_time'] = 0
                    features['std_inter_access_time'] = 0
                    features['min_inter_access_time'] = 0
                    features['max_inter_access_time'] = 0
                    features['cv_inter_access_time'] = 0
                
                # Operation type ratios
                get_count = sum(1 for access in recent_key_accesses 
                              if access['operation'] == 'GET')
                put_count = sum(1 for access in recent_key_accesses 
                              if access['operation'] == 'PUT')
                del_count = sum(1 for access in recent_key_accesses 
                              if access['operation'] == 'DEL')
                
                total = get_count + put_count + del_count
                if total > 0:
                    features['get_ratio'] = get_count / total
                    features['put_ratio'] = put_count / total
                    features['del_ratio'] = del_count / total
                else:
                    features['get_ratio'] = 0
                    features['put_ratio'] = 0
                    features['del_ratio'] = 0
            
            # Update feature cache
            self.feature_cache[key] = (current_time, features)
            
            # Update stats
            extraction_time = time.time() - start_time
            self.stats['feature_extraction_time'].append(extraction_time)
            self.stats['features_per_second'] = (
                0.9 * self.stats['features_per_second'] + 
                0.1 * (1.0 / max(extraction_time, 1e-6))
            )
            self.stats['cache_hit_rate'] = 0.99 * self.stats['cache_hit_rate']
            
            self.num_updates += 1
            self.last_update_time = current_time
            
            return features
    
    def extract_sequence_features(self, sequence_length: int = 10) -> Dict[str, Any]:
        """
        Extract features from the recent access sequence
        
        Args:
            sequence_length: Number of recent accesses to use
            
        Returns:
            Dictionary of sequence features
        """
        with self.lock:
            # Get recent access history
            recent_accesses = list(self.access_history)[-sequence_length:]
            
            if len(recent_accesses) < 2:
                return {}
            
            # Extract sequence
            keys = [access['key'] for access in recent_accesses]
            timestamps = [access['timestamp'] for access in recent_accesses]
            operations = [access['operation'] for access in recent_accesses]
            
            # Calculate basic sequence features
            features = {}
            
            # Time-based features
            features['sequence_duration'] = timestamps[-1] - timestamps[0]
            features['mean_inter_access_time'] = features['sequence_duration'] / (len(timestamps) - 1)
            
            # Key repetition features
            unique_keys = set(keys)
            features['unique_key_ratio'] = len(unique_keys) / len(keys)
            
            # Most common key
            key_counts = {}
            for key in keys:
                key_counts[key] = key_counts.get(key, 0) + 1
            
            most_common_key = max(key_counts.items(), key=lambda x: x[1])[0]
            features['most_common_key'] = most_common_key
            features['most_common_key_ratio'] = key_counts[most_common_key] / len(keys)
            
            # Operation type features
            features['get_ratio'] = operations.count('GET') / len(operations)
            features['put_ratio'] = operations.count('PUT') / len(operations)
            features['del_ratio'] = operations.count('DEL') / len(operations)
            
            # Check for repeating patterns
            has_pattern = False
            pattern_length = 0
            
            # Try pattern lengths from 2 to half the sequence length
            for length in range(2, len(keys) // 2 + 1):
                # Check if last 'length' keys match previous 'length' keys
                if keys[-length:] == keys[-2*length:-length]:
                    has_pattern = True
                    pattern_length = length
                    break
            
            features['has_repeating_pattern'] = int(has_pattern)
            features['pattern_length'] = pattern_length
            
            return features
    
    def extract_features_gpu(self, key: str) -> Dict[str, float]:
        """
        Extract features using GPU acceleration
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary of feature name to value
        """
        if not self.has_gpu:
            logger.warning("GPU libraries not available, using CPU implementation")
            return self.extract_features_for_key(key)
        
        try:
            import cupy as cp
            import cudf
            
            with self.lock:
                # Check if key exists
                if key not in self.key_stats:
                    return {}
                
                # Check cache
                current_time = time.time()
                if key in self.feature_cache:
                    cache_time, features = self.feature_cache[key]
                    if current_time - cache_time < self.update_interval:
                        return features
                
                # Convert access history to cuDF DataFrame for GPU processing
                accesses = list(self.access_history)
                if not accesses:
                    return {}
                
                # Create DataFrame
                df = pd.DataFrame(accesses)
                gdf = cudf.DataFrame.from_pandas(df)
                
                # Filter for target key
                key_gdf = gdf[gdf['key'] == key]
                
                if len(key_gdf) == 0:
                    return {}
                
                # Extract features
                features = {}
                
                # Temporal features
                features['recency'] = current_time - self.key_stats[key]['last_access_time']
                features['access_count'] = self.key_stats[key]['access_count']
                
                # Calculate inter-access time statistics if we have enough data
                inter_access_times = self.key_stats[key]['inter_access_times']
                if len(inter_access_times) > 1:
                    # Use GPU to calculate statistics
                    iat_array = cp.array(inter_access_times)
                    features['mean_inter_access_time'] = float(cp.mean(iat_array))
                    features['std_inter_access_time'] = float(cp.std(iat_array))
                    features['min_inter_access_time'] = float(cp.min(iat_array))
                    features['max_inter_access_time'] = float(cp.max(iat_array))
                    features['cv_inter_access_time'] = (
                        features['std_inter_access_time'] / features['mean_inter_access_time']
                        if features['mean_inter_access_time'] > 0 else 0
                    )
                else:
                    # Default values for new keys
                    features['mean_inter_access_time'] = 0
                    features['std_inter_access_time'] = 0
                    features['min_inter_access_time'] = 0
                    features['max_inter_access_time'] = 0
                    features['cv_inter_access_time'] = 0
                
                # Operation type ratios
                if len(key_gdf) > 0:
                    get_count = len(key_gdf[key_gdf['operation'] == 'GET'])
                    put_count = len(key_gdf[key_gdf['operation'] == 'PUT'])
                    del_count = len(key_gdf[key_gdf['operation'] == 'DEL'])
                    
                    total = get_count + put_count + del_count
                    if total > 0:
                        features['get_ratio'] = get_count / total
                        features['put_ratio'] = put_count / total
                        features['del_ratio'] = del_count / total
                    else:
                        features['get_ratio'] = 0
                        features['put_ratio'] = 0
                        features['del_ratio'] = 0
                
                # Update feature cache
                self.feature_cache[key] = (current_time, features)
                
                return features
                
        except Exception as e:
            logger.error(f"Error in GPU feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            return self.extract_features_for_key(key)
    
    def predict_next_access(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next keys to be accessed
        
        Args:
            top_n: Number of predictions to return
            
        Returns:
            List of (key, probability) tuples
        """
        with self.lock:
            if len(self.access_history) < 2:
                return []
            
            # Get recent keys
            recent_keys = [access['key'] for access in self.access_history][-10:]
            
            # Calculate prediction scores based on:
            # 1. Recency (more recent = higher score)
            # 2. Frequency (more frequent = higher score)
            # 3. Periodicity (regular access pattern = higher score)
            
            scores = {}
            current_time = time.time()
            
            for key, stats in self.key_stats.items():
                # Skip keys with no recent accesses
                if current_time - stats['last_access_time'] > 3600:  # 1 hour cutoff
                    continue
                
                # Base score
                score = 0.0
                
                # Recency score (exponential decay)
                time_since_last = current_time - stats['last_access_time']
                recency_score = np.exp(-0.01 * time_since_last)  # Half-life ~70 seconds
                
                # Frequency score (logarithmic scaling)
                frequency_score = np.log1p(stats['access_count']) / 10.0  # Normalize
                
                # Periodicity score
                periodicity_score = 0.0
                if len(stats['inter_access_times']) > 1:
                    # Calculate coefficient of variation (lower = more regular)
                    mean_iat = np.mean(stats['inter_access_times'])
                    std_iat = np.std(stats['inter_access_times'])
                    if mean_iat > 0:
                        cv = std_iat / mean_iat
                        # Invert CV: lower CV = higher score
                        periodicity_score = np.exp(-cv) / 2.0  # Max 0.5
                        
                        # If due for access, boost score
                        if len(stats['inter_access_times']) >= 3:
                            expected_time = stats['last_access_time'] + mean_iat
                            if current_time >= expected_time:
                                # Boost score if we're past the expected time
                                time_overdue = current_time - expected_time
                                overdue_factor = min(time_overdue / mean_iat, 2.0)
                                periodicity_score *= (1.0 + overdue_factor)
                
                # Pattern score (if key appears in recent sequence)
                pattern_score = 0.0
                for i, recent_key in enumerate(reversed(recent_keys)):
                    if recent_key == key:
                        # More recent occurrence gets higher score
                        pattern_score = 0.3 * (0.9 ** i)
                        break
                
                # Combine scores
                score = recency_score * 0.4 + frequency_score * 0.3 + periodicity_score * 0.2 + pattern_score * 0.1
                
                scores[key] = score
            
            # Sort by score (descending)
            sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N predictions
            return sorted_predictions[:top_n]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feature extractor
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate average feature extraction time
            if self.stats['feature_extraction_time']:
                stats['avg_feature_extraction_time'] = np.mean(self.stats['feature_extraction_time'])
                stats['max_feature_extraction_time'] = np.max(self.stats['feature_extraction_time'])
                stats['min_feature_extraction_time'] = np.min(self.stats['feature_extraction_time'])
            else:
                stats['avg_feature_extraction_time'] = 0
                stats['max_feature_extraction_time'] = 0
                stats['min_feature_extraction_time'] = 0
            
            # Add extractor state
            stats['window_size'] = self.window_size
            stats['update_interval'] = self.update_interval
            stats['num_keys_tracked'] = len(self.key_stats)
            stats['num_updates'] = self.num_updates
            stats['running'] = self.running
            stats['has_gpu'] = self.has_gpu
            
            return stats
    
    def save_state(self, filepath: str) -> None:
        """
        Save extractor state to file
        
        Args:
            filepath: Path to save file
        """
        with self.lock:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Copy state
            state = {
                'key_stats': dict(self.key_stats),
                'feature_cache': self.feature_cache,
                'stats': self.stats,
                'window_size': self.window_size,
                'update_interval': self.update_interval,
                'num_updates': self.num_updates,
                'last_update_time': self.last_update_time
            }
            
            # Don't save deque objects directly
            for key, stats in state['key_stats'].items():
                if 'inter_access_times' in stats:
                    state['key_stats'][key]['inter_access_times'] = list(stats['inter_access_times'])
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Saved extractor state to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load extractor state from file
        
        Args:
            filepath: Path to load file
        """
        with self.lock:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.feature_cache = state['feature_cache']
            self.stats = state['stats']
            self.window_size = state['window_size']
            self.update_interval = state['update_interval']
            self.num_updates = state['num_updates']
            self.last_update_time = state['last_update_time']
            
            # Convert lists back to deques
            self.key_stats = defaultdict(lambda: {
                'access_count': 0,
                'last_access_time': 0,
                'inter_access_times': deque(maxlen=10),
                'last_prediction_time': 0,
                'prediction_success_rate': 0.0
            })
            
            for key, stats in state['key_stats'].items():
                self.key_stats[key] = stats.copy()
                if 'inter_access_times' in stats:
                    self.key_stats[key]['inter_access_times'] = deque(
                        stats['inter_access_times'], maxlen=10)
            
            logger.info(f"Loaded extractor state from {filepath}")


class ZeroCopyFeatureExtractor:
    """
    Feature extractor optimized for Predis' zero-copy memory interface
    
    This extractor is designed to work with the three memory access strategies:
    1. GPU-Direct pathway for lowest-latency access
    2. Optimized UVM integration with ML-driven page placement
    3. Custom peer mapping with explicit coherence control
    """
    
    def __init__(self, 
                strategy: str = 'auto',
                device_id: int = 0,
                max_batch_size: int = 128):
        """
        Initialize zero-copy feature extractor
        
        Args:
            strategy: Memory access strategy ('gpu_direct', 'uvm', 'peer_mapping', or 'auto')
            device_id: GPU device ID
            max_batch_size: Maximum batch size for feature extraction
        """
        self.strategy = strategy
        self.device_id = device_id
        self.max_batch_size = max_batch_size
        self.has_gpu = False
        self.extractor = RealtimeFeatureExtractor(use_gpu=True)
        
        # Feature buffer for zero-copy access
        self.feature_buffer = None
        self.buffer_keys = []
        
        # Try to import GPU libraries
        try:
            import cupy as cp
            self.has_gpu = True
            
            # Initialize device
            cp.cuda.Device(device_id).use()
            
            # Select strategy if 'auto'
            if strategy == 'auto':
                # Check if GPU-Direct is available
                try:
                    # Try to allocate a small array using GPU-Direct
                    # This would check for PCIe BAR1 or NVLink support
                    mem_info = cp.cuda.runtime.memGetInfo()
                    if mem_info[0] / mem_info[1] > 0.1:  # If > 10% free memory
                        self.strategy = 'gpu_direct'
                    else:
                        self.strategy = 'uvm'
                except:
                    self.strategy = 'uvm'
            
            logger.info(f"Using {self.strategy} memory access strategy")
            
        except ImportError:
            logger.warning("GPU libraries not available, falling back to CPU implementation")
            self.strategy = 'cpu'
    
    def extract_features(self, 
                        keys: List[str], 
                        sync: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Extract features for a batch of keys using zero-copy memory interface
        
        Args:
            keys: List of cache keys
            sync: Whether to synchronize after extraction
            
        Returns:
            Dictionary mapping keys to feature dictionaries
        """
        if not self.has_gpu or self.strategy == 'cpu':
            # Fall back to CPU implementation
            return {key: self.extractor.extract_features_for_key(key) for key in keys}
        
        try:
            import cupy as cp
            
            # Limit batch size
            if len(keys) > self.max_batch_size:
                keys = keys[:self.max_batch_size]
            
            # Extract features for each key
            features = {}
            
            if self.strategy == 'gpu_direct':
                # GPU-Direct strategy for lowest latency
                # Extract features directly on GPU
                for key in keys:
                    features[key] = self.extractor.extract_features_gpu(key)
                
                # Allocate feature buffer if needed
                if self.feature_buffer is None or len(self.buffer_keys) != len(keys):
                    # Free old buffer if it exists
                    if self.feature_buffer is not None:
                        del self.feature_buffer
                    
                    # Allocate new buffer
                    feature_array = self._features_to_array(features)
                    self.feature_buffer = cp.asarray(feature_array)
                    self.buffer_keys = list(keys)
                else:
                    # Update existing buffer
                    feature_array = self._features_to_array(features)
                    self.feature_buffer.set(feature_array)
                    self.buffer_keys = list(keys)
                
                # Synchronize if requested
                if sync:
                    cp.cuda.Stream.null.synchronize()
            
            elif self.strategy == 'uvm':
                # UVM strategy for larger data
                # Extract features on CPU
                for key in keys:
                    features[key] = self.extractor.extract_features_for_key(key)
                
                # Allocate UVM feature buffer if needed
                if self.feature_buffer is None or len(self.buffer_keys) != len(keys):
                    # Free old buffer if it exists
                    if self.feature_buffer is not None:
                        del self.feature_buffer
                    
                    # Allocate new buffer with UVM memory
                    feature_array = self._features_to_array(features)
                    self.feature_buffer = cp.asarray(feature_array)
                    self.buffer_keys = list(keys)
                else:
                    # Update existing buffer
                    feature_array = self._features_to_array(features)
                    self.feature_buffer.set(feature_array)
                    self.buffer_keys = list(keys)
                
                # Prefetch to GPU
                if hasattr(self.feature_buffer, 'prefetch'):
                    self.feature_buffer.prefetch(cp.cuda.runtime.cudaMemcpyHostToDevice)
                
                # Synchronize if requested
                if sync:
                    cp.cuda.Stream.null.synchronize()
            
            else:  # peer_mapping
                # Custom peer mapping strategy
                # Extract features on CPU
                for key in keys:
                    features[key] = self.extractor.extract_features_for_key(key)
                
                # For peer mapping, we use a different approach
                # Instead of a single buffer, use a dictionary of arrays
                feature_dict = {}
                
                for feature_name in self._get_feature_names(features):
                    # Extract values for this feature
                    values = np.array([
                        features[key].get(feature_name, 0.0) for key in keys
                    ], dtype=np.float32)
                    
                    # Create GPU array
                    feature_dict[feature_name] = cp.asarray(values)
                
                # Store reference to feature dictionary
                self.feature_buffer = feature_dict
                self.buffer_keys = list(keys)
                
                # Synchronize if requested
                if sync:
                    cp.cuda.Stream.null.synchronize()
            
            return features
            
        except Exception as e:
            logger.error(f"Error in zero-copy feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            return {key: self.extractor.extract_features_for_key(key) for key in keys}
    
    def _features_to_array(self, features: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array
        
        Args:
            features: Dictionary mapping keys to feature dictionaries
            
        Returns:
            NumPy array with features
        """
        if not features:
            return np.array([], dtype=np.float32)
        
        # Get all feature names
        feature_names = self._get_feature_names(features)
        
        # Create array
        keys = list(features.keys())
        array = np.zeros((len(keys), len(feature_names)), dtype=np.float32)
        
        # Fill array
        for i, key in enumerate(keys):
            for j, feature_name in enumerate(feature_names):
                array[i, j] = features[key].get(feature_name, 0.0)
        
        return array
    
    def _get_feature_names(self, features: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Get all feature names from feature dictionary
        
        Args:
            features: Dictionary mapping keys to feature dictionaries
            
        Returns:
            List of feature names
        """
        # Collect all feature names
        feature_names = set()
        for key, key_features in features.items():
            feature_names.update(key_features.keys())
        
        # Sort for consistent order
        return sorted(feature_names)
    
    def get_feature_buffer(self) -> Any:
        """
        Get the zero-copy feature buffer
        
        Returns:
            Feature buffer (CuPy array or dictionary of arrays)
        """
        return self.feature_buffer
    
    def get_buffer_keys(self) -> List[str]:
        """
        Get the keys in the current feature buffer
        
        Returns:
            List of keys
        """
        return self.buffer_keys
    
    def get_strategy(self) -> str:
        """
        Get the current memory access strategy
        
        Returns:
            Memory access strategy
        """
        return self.strategy
    
    def select_optimal_strategy(self, data_size_mb: float) -> str:
        """
        Select optimal memory access strategy based on data size
        
        Args:
            data_size_mb: Estimated data size in MB
            
        Returns:
            Optimal strategy
        """
        if not self.has_gpu:
            return 'cpu'
        
        if data_size_mb < 50:
            # Small data, use GPU-Direct for lowest latency
            strategy = 'gpu_direct'
        elif data_size_mb < 500:
            # Medium data, use UVM
            strategy = 'uvm'
        else:
            # Large data, use peer mapping
            strategy = 'peer_mapping'
        
        self.strategy = strategy
        logger.info(f"Selected {strategy} strategy for {data_size_mb:.2f} MB data")
        
        return strategy
