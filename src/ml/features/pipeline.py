"""
Feature Engineering Pipeline for Predis ML

This module provides an integrated pipeline for feature engineering that combines
temporal, sequential, and real-time features for Predis' ML-driven prefetching system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import os
from pathlib import Path
import json
import threading

# Import Predis feature extraction components
from .extractors import FeatureExtractor, AccessPatternFeatureExtractor, GPUOptimizedFeatureExtractor
from .temporal import TemporalFeatureGenerator, TemporalPatternDetector
from .sequential import SequenceFeatureGenerator, SequentialPatternMiner
from .realtime import RealtimeFeatureExtractor, ZeroCopyFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Integrated pipeline for feature engineering
    
    This pipeline combines multiple feature extraction approaches to create
    a comprehensive feature set for ML-driven prefetching.
    """
    
    def __init__(self, 
                output_dir: str = "../../../data/ml/features",
                use_gpu: bool = True,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineering pipeline
        
        Args:
            output_dir: Directory for output features
            use_gpu: Whether to use GPU acceleration
            config: Configuration parameters
        """
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default configuration
        self.config = {
            'temporal': {
                'enabled': True,
                'window_size': 24 * 3600  # 1 day
            },
            'sequential': {
                'enabled': True,
                'sequence_length': 10,
                'min_pattern_length': 2,
                'max_pattern_length': 5
            },
            'realtime': {
                'enabled': True,
                'window_size': 100,
                'update_interval': 0.01
            },
            'zero_copy': {
                'enabled': use_gpu,
                'strategy': 'auto',
                'max_batch_size': 128
            },
            'validation': {
                'enabled': True,
                'validation_fraction': 0.2
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Initialize extractors
        self._init_extractors()
        
        # Pipeline state
        self.is_initialized = False
        self.is_running = False
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'extraction_time': [],
            'features_per_second': 0,
            'total_features_extracted': 0,
            'last_extraction_time': 0
        }
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with provided values"""
        for section, section_config in config.items():
            if section in self.config:
                if isinstance(section_config, dict):
                    self.config[section].update(section_config)
                else:
                    self.config[section] = section_config
    
    def _init_extractors(self) -> None:
        """Initialize feature extractors based on configuration"""
        # Temporal features
        if self.config['temporal']['enabled']:
            self.temporal_extractor = TemporalFeatureGenerator(use_gpu=self.use_gpu)
            self.temporal_detector = TemporalPatternDetector()
        else:
            self.temporal_extractor = None
            self.temporal_detector = None
        
        # Sequential features
        if self.config['sequential']['enabled']:
            self.sequence_extractor = SequenceFeatureGenerator(
                sequence_length=self.config['sequential']['sequence_length'],
                use_gpu=self.use_gpu
            )
            self.sequence_miner = SequentialPatternMiner(
                min_pattern_length=self.config['sequential']['min_pattern_length'],
                max_pattern_length=self.config['sequential']['max_pattern_length']
            )
        else:
            self.sequence_extractor = None
            self.sequence_miner = None
        
        # Real-time features
        if self.config['realtime']['enabled']:
            self.realtime_extractor = RealtimeFeatureExtractor(
                window_size=self.config['realtime']['window_size'],
                update_interval=self.config['realtime']['update_interval'],
                use_gpu=self.use_gpu
            )
        else:
            self.realtime_extractor = None
        
        # Zero-copy features
        if self.config['zero_copy']['enabled']:
            self.zero_copy_extractor = ZeroCopyFeatureExtractor(
                strategy=self.config['zero_copy']['strategy'],
                max_batch_size=self.config['zero_copy']['max_batch_size']
            )
        else:
            self.zero_copy_extractor = None
        
        # Main access pattern extractor
        self.access_extractor = AccessPatternFeatureExtractor(output_dir=self.output_dir)
        
        # GPU-optimized extractor
        if self.use_gpu:
            self.gpu_extractor = GPUOptimizedFeatureExtractor(output_dir=self.output_dir)
        else:
            self.gpu_extractor = None
    
    def initialize(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize pipeline with historical data
        
        Args:
            df: DataFrame with historical access logs
        """
        with self.lock:
            if df is not None:
                logger.info(f"Initializing pipeline with {len(df)} historical records")
                
                # Initialize extractors with historical data
                if self.temporal_extractor:
                    # Extract temporal features
                    logger.info("Extracting temporal features from historical data")
                    df = self.temporal_extractor.extract_all_temporal_features(df)
                    
                    # Detect temporal patterns
                    logger.info("Detecting temporal patterns")
                    self.temporal_detector.detect_key_patterns(df)
                
                if self.sequence_extractor:
                    # Extract sequence features
                    logger.info("Extracting sequence features from historical data")
                    df = self.sequence_extractor.extract_all_sequence_features(df)
                    
                    # Mine sequential patterns
                    logger.info("Mining sequential patterns")
                    self.sequence_miner.mine_patterns(df)
                
                if self.realtime_extractor:
                    # Initialize real-time extractor with recent data
                    logger.info("Initializing real-time extractor")
                    
                    # Sort by timestamp
                    df = df.sort_values('timestamp')
                    
                    # Add recent accesses to real-time extractor
                    recent_df = df.iloc[-min(len(df), 1000):]
                    for _, row in recent_df.iterrows():
                        self.realtime_extractor.add_access(
                            row['key'], 
                            row.get('operation', 'GET'), 
                            row['timestamp']
                        )
                    
                    # Start real-time extraction
                    self.realtime_extractor.start()
            
            self.is_initialized = True
            logger.info("Feature engineering pipeline initialized")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from access logs
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with all extracted features
        """
        start_time = time.time()
        
        with self.lock:
            # Initialize if not already done
            if not self.is_initialized:
                self.initialize(df)
            
            # Make copy to avoid modifying original
            result = df.copy()
            
            # Sort by timestamp
            result = result.sort_values('timestamp')
            
            # Extract features using different extractors
            if self.use_gpu and self.gpu_extractor:
                # Use GPU-optimized extractor for all features
                logger.info("Using GPU-optimized feature extraction")
                result = self.gpu_extractor.extract_features(result)
            else:
                # Use individual extractors
                
                # Access pattern features (base features)
                logger.info("Extracting access pattern features")
                result = self.access_extractor.extract_features(result)
                
                # Temporal features
                if self.temporal_extractor:
                    logger.info("Extracting temporal features")
                    result = self.temporal_extractor.extract_all_temporal_features(result)
                
                # Sequential features
                if self.sequence_extractor:
                    logger.info("Extracting sequential features")
                    result = self.sequence_extractor.extract_all_sequence_features(result)
            
            # Update metrics
            extraction_time = time.time() - start_time
            self.metrics['extraction_time'].append(extraction_time)
            self.metrics['features_per_second'] = len(result) / extraction_time
            self.metrics['total_features_extracted'] += len(result)
            self.metrics['last_extraction_time'] = time.time()
            
            logger.info(f"Extracted features for {len(result)} records in {extraction_time:.2f}s")
            
            return result
    
    def extract_realtime_features(self, 
                                keys: List[str],
                                use_zero_copy: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Extract features in real-time for a batch of keys
        
        Args:
            keys: List of cache keys
            use_zero_copy: Whether to use zero-copy memory interface
            
        Returns:
            Dictionary mapping keys to feature dictionaries
        """
        if not self.is_initialized:
            logger.warning("Pipeline not initialized. Call initialize() first.")
            return {}
        
        # Check if real-time extractors are available
        if use_zero_copy and self.zero_copy_extractor:
            # Use zero-copy extractor for real-time features
            return self.zero_copy_extractor.extract_features(keys)
        elif self.realtime_extractor:
            # Use regular real-time extractor
            return {key: self.realtime_extractor.extract_features_for_key(key) for key in keys}
        else:
            logger.warning("No real-time extractor available")
            return {}
    
    def add_access(self, key: str, operation: str, timestamp: Optional[float] = None) -> None:
        """
        Add a new access event to the real-time extractor
        
        Args:
            key: Cache key
            operation: Operation type (GET, PUT, DEL)
            timestamp: Access timestamp (default: current time)
        """
        if self.realtime_extractor:
            self.realtime_extractor.add_access(key, operation, timestamp)
    
    def predict_next_access(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next keys to be accessed
        
        Args:
            top_n: Number of predictions to return
            
        Returns:
            List of (key, probability) tuples
        """
        if not self.is_initialized:
            logger.warning("Pipeline not initialized. Call initialize() first.")
            return []
        
        if self.realtime_extractor:
            return self.realtime_extractor.predict_next_access(top_n)
        else:
            logger.warning("No real-time extractor available for prediction")
            return []
    
    def get_prefetchable_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        Get keys suitable for prefetching with prefetch metadata
        
        Returns:
            Dictionary mapping keys to prefetch metadata
        """
        prefetchable = {}
        
        # Get keys with temporal patterns
        if self.temporal_detector:
            temporal_keys = self.temporal_detector.get_prefetchable_keys(confidence_threshold=0.7)
            for key in temporal_keys:
                if key not in prefetchable:
                    prefetchable[key] = {'patterns': []}
                
                prefetchable[key]['patterns'].append({
                    'type': 'temporal',
                    'confidence': 0.7,  # Default confidence
                    'next_access': self.temporal_detector.get_next_access_time(key, time.time())
                })
        
        # Get keys with sequential patterns
        if self.sequence_miner:
            try:
                sequential_patterns = self.sequence_miner.get_prefetchable_patterns(min_confidence=0.5)
                for pattern in sequential_patterns:
                    # Last key in pattern is the one to prefetch
                    key = pattern['pattern'][-1]
                    
                    if key not in prefetchable:
                        prefetchable[key] = {'patterns': []}
                    
                    prefetchable[key]['patterns'].append({
                        'type': 'sequential',
                        'confidence': pattern['confidence'],
                        'trigger_pattern': pattern['pattern'][:-1]  # Pattern that triggers prefetch
                    })
            except Exception as e:
                logger.error(f"Error getting sequential patterns: {e}")
        
        # Add metadata for each prefetchable key
        for key in prefetchable:
            # Combine confidence from all patterns
            patterns = prefetchable[key]['patterns']
            if patterns:
                # Use max confidence as the overall confidence
                prefetchable[key]['confidence'] = max(p.get('confidence', 0) for p in patterns)
                
                # Add timestamp
                prefetchable[key]['timestamp'] = time.time()
        
        return prefetchable
    
    def save_pipeline_state(self, filename: Optional[str] = None) -> str:
        """
        Save pipeline state to file
        
        Args:
            filename: Optional filename, default is timestamped state file
            
        Returns:
            Path to saved state file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"pipeline_state_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create state dictionary
        state = {
            'config': self.config,
            'metrics': {
                'total_features_extracted': self.metrics['total_features_extracted'],
                'features_per_second': self.metrics['features_per_second'],
                'last_extraction_time': self.metrics['last_extraction_time']
            },
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'timestamp': time.time()
        }
        
        # Save state file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved pipeline state to {filepath}")
        
        # Save component states
        components_dir = os.path.join(self.output_dir, "components")
        os.makedirs(components_dir, exist_ok=True)
        
        if self.realtime_extractor:
            rt_path = os.path.join(components_dir, f"realtime_{timestamp}.pkl")
            self.realtime_extractor.save_state(rt_path)
        
        return filepath
    
    def load_pipeline_state(self, filepath: str) -> None:
        """
        Load pipeline state from file
        
        Args:
            filepath: Path to state file
        """
        # Load state file
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Update configuration
        self._update_config(state['config'])
        
        # Reinitialize extractors with new config
        self._init_extractors()
        
        # Restore metrics
        self.metrics['total_features_extracted'] = state['metrics']['total_features_extracted']
        self.metrics['features_per_second'] = state['metrics']['features_per_second']
        self.metrics['last_extraction_time'] = state['metrics']['last_extraction_time']
        
        # Restore state flags
        self.is_initialized = state['is_initialized']
        self.is_running = state['is_running']
        
        # Try to load component states
        components_dir = os.path.join(self.output_dir, "components")
        
        # Extract timestamp from filepath
        timestamp = os.path.basename(filepath).split('_')[-1].split('.')[0]
        
        # Try to load realtime extractor state
        rt_path = os.path.join(components_dir, f"realtime_{timestamp}.pkl")
        if os.path.exists(rt_path) and self.realtime_extractor:
            self.realtime_extractor.load_state(rt_path)
        
        logger.info(f"Loaded pipeline state from {filepath}")


class ZeroCopyFeaturePipeline:
    """
    Feature pipeline optimized for Predis' zero-copy memory interface
    
    This pipeline is specifically designed to work with Predis' three memory access strategies:
    1. GPU-Direct pathway for lowest-latency access
    2. Optimized UVM integration with ML-driven page placement
    3. Custom peer mapping with explicit coherence control
    """
    
    def __init__(self, 
                strategy: str = 'auto',
                device_id: int = 0,
                output_dir: str = "../../../data/ml/features"):
        """
        Initialize zero-copy feature pipeline
        
        Args:
            strategy: Memory access strategy ('gpu_direct', 'uvm', 'peer_mapping', or 'auto')
            device_id: GPU device ID
            output_dir: Directory for output features
        """
        self.strategy = strategy
        self.device_id = device_id
        self.output_dir = output_dir
        self.has_gpu = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize zero-copy extractors
        self.zero_copy_extractor = ZeroCopyFeatureExtractor(strategy, device_id)
        self.realtime_extractor = RealtimeFeatureExtractor(use_gpu=True)
        
        # Feature cache for previously computed features
        self.feature_cache = {}
        self.max_cache_size = 10000
        
        # Try to import GPU libraries
        try:
            import cupy as cp
            self.has_gpu = True
            
            # Initialize device
            cp.cuda.Device(device_id).use()
            
            # Update strategy if auto
            if strategy == 'auto':
                self.strategy = self.zero_copy_extractor.get_strategy()
            
        except ImportError:
            logger.warning("GPU libraries not available, falling back to CPU implementation")
            self.strategy = 'cpu'
    
    def extract_features(self, 
                        df: pd.DataFrame, 
                        batch_size: int = 1000) -> Dict[str, Any]:
        """
        Extract features from access logs using zero-copy memory interface
        
        Args:
            df: DataFrame with access logs
            batch_size: Batch size for feature extraction
            
        Returns:
            Dictionary with extracted features and metadata
        """
        if not self.has_gpu or self.strategy == 'cpu':
            # Fall back to regular pipeline for CPU
            pipeline = FeatureEngineeringPipeline(
                output_dir=self.output_dir,
                use_gpu=False
            )
            feature_df = pipeline.extract_features(df)
            
            return {
                'features': feature_df,
                'strategy': 'cpu',
                'device_id': -1
            }
        
        try:
            import cupy as cp
            
            # Initialize real-time extractor with data
            self._init_realtime_extractor(df)
            
            # Determine optimal strategy based on data size
            data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            strategy = self.zero_copy_extractor.select_optimal_strategy(data_size_mb)
            
            # Process in batches to avoid OOM
            num_batches = (len(df) + batch_size - 1) // batch_size
            
            if strategy == 'gpu_direct':
                # For GPU-Direct, process all data on GPU at once if possible
                logger.info(f"Using GPU-Direct strategy for {data_size_mb:.2f} MB data")
                
                # Create cuDF DataFrame
                import cudf
                gdf = cudf.DataFrame.from_pandas(df)
                
                # Extract features
                feature_dict = self._extract_gpu_direct_features(gdf)
                
                # Convert back to pandas for return
                feature_df = pd.DataFrame(feature_dict)
                
                return {
                    'features': feature_df,
                    'strategy': 'gpu_direct',
                    'device_id': self.device_id
                }
                
            elif strategy == 'uvm':
                # For UVM, process in batches with page hints
                logger.info(f"Using UVM strategy for {data_size_mb:.2f} MB data")
                
                # Extract features for all keys
                all_keys = df['key'].unique()
                
                # Process in batches
                features = {}
                for i in range(0, len(all_keys), batch_size):
                    batch_keys = all_keys[i:i+batch_size]
                    batch_features = self.zero_copy_extractor.extract_features(
                        batch_keys, sync=True)
                    features.update(batch_features)
                
                # Create feature DataFrame
                feature_dicts = []
                for key in df['key']:
                    if key in features:
                        feature_dict = features[key].copy()
                        feature_dict['key'] = key
                        feature_dicts.append(feature_dict)
                
                feature_df = pd.DataFrame(feature_dicts)
                
                return {
                    'features': feature_df,
                    'strategy': 'uvm',
                    'device_id': self.device_id
                }
                
            else:  # peer_mapping
                # For peer mapping, use interleaved access
                logger.info(f"Using peer mapping strategy for {data_size_mb:.2f} MB data")
                
                # Create batches of keys
                all_keys = df['key'].unique()
                key_batches = [all_keys[i:i+batch_size] for i in range(0, len(all_keys), batch_size)]
                
                # Extract features for each batch
                all_features = {}
                for batch in key_batches:
                    batch_features = self.zero_copy_extractor.extract_features(
                        batch, sync=True)
                    all_features.update(batch_features)
                
                # Create feature DataFrame
                feature_dicts = []
                for key in df['key']:
                    if key in all_features:
                        feature_dict = all_features[key].copy()
                        feature_dict['key'] = key
                        feature_dicts.append(feature_dict)
                
                feature_df = pd.DataFrame(feature_dicts)
                
                return {
                    'features': feature_df,
                    'strategy': 'peer_mapping',
                    'device_id': self.device_id
                }
            
        except Exception as e:
            logger.error(f"Error in zero-copy feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            
            # Fall back to regular pipeline
            pipeline = FeatureEngineeringPipeline(
                output_dir=self.output_dir,
                use_gpu=False
            )
            feature_df = pipeline.extract_features(df)
            
            return {
                'features': feature_df,
                'strategy': 'cpu',
                'device_id': -1
            }
    
    def _init_realtime_extractor(self, df: pd.DataFrame) -> None:
        """Initialize real-time extractor with data"""
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add recent accesses to real-time extractor
        recent_df = df.iloc[-min(len(df), 1000):]
        for _, row in recent_df.iterrows():
            self.realtime_extractor.add_access(
                row['key'], 
                row.get('operation', 'GET'), 
                row['timestamp']
            )
        
        # Start real-time extraction
        self.realtime_extractor.start()
    
    def _extract_gpu_direct_features(self, gdf: Any) -> Dict[str, List[float]]:
        """Extract features using GPU-Direct strategy"""
        # This would use cuDF and cuPy for direct GPU computation
        # For now, simulate by processing each key
        keys = gdf['key'].unique().to_pandas().tolist()
        
        # Extract features for all keys
        features = self.zero_copy_extractor.extract_features(keys, sync=True)
        
        # Convert to dictionary of lists for DataFrame construction
        feature_dict = defaultdict(list)
        
        # Add key column
        feature_dict['key'] = []
        
        # Get all feature names
        feature_names = set()
        for key_features in features.values():
            feature_names.update(key_features.keys())
        
        # Initialize lists for each feature
        for feature_name in feature_names:
            feature_dict[feature_name] = []
        
        # Fill dictionary
        for key, row in zip(gdf['key'], range(len(gdf))):
            feature_dict['key'].append(key)
            
            # Get features for this key
            key_features = features.get(key, {})
            
            # Add each feature
            for feature_name in feature_names:
                feature_dict[feature_name].append(key_features.get(feature_name, 0.0))
        
        return feature_dict
    
    def extract_realtime_features(self, keys: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract real-time features for a batch of keys
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to feature dictionaries
        """
        # Use zero-copy extractor for real-time features
        return self.zero_copy_extractor.extract_features(keys)
    
    def add_access(self, key: str, operation: str, timestamp: Optional[float] = None) -> None:
        """
        Add a new access event to the real-time extractor
        
        Args:
            key: Cache key
            operation: Operation type (GET, PUT, DEL)
            timestamp: Access timestamp (default: current time)
        """
        self.realtime_extractor.add_access(key, operation, timestamp)
    
    def predict_next_access(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next keys to be accessed
        
        Args:
            top_n: Number of predictions to return
            
        Returns:
            List of (key, probability) tuples
        """
        return self.realtime_extractor.predict_next_access(top_n)
    
    def get_feature_buffer(self) -> Any:
        """
        Get the zero-copy feature buffer
        
        Returns:
            Feature buffer (GPU memory pointer)
        """
        return self.zero_copy_extractor.get_feature_buffer()
    
    def get_strategy(self) -> str:
        """
        Get the current memory access strategy
        
        Returns:
            Memory access strategy
        """
        return self.strategy
