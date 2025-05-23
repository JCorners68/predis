"""
Feature Extraction Pipelines for Predis ML

This module provides feature extraction pipelines for Predis' ML-driven prefetching,
optimized to work with the multi-strategy zero-copy memory interface system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Base class for feature extraction pipelines"""
    
    def __init__(self, name: str, output_dir: str = "../../../data/real/features"):
        """
        Initialize feature extractor
        
        Args:
            name: Extractor name
            output_dir: Directory to save extracted features
        """
        self.name = name
        self.output_dir = output_dir
        self.feature_transformers = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_transformer(self, 
                       transformer_func: Callable[[pd.DataFrame], pd.DataFrame], 
                       name: str) -> None:
        """
        Add a feature transformer to the pipeline
        
        Args:
            transformer_func: Function that transforms the dataframe
            name: Name of the transformer
        """
        self.feature_transformers.append({
            'func': transformer_func,
            'name': name
        })
        logger.info(f"Added transformer '{name}' to extractor '{self.name}'")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from access log dataframe
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting features using '{self.name}' pipeline")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply each transformer in sequence
        for transformer in self.feature_transformers:
            logger.info(f"Applying transformer: {transformer['name']}")
            start_time = time.time()
            
            result_df = transformer['func'](result_df)
            
            duration = time.time() - start_time
            logger.info(f"Transformer '{transformer['name']}' applied in {duration:.2f}s")
        
        return result_df
    
    def save_features(self, 
                     feature_df: pd.DataFrame, 
                     name: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save extracted features to disk
        
        Args:
            feature_df: DataFrame with extracted features
            name: Optional name for the feature set
            metadata: Optional metadata
            
        Returns:
            Path to saved features
        """
        if name is None:
            name = self.name
        
        # Generate timestamp for filename
        timestamp = int(time.time())
        filename = f"{name}_features_{timestamp}"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save in multiple formats for flexibility
        feature_df.to_csv(f"{output_path}.csv", index=False)
        feature_df.to_parquet(f"{output_path}.parquet")
        
        # Save metadata
        meta = {
            'name': name,
            'extractor': self.name,
            'timestamp': timestamp,
            'num_samples': len(feature_df),
            'num_features': len(feature_df.columns),
            'feature_columns': list(feature_df.columns),
            'transformers': [t['name'] for t in self.feature_transformers]
        }
        
        if metadata:
            meta.update(metadata)
        
        import json
        with open(f"{output_path}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Saved {len(feature_df)} feature samples to {output_path}")
        return output_path


class AccessPatternFeatureExtractor(FeatureExtractor):
    """Extracts features from cache access patterns"""
    
    def __init__(self, output_dir: str = "../../../data/real/features"):
        """Initialize with standard transformers for access patterns"""
        super().__init__("access_pattern_extractor", output_dir)
        
        # Add standard transformers
        self.add_transformer(self._add_temporal_features, "temporal_features")
        self.add_transformer(self._add_key_frequency_features, "key_frequency")
        self.add_transformer(self._add_recency_features, "recency_features")
        self.add_transformer(self._add_operation_features, "operation_features")
    
    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (time of day, day of week, etc.)"""
        result = df.copy()
        
        # Ensure datetime column exists
        if 'datetime' not in result.columns and 'timestamp' in result.columns:
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='s')
        
        # Extract time components
        result['hour_of_day'] = result['datetime'].dt.hour
        result['minute_of_hour'] = result['datetime'].dt.minute
        result['day_of_week'] = result['datetime'].dt.dayofweek
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        result['is_business_hours'] = ((result['hour_of_day'] >= 9) & 
                                      (result['hour_of_day'] <= 17) &
                                      ~result['is_weekend']).astype(int)
        
        return result
    
    @staticmethod
    def _add_key_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to key access frequency"""
        result = df.copy()
        
        # Calculate global key frequency
        key_counts = result['key'].value_counts()
        result['key_frequency'] = result['key'].map(key_counts)
        
        # Calculate normalized frequency (percentile)
        key_rank = key_counts.rank(pct=True)
        result['key_frequency_percentile'] = result['key'].map(key_rank)
        
        # Calculate rolling window frequency (last 1000 accesses)
        result = result.sort_values('timestamp')
        result['rolling_frequency'] = result.groupby('key')['key'].transform(
            lambda x: x.rolling(1000, min_periods=1).count())
        
        return result
    
    @staticmethod
    def _add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to recency of access"""
        result = df.copy()
        
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Calculate time since last access for each key
        result['time_since_last_access'] = result.groupby('key')['timestamp'].diff()
        
        # Fill NaN (first access) with a large value
        max_time = result['timestamp'].max() - result['timestamp'].min()
        result['time_since_last_access'] = result['time_since_last_access'].fillna(max_time)
        
        # Calculate exponential decay of recency (recent = higher value)
        # Half-life of 1 hour (3600 seconds)
        half_life = 3600
        decay_factor = np.log(2) / half_life
        result['recency_score'] = np.exp(-decay_factor * result['time_since_last_access'])
        
        # Calculate access count for each key
        result['access_count'] = result.groupby('key').cumcount() + 1
        
        # Calculate inter-access time statistics
        result['avg_interaccess_time'] = result.groupby('key')['time_since_last_access'].transform('mean')
        result['std_interaccess_time'] = result.groupby('key')['time_since_last_access'].transform('std')
        
        # Fill NaN for std (single access) with 0
        result['std_interaccess_time'] = result['std_interaccess_time'].fillna(0)
        
        return result
    
    @staticmethod
    def _add_operation_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to operation type"""
        result = df.copy()
        
        # Check if operation column exists
        if 'operation' in result.columns:
            # One-hot encode operation type
            result['is_get'] = (result['operation'] == 'GET').astype(int)
            result['is_put'] = (result['operation'] == 'PUT').astype(int)
            result['is_del'] = (result['operation'] == 'DEL').astype(int)
            
            # Calculate operation type ratios for each key
            result['get_ratio'] = result.groupby('key')['is_get'].transform('mean')
            result['put_ratio'] = result.groupby('key')['is_put'].transform('mean')
            result['write_heavy'] = (result['put_ratio'] > 0.5).astype(int)
        
        return result


class SequenceFeatureExtractor(FeatureExtractor):
    """Extracts sequence-based features for time series prediction"""
    
    def __init__(self, 
                sequence_length: int = 10,
                output_dir: str = "../../../data/real/features"):
        """
        Initialize sequence feature extractor
        
        Args:
            sequence_length: Number of timesteps in each sequence
            output_dir: Directory to save extracted features
        """
        super().__init__("sequence_extractor", output_dir)
        self.sequence_length = sequence_length
        
        # Add standard transformers
        self.add_transformer(self._prepare_sequence_data, "prepare_sequences")
        self.add_transformer(self._add_key_cooccurrence, "key_cooccurrence")
        self.add_transformer(self._add_sequence_patterns, "sequence_patterns")
    
    def _prepare_sequence_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for sequence processing"""
        result = df.copy()
        
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Add sequence ID (sliding window)
        result['sequence_id'] = np.arange(len(result))
        
        # Add previous keys in sequence
        for i in range(1, self.sequence_length + 1):
            result[f'prev_key_{i}'] = result['key'].shift(i)
        
        # Drop rows with missing previous keys
        result = result.dropna(subset=[f'prev_key_{self.sequence_length}'])
        
        return result
    
    @staticmethod
    def _add_key_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on key co-occurrence"""
        result = df.copy()
        
        # Get all 'prev_key_X' columns
        prev_key_cols = [col for col in result.columns if col.startswith('prev_key_')]
        
        # Calculate how often each previous key appears with the current key
        for col in prev_key_cols:
            # For each previous key position, calculate how often it predicts the current key
            result[f'{col}_cooccurrence'] = 0.0
            
            # Group by previous key and calculate probability of current key
            cooccurrence = result.groupby(col)['key'].value_counts(normalize=True).reset_index()
            cooccurrence.columns = [col, 'key', 'probability']
            
            # Map probabilities back to original dataframe
            result = result.merge(
                cooccurrence, on=[col, 'key'], how='left')
            
            # Rename probability column
            result = result.rename(columns={'probability': f'{col}_cooccurrence'})
            
            # Fill NaN with 0
            result[f'{col}_cooccurrence'] = result[f'{col}_cooccurrence'].fillna(0)
        
        return result
    
    @staticmethod
    def _add_sequence_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on sequence patterns"""
        result = df.copy()
        
        # Get all 'prev_key_X' columns
        prev_key_cols = [col for col in result.columns if col.startswith('prev_key_')]
        
        # Check for repeated keys in sequence
        result['has_repeated_prev'] = result.apply(
            lambda row: len(set([row[col] for col in prev_key_cols])) < len(prev_key_cols),
            axis=1
        ).astype(int)
        
        # Check for sequential patterns (current key = prev_key_1)
        result['is_sequential'] = (result['key'] == result['prev_key_1']).astype(int)
        
        # Check for cyclic patterns (current key = prev_key_n for some n)
        for i, col in enumerate(prev_key_cols, 1):
            result[f'matches_prev_{i}'] = (result['key'] == result[col]).astype(int)
        
        # Calculate cycle length (position of previous occurrence of current key)
        result['cycle_length'] = 0
        for i, col in enumerate(prev_key_cols, 1):
            mask = (result['key'] == result[col]) & (result['cycle_length'] == 0)
            result.loc[mask, 'cycle_length'] = i
        
        return result
    
    def create_sequence_features(self, 
                                df: pd.DataFrame, 
                                target_column: str = 'key') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence features for time series prediction
        
        Args:
            df: DataFrame with access logs
            target_column: Column to predict
            
        Returns:
            Tuple of (X, y) with X as sequence features and y as target labels
        """
        # Extract features
        feature_df = self.extract_features(df)
        
        # Select feature columns (exclude target and string columns)
        feature_cols = [col for col in feature_df.columns 
                      if col != target_column
                      and not col.startswith('prev_key_')
                      and not col.startswith('key')
                      and not col.startswith('operation')
                      and not col.startswith('datetime')
                      and not pd.api.types.is_string_dtype(feature_df[col])]
        
        # Create sequences
        X = np.zeros((len(feature_df), self.sequence_length, len(feature_cols)))
        
        for i, feature_idx in enumerate(feature_cols):
            for j in range(self.sequence_length):
                # Current feature values go in the first position of each sequence
                # Then previous values in reverse order
                if j == 0:
                    X[:, j, i] = feature_df[feature_idx].values
                else:
                    # Get values from previous rows
                    X[:, j, i] = feature_df[feature_idx].shift(j).fillna(0).values
        
        # Target is the next key to be accessed
        if target_column == 'key':
            # For key prediction, use hash values
            y = np.array([hash(k) % (2**32) for k in feature_df[target_column]])
        else:
            y = feature_df[target_column].values
        
        return X, y


class RelationshipFeatureExtractor(FeatureExtractor):
    """Extracts features based on key relationships"""
    
    def __init__(self, output_dir: str = "../../../data/real/features"):
        """Initialize relationship feature extractor"""
        super().__init__("relationship_extractor", output_dir)
        
        # Add standard transformers
        self.add_transformer(self._add_key_relationships, "key_relationships")
        self.add_transformer(self._add_transition_probabilities, "transition_probs")
        self.add_transformer(self._add_access_graphs, "access_graphs")
    
    @staticmethod
    def _add_key_relationships(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on key relationships"""
        result = df.copy()
        
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Add previous and next keys
        result['prev_key'] = result['key'].shift(1)
        result['next_key'] = result['key'].shift(-1)
        
        # For first and last rows, use NaN
        result['prev_key'] = result['prev_key'].fillna('__START__')
        result['next_key'] = result['next_key'].fillna('__END__')
        
        # Calculate key transition counts
        transitions = result.groupby(['key', 'next_key']).size().reset_index()
        transitions.columns = ['key', 'next_key', 'transition_count']
        
        # Calculate outgoing transition counts for each key
        key_outgoing = result.groupby('key').size().reset_index()
        key_outgoing.columns = ['key', 'outgoing_count']
        
        # Merge transition counts back to original dataframe
        result = result.merge(transitions, on=['key', 'next_key'], how='left')
        result = result.merge(key_outgoing, on='key', how='left')
        
        # Calculate transition probability
        result['transition_prob'] = result['transition_count'] / result['outgoing_count']
        result['transition_prob'] = result['transition_prob'].fillna(0)
        
        return result
    
    @staticmethod
    def _add_transition_probabilities(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on transition probabilities"""
        result = df.copy()
        
        # Calculate global transition matrix (as a sparse representation)
        transitions = result.groupby(['prev_key', 'key']).size().reset_index()
        transitions.columns = ['prev_key', 'key', 'count']
        
        # Calculate total occurrences of each previous key
        prev_key_counts = transitions.groupby('prev_key')['count'].sum().reset_index()
        prev_key_counts.columns = ['prev_key', 'total_count']
        
        # Calculate conditional probability P(key | prev_key)
        transitions = transitions.merge(prev_key_counts, on='prev_key', how='left')
        transitions['cond_prob'] = transitions['count'] / transitions['total_count']
        
        # Keep only the top 3 most likely next keys for each previous key
        top_transitions = (transitions
                          .sort_values(['prev_key', 'cond_prob'], ascending=[True, False])
                          .groupby('prev_key')
                          .head(3)
                          .reset_index(drop=True))
        
        # Add indicator features for each key
        for i in range(1, 4):
            # Merge top transition probabilities
            result[f'top{i}_next_key_prob'] = 0.0
            
            # Only keep rows where we match the previous key and current key is in top N
            if i <= len(top_transitions):
                # Get transitions for rank i
                rank_i = (top_transitions
                         .sort_values(['prev_key', 'cond_prob'], ascending=[True, False])
                         .groupby('prev_key')
                         .nth(i-1)
                         .reset_index())
                
                # Create a mapping from (prev_key, key) to conditional probability
                prob_map = {}
                for _, row in rank_i.iterrows():
                    prob_map[(row['prev_key'], row['key'])] = row['cond_prob']
                
                # Apply mapping to the result dataframe
                result[f'top{i}_next_key_prob'] = result.apply(
                    lambda row: prob_map.get((row['prev_key'], row['key']), 0.0),
                    axis=1
                )
        
        return result
    
    @staticmethod
    def _add_access_graphs(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on access graphs"""
        result = df.copy()
        
        # Calculate key centrality in access graph
        # For each key, count how many unique keys transition to/from it
        incoming = result.groupby('key')['prev_key'].nunique().reset_index()
        incoming.columns = ['key', 'incoming_centrality']
        
        outgoing = result.groupby('prev_key')['key'].nunique().reset_index()
        outgoing.columns = ['key', 'outgoing_centrality']
        
        # Merge centrality metrics back to original dataframe
        result = result.merge(incoming, on='key', how='left')
        result = result.merge(outgoing, on='key', how='left')
        
        # Fill NaN with 0
        result['incoming_centrality'] = result['incoming_centrality'].fillna(0)
        result['outgoing_centrality'] = result['outgoing_centrality'].fillna(0)
        
        # Calculate total centrality
        result['total_centrality'] = result['incoming_centrality'] + result['outgoing_centrality']
        
        # Normalize centrality scores
        max_centrality = result['total_centrality'].max()
        if max_centrality > 0:
            result['normalized_centrality'] = result['total_centrality'] / max_centrality
        else:
            result['normalized_centrality'] = 0.0
        
        return result


class GPUOptimizedFeatureExtractor(FeatureExtractor):
    """
    GPU-optimized feature extractor that leverages Predis' zero-copy memory interface
    
    This extractor is designed to work efficiently with Predis' three memory access strategies:
    1. GPU-Direct pathway for lowest-latency access
    2. Optimized UVM integration with ML-driven page placement
    3. Custom peer mapping with explicit coherence control
    """
    
    def __init__(self, 
                output_dir: str = "../../../data/real/features",
                device_id: int = 0,
                use_zero_copy: bool = True):
        """
        Initialize GPU-optimized feature extractor
        
        Args:
            output_dir: Directory to save extracted features
            device_id: GPU device ID
            use_zero_copy: Whether to use zero-copy memory interface
        """
        super().__init__("gpu_optimized_extractor", output_dir)
        self.device_id = device_id
        self.use_zero_copy = use_zero_copy
        
        # Add standard transformers
        self.add_transformer(self._prepare_gpu_data, "prepare_gpu_data")
        self.add_transformer(self._extract_gpu_features, "gpu_features")
        
        # Try to import GPU libraries
        self.has_gpu = False
        try:
            import cupy as cp
            import cudf
            self.has_gpu = True
            logger.info("GPU libraries available, using GPU acceleration")
        except ImportError:
            logger.warning("GPU libraries not available, falling back to CPU implementation")
    
    def _prepare_gpu_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for GPU processing"""
        if not self.has_gpu:
            logger.warning("GPU libraries not available, using CPU preprocessing")
            return df
        
        try:
            import cupy as cp
            import cudf
            
            # Convert to cuDF DataFrame for GPU processing
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Sort by timestamp
            gdf = gdf.sort_values('timestamp')
            
            # Compute key hashes for faster processing
            if 'key' in gdf.columns:
                gdf['key_hash'] = gdf['key'].hash_values() % (2**32)
            
            # Add temporal features
            if 'timestamp' in gdf.columns:
                # Convert timestamp to datetime
                gdf['datetime'] = gdf['timestamp'].astype('datetime64[s]')
                
                # Extract time components
                gdf['hour_of_day'] = gdf['datetime'].dt.hour
                gdf['day_of_week'] = gdf['datetime'].dt.dayofweek
            
            # Convert back to pandas for compatibility with other transformers
            return gdf.to_pandas()
            
        except Exception as e:
            logger.error(f"Error in GPU preprocessing: {e}")
            logger.warning("Falling back to CPU implementation")
            return df
    
    def _extract_gpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features using GPU acceleration"""
        if not self.has_gpu:
            logger.warning("GPU libraries not available, using CPU feature extraction")
            # Use regular feature extraction instead
            extractor = AccessPatternFeatureExtractor()
            return extractor.extract_features(df)
        
        try:
            import cupy as cp
            import cudf
            
            # Convert to cuDF DataFrame for GPU processing
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Key frequency features (optimized for GPU)
            key_value_counts = gdf['key'].value_counts()
            gdf['key_frequency'] = gdf['key'].map_values(key_value_counts)
            
            # Recency features
            gdf = gdf.sort_values('timestamp')
            
            # Calculate time since last access using GPU operations
            # Group by key and compute diff
            gdf['time_since_last_access'] = gdf.groupby('key')['timestamp'].diff()
            
            # Fill NaN with a large value
            max_time = gdf['timestamp'].max() - gdf['timestamp'].min()
            gdf['time_since_last_access'] = gdf['time_since_last_access'].fillna(max_time)
            
            # Calculate exponential decay of recency using GPU
            half_life = 3600
            decay_factor = np.log(2) / half_life
            gdf['recency_score'] = cp.exp(-decay_factor * gdf['time_since_last_access'].values)
            
            # Calculate access count for each key
            gdf['access_count'] = gdf.groupby('key').cumcount() + 1
            
            # Convert back to pandas for compatibility
            result_df = gdf.to_pandas()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in GPU feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            
            # Use regular feature extraction instead
            extractor = AccessPatternFeatureExtractor()
            return extractor.extract_features(df)
    
    def extract_features_zero_copy(self, 
                                 data_path: str, 
                                 output_path: Optional[str] = None) -> str:
        """
        Extract features using zero-copy memory interface
        
        Args:
            data_path: Path to input data
            output_path: Path to save output features
            
        Returns:
            Path to saved features
        """
        if not self.has_gpu or not self.use_zero_copy:
            logger.warning("Zero-copy interface not available, using standard extraction")
            # Load data normally
            df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
            
            # Extract features
            feature_df = self.extract_features(df)
            
            # Save features
            return self.save_features(feature_df, "cpu_features")
        
        try:
            import cupy as cp
            import cudf
            from ..data.utils.exporters import ZeroCopyExporter
            
            logger.info("Using zero-copy memory interface for feature extraction")
            
            # Determine optimal memory access strategy based on data size
            # First, get file size
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            
            if file_size_mb < 50:
                # Small data, use GPU-Direct for lowest latency
                logger.info("Using GPU-Direct pathway for small dataset")
                access_strategy = "gpu_direct"
            elif file_size_mb < 500:
                # Medium data, use Optimized UVM
                logger.info("Using Optimized UVM for medium dataset")
                access_strategy = "uvm"
            else:
                # Large data, use Custom peer mapping
                logger.info("Using Custom peer mapping for large dataset")
                access_strategy = "peer_mapping"
            
            # Load data using appropriate strategy
            # This would normally use Predis' zero-copy interface
            # For now, we'll simulate by loading with standard methods
            if data_path.endswith('.parquet'):
                gdf = cudf.read_parquet(data_path)
            else:
                gdf = cudf.read_csv(data_path)
            
            # Extract features (using GPU)
            # Convert to pandas temporarily for our extraction pipeline
            df = gdf.to_pandas()
            feature_df = self.extract_features(df)
            
            # Convert back to GPU for zero-copy export
            feature_gdf = cudf.DataFrame.from_pandas(feature_df)
            
            # Export using zero-copy exporter
            exporter = ZeroCopyExporter()
            
            if output_path is None:
                output_name = f"gpu_features_{int(time.time())}"
            else:
                output_name = os.path.basename(output_path).split('.')[0]
            
            # Export using the determined strategy
            if access_strategy == "gpu_direct":
                output_path = exporter.export_for_gpu_direct(
                    feature_gdf.values, output_name)
            elif access_strategy == "uvm":
                # Generate page access hints based on feature importance
                # This would be more sophisticated in practice
                num_pages = (feature_gdf.values.nbytes + 4095) // 4096
                page_hints = np.ones(num_pages) * 0.5  # Default medium priority
                
                output_path = exporter.export_for_uvm(
                    feature_gdf.values, output_name, page_access_hints=page_hints)
            else:  # peer_mapping
                output_path = exporter.export_for_peer_mapping(
                    feature_gdf.values, output_name, coherence_level=1)
            
            logger.info(f"Exported features using zero-copy interface: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in zero-copy feature extraction: {e}")
            logger.warning("Falling back to standard extraction")
            
            # Load data normally
            df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
            
            # Extract features
            feature_df = self.extract_features(df)
            
            # Save features
            return self.save_features(feature_df, "cpu_fallback_features")
