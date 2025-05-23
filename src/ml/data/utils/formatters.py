"""
Data Format Converters for Predis ML

This module provides utilities for converting data between different formats,
ensuring compatibility with Predis' ML components and zero-copy memory interface.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AccessLogFormatter:
    """Converts access logs between different formats"""
    
    @staticmethod
    def to_dataframe(access_logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert access logs to pandas DataFrame
        
        Args:
            access_logs: List of access log dictionaries
            
        Returns:
            DataFrame with standardized format
        """
        df = pd.DataFrame(access_logs)
        
        # Ensure standard columns exist
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    @staticmethod
    def to_gpu_tensor(df: pd.DataFrame, device_id: int = 0) -> Dict[str, Any]:
        """
        Convert DataFrame to GPU-compatible tensors for ML processing
        
        Args:
            df: DataFrame containing access logs
            device_id: GPU device ID
            
        Returns:
            Dictionary of GPU tensors compatible with Predis' zero-copy memory interface
        """
        try:
            import cupy as cp
            import cudf
            
            # Set GPU device
            cp.cuda.Device(device_id).use()
            
            # Convert to cuDF DataFrame first for efficient conversion
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Extract key features
            tensor_dict = {
                'timestamps': cp.array(gdf['timestamp'].values),
                'key_hashes': cp.array([hash(k) % (2**32) for k in df['key']]),
            }
            
            # Add operation type as one-hot encoding
            if 'operation' in gdf.columns:
                op_types = {'GET': 0, 'PUT': 1, 'DEL': 2}
                tensor_dict['operation'] = cp.array([op_types.get(op, 0) for op in df['operation']])
            
            # Add any numerical columns
            for col in gdf.columns:
                if col not in ['timestamp', 'key', 'operation', 'datetime'] and pd.api.types.is_numeric_dtype(gdf[col]):
                    tensor_dict[col] = cp.array(gdf[col].values)
            
            logger.info(f"Converted DataFrame to GPU tensors on device {device_id}")
            return tensor_dict
            
        except ImportError:
            logger.warning("cupy or cudf not available. Falling back to NumPy arrays")
            
            # Fall back to NumPy arrays
            tensor_dict = {
                'timestamps': np.array(df['timestamp'].values),
                'key_hashes': np.array([hash(k) % (2**32) for k in df['key']]),
            }
            
            # Add operation type as one-hot encoding
            if 'operation' in df.columns:
                op_types = {'GET': 0, 'PUT': 1, 'DEL': 2}
                tensor_dict['operation'] = np.array([op_types.get(op, 0) for op in df['operation']])
            
            # Add any numerical columns
            for col in df.columns:
                if col not in ['timestamp', 'key', 'operation', 'datetime'] and pd.api.types.is_numeric_dtype(df[col]):
                    tensor_dict[col] = np.array(df[col].values)
            
            return tensor_dict
    
    @staticmethod
    def from_predis_logs(log_file: str) -> pd.DataFrame:
        """
        Convert Predis server logs to standardized DataFrame
        
        Args:
            log_file: Path to Predis log file
            
        Returns:
            DataFrame with standardized format
        """
        # Read log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse log lines
        access_logs = []
        
        for line in lines:
            try:
                # Assuming log format: [timestamp] operation key value_size
                parts = line.strip().split()
                if len(parts) >= 4:
                    timestamp_str = parts[0].strip('[]')
                    operation = parts[1]
                    key = parts[2]
                    value_size = int(parts[3])
                    
                    # Convert timestamp to float
                    timestamp = float(timestamp_str)
                    
                    access_logs.append({
                        'timestamp': timestamp,
                        'operation': operation,
                        'key': key,
                        'value_size': value_size
                    })
            except Exception as e:
                logger.warning(f"Error parsing log line: {line}. Error: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(access_logs)
        
        # Add datetime column
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        logger.info(f"Converted {len(df)} Predis log entries to DataFrame")
        return df


class FeatureFormatConverter:
    """Converts feature data between different formats"""
    
    @staticmethod
    def to_training_format(df: pd.DataFrame, 
                          sequence_length: int = 10, 
                          target_column: str = 'key',
                          feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to sequence format for training
        
        Args:
            df: DataFrame containing access logs
            sequence_length: Number of timesteps in each sequence
            target_column: Column to use as prediction target
            feature_columns: List of columns to use as features
            
        Returns:
            Tuple of (X, y) with X as features and y as target labels
        """
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = [col for col in df.columns 
                             if col != target_column and pd.api.types.is_numeric_dtype(df[col])]
        
        # Sort by timestamp to ensure correct sequence
        df = df.sort_values('timestamp')
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            # Extract sequence
            seq = df.iloc[i:i+sequence_length][feature_columns].values
            # Target is the next key after the sequence
            target = df.iloc[i+sequence_length][target_column]
            
            sequences.append(seq)
            targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        
        # For key targets, convert to hash values for numerical processing
        if target_column == 'key':
            y = np.array([hash(t) % (2**32) for t in targets])
        else:
            y = np.array(targets)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    @staticmethod
    def prepare_zero_copy_data(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Prepare data for use with Predis' zero-copy memory interface
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Dictionary with data formatted for zero-copy memory interface
        """
        try:
            import cupy as cp
            
            # Transfer to GPU using zero-copy when possible
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)
            
            # Create dictionary with metadata for the zero-copy interface
            zero_copy_data = {
                'features': X_gpu,
                'targets': y_gpu,
                'feature_shape': X.shape,
                'target_shape': y.shape,
                'dtype': str(X.dtype),
                'is_gpu_tensor': True
            }
            
            logger.info(f"Prepared zero-copy data with {X.shape[0]} samples")
            return zero_copy_data
            
        except ImportError:
            logger.warning("cupy not available. Preparing CPU data instead")
            
            # Fall back to CPU arrays
            zero_copy_data = {
                'features': X,
                'targets': y,
                'feature_shape': X.shape,
                'target_shape': y.shape,
                'dtype': str(X.dtype),
                'is_gpu_tensor': False
            }
            
            return zero_copy_data


class ModelInputFormatter:
    """Formats data for specific ML models"""
    
    @staticmethod
    def format_for_lstm(sequences: np.ndarray, batch_size: int = 32) -> Dict[str, Any]:
        """
        Format sequence data for LSTM models
        
        Args:
            sequences: Sequence data (n_samples, seq_length, n_features)
            batch_size: Batch size for model input
            
        Returns:
            Dictionary with formatted data
        """
        # Check if we need to reshape
        if len(sequences.shape) == 2:
            # Add feature dimension
            sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
        
        # Calculate steps per epoch
        steps_per_epoch = (sequences.shape[0] + batch_size - 1) // batch_size
        
        formatted_data = {
            'sequences': sequences,
            'n_samples': sequences.shape[0],
            'sequence_length': sequences.shape[1],
            'n_features': sequences.shape[2],
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch
        }
        
        return formatted_data
    
    @staticmethod
    def format_for_ngboost(features: np.ndarray, is_sequence: bool = True) -> np.ndarray:
        """
        Format features for NGBoost models
        
        Args:
            features: Feature data
            is_sequence: Whether data is in sequence format
            
        Returns:
            Formatted features
        """
        if is_sequence:
            # NGBoost doesn't handle sequences directly, so we flatten
            # For each sequence, we'll use the last timestep plus some aggregated features
            n_samples = features.shape[0]
            n_seq = features.shape[1]
            n_features = features.shape[2]
            
            # Create new feature array
            new_features = np.zeros((n_samples, n_features * 2))
            
            for i in range(n_samples):
                # Use last timestep directly
                new_features[i, :n_features] = features[i, -1, :]
                
                # Add aggregated features (mean across sequence)
                new_features[i, n_features:] = np.mean(features[i, :, :], axis=0)
            
            return new_features
        else:
            # Already in the right format
            return features
    
    @staticmethod
    def format_for_xgboost(features: np.ndarray, is_sequence: bool = True) -> np.ndarray:
        """
        Format features for XGBoost models
        
        Args:
            features: Feature data
            is_sequence: Whether data is in sequence format
            
        Returns:
            Formatted features
        """
        # XGBoost formatting is the same as NGBoost for our purposes
        return ModelInputFormatter.format_for_ngboost(features, is_sequence)
