"""
Data Export Utilities for Predis ML

This module provides utilities for exporting data in formats compatible with
Predis' multi-strategy zero-copy memory interface system and ML training pipelines.
"""

import numpy as np
import pandas as pd
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZeroCopyExporter:
    """
    Exports data in formats optimized for Predis' multi-strategy zero-copy memory interface
    
    This exporter is designed to work with Predis' three memory access pathways:
    1. GPU-Direct pathway for lowest-latency access via PCIe BAR1 or NVLink
    2. Optimized UVM integration with ML-driven page placement
    3. Custom peer mapping with explicit coherence control
    """
    
    def __init__(self, base_path: str = "../../../data"):
        """
        Initialize exporter
        
        Args:
            base_path: Base directory for data exports
        """
        self.base_path = base_path
        
        # Create base directories if they don't exist
        for subdir in ['ml_features', 'ml_models', 'zero_copy_exports']:
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    
    def export_for_gpu_direct(self, data: np.ndarray, 
                             name: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export data optimized for GPU-Direct pathway
        
        Args:
            data: NumPy array to export
            name: Identifier for the data
            metadata: Additional metadata to include
            
        Returns:
            Path to the exported file
        """
        try:
            import cupy as cp
            
            # Create output directory
            output_dir = os.path.join(self.base_path, 'zero_copy_exports', 'gpu_direct')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.gpumem"
            output_path = os.path.join(output_dir, filename)
            
            # Create metadata
            meta = {
                'name': name,
                'timestamp': timestamp,
                'shape': data.shape,
                'dtype': str(data.dtype),
                'access_strategy': 'gpu_direct',
                'memory_layout': 'contiguous'
            }
            
            if metadata:
                meta.update(metadata)
            
            # Write metadata file
            with open(f"{output_path}.meta", 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Convert to CuPy array and export in optimized format
            if not isinstance(data, cp.ndarray):
                data_gpu = cp.asarray(data)
            else:
                data_gpu = data
            
            # Export in a memory-mapped format for efficient GPU-Direct access
            with open(output_path, 'wb') as f:
                # Write header with shape and dtype information
                header = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'alignment': 256  # Align for optimal PCIe transfers
                }
                header_bytes = json.dumps(header).encode('utf-8')
                f.write(len(header_bytes).to_bytes(8, byteorder='little'))
                f.write(header_bytes)
                
                # Write data with page alignment for optimal PCIe access
                f.write(b'\x00' * (256 - (f.tell() % 256) if f.tell() % 256 != 0 else 0))
                cp.save(f, data_gpu)
            
            logger.info(f"Exported data for GPU-Direct access: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("CuPy not available. Falling back to NumPy export.")
            return self.export_for_peer_mapping(data, name, metadata)
    
    def export_for_uvm(self, data: np.ndarray, 
                      name: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      page_access_hints: Optional[np.ndarray] = None) -> str:
        """
        Export data optimized for UVM (Unified Virtual Memory) with ML-driven page placement
        
        Args:
            data: NumPy array to export
            name: Identifier for the data
            metadata: Additional metadata to include
            page_access_hints: Optional hints for page placement (0-1 values, higher means more likely to access)
            
        Returns:
            Path to the exported file
        """
        # Create output directory
        output_dir = os.path.join(self.base_path, 'zero_copy_exports', 'uvm')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.uvmmem"
        output_path = os.path.join(output_dir, filename)
        
        # Create metadata
        meta = {
            'name': name,
            'timestamp': timestamp,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'access_strategy': 'uvm',
            'page_size': 4096  # Standard page size
        }
        
        if metadata:
            meta.update(metadata)
        
        # Include page access hints if provided
        if page_access_hints is not None:
            if page_access_hints.shape[0] * 4096 < np.prod(data.shape) * data.itemsize:
                # Calculate how many bytes each hint covers
                bytes_per_hint = (np.prod(data.shape) * data.itemsize) / page_access_hints.shape[0]
                meta['bytes_per_hint'] = bytes_per_hint
            else:
                # Standard page size
                meta['bytes_per_hint'] = 4096
            
            # Save page access hints in metadata file
            page_hints_file = f"{output_path}.hints"
            np.save(page_hints_file, page_access_hints)
            meta['page_hints_file'] = page_hints_file
        
        # Write metadata file
        with open(f"{output_path}.meta", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Export data in UVM-optimized format
        with open(output_path, 'wb') as f:
            # Write header with shape, dtype, and UVM-specific information
            header = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'page_size': 4096,
                'has_hints': page_access_hints is not None
            }
            header_bytes = json.dumps(header).encode('utf-8')
            f.write(len(header_bytes).to_bytes(8, byteorder='little'))
            f.write(header_bytes)
            
            # Align to page boundary
            f.write(b'\x00' * (4096 - (f.tell() % 4096) if f.tell() % 4096 != 0 else 0))
            
            # Write the data
            np.save(f, data)
        
        logger.info(f"Exported data for UVM access: {output_path}")
        return output_path
    
    def export_for_peer_mapping(self, data: np.ndarray, 
                               name: str, 
                               metadata: Optional[Dict[str, Any]] = None,
                               coherence_level: int = 2) -> str:
        """
        Export data optimized for custom peer mapping with explicit coherence control
        
        Args:
            data: NumPy array to export
            name: Identifier for the data
            metadata: Additional metadata to include
            coherence_level: Coherence level (0=None, 1=Relaxed, 2=Full)
            
        Returns:
            Path to the exported file
        """
        # Create output directory
        output_dir = os.path.join(self.base_path, 'zero_copy_exports', 'peer_mapping')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}_c{coherence_level}.pmem"
        output_path = os.path.join(output_dir, filename)
        
        # Create metadata
        meta = {
            'name': name,
            'timestamp': timestamp,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'access_strategy': 'peer_mapping',
            'coherence_level': coherence_level,
            'memory_layout': 'interleaved' if coherence_level == 1 else 'contiguous'
        }
        
        if metadata:
            meta.update(metadata)
        
        # Write metadata file
        with open(f"{output_path}.meta", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Export data in peer-mapping-optimized format
        with open(output_path, 'wb') as f:
            # Write header
            header = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'coherence_level': coherence_level,
            }
            header_bytes = json.dumps(header).encode('utf-8')
            f.write(len(header_bytes).to_bytes(8, byteorder='little'))
            f.write(header_bytes)
            
            # Align to optimal boundary
            alignment = 4096 if coherence_level == 2 else 256
            f.write(b'\x00' * (alignment - (f.tell() % alignment) if f.tell() % alignment != 0 else 0))
            
            # Write the data
            if coherence_level == 1:
                # For relaxed coherence, use interleaved layout
                self._write_interleaved(f, data)
            else:
                # For full coherence, use standard layout
                np.save(f, data)
        
        logger.info(f"Exported data for peer mapping access: {output_path}")
        return output_path
    
    def _write_interleaved(self, file: BinaryIO, data: np.ndarray) -> None:
        """
        Write data in interleaved format for optimal peer mapping with relaxed coherence
        
        Args:
            file: Open binary file object
            data: NumPy array to write
        """
        # Calculate item size in bytes
        item_size = data.itemsize
        
        # Flatten data for interleaving
        flat_data = data.reshape(-1)
        
        # Calculate interleave size (64KB blocks)
        interleave_size = 65536 // item_size
        
        # Interleave blocks
        for i in range(0, len(flat_data), interleave_size):
            block = flat_data[i:i+interleave_size]
            block.tofile(file)
            
            # Add padding between blocks for optimal memory access
            padding_size = 256 - (len(block) * item_size % 256) if (len(block) * item_size % 256) != 0 else 0
            file.write(b'\x00' * padding_size)
    
    def auto_export(self, data: np.ndarray, 
                   name: str, 
                   metadata: Optional[Dict[str, Any]] = None,
                   access_pattern: Optional[str] = None) -> Dict[str, str]:
        """
        Automatically export data in optimal format based on access pattern
        
        Args:
            data: NumPy array to export
            name: Identifier for the data
            metadata: Additional metadata to include
            access_pattern: Optional hint for access pattern ('random', 'sequential', 'strided')
            
        Returns:
            Dictionary with paths to exported files for each strategy
        """
        export_paths = {}
        
        # Add access pattern to metadata
        if metadata is None:
            metadata = {}
        
        if access_pattern:
            metadata['access_pattern'] = access_pattern
        
        # Determine optimal export strategies based on data size and access pattern
        data_size_mb = data.nbytes / (1024 * 1024)
        
        # For small data (<50MB), use GPU-Direct for lowest latency
        if data_size_mb < 50:
            export_paths['gpu_direct'] = self.export_for_gpu_direct(data, name, metadata)
            
            # Also export in other formats for flexibility
            if access_pattern != 'sequential':
                # For random access, UVM with hints can be beneficial
                page_hints = self._generate_access_hints(data.shape, access_pattern)
                export_paths['uvm'] = self.export_for_uvm(data, name, metadata, page_hints)
            
            export_paths['peer_mapping'] = self.export_for_peer_mapping(
                data, name, metadata, coherence_level=2)
        
        # For medium data (50MB-500MB), use strategy based on access pattern
        elif data_size_mb < 500:
            if access_pattern == 'sequential':
                # Sequential access benefits from GPU-Direct
                export_paths['gpu_direct'] = self.export_for_gpu_direct(data, name, metadata)
                export_paths['peer_mapping'] = self.export_for_peer_mapping(
                    data, name, metadata, coherence_level=1)
            elif access_pattern == 'random':
                # Random access benefits from UVM with hints
                page_hints = self._generate_access_hints(data.shape, 'random')
                export_paths['uvm'] = self.export_for_uvm(data, name, metadata, page_hints)
                export_paths['peer_mapping'] = self.export_for_peer_mapping(
                    data, name, metadata, coherence_level=2)
            else:
                # Mixed/unknown pattern, use all strategies
                export_paths['gpu_direct'] = self.export_for_gpu_direct(data, name, metadata)
                page_hints = self._generate_access_hints(data.shape, 'mixed')
                export_paths['uvm'] = self.export_for_uvm(data, name, metadata, page_hints)
                export_paths['peer_mapping'] = self.export_for_peer_mapping(
                    data, name, metadata, coherence_level=1)
        
        # For large data (>500MB), use UVM and peer mapping
        else:
            # Large data is unlikely to fit in GPU memory, so UVM is preferred
            page_hints = self._generate_access_hints(data.shape, access_pattern or 'mixed')
            export_paths['uvm'] = self.export_for_uvm(data, name, metadata, page_hints)
            
            # Also use peer mapping with relaxed coherence for parts that fit
            export_paths['peer_mapping'] = self.export_for_peer_mapping(
                data, name, metadata, coherence_level=1)
        
        logger.info(f"Auto-exported data '{name}' using {len(export_paths)} strategies")
        return export_paths
    
    def _generate_access_hints(self, shape: Tuple[int, ...], 
                              access_pattern: Optional[str] = None) -> np.ndarray:
        """
        Generate page access hints for UVM optimization
        
        Args:
            shape: Shape of the data
            access_pattern: Access pattern hint ('random', 'sequential', 'strided', 'mixed')
            
        Returns:
            NumPy array with page access probabilities (0-1)
        """
        # Calculate total size in bytes
        total_size = np.prod(shape) * 4  # Assuming float32 data
        
        # Calculate number of pages (4KB each)
        num_pages = (total_size + 4095) // 4096
        
        # Generate hints based on access pattern
        if access_pattern == 'sequential':
            # Sequential access - gradually decreasing probability
            hints = np.linspace(1.0, 0.1, num_pages)
        elif access_pattern == 'random':
            # Random access - uniform distribution with some hot spots
            hints = np.random.uniform(0.1, 0.5, num_pages)
            # Add some hot spots (20% of pages get higher probability)
            hot_spots = np.random.choice(num_pages, size=num_pages // 5, replace=False)
            hints[hot_spots] = np.random.uniform(0.7, 1.0, size=len(hot_spots))
        elif access_pattern == 'strided':
            # Strided access - periodic pattern
            hints = np.ones(num_pages) * 0.3
            # Every 8th page is accessed more frequently
            hints[::8] = 0.9
        else:
            # Mixed/unknown pattern - use moderate values with some variation
            hints = np.random.uniform(0.4, 0.6, num_pages)
        
        return hints


class MLDataExporter:
    """
    Exports data for ML training and inference
    
    This class handles standard ML formats as well as optimized exports
    for Predis' ML-driven prefetching system.
    """
    
    def __init__(self, base_path: str = "../../../data"):
        """
        Initialize exporter
        
        Args:
            base_path: Base directory for data exports
        """
        self.base_path = base_path
        self.zero_copy_exporter = ZeroCopyExporter(base_path)
        
        # Create base directories if they don't exist
        for subdir in ['ml_features', 'ml_models', 'ml_datasets']:
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    
    def export_training_data(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            name: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Export training data (features and targets)
        
        Args:
            X: Feature array
            y: Target array
            name: Dataset name
            metadata: Additional metadata
            
        Returns:
            Dictionary with paths to exported files
        """
        # Create output directory
        output_dir = os.path.join(self.base_path, 'ml_datasets')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        basename = f"{name}_{timestamp}"
        
        # Create metadata
        meta = {
            'name': name,
            'timestamp': timestamp,
            'n_samples': len(X),
            'feature_shape': X.shape,
            'target_shape': y.shape,
            'feature_dtype': str(X.dtype),
            'target_dtype': str(y.dtype),
        }
        
        if metadata:
            meta.update(metadata)
        
        # Export standard formats for compatibility
        export_paths = {}
        
        # Numpy format
        np_path = os.path.join(output_dir, f"{basename}.npz")
        np.savez_compressed(np_path, X=X, y=y)
        export_paths['numpy'] = np_path
        
        # Pickle format
        pkl_path = os.path.join(output_dir, f"{basename}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'metadata': meta}, f)
        export_paths['pickle'] = pkl_path
        
        # Write metadata file
        meta_path = os.path.join(output_dir, f"{basename}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        export_paths['metadata'] = meta_path
        
        # Export using zero-copy format for optimal GPU access
        # Determine access pattern based on data type
        if len(X.shape) == 3:  # Sequence data
            access_pattern = 'sequential'
        elif len(X.shape) == 2 and X.shape[1] > 100:  # Wide feature matrix
            access_pattern = 'strided'
        else:
            access_pattern = 'mixed'
        
        # Export features using zero-copy exporter
        X_paths = self.zero_copy_exporter.auto_export(
            X, f"{name}_features", metadata, access_pattern)
        export_paths['features_zero_copy'] = X_paths
        
        # Export targets using zero-copy exporter
        y_paths = self.zero_copy_exporter.auto_export(
            y, f"{name}_targets", metadata, 'random')  # Targets are often accessed randomly
        export_paths['targets_zero_copy'] = y_paths
        
        logger.info(f"Exported training data '{name}' with {len(X)} samples")
        return export_paths
    
    def export_inference_features(self, 
                                 features: np.ndarray, 
                                 name: str,
                                 real_time: bool = True) -> Dict[str, Any]:
        """
        Export features for inference optimized for real-time prediction
        
        Args:
            features: Feature array
            name: Feature set name
            real_time: Whether this is for real-time inference
            
        Returns:
            Dictionary with export information
        """
        # For real-time inference, optimize for GPU-Direct pathway
        # to achieve lowest latency
        if real_time:
            # Export using GPU-Direct for lowest latency
            output_path = self.zero_copy_exporter.export_for_gpu_direct(
                features, f"{name}_rt", {'real_time': True})
            
            return {
                'path': output_path,
                'strategy': 'gpu_direct',
                'shape': features.shape,
                'dtype': str(features.dtype),
                'real_time': True
            }
        else:
            # For non-real-time, use auto-export to optimize based on size and access pattern
            paths = self.zero_copy_exporter.auto_export(
                features, name, {'real_time': False})
            
            return {
                'paths': paths,
                'shape': features.shape,
                'dtype': str(features.dtype),
                'real_time': False
            }
    
    def export_model_weights(self, 
                            weights: Dict[str, np.ndarray], 
                            model_name: str,
                            model_type: str,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export model weights optimized for fast inference
        
        Args:
            weights: Dictionary of weight arrays
            model_name: Name of the model
            model_type: Type of model (e.g., 'lstm', 'ngboost', 'xgboost')
            metadata: Additional metadata
            
        Returns:
            Path to the exported weights
        """
        # Create output directory
        output_dir = os.path.join(self.base_path, 'ml_models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{model_name}_{model_type}_{timestamp}.weights"
        output_path = os.path.join(output_dir, filename)
        
        # Create metadata
        meta = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': timestamp,
            'num_weights': len(weights),
            'weight_shapes': {k: v.shape for k, v in weights.items()},
            'weight_dtypes': {k: str(v.dtype) for k, v in weights.items()},
        }
        
        if metadata:
            meta.update(metadata)
        
        # Write metadata file
        with open(f"{output_path}.meta", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # For each weight matrix, export using optimal strategy
        for weight_name, weight_array in weights.items():
            # Determine optimal strategy based on size and model type
            if weight_array.nbytes < 10 * 1024 * 1024:  # < 10MB
                # Small weights benefit from GPU-Direct
                self.zero_copy_exporter.export_for_gpu_direct(
                    weight_array, 
                    f"{model_name}_{weight_name}", 
                    {'model': model_name, 'layer': weight_name}
                )
            else:
                # Larger weights benefit from UVM with hints
                # Generate access hints based on model type and layer name
                if 'lstm' in model_type.lower() and 'recurrent' in weight_name.lower():
                    # LSTM recurrent weights have sequential access patterns
                    access_pattern = 'sequential'
                elif 'embedding' in weight_name.lower():
                    # Embedding layers have random access
                    access_pattern = 'random'
                else:
                    # Default to mixed
                    access_pattern = 'mixed'
                
                page_hints = self.zero_copy_exporter._generate_access_hints(
                    weight_array.shape, access_pattern)
                
                self.zero_copy_exporter.export_for_uvm(
                    weight_array,
                    f"{model_name}_{weight_name}",
                    {'model': model_name, 'layer': weight_name},
                    page_hints
                )
        
        # Also save all weights together in standard format for compatibility
        with open(output_path, 'wb') as f:
            pickle.dump(weights, f)
        
        logger.info(f"Exported model weights for {model_name} ({model_type})")
        return output_path
