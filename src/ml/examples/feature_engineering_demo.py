#!/usr/bin/env python3
"""
Feature Engineering Pipeline Demo for Predis ML

This script demonstrates the complete feature engineering pipeline for Predis ML,
integrating synthetic data generation, feature extraction, and zero-copy optimization.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path to import Predis modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Predis ML components
from ml.data.synthetic.generators import (
    generate_zipfian_access_pattern,
    generate_temporal_access_pattern,
    generate_ml_training_access_pattern,
    generate_hft_access_pattern,
    generate_gaming_access_pattern
)
from ml.data.synthetic.workloads import (
    WebServiceWorkload,
    DatabaseWorkload,
    MLTrainingWorkload,
    GamingWorkload,
    CombinedWorkloadGenerator
)
from ml.data.synthetic.validation import SyntheticDataValidator
from ml.data.utils.formatters import AccessLogFormatter, FeatureFormatConverter
from ml.data.utils.exporters import MLDataExporter, ZeroCopyExporter
from ml.features.extractors import AccessPatternFeatureExtractor, GPUOptimizedFeatureExtractor
from ml.features.temporal import TemporalFeatureGenerator
from ml.features.sequential import SequenceFeatureGenerator, SequentialPatternMiner
from ml.features.realtime import RealtimeFeatureExtractor, ZeroCopyFeatureExtractor
from ml.features.pipeline import FeatureEngineeringPipeline, ZeroCopyFeaturePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_arg_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description="Predis ML Feature Engineering Demo")
    parser.add_argument("--workload", choices=["web", "database", "ml", "gaming", "combined"],
                       default="combined", help="Workload type to generate")
    parser.add_argument("--num-keys", type=int, default=10000,
                       help="Number of unique keys to generate")
    parser.add_argument("--num-accesses", type=int, default=100000,
                       help="Number of access events to generate")
    parser.add_argument("--output-dir", type=str, default="../../../data/demo",
                       help="Output directory for generated data and features")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU acceleration if available")
    parser.add_argument("--validate", action="store_true",
                       help="Validate synthetic data")
    parser.add_argument("--zero-copy", action="store_true",
                       help="Use zero-copy memory interface for feature extraction")
    parser.add_argument("--real-time", action="store_true",
                       help="Demonstrate real-time feature extraction")
    return parser


def generate_synthetic_data(args):
    """Generate synthetic data based on specified workload"""
    logger.info(f"Generating synthetic {args.workload} workload data...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate workload data
    if args.workload == "web":
        workload = WebServiceWorkload(
            num_keys=args.num_keys,
            traffic_pattern="diurnal"
        )
        df = workload.generate(args.num_accesses)
    
    elif args.workload == "database":
        workload = DatabaseWorkload(
            num_keys=args.num_keys,
            read_write_ratio=0.8
        )
        df = workload.generate(args.num_accesses)
    
    elif args.workload == "ml":
        workload = MLTrainingWorkload(
            num_keys=args.num_keys,
            batch_size=32,
            num_epochs=5
        )
        df = workload.generate(args.num_accesses)
    
    elif args.workload == "gaming":
        workload = GamingWorkload(
            num_keys=args.num_keys,
            num_players=100
        )
        df = workload.generate(args.num_accesses)
    
    else:  # combined
        combined_gen = CombinedWorkloadGenerator(
            num_keys=args.num_keys,
            workload_weights={
                "web": 0.4,
                "database": 0.3,
                "ml": 0.2,
                "gaming": 0.1
            }
        )
        df = combined_gen.generate(args.num_accesses)
    
    # Save generated data
    output_path = os.path.join(args.output_dir, f"{args.workload}_workload.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} access events with {df['key'].nunique()} unique keys")
    logger.info(f"Data saved to {output_path}")
    
    return df, output_path


def validate_synthetic_data(df, args):
    """Validate the synthetic data"""
    logger.info("Validating synthetic data...")
    
    # Create validator
    validator = SyntheticDataValidator(df)
    
    # Run validations
    validation_results = validator.run_all_validations()
    
    # Save validation report
    report_path = os.path.join(args.output_dir, f"{args.workload}_validation_report.json")
    validator.save_validation_report(report_path)
    
    # Generate validation visualizations
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    validator.generate_visualizations(viz_dir)
    
    logger.info(f"Validation report saved to {report_path}")
    logger.info(f"Validation visualizations saved to {viz_dir}")
    
    return validation_results


def extract_features(df, args):
    """Extract features from the data"""
    logger.info("Extracting features...")
    
    # Determine feature extraction approach
    if args.zero_copy and args.use_gpu:
        logger.info("Using zero-copy feature pipeline...")
        
        # Create zero-copy pipeline
        pipeline = ZeroCopyFeaturePipeline(
            strategy='auto',
            output_dir=os.path.join(args.output_dir, "features")
        )
        
        # Extract features
        result = pipeline.extract_features(df)
        
        # Get the features
        feature_df = result['features']
        
        logger.info(f"Extracted features using {result['strategy']} strategy")
        
    else:
        # Create standard pipeline
        pipeline = FeatureEngineeringPipeline(
            output_dir=os.path.join(args.output_dir, "features"),
            use_gpu=args.use_gpu
        )
        
        # Initialize pipeline with data
        pipeline.initialize(df)
        
        # Extract features
        feature_df = pipeline.extract_features(df)
    
    # Save features
    feature_path = os.path.join(args.output_dir, "features", f"{args.workload}_features.csv")
    feature_df.to_csv(feature_path, index=False)
    
    logger.info(f"Extracted {len(feature_df.columns) - 1} features for {len(feature_df)} records")
    logger.info(f"Features saved to {feature_path}")
    
    return feature_df, pipeline


def demonstrate_realtime_extraction(df, pipeline, args):
    """Demonstrate real-time feature extraction"""
    logger.info("Demonstrating real-time feature extraction...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Get a sample of recent keys to predict
    recent_keys = df['key'].iloc[-100:].unique()[:10]
    
    logger.info(f"Extracting real-time features for {len(recent_keys)} keys...")
    
    # Time the extraction
    start_time = time.time()
    
    # Extract features for these keys
    if hasattr(pipeline, 'extract_realtime_features'):
        features = pipeline.extract_realtime_features(recent_keys)
    else:
        features = {key: pipeline.realtime_extractor.extract_features_for_key(key) 
                  for key in recent_keys}
    
    extraction_time = time.time() - start_time
    
    # Show results
    logger.info(f"Real-time extraction completed in {extraction_time:.6f} seconds")
    logger.info(f"Average time per key: {extraction_time / len(recent_keys):.6f} seconds")
    
    # Predict next accesses
    logger.info("Predicting next keys to be accessed...")
    predictions = pipeline.predict_next_access(top_n=5)
    
    # Show predictions
    for key, probability in predictions:
        logger.info(f"  Key: {key}, Probability: {probability:.4f}")
    
    return features, predictions


def prepare_for_ml_training(feature_df, args):
    """Prepare features for ML training"""
    logger.info("Preparing data for ML training...")
    
    # Configure for ML model input
    converter = FeatureFormatConverter()
    
    # Select feature columns (exclude non-numeric and identifier columns)
    feature_cols = [col for col in feature_df.columns 
                   if col not in ['key', 'operation', 'timestamp', 'datetime']
                   and pd.api.types.is_numeric_dtype(feature_df[col])]
    
    # Create sequence data for training
    X, y = converter.to_training_format(
        feature_df, 
        sequence_length=10,
        target_column='key',
        feature_columns=feature_cols
    )
    
    logger.info(f"Prepared {X.shape[0]} training sequences with {X.shape[2]} features")
    
    # Export training data
    exporter = MLDataExporter(base_path=args.output_dir)
    export_paths = exporter.export_training_data(
        X, y, f"{args.workload}_training_data"
    )
    
    logger.info(f"Training data exported to:")
    for format_name, path in export_paths.items():
        if isinstance(path, dict):
            for strategy, strategy_path in path.items():
                logger.info(f"  - {format_name}/{strategy}: {strategy_path}")
        else:
            logger.info(f"  - {format_name}: {path}")
    
    # Prepare for zero-copy if requested
    if args.zero_copy and args.use_gpu:
        logger.info("Preparing data for zero-copy memory interface...")
        
        zero_copy_data = converter.prepare_zero_copy_data(X, y)
        
        # Export using zero-copy exporter
        zero_copy_exporter = ZeroCopyExporter(base_path=args.output_dir)
        
        # Choose strategy based on data size
        data_size_mb = X.nbytes / (1024 * 1024)
        
        if data_size_mb < 50:
            # Small data, use GPU-Direct
            zero_copy_path = zero_copy_exporter.export_for_gpu_direct(
                X, f"{args.workload}_X_gpu_direct"
            )
        elif data_size_mb < 500:
            # Medium data, use UVM with hints
            page_hints = zero_copy_exporter._generate_access_hints(X.shape, 'sequential')
            zero_copy_path = zero_copy_exporter.export_for_uvm(
                X, f"{args.workload}_X_uvm", page_access_hints=page_hints
            )
        else:
            # Large data, use peer mapping
            zero_copy_path = zero_copy_exporter.export_for_peer_mapping(
                X, f"{args.workload}_X_peer_mapping", coherence_level=1
            )
        
        logger.info(f"Zero-copy data exported to: {zero_copy_path}")
    
    return X, y


def main():
    """Main function to run the demo"""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print(" Predis ML Feature Engineering Pipeline Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Generate synthetic data
    df, data_path = generate_synthetic_data(args)
    
    # Validate synthetic data if requested
    if args.validate:
        validation_results = validate_synthetic_data(df, args)
    
    # Extract features
    feature_df, pipeline = extract_features(df, args)
    
    # Demonstrate real-time extraction if requested
    if args.real_time:
        realtime_features, predictions = demonstrate_realtime_extraction(df, pipeline, args)
    
    # Prepare for ML training
    X, y = prepare_for_ml_training(feature_df, args)
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(" Summary ".center(80, "="))
    print("="*80)
    print(f"Workload type:       {args.workload}")
    print(f"Number of keys:      {args.num_keys}")
    print(f"Number of accesses:  {args.num_accesses}")
    print(f"Generated data size: {os.path.getsize(data_path) / (1024*1024):.2f} MB")
    print(f"Number of features:  {len(feature_df.columns) - 1}")
    print(f"Total time:          {total_time:.2f} seconds")
    print("="*80 + "\n")
    
    print("Done! The feature engineering pipeline demo has completed successfully.")
    print(f"All outputs have been saved to: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
