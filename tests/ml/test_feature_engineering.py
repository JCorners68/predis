#!/usr/bin/env python3
"""
Unit Tests for Predis ML Feature Engineering

This module contains unit tests for the feature engineering components
of the Predis ML-driven predictive prefetching system.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import Predis modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Predis ML components
from src.ml.data.synthetic.generators import generate_zipfian_access_pattern
from src.ml.features.extractors import AccessPatternFeatureExtractor
from src.ml.features.temporal import TemporalFeatureGenerator
from src.ml.features.sequential import SequenceFeatureGenerator, SequentialPatternMiner
from src.ml.features.pipeline import FeatureEngineeringPipeline


class TestAccessPatternFeatureExtractor(unittest.TestCase):
    """Tests for the AccessPatternFeatureExtractor class"""
    
    def setUp(self):
        """Set up test data"""
        # Generate synthetic access logs
        access_logs = generate_zipfian_access_pattern(num_keys=100, num_accesses=1000)
        self.df = pd.DataFrame(access_logs)
        
        # Create extractor
        self.extractor = AccessPatternFeatureExtractor()
    
    def test_extract_features(self):
        """Test basic feature extraction"""
        # Extract features
        feature_df = self.extractor.extract_features(self.df)
        
        # Check that features were extracted
        self.assertGreater(len(feature_df.columns), len(self.df.columns))
        
        # Check for specific features
        expected_features = ['key_frequency', 'recency_score']
        for feature in expected_features:
            self.assertIn(feature, feature_df.columns)
        
        # Check that all rows were preserved
        self.assertEqual(len(feature_df), len(self.df))


class TestTemporalFeatureGenerator(unittest.TestCase):
    """Tests for the TemporalFeatureGenerator class"""
    
    def setUp(self):
        """Set up test data"""
        # Generate synthetic access logs with timestamps
        access_logs = generate_zipfian_access_pattern(num_keys=100, num_accesses=1000)
        self.df = pd.DataFrame(access_logs)
        
        # Ensure datetime column exists
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
        
        # Create generator
        self.generator = TemporalFeatureGenerator(use_gpu=False)
    
    def test_extract_time_of_day_features(self):
        """Test extraction of time-of-day features"""
        # Extract features
        feature_df = self.generator.extract_time_of_day_features(self.df)
        
        # Check for specific features
        expected_features = ['hour_of_day', 'normalized_time_of_day', 'is_business_hours']
        for feature in expected_features:
            self.assertIn(feature, feature_df.columns)
    
    def test_extract_all_temporal_features(self):
        """Test extraction of all temporal features"""
        # Extract all features
        feature_df = self.generator.extract_all_temporal_features(self.df)
        
        # Check that features were extracted
        self.assertGreater(len(feature_df.columns), len(self.df.columns))
        
        # Check for specific feature groups
        time_features = ['hour_of_day', 'minute_of_hour']
        day_features = ['day_of_week', 'is_weekend']
        seasonality_features = ['month', 'quarter']
        
        for feature in time_features + day_features + seasonality_features:
            self.assertIn(feature, feature_df.columns)


class TestSequenceFeatureGenerator(unittest.TestCase):
    """Tests for the SequenceFeatureGenerator class"""
    
    def setUp(self):
        """Set up test data"""
        # Generate synthetic access logs
        access_logs = generate_zipfian_access_pattern(num_keys=100, num_accesses=1000)
        self.df = pd.DataFrame(access_logs)
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')
        
        # Create generator
        self.generator = SequenceFeatureGenerator(sequence_length=5, use_gpu=False)
    
    def test_add_previous_keys(self):
        """Test adding previous keys to the dataframe"""
        # Add previous keys
        feature_df = self.generator.add_previous_keys(self.df)
        
        # Check for previous key columns
        for i in range(1, 6):
            self.assertIn(f'prev_key_{i}', feature_df.columns)
    
    def test_add_sequence_features(self):
        """Test adding sequence-based features"""
        # Add sequence features
        feature_df = self.generator.add_previous_keys(self.df)
        feature_df = self.generator.add_sequence_features(feature_df)
        
        # Check for specific sequence features
        expected_features = ['key_repeat_count', 'repeat_distance']
        for feature in expected_features:
            self.assertIn(feature, feature_df.columns)


class TestFeatureEngineeringPipeline(unittest.TestCase):
    """Tests for the FeatureEngineeringPipeline class"""
    
    def setUp(self):
        """Set up test data"""
        # Generate synthetic access logs
        access_logs = generate_zipfian_access_pattern(num_keys=100, num_accesses=1000)
        self.df = pd.DataFrame(access_logs)
        
        # Create pipeline
        self.pipeline = FeatureEngineeringPipeline(use_gpu=False)
    
    def test_extract_features(self):
        """Test the complete feature extraction pipeline"""
        # Initialize pipeline
        self.pipeline.initialize(self.df)
        
        # Extract features
        feature_df = self.pipeline.extract_features(self.df)
        
        # Check that features were extracted
        self.assertGreater(len(feature_df.columns), len(self.df.columns))
        
        # Check that all rows were preserved
        self.assertEqual(len(feature_df), len(self.df))
        
        # Check that the original key column is preserved
        self.assertIn('key', feature_df.columns)
    
    def test_predict_next_access(self):
        """Test prediction of next access"""
        # Initialize pipeline
        self.pipeline.initialize(self.df)
        
        # Predict next access
        predictions = self.pipeline.predict_next_access(top_n=3)
        
        # Check that predictions were made
        self.assertIsInstance(predictions, list)
        self.assertLessEqual(len(predictions), 3)
        
        # Check prediction format
        if predictions:
            key, probability = predictions[0]
            self.assertIsInstance(key, str)
            self.assertIsInstance(probability, float)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == '__main__':
    unittest.main()
