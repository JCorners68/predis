"""
Sequence-Based Feature Engineering for Predis ML

This module provides sequence-based feature extraction for Predis' ML-driven prefetching,
focusing on identifying sequential patterns in cache access logs that can be leveraged 
by the multi-strategy zero-copy memory interface system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
import time
from collections import defaultdict, Counter
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceFeatureGenerator:
    """
    Generates features from access sequence patterns
    
    This class is designed to identify sequential patterns in cache accesses
    that can be leveraged for predictive prefetching.
    """
    
    def __init__(self, sequence_length: int = 10, use_gpu: bool = True):
        """
        Initialize sequence feature generator
        
        Args:
            sequence_length: Length of sequences to analyze
            use_gpu: Whether to use GPU acceleration when available
        """
        self.sequence_length = sequence_length
        self.use_gpu = use_gpu
        self.has_gpu = False
        self.key_vocab = {}  # Maps keys to integer IDs
        self.reverse_vocab = {}  # Maps integer IDs back to keys
        self.next_key_id = 0
        
        # Try to import GPU libraries if requested
        if use_gpu:
            try:
                import cupy as cp
                import cudf
                self.has_gpu = True
                logger.info("GPU libraries available, using GPU acceleration")
            except ImportError:
                logger.warning("GPU libraries not available, falling back to CPU implementation")
    
    def extract_all_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all sequence-based features from access logs
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added sequence features
        """
        result = df.copy()
        
        # Sort by timestamp to ensure correct sequence
        result = result.sort_values('timestamp')
        
        # Build key vocabulary if not already built
        if not self.key_vocab:
            self._build_key_vocabulary(result)
        
        # Extract features
        result = self.add_previous_keys(result)
        result = self.add_sequence_features(result)
        result = self.add_ngram_features(result)
        result = self.add_key_transition_features(result)
        
        return result
    
    def _build_key_vocabulary(self, df: pd.DataFrame) -> None:
        """
        Build vocabulary mapping keys to integer IDs
        
        Args:
            df: DataFrame with access logs
        """
        # Get unique keys
        unique_keys = df['key'].unique()
        
        # Assign IDs to keys
        for key in unique_keys:
            self.key_vocab[key] = self.next_key_id
            self.reverse_vocab[self.next_key_id] = key
            self.next_key_id += 1
        
        logger.info(f"Built key vocabulary with {len(self.key_vocab)} unique keys")
    
    def _key_to_id(self, key: str) -> int:
        """Convert key to integer ID, adding to vocabulary if needed"""
        if key not in self.key_vocab:
            self.key_vocab[key] = self.next_key_id
            self.reverse_vocab[self.next_key_id] = key
            self.next_key_id += 1
        return self.key_vocab[key]
    
    def add_previous_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add previous keys in the access sequence
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added previous key columns
        """
        result = df.copy()
        
        # Ensure data is sorted by timestamp
        result = result.sort_values('timestamp')
        
        # Add previous keys as separate columns
        for i in range(1, self.sequence_length + 1):
            result[f'prev_key_{i}'] = result['key'].shift(i)
        
        # Add previous key IDs for numeric processing
        for i in range(1, self.sequence_length + 1):
            col_name = f'prev_key_{i}'
            if col_name in result.columns:
                result[f'prev_key_id_{i}'] = result[col_name].apply(
                    lambda k: self._key_to_id(k) if pd.notna(k) else -1)
        
        # Current key ID
        result['key_id'] = result['key'].apply(self._key_to_id)
        
        return result
    
    def add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sequence-based features
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added sequence features
        """
        result = df.copy()
        
        # Check if we have the necessary previous key columns
        prev_key_cols = [f'prev_key_{i}' for i in range(1, self.sequence_length + 1)]
        if not all(col in result.columns for col in prev_key_cols):
            logger.warning("Previous key columns not found. Run add_previous_keys first.")
            result = self.add_previous_keys(result)
        
        # Add key repetition features
        result['key_repeat_count'] = 0
        
        for i in range(1, self.sequence_length + 1):
            prev_col = f'prev_key_{i}'
            # Count how many times the current key appears in the previous sequence
            result['key_repeat_count'] += (result['key'] == result[prev_col]).astype(int)
        
        # Add features for repeated patterns
        # Check if the current key matches a key from earlier in the sequence
        for i in range(1, self.sequence_length + 1):
            result[f'matches_prev_{i}'] = (result['key'] == result[f'prev_key_{i}']).astype(int)
        
        # Find most recent occurrence of current key in sequence
        result['repeat_distance'] = 0
        for i in range(1, self.sequence_length + 1):
            mask = (result['key'] == result[f'prev_key_{i}']) & (result['repeat_distance'] == 0)
            result.loc[mask, 'repeat_distance'] = i
        
        # Replace 0 with sequence_length + 1 for keys that don't repeat
        result.loc[result['repeat_distance'] == 0, 'repeat_distance'] = self.sequence_length + 1
        
        # Add feature for detecting cycling patterns
        # A cycling pattern is where the sequence [A, B, C, A, B, C] repeats
        result['has_cycle'] = 0
        
        # Check for cycles of different lengths
        for cycle_len in range(2, self.sequence_length // 2 + 1):
            # Check if the current sequence matches a shifted version of itself
            cycle_match = True
            for i in range(cycle_len):
                if i + cycle_len >= self.sequence_length:
                    cycle_match = False
                    break
                
                curr_idx = i + 1
                prev_idx = i + cycle_len + 1
                
                if f'prev_key_id_{curr_idx}' not in result.columns or \
                   f'prev_key_id_{prev_idx}' not in result.columns:
                    cycle_match = False
                    break
                
                if not np.array_equal(
                    result[f'prev_key_id_{curr_idx}'].values,
                    result[f'prev_key_id_{prev_idx}'].values
                ):
                    cycle_match = False
                    break
            
            if cycle_match:
                result['has_cycle'] = 1
                result['cycle_length'] = cycle_len
                break
        
        return result
    
    def add_ngram_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add n-gram based features from the access sequence
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added n-gram features
        """
        result = df.copy()
        
        # Check if we have the necessary previous key columns
        prev_key_cols = [f'prev_key_id_{i}' for i in range(1, self.sequence_length + 1)]
        if not all(col in result.columns for col in prev_key_cols):
            logger.warning("Previous key ID columns not found.")
            result = self.add_previous_keys(result)
            prev_key_cols = [f'prev_key_id_{i}' for i in range(1, self.sequence_length + 1)]
        
        # Initialize n-gram counters
        bigram_counter = Counter()
        trigram_counter = Counter()
        
        # Create sequences for n-gram analysis
        sequences = []
        for _, row in result.iterrows():
            # Create sequence of previous keys and current key
            sequence = []
            for i in range(self.sequence_length, 0, -1):
                key_id = row.get(f'prev_key_id_{i}', -1)
                if key_id >= 0:
                    sequence.append(key_id)
            
            # Add current key
            sequence.append(row['key_id'])
            sequences.append(sequence)
            
            # Count bigrams and trigrams
            for i in range(len(sequence) - 1):
                # Bigrams
                bigram = (sequence[i], sequence[i+1])
                bigram_counter[bigram] += 1
                
                # Trigrams
                if i < len(sequence) - 2:
                    trigram = (sequence[i], sequence[i+1], sequence[i+2])
                    trigram_counter[trigram] += 1
        
        # Find frequent n-grams (top 1000)
        top_bigrams = dict(bigram_counter.most_common(1000))
        top_trigrams = dict(trigram_counter.most_common(1000))
        
        # Convert to dictionaries for faster lookup
        bigram_dict = defaultdict(int)
        for (k1, k2), count in top_bigrams.items():
            bigram_dict[(k1, k2)] = count
        
        trigram_dict = defaultdict(int)
        for (k1, k2, k3), count in top_trigrams.items():
            trigram_dict[(k1, k2, k3)] = count
        
        # Add features for common n-grams
        result['bigram_count'] = 0
        result['trigram_count'] = 0
        
        # Calculate for each row
        for idx, row in result.iterrows():
            # Get current key ID
            current_key_id = row['key_id']
            
            # Check bigrams
            if idx > 0 and 'prev_key_id_1' in row and row['prev_key_id_1'] >= 0:
                prev_key_id = row['prev_key_id_1']
                bigram = (prev_key_id, current_key_id)
                result.loc[idx, 'bigram_count'] = bigram_dict[bigram]
            
            # Check trigrams
            if idx > 1 and 'prev_key_id_1' in row and 'prev_key_id_2' in row and \
               row['prev_key_id_1'] >= 0 and row['prev_key_id_2'] >= 0:
                prev1 = row['prev_key_id_1']
                prev2 = row['prev_key_id_2']
                trigram = (prev2, prev1, current_key_id)
                result.loc[idx, 'trigram_count'] = trigram_dict[trigram]
        
        # Normalize counts
        max_bigram = result['bigram_count'].max()
        max_trigram = result['trigram_count'].max()
        
        if max_bigram > 0:
            result['bigram_score'] = result['bigram_count'] / max_bigram
        else:
            result['bigram_score'] = 0
            
        if max_trigram > 0:
            result['trigram_score'] = result['trigram_count'] / max_trigram
        else:
            result['trigram_score'] = 0
        
        return result
    
    def add_key_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on key transition probabilities
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added transition features
        """
        result = df.copy()
        
        # Ensure data is sorted by timestamp
        result = result.sort_values('timestamp')
        
        # Calculate transition counts
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Count transitions from each key to next key
        prev_key = None
        for key in result['key']:
            if prev_key is not None:
                transition_counts[prev_key][key] += 1
            prev_key = key
        
        # Calculate transition probabilities
        transition_probs = {}
        for key1, transitions in transition_counts.items():
            total = sum(transitions.values())
            probs = {key2: count / total for key2, count in transitions.items()}
            transition_probs[key1] = probs
        
        # Add transition probability features
        result['next_key_prob'] = 0.0
        result['prev_key_prob'] = 0.0
        
        # Calculate for each row
        for i in range(len(result) - 1):
            current_key = result.iloc[i]['key']
            next_key = result.iloc[i + 1]['key']
            
            # Probability of next key given current key
            if current_key in transition_probs and next_key in transition_probs[current_key]:
                result.iloc[i, result.columns.get_loc('next_key_prob')] = \
                    transition_probs[current_key][next_key]
        
        # Calculate for backward transitions
        for i in range(1, len(result)):
            current_key = result.iloc[i]['key']
            prev_key = result.iloc[i - 1]['key']
            
            # Probability of current key given previous key
            if prev_key in transition_probs and current_key in transition_probs[prev_key]:
                result.iloc[i, result.columns.get_loc('prev_key_prob')] = \
                    transition_probs[prev_key][current_key]
        
        return result
    
    def extract_gpu_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sequence features using GPU acceleration
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added sequence features
        """
        if not self.has_gpu:
            logger.warning("GPU libraries not available, using CPU implementation")
            return self.extract_all_sequence_features(df)
        
        try:
            import cupy as cp
            import cudf
            
            # Convert to cuDF DataFrame for GPU processing
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Ensure data is sorted by timestamp
            gdf = gdf.sort_values('timestamp')
            
            # Build key vocabulary if not already built
            if not self.key_vocab:
                self._build_key_vocabulary(df)
            
            # Add key IDs
            gdf['key_id'] = gdf['key'].apply(lambda k: self._key_to_id(k))
            
            # Add previous key IDs
            for i in range(1, self.sequence_length + 1):
                gdf[f'prev_key_{i}'] = gdf['key'].shift(i)
                gdf[f'prev_key_id_{i}'] = gdf[f'prev_key_{i}'].apply(
                    lambda k: self._key_to_id(k) if k is not None else -1)
            
            # Add key repetition features using GPU
            gdf['key_repeat_count'] = 0
            for i in range(1, self.sequence_length + 1):
                gdf['key_repeat_count'] += (gdf['key'] == gdf[f'prev_key_{i}']).astype('int8')
            
            # Convert back to pandas for further processing
            result = gdf.to_pandas()
            
            # Add more complex features using the CPU implementation
            # These features are more difficult to implement efficiently on GPU
            result = self.add_sequence_features(result)
            result = self.add_ngram_features(result)
            result = self.add_key_transition_features(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GPU feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            return self.extract_all_sequence_features(df)


class SequenceVectorizer:
    """
    Vectorizes key sequences for ML model input
    
    This class converts key access sequences into numerical feature vectors
    that can be used as input to machine learning models.
    """
    
    def __init__(self, 
                sequence_length: int = 10, 
                max_features: int = 1000,
                use_tfidf: bool = True):
        """
        Initialize sequence vectorizer
        
        Args:
            sequence_length: Length of sequences to consider
            max_features: Maximum number of features to extract
            use_tfidf: Whether to use TF-IDF weighting
        """
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.key_vocab = {}
        self.reverse_vocab = {}
        self.next_key_id = 0
    
    def _build_key_vocabulary(self, keys: List[str]) -> None:
        """
        Build vocabulary mapping keys to integer IDs
        
        Args:
            keys: List of cache keys
        """
        # Get unique keys
        unique_keys = set(keys)
        
        # Assign IDs to keys
        for key in unique_keys:
            if key not in self.key_vocab:
                self.key_vocab[key] = self.next_key_id
                self.reverse_vocab[self.next_key_id] = key
                self.next_key_id += 1
        
        logger.info(f"Built key vocabulary with {len(self.key_vocab)} unique keys")
    
    def _key_to_id(self, key: str) -> int:
        """Convert key to integer ID, adding to vocabulary if needed"""
        if key not in self.key_vocab:
            self.key_vocab[key] = self.next_key_id
            self.reverse_vocab[self.next_key_id] = key
            self.next_key_id += 1
        return self.key_vocab[key]
    
    def _create_sequence_documents(self, df: pd.DataFrame) -> List[str]:
        """
        Create sequence 'documents' for vectorization
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            List of sequence documents
        """
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Create sequence documents
        documents = []
        
        # Use sliding window to create sequences
        for i in range(len(df) - self.sequence_length):
            # Get sequence of keys
            sequence = df['key'].iloc[i:i+self.sequence_length].tolist()
            
            # Convert keys to IDs and then to strings
            sequence_ids = [str(self._key_to_id(key)) for key in sequence]
            
            # Join IDs into a single document
            document = ' '.join(sequence_ids)
            documents.append(document)
        
        return documents
    
    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Fit vectorizer and transform sequences to feature matrix
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            Sparse matrix of sequence features
        """
        # Build vocabulary
        self._build_key_vocabulary(df['key'].tolist())
        
        # Create sequence documents
        documents = self._create_sequence_documents(df)
        
        # Create and fit vectorizer
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 3)  # Include unigrams, bigrams, and trigrams
            )
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 3),
                use_idf=False,
                norm=None  # No normalization for count vectorization
            )
        
        # Fit and transform
        features = self.vectorizer.fit_transform(documents)
        
        logger.info(f"Vectorized {len(documents)} sequences into {features.shape[1]} features")
        return features
    
    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Transform new sequences to feature matrix using fitted vectorizer
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            Sparse matrix of sequence features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Update vocabulary with any new keys
        self._build_key_vocabulary(df['key'].tolist())
        
        # Create sequence documents
        documents = self._create_sequence_documents(df)
        
        # Transform
        features = self.vectorizer.transform(documents)
        
        return features
    
    def save(self, filepath: str) -> None:
        """
        Save vectorizer to file
        
        Args:
            filepath: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save vectorizer and vocabularies
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'key_vocab': self.key_vocab,
                'reverse_vocab': self.reverse_vocab,
                'next_key_id': self.next_key_id,
                'sequence_length': self.sequence_length,
                'max_features': self.max_features,
                'use_tfidf': self.use_tfidf
            }, f)
        
        logger.info(f"Saved sequence vectorizer to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SequenceVectorizer':
        """
        Load vectorizer from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded SequenceVectorizer
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            sequence_length=data['sequence_length'],
            max_features=data['max_features'],
            use_tfidf=data['use_tfidf']
        )
        
        # Restore state
        instance.vectorizer = data['vectorizer']
        instance.key_vocab = data['key_vocab']
        instance.reverse_vocab = data['reverse_vocab']
        instance.next_key_id = data['next_key_id']
        
        logger.info(f"Loaded sequence vectorizer from {filepath}")
        return instance


class SequentialPatternMiner:
    """
    Mines sequential patterns from access logs
    
    This class identifies frequent sequential patterns that can be
    used for predictive prefetching.
    """
    
    def __init__(self, 
                min_support: float = 0.01, 
                max_pattern_length: int = 5,
                min_pattern_length: int = 2):
        """
        Initialize sequential pattern miner
        
        Args:
            min_support: Minimum support threshold (0-1)
            max_pattern_length: Maximum pattern length to mine
            min_pattern_length: Minimum pattern length to mine
        """
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.min_pattern_length = min_pattern_length
        self.patterns = []
        self.pattern_support = {}
        self.key_vocab = {}
        self.reverse_vocab = {}
        self.next_key_id = 0
    
    def _build_key_vocabulary(self, keys: List[str]) -> None:
        """
        Build vocabulary mapping keys to integer IDs
        
        Args:
            keys: List of cache keys
        """
        # Get unique keys
        unique_keys = set(keys)
        
        # Assign IDs to keys
        for key in unique_keys:
            if key not in self.key_vocab:
                self.key_vocab[key] = self.next_key_id
                self.reverse_vocab[self.next_key_id] = key
                self.next_key_id += 1
    
    def _key_to_id(self, key: str) -> int:
        """Convert key to integer ID, adding to vocabulary if needed"""
        if key not in self.key_vocab:
            self.key_vocab[key] = self.next_key_id
            self.reverse_vocab[self.next_key_id] = key
            self.next_key_id += 1
        return self.key_vocab[key]
    
    def _create_sequences(self, df: pd.DataFrame) -> List[List[int]]:
        """
        Create sequences of key IDs from access logs
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            List of key ID sequences
        """
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Convert keys to IDs
        key_ids = [self._key_to_id(key) for key in df['key']]
        
        # Create sequences using sliding window
        sequences = []
        for i in range(len(key_ids) - self.max_pattern_length + 1):
            sequence = key_ids[i:i+self.max_pattern_length]
            sequences.append(sequence)
        
        return sequences
    
    def _count_subsequences(self, sequences: List[List[int]]) -> Dict[Tuple[int, ...], int]:
        """
        Count occurrences of all possible subsequences
        
        Args:
            sequences: List of key ID sequences
            
        Returns:
            Dictionary mapping subsequences to counts
        """
        # Count subsequences
        counts = defaultdict(int)
        
        # For each sequence
        for sequence in sequences:
            # Generate all subsequences of length min_pattern_length to max_pattern_length
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                for i in range(len(sequence) - length + 1):
                    subsequence = tuple(sequence[i:i+length])
                    counts[subsequence] += 1
        
        return counts
    
    def mine_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Mine sequential patterns from access logs
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            List of patterns with support and confidence
        """
        # Build vocabulary
        self._build_key_vocabulary(df['key'].tolist())
        
        # Create sequences
        sequences = self._create_sequences(df)
        
        # Count subsequences
        counts = self._count_subsequences(sequences)
        
        # Calculate support
        total_sequences = len(sequences)
        pattern_support = {pattern: count / total_sequences 
                          for pattern, count in counts.items()}
        
        # Filter by minimum support
        frequent_patterns = {pattern: support 
                           for pattern, support in pattern_support.items() 
                           if support >= self.min_support}
        
        # Convert to list of dictionaries for easier use
        self.patterns = []
        for pattern, support in frequent_patterns.items():
            # Convert pattern IDs back to keys
            key_pattern = [self.reverse_vocab[key_id] for key_id in pattern]
            
            self.patterns.append({
                'pattern': key_pattern,
                'pattern_ids': pattern,
                'support': support,
                'length': len(pattern)
            })
        
        # Sort by support (descending)
        self.patterns.sort(key=lambda x: x['support'], reverse=True)
        
        # Store pattern support for later use
        self.pattern_support = frequent_patterns
        
        logger.info(f"Mined {len(self.patterns)} sequential patterns")
        return self.patterns
    
    def get_next_key_predictions(self, 
                                sequence: List[str], 
                                top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next keys based on a given sequence
        
        Args:
            sequence: List of recent keys
            top_n: Number of predictions to return
            
        Returns:
            List of (predicted_key, confidence) tuples
        """
        if not self.patterns:
            raise ValueError("No patterns mined yet. Call mine_patterns first.")
        
        # Convert sequence to IDs
        sequence_ids = [self._key_to_id(key) if key in self.key_vocab else -1 
                      for key in sequence]
        
        # Remove unknown keys
        sequence_ids = [key_id for key_id in sequence_ids if key_id >= 0]
        
        # Map to store predictions with confidences
        predictions = defaultdict(float)
        
        # Check if sequence matches beginning of any pattern
        for pattern in self.patterns:
            pattern_ids = pattern['pattern_ids']
            
            # Try different alignments of the sequence with the pattern
            for i in range(1, len(pattern_ids)):
                # Check if sequence ends with the beginning of this pattern
                if len(sequence_ids) >= i:
                    seq_suffix = sequence_ids[-i:]
                    pattern_prefix = pattern_ids[:i]
                    
                    if seq_suffix == list(pattern_prefix):
                        # Sequence matches pattern prefix, predict next key
                        if i < len(pattern_ids):
                            next_key_id = pattern_ids[i]
                            next_key = self.reverse_vocab[next_key_id]
                            
                            # Confidence based on pattern support and match length
                            confidence = pattern['support'] * (i / len(pattern_ids))
                            
                            # Update prediction confidence (take max if multiple matches)
                            predictions[next_key] = max(predictions[next_key], confidence)
        
        # Sort predictions by confidence
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N predictions
        return sorted_predictions[:top_n]
    
    def get_prefetchable_patterns(self, 
                                min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get patterns suitable for prefetching
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of prefetchable patterns
        """
        if not self.patterns:
            raise ValueError("No patterns mined yet. Call mine_patterns first.")
        
        prefetchable = []
        
        for pattern in self.patterns:
            # Calculate confidence by comparing this pattern's support
            # with the support of its prefix
            pattern_ids = pattern['pattern_ids']
            
            if len(pattern_ids) > 1:
                prefix = pattern_ids[:-1]
                if prefix in self.pattern_support:
                    prefix_support = self.pattern_support[prefix]
                    confidence = pattern['support'] / prefix_support
                    
                    if confidence >= min_confidence:
                        prefetchable.append({
                            'pattern': pattern['pattern'],
                            'support': pattern['support'],
                            'confidence': confidence,
                            'length': pattern['length']
                        })
        
        # Sort by confidence (descending)
        prefetchable.sort(key=lambda x: x['confidence'], reverse=True)
        
        return prefetchable
    
    def save(self, filepath: str) -> None:
        """
        Save miner to file
        
        Args:
            filepath: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save miner state
        with open(filepath, 'wb') as f:
            pickle.dump({
                'patterns': self.patterns,
                'pattern_support': self.pattern_support,
                'key_vocab': self.key_vocab,
                'reverse_vocab': self.reverse_vocab,
                'next_key_id': self.next_key_id,
                'min_support': self.min_support,
                'max_pattern_length': self.max_pattern_length,
                'min_pattern_length': self.min_pattern_length
            }, f)
        
        logger.info(f"Saved sequential pattern miner to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SequentialPatternMiner':
        """
        Load miner from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded SequentialPatternMiner
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            min_support=data['min_support'],
            max_pattern_length=data['max_pattern_length'],
            min_pattern_length=data['min_pattern_length']
        )
        
        # Restore state
        instance.patterns = data['patterns']
        instance.pattern_support = data['pattern_support']
        instance.key_vocab = data['key_vocab']
        instance.reverse_vocab = data['reverse_vocab']
        instance.next_key_id = data['next_key_id']
        
        logger.info(f"Loaded sequential pattern miner from {filepath}")
        return instance
