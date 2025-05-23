"""
Temporal Feature Engineering for Predis ML

This module provides temporal feature extraction for Predis' ML-driven prefetching,
focusing on time-based patterns in cache access logs.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
from datetime import datetime, timedelta
from scipy import stats
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalFeatureGenerator:
    """
    Generates temporal features from access logs
    
    This class is optimized to work with Predis' multi-strategy zero-copy memory interface
    by focusing on efficient feature computation and memory access patterns.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize temporal feature generator
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        self.use_gpu = use_gpu
        self.has_gpu = False
        
        # Try to import GPU libraries if requested
        if use_gpu:
            try:
                import cupy as cp
                import cudf
                self.has_gpu = True
                logger.info("GPU libraries available, using GPU acceleration")
            except ImportError:
                logger.warning("GPU libraries not available, falling back to CPU implementation")
    
    def extract_all_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temporal features from access logs
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added temporal features
        """
        result = df.copy()
        
        # Ensure timestamp and datetime columns exist
        if 'datetime' not in result.columns and 'timestamp' in result.columns:
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='s')
        
        # Extract basic time features
        result = self.extract_time_of_day_features(result)
        result = self.extract_day_of_week_features(result)
        result = self.extract_seasonality_features(result)
        result = self.extract_temporal_locality_features(result)
        result = self.extract_access_rate_features(result)
        
        return result
    
    def extract_time_of_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time of day features
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added time of day features
        """
        result = df.copy()
        
        # Ensure datetime column exists
        if 'datetime' not in result.columns and 'timestamp' in result.columns:
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='s')
        
        # Extract time components
        result['hour_of_day'] = result['datetime'].dt.hour
        result['minute_of_hour'] = result['datetime'].dt.minute
        result['second_of_minute'] = result['datetime'].dt.second
        
        # Normalized time of day (0-1)
        result['normalized_time_of_day'] = (
            result['hour_of_day'] * 3600 + 
            result['minute_of_hour'] * 60 + 
            result['second_of_minute']
        ) / 86400.0
        
        # Time period categories
        result['time_period'] = pd.cut(
            result['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # One-hot encode time periods for ML model compatibility
        result['is_morning'] = (result['time_period'] == 'morning').astype(int)
        result['is_afternoon'] = (result['time_period'] == 'afternoon').astype(int)
        result['is_evening'] = (result['time_period'] == 'evening').astype(int)
        result['is_night'] = (result['time_period'] == 'night').astype(int)
        
        # Business hours indicator
        result['is_business_hours'] = (
            (result['hour_of_day'] >= 9) & 
            (result['hour_of_day'] < 18) &
            (result['datetime'].dt.dayofweek < 5)  # Monday-Friday
        ).astype(int)
        
        # Circular encoding of time (for continuity between 23:59 and 00:00)
        hours_rad = result['hour_of_day'] * 2 * np.pi / 24
        result['hour_sin'] = np.sin(hours_rad)
        result['hour_cos'] = np.cos(hours_rad)
        
        # Minute circular encoding
        minutes_rad = result['minute_of_hour'] * 2 * np.pi / 60
        result['minute_sin'] = np.sin(minutes_rad)
        result['minute_cos'] = np.cos(minutes_rad)
        
        return result
    
    def extract_day_of_week_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract day of week features
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added day of week features
        """
        result = df.copy()
        
        # Ensure datetime column exists
        if 'datetime' not in result.columns and 'timestamp' in result.columns:
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='s')
        
        # Extract day of week (0=Monday, 6=Sunday)
        result['day_of_week'] = result['datetime'].dt.dayofweek
        
        # Weekend indicator
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        # Day type (categorical for readability)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        result['day_name'] = result['day_of_week'].map(lambda x: day_names[x])
        
        # Circular encoding of day of week
        days_rad = result['day_of_week'] * 2 * np.pi / 7
        result['day_sin'] = np.sin(days_rad)
        result['day_cos'] = np.cos(days_rad)
        
        # Workday vs. weekend patterns
        result['is_workday'] = (~result['is_weekend']).astype(int)
        
        return result
    
    def extract_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract seasonality features (day of month, month, quarter, etc.)
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added seasonality features
        """
        result = df.copy()
        
        # Ensure datetime column exists
        if 'datetime' not in result.columns and 'timestamp' in result.columns:
            result['datetime'] = pd.to_datetime(result['timestamp'], unit='s')
        
        # Extract date components
        result['day_of_month'] = result['datetime'].dt.day
        result['month'] = result['datetime'].dt.month
        result['quarter'] = result['datetime'].dt.quarter
        result['year'] = result['datetime'].dt.year
        result['week_of_year'] = result['datetime'].dt.isocalendar().week
        
        # Month-related features
        result['is_month_start'] = result['datetime'].dt.is_month_start.astype(int)
        result['is_month_end'] = result['datetime'].dt.is_month_end.astype(int)
        result['days_in_month'] = result['datetime'].dt.days_in_month
        
        # Normalized day of month (0-1)
        result['normalized_day_of_month'] = (result['day_of_month'] - 1) / (result['days_in_month'] - 1)
        
        # Quarter and month circular encoding
        months_rad = (result['month'] - 1) * 2 * np.pi / 12
        result['month_sin'] = np.sin(months_rad)
        result['month_cos'] = np.cos(months_rad)
        
        quarters_rad = (result['quarter'] - 1) * 2 * np.pi / 4
        result['quarter_sin'] = np.sin(quarters_rad)
        result['quarter_cos'] = np.cos(quarters_rad)
        
        return result
    
    def extract_temporal_locality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal locality features (time between accesses)
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added temporal locality features
        """
        result = df.copy()
        
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Calculate time since last access for each key
        result['time_since_last_access'] = result.groupby('key')['timestamp'].diff()
        
        # Fill NaN (first access) with a large value
        max_time = result['timestamp'].max() - result['timestamp'].min()
        result['time_since_last_access'] = result['time_since_last_access'].fillna(max_time)
        
        # Calculate inter-access time statistics for each key
        result['mean_access_interval'] = result.groupby('key')['time_since_last_access'].transform('mean')
        result['std_access_interval'] = result.groupby('key')['time_since_last_access'].transform('std')
        result['min_access_interval'] = result.groupby('key')['time_since_last_access'].transform('min')
        result['max_access_interval'] = result.groupby('key')['time_since_last_access'].transform('max')
        
        # Fill NaN values for std (single access) with 0
        result['std_access_interval'] = result['std_access_interval'].fillna(0)
        
        # Calculate z-score of current access interval
        # This helps identify unusual access patterns
        result['access_interval_zscore'] = (
            (result['time_since_last_access'] - result['mean_access_interval']) / 
            result['std_access_interval'].replace(0, 1)  # Avoid division by zero
        )
        
        # Calculate regularity score (lower std deviation = more regular)
        # Normalized to 0-1 range, where 1 = perfectly regular
        result['regularity_score'] = 1.0 / (1.0 + result['std_access_interval'] / 
                                          (result['mean_access_interval'] + 1e-6))
        
        # Calculate exponential decay of recency (recent = higher value)
        # Half-life of 1 hour (3600 seconds)
        half_life = 3600
        decay_factor = np.log(2) / half_life
        result['recency_score'] = np.exp(-decay_factor * result['time_since_last_access'])
        
        # Calculate time until next access (forward-looking)
        result['time_until_next_access'] = result.groupby('key')['timestamp'].diff(-1).abs()
        
        # Fill NaN (last access) with a large value
        result['time_until_next_access'] = result['time_until_next_access'].fillna(max_time)
        
        return result
    
    def extract_access_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract access rate features (frequency over time)
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added access rate features
        """
        result = df.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in result.columns:
            logger.warning("No timestamp column found, cannot calculate access rates")
            return result
        
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Calculate overall access count for each key
        result['access_count'] = result.groupby('key').cumcount() + 1
        
        # Calculate access rates over different time windows
        for window in [60, 300, 3600, 86400]:  # 1min, 5min, 1hr, 1day in seconds
            window_name = {60: '1min', 300: '5min', 3600: '1hr', 86400: '1day'}[window]
            
            # For each access, count how many accesses to the same key occurred 
            # in the previous time window
            result[f'access_rate_{window_name}'] = 0
            
            # Group by key and calculate rolling access counts
            for key, group in result.groupby('key'):
                # Calculate time difference from current row to all previous rows
                for i in range(len(group)):
                    current_time = group.iloc[i]['timestamp']
                    window_start = current_time - window
                    
                    # Count accesses in the time window
                    window_count = sum((group['timestamp'] >= window_start) & 
                                      (group['timestamp'] < current_time))
                    
                    # Store count in the result
                    idx = group.iloc[i].name
                    result.loc[idx, f'access_rate_{window_name}'] = window_count
        
        # Calculate acceleration (change in access rate)
        result['access_acceleration_1hr'] = (
            result['access_rate_1hr'] - result['access_rate_1day'] / 24
        )
        
        # Calculate short-term vs long-term access ratio
        # High values indicate increasing popularity
        result['short_long_ratio'] = (
            result['access_rate_1hr'] / (result['access_rate_1day'] / 24 + 0.1)
        )
        
        return result
    
    def extract_gpu_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features using GPU acceleration
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            DataFrame with added temporal features
        """
        if not self.has_gpu:
            logger.warning("GPU libraries not available, using CPU implementation")
            return self.extract_all_temporal_features(df)
        
        try:
            import cupy as cp
            import cudf
            
            # Convert to cuDF DataFrame for GPU processing
            gdf = cudf.DataFrame.from_pandas(df)
            
            # Ensure timestamp and datetime columns exist
            if 'datetime' not in gdf.columns and 'timestamp' in gdf.columns:
                gdf['datetime'] = gdf['timestamp'].astype('datetime64[s]')
            
            # Extract time components
            gdf['hour_of_day'] = gdf['datetime'].dt.hour
            gdf['day_of_week'] = gdf['datetime'].dt.dayofweek
            gdf['month'] = gdf['datetime'].dt.month
            
            # Calculate time-based features
            # Weekend indicator
            gdf['is_weekend'] = (gdf['day_of_week'] >= 5).astype('int8')
            
            # Business hours indicator
            gdf['is_business_hours'] = (
                (gdf['hour_of_day'] >= 9) & 
                (gdf['hour_of_day'] < 18) &
                (gdf['day_of_week'] < 5)
            ).astype('int8')
            
            # Circular encoding using GPU
            hours_rad = gdf['hour_of_day'] * 2 * np.pi / 24
            gdf['hour_sin'] = cp.sin(hours_rad.values)
            gdf['hour_cos'] = cp.cos(hours_rad.values)
            
            days_rad = gdf['day_of_week'] * 2 * np.pi / 7
            gdf['day_sin'] = cp.sin(days_rad.values)
            gdf['day_cos'] = cp.cos(days_rad.values)
            
            # Sort by timestamp for sequential calculations
            gdf = gdf.sort_values('timestamp')
            
            # Calculate time since last access using GPU operations
            gdf['time_since_last_access'] = gdf.groupby('key')['timestamp'].diff()
            
            # Fill NaN with a large value
            max_time = gdf['timestamp'].max() - gdf['timestamp'].min()
            gdf['time_since_last_access'] = gdf['time_since_last_access'].fillna(max_time)
            
            # Exponential decay of recency using GPU
            half_life = 3600
            decay_factor = np.log(2) / half_life
            gdf['recency_score'] = cp.exp(-decay_factor * gdf['time_since_last_access'].values)
            
            # Convert back to pandas for compatibility with rest of pipeline
            result_df = gdf.to_pandas()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in GPU feature extraction: {e}")
            logger.warning("Falling back to CPU implementation")
            return self.extract_all_temporal_features(df)


class SeasonalDecomposer:
    """
    Decomposes time series into trend, seasonality, and residual components
    
    This is useful for identifying regular patterns in cache access logs
    and can help improve prefetching accuracy.
    """
    
    def __init__(self, 
                period: Optional[int] = None, 
                min_periods: int = 2):
        """
        Initialize seasonal decomposer
        
        Args:
            period: Seasonality period (in number of observations)
                   If None, will try to detect automatically
            min_periods: Minimum number of observations to calculate stats
        """
        self.period = period
        self.min_periods = min_periods
        self.decomposition_results = {}
    
    def detect_period(self, 
                     time_series: pd.Series, 
                     max_period: int = 24) -> int:
        """
        Detect seasonality period from time series
        
        Args:
            time_series: Time series data
            max_period: Maximum period to check
            
        Returns:
            Detected period length
        """
        # Compute autocorrelation for different lags
        autocorr = []
        for lag in range(1, min(max_period + 1, len(time_series) // 2)):
            # Calculate autocorrelation for this lag
            series1 = time_series[lag:]
            series2 = time_series[:-lag]
            corr = np.corrcoef(series1, series2)[0, 1]
            autocorr.append(corr)
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i + 1, autocorr[i]))
        
        # Sort peaks by correlation strength
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            # Return period with strongest autocorrelation
            return peaks[0][0]
        else:
            # Default to 24 hours if no clear pattern
            return 24
    
    def decompose(self, 
                 time_series: pd.Series, 
                 period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonality, and residual
        
        Args:
            time_series: Time series data
            period: Seasonality period (overrides instance setting if provided)
            
        Returns:
            Dictionary with trend, seasonality, and residual components
        """
        # Use provided period or instance period or detect automatically
        if period is not None:
            self.period = period
        elif self.period is None:
            self.period = self.detect_period(time_series)
        
        # Ensure time series is sorted
        time_series = time_series.sort_index()
        
        # Extract trend using moving average
        trend = time_series.rolling(
            window=self.period, 
            center=True, 
            min_periods=self.min_periods
        ).mean()
        
        # Deal with missing values at the edges
        # Forward and backward fill
        trend = trend.fillna(method='ffill').fillna(method='bfill')
        
        # Detrend the time series
        detrended = time_series - trend
        
        # Extract seasonality by averaging over periods
        # Group values by their position in the cycle
        position_in_cycle = np.arange(len(time_series)) % self.period
        seasonality_by_position = {}
        
        for pos in range(self.period):
            mask = position_in_cycle == pos
            if sum(mask) >= self.min_periods:
                seasonality_by_position[pos] = detrended[mask].mean()
            else:
                seasonality_by_position[pos] = 0
        
        # Construct seasonality component
        seasonality = pd.Series(
            [seasonality_by_position[pos] for pos in position_in_cycle],
            index=time_series.index
        )
        
        # Calculate residual
        residual = time_series - trend - seasonality
        
        # Store results
        self.decomposition_results = {
            'original': time_series,
            'trend': trend,
            'seasonality': seasonality,
            'residual': residual,
            'period': self.period
        }
        
        return self.decomposition_results
    
    def get_seasonality_strength(self) -> float:
        """
        Calculate strength of seasonality component
        
        Returns:
            Seasonality strength (0-1, higher means stronger seasonality)
        """
        if not self.decomposition_results:
            raise ValueError("Must run decompose() first")
        
        # Calculate variances
        var_seasonality = np.var(self.decomposition_results['seasonality'])
        var_residual = np.var(self.decomposition_results['residual'])
        var_detrended = var_seasonality + var_residual
        
        # Calculate strength as proportion of variance explained by seasonality
        if var_detrended > 0:
            strength = var_seasonality / var_detrended
            return max(0, min(1, strength))  # Clamp to 0-1 range
        else:
            return 0.0
    
    def forecast_next_values(self, steps: int = 10) -> pd.Series:
        """
        Forecast future values based on trend and seasonality
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Series with forecasted values
        """
        if not self.decomposition_results:
            raise ValueError("Must run decompose() first")
        
        # Get components
        original = self.decomposition_results['original']
        trend = self.decomposition_results['trend']
        seasonality = self.decomposition_results['seasonality']
        
        # Calculate average trend change
        trend_diff = trend.diff().mean()
        
        # Create new index for forecast
        last_idx = original.index[-1]
        if isinstance(last_idx, pd.Timestamp):
            # For time-based index, extend by appropriate time unit
            try:
                freq = pd.infer_freq(original.index)
                if freq:
                    new_idx = pd.date_range(start=last_idx + pd.Timedelta(1, unit=freq), 
                                           periods=steps, freq=freq)
                else:
                    # If frequency can't be inferred, use mean time difference
                    avg_diff = np.mean(np.diff(original.index.astype(np.int64))) / 10**9
                    new_idx = [last_idx + pd.Timedelta(seconds=avg_diff * (i+1)) 
                              for i in range(steps)]
            except:
                # Fallback: assume seconds
                new_idx = [last_idx + pd.Timedelta(seconds=i+1) for i in range(steps)]
        else:
            # For integer index, just increment
            new_idx = range(last_idx + 1, last_idx + steps + 1)
        
        # Forecast trend (extrapolate)
        forecast_trend = [trend.iloc[-1] + trend_diff * (i+1) for i in range(steps)]
        
        # Forecast seasonality (repeat pattern)
        last_position = len(original) % self.period
        forecast_seasonality = [
            seasonality.iloc[(last_position + i) % self.period] 
            for i in range(steps)
        ]
        
        # Combine trend and seasonality
        forecast = pd.Series(
            [t + s for t, s in zip(forecast_trend, forecast_seasonality)],
            index=new_idx
        )
        
        return forecast


class TemporalPatternDetector:
    """
    Detects temporal patterns in cache access logs
    
    This helps identify keys with predictable access patterns
    that can benefit from prefetching.
    """
    
    def __init__(self, min_pattern_length: int = 3, min_occurrences: int = 2):
        """
        Initialize pattern detector
        
        Args:
            min_pattern_length: Minimum length of patterns to detect
            min_occurrences: Minimum number of occurrences to consider a pattern
        """
        self.min_pattern_length = min_pattern_length
        self.min_occurrences = min_occurrences
        self.patterns = {}
    
    def detect_key_patterns(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect temporal patterns for each key
        
        Args:
            df: DataFrame with access logs
            
        Returns:
            Dictionary of detected patterns by key
        """
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a timestamp column")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Initialize patterns dictionary
        self.patterns = {}
        
        # Process each key separately
        for key, group in df.groupby('key'):
            if len(group) >= self.min_pattern_length * self.min_occurrences:
                # Detect patterns for this key
                key_patterns = self._detect_patterns_for_key(group)
                if key_patterns:
                    self.patterns[key] = key_patterns
        
        logger.info(f"Detected patterns for {len(self.patterns)} keys")
        return self.patterns
    
    def _detect_patterns_for_key(self, key_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect temporal patterns for a specific key
        
        Args:
            key_df: DataFrame with access logs for a single key
            
        Returns:
            List of detected patterns
        """
        # Calculate inter-access intervals
        key_df = key_df.sort_values('timestamp')
        intervals = np.diff(key_df['timestamp'].values)
        
        # List to store detected patterns
        patterns = []
        
        # Look for regular interval patterns
        # Group similar intervals together
        interval_groups = self._group_similar_intervals(intervals)
        
        for interval, count in interval_groups.items():
            if count >= self.min_occurrences:
                # We found a regular interval pattern
                patterns.append({
                    'type': 'regular_interval',
                    'interval': interval,
                    'count': count,
                    'confidence': count / len(intervals)
                })
        
        # Look for time-of-day patterns
        if 'hour_of_day' in key_df.columns:
            hour_counts = key_df['hour_of_day'].value_counts()
            total_days = (key_df['timestamp'].max() - key_df['timestamp'].min()) / 86400
            
            for hour, count in hour_counts.items():
                # If key is accessed at the same hour repeatedly
                if count >= self.min_occurrences and count >= total_days * 0.5:
                    patterns.append({
                        'type': 'time_of_day',
                        'hour': int(hour),
                        'count': int(count),
                        'confidence': count / len(key_df)
                    })
        
        # Look for day-of-week patterns
        if 'day_of_week' in key_df.columns:
            day_counts = key_df['day_of_week'].value_counts()
            total_weeks = (key_df['timestamp'].max() - key_df['timestamp'].min()) / (86400 * 7)
            
            for day, count in day_counts.items():
                # If key is accessed on the same day repeatedly
                if count >= self.min_occurrences and count >= total_weeks * 0.5:
                    patterns.append({
                        'type': 'day_of_week',
                        'day': int(day),
                        'count': int(count),
                        'confidence': count / len(key_df)
                    })
        
        # Sort patterns by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        return patterns
    
    def _group_similar_intervals(self, 
                                intervals: np.ndarray, 
                                tolerance: float = 0.1) -> Dict[float, int]:
        """
        Group similar intervals together
        
        Args:
            intervals: Array of time intervals
            tolerance: Relative tolerance for grouping
            
        Returns:
            Dictionary of {interval_value: count}
        """
        # Sort intervals
        sorted_intervals = np.sort(intervals)
        
        # Initialize groups
        groups = {}
        
        for interval in sorted_intervals:
            # Check if interval fits in any existing group
            found_group = False
            
            for group_interval in list(groups.keys()):
                # Check if interval is within tolerance of group
                if abs(interval - group_interval) / group_interval <= tolerance:
                    # Update group count
                    groups[group_interval] += 1
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                groups[interval] = 1
        
        # Merge small groups into larger ones
        merged_groups = {}
        for interval, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
            if count >= self.min_occurrences:
                merged_groups[interval] = count
            else:
                # Try to merge with closest larger group
                closest_interval = None
                min_diff = float('inf')
                
                for group_interval in merged_groups:
                    diff = abs(interval - group_interval) / group_interval
                    if diff < min_diff and diff <= tolerance:
                        min_diff = diff
                        closest_interval = group_interval
                
                if closest_interval is not None:
                    # Merge with closest group
                    merged_groups[closest_interval] += count
        
        return merged_groups
    
    def get_prefetchable_keys(self, confidence_threshold: float = 0.7) -> List[str]:
        """
        Get keys with high-confidence temporal patterns that can be prefetched
        
        Args:
            confidence_threshold: Minimum confidence for prefetchable patterns
            
        Returns:
            List of keys suitable for prefetching
        """
        prefetchable = []
        
        for key, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern['confidence'] >= confidence_threshold:
                    prefetchable.append(key)
                    break  # One good pattern is enough
        
        return prefetchable
    
    def get_next_access_time(self, 
                            key: str, 
                            current_time: float) -> Optional[float]:
        """
        Predict next access time for a key based on detected patterns
        
        Args:
            key: Cache key
            current_time: Current timestamp
            
        Returns:
            Predicted next access timestamp or None if unpredictable
        """
        if key not in self.patterns:
            return None
        
        # Get patterns for this key
        key_patterns = self.patterns[key]
        
        # Find the highest confidence pattern
        best_pattern = max(key_patterns, key=lambda p: p['confidence'])
        
        # Predict based on pattern type
        if best_pattern['type'] == 'regular_interval':
            # Regular interval pattern
            return current_time + best_pattern['interval']
        
        elif best_pattern['type'] == 'time_of_day':
            # Time of day pattern
            current_dt = datetime.fromtimestamp(current_time)
            target_hour = best_pattern['hour']
            
            # Calculate next occurrence of this hour
            if current_dt.hour < target_hour:
                # Later today
                next_dt = current_dt.replace(hour=target_hour, minute=0, second=0)
            else:
                # Tomorrow
                next_dt = (current_dt + timedelta(days=1)).replace(
                    hour=target_hour, minute=0, second=0)
            
            return next_dt.timestamp()
        
        elif best_pattern['type'] == 'day_of_week':
            # Day of week pattern
            current_dt = datetime.fromtimestamp(current_time)
            target_day = best_pattern['day']  # 0=Monday, 6=Sunday
            
            # Calculate days until next occurrence
            current_day = current_dt.weekday()
            days_ahead = (target_day - current_day) % 7
            
            if days_ahead == 0 and current_dt.hour >= 12:
                # If it's already this day and afternoon, go to next week
                days_ahead = 7
            
            next_dt = (current_dt + timedelta(days=days_ahead)).replace(
                hour=9, minute=0, second=0)  # Assume morning access
            
            return next_dt.timestamp()
        
        return None
