"""
Synthetic Data Validation Utilities

This module provides tools for validating synthetic data patterns to ensure
they meet requirements for ML training and accurately simulate real-world scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticDataValidator:
    """Validates synthetic data against expected real-world characteristics"""
    
    def __init__(self, data_path: str, output_dir: str = "../../../data/synthetic/validation"):
        """
        Initialize validator
        
        Args:
            data_path: Path to data file (.csv or .parquet)
            output_dir: Directory to save validation results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.validation_results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        logger.info(f"Loading data from {self.data_path}")
        
        if self.data_path.endswith(".csv"):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(".parquet"):
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
            
        logger.info(f"Loaded {len(self.df)} records")
        return self.df
        
    def validate_zipfian_distribution(self) -> Dict[str, Any]:
        """Validate Zipfian distribution (power law) in key access patterns"""
        if self.df is None:
            self.load_data()
            
        logger.info("Validating Zipfian distribution")
        results = {}
        
        # Calculate key frequency distribution
        key_counts = self.df['key'].value_counts()
        
        # Calculate empirical power law exponent
        # Log-log regression to estimate alpha
        x = np.log(np.arange(1, len(key_counts) + 1))
        y = np.log(key_counts.values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        results['power_law_exponent'] = -slope  # Should be close to 1.0 for Zipfian
        results['power_law_r_squared'] = r_value**2
        results['power_law_p_value'] = p_value
        
        # 80/20 rule check
        top_20_percent = int(len(key_counts) * 0.2)
        top_20_traffic = key_counts.iloc[:top_20_percent].sum()
        total_traffic = len(self.df)
        results['top_20_percent_traffic'] = top_20_traffic / total_traffic
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        plt.loglog(np.arange(1, len(key_counts) + 1), key_counts.values, 'o', markersize=3, alpha=0.5)
        plt.loglog(np.arange(1, len(key_counts) + 1), np.exp(intercept + slope * x), 'r-', 
                  label=f'Power Law (α={-slope:.2f})')
        plt.xlabel('Rank (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Key Popularity Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/zipfian_validation.png")
        plt.close()
        
        # Validation criteria
        results['is_valid_zipfian'] = (
            results['power_law_r_squared'] > 0.8 and 
            results['power_law_p_value'] < 0.05 and
            0.7 < results['top_20_percent_traffic'] < 0.9
        )
        
        self.validation_results['zipfian'] = results
        return results
    
    def validate_temporal_patterns(self) -> Dict[str, Any]:
        """Validate temporal patterns (daily/weekly cycles)"""
        if self.df is None:
            self.load_data()
            
        logger.info("Validating temporal patterns")
        results = {}
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' in self.df.columns:
            if 'datetime' not in self.df.columns:
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
                
            # Extract time components
            self.df['hour_of_day'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
            
            # Hourly distribution
            hourly_traffic = self.df.groupby('hour_of_day').size()
            results['peak_hour'] = hourly_traffic.idxmax()
            results['peak_hour_traffic'] = hourly_traffic.max()
            results['off_peak_hour'] = hourly_traffic.idxmin()
            results['off_peak_traffic'] = hourly_traffic.min()
            results['peak_to_offpeak_ratio'] = results['peak_hour_traffic'] / results['off_peak_traffic']
            
            # Daily distribution
            if 'day_of_week' in self.df.columns:
                daily_traffic = self.df.groupby('day_of_week').size()
                results['weekday_avg'] = daily_traffic.iloc[:5].mean()  # Mon-Fri
                results['weekend_avg'] = daily_traffic.iloc[5:].mean()  # Sat-Sun
                results['weekday_to_weekend_ratio'] = results['weekday_avg'] / results['weekend_avg']
            
            # Generate visualization
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            sns.barplot(x=hourly_traffic.index, y=hourly_traffic.values)
            plt.title('Hourly Traffic Distribution')
            plt.xlabel('Hour of Day')
            plt.ylabel('Access Count')
            
            if 'day_of_week' in self.df.columns:
                plt.subplot(2, 1, 2)
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                sns.barplot(x=[day_names[i] for i in daily_traffic.index], y=daily_traffic.values)
                plt.title('Daily Traffic Distribution')
                plt.xlabel('Day of Week')
                plt.ylabel('Access Count')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/temporal_validation.png")
            plt.close()
            
            # Validation criteria for business hour patterns
            # Expecting higher traffic during business hours (9-17)
            business_hours_traffic = hourly_traffic.loc[9:17].mean()
            non_business_hours_traffic = hourly_traffic.drop(range(9, 18)).mean()
            results['business_hours_ratio'] = business_hours_traffic / non_business_hours_traffic
            
            results['is_valid_temporal'] = (
                results['peak_to_offpeak_ratio'] > 2.0 and
                results['business_hours_ratio'] > 1.5
            )
        else:
            results['is_valid_temporal'] = False
            results['error'] = "No timestamp column found"
            
        self.validation_results['temporal'] = results
        return results
    
    def validate_ml_training_patterns(self) -> Dict[str, Any]:
        """Validate ML training patterns (batch behavior)"""
        if self.df is None:
            self.load_data()
            
        logger.info("Validating ML training patterns")
        results = {}
        
        # Check if this dataset has ML training patterns
        ml_patterns = self.df[self.df['workload_type'] == 'ml_training'] if 'workload_type' in self.df.columns else None
        
        if ml_patterns is not None and len(ml_patterns) > 0:
            # Check for batch patterns
            if 'batch' in ml_patterns.columns and 'epoch' in ml_patterns.columns:
                # Calculate batch sizes
                batch_sizes = ml_patterns.groupby(['epoch', 'batch']).size()
                results['avg_batch_size'] = batch_sizes.mean()
                results['batch_size_std'] = batch_sizes.std()
                results['batch_size_cv'] = results['batch_size_std'] / results['avg_batch_size']
                
                # Check for sequential access within batches
                if 'position_in_batch' in ml_patterns.columns:
                    # For sequential access, we expect position_in_batch to closely follow access order
                    ml_patterns = ml_patterns.sort_values(['epoch', 'batch', 'timestamp'])
                    ml_patterns['expected_position'] = ml_patterns.groupby(['epoch', 'batch']).cumcount()
                    position_correlation = ml_patterns['position_in_batch'].corr(ml_patterns['expected_position'])
                    results['position_correlation'] = position_correlation
                
                # Visualize epoch and batch patterns
                plt.figure(figsize=(12, 6))
                
                # Plot batch sizes
                plt.subplot(1, 2, 1)
                sns.histplot(batch_sizes, kde=True)
                plt.title('Batch Size Distribution')
                plt.xlabel('Batch Size')
                
                # Plot sequential access pattern for a sample batch
                plt.subplot(1, 2, 2)
                sample_batch = ml_patterns[(ml_patterns['epoch'] == 0) & (ml_patterns['batch'] == 0)]
                if len(sample_batch) > 0:
                    plt.scatter(sample_batch['position_in_batch'], sample_batch['timestamp'], alpha=0.5)
                    plt.title('Sequential Access Pattern (Epoch 0, Batch 0)')
                    plt.xlabel('Position in Batch')
                    plt.ylabel('Timestamp')
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/ml_training_validation.png")
                plt.close()
                
                # Validation criteria
                results['is_valid_ml_training'] = (
                    results['batch_size_cv'] < 0.1 and  # Consistent batch sizes
                    ('position_correlation' not in results or results['position_correlation'] > 0.9)  # Sequential access
                )
            else:
                results['is_valid_ml_training'] = False
                results['error'] = "Missing batch or epoch information"
        else:
            results['is_valid_ml_training'] = False
            results['error'] = "No ML training patterns found"
            
        self.validation_results['ml_training'] = results
        return results
    
    def validate_hft_patterns(self) -> Dict[str, Any]:
        """Validate HFT (High-Frequency Trading) patterns"""
        if self.df is None:
            self.load_data()
            
        logger.info("Validating HFT patterns")
        results = {}
        
        # Check if this dataset has HFT patterns
        hft_patterns = self.df[self.df['workload_type'] == 'hft'] if 'workload_type' in self.df.columns else None
        
        if hft_patterns is not None and len(hft_patterns) > 0:
            # Check for hot symbols concentration
            if 'is_hot_symbol' in hft_patterns.columns:
                hot_symbol_ratio = hft_patterns['is_hot_symbol'].mean()
                results['hot_symbol_ratio'] = hot_symbol_ratio
            else:
                # Try to infer hot symbols from key distribution
                symbol_counts = hft_patterns['key'].value_counts()
                top_10_percent = int(len(symbol_counts) * 0.1)
                top_10_traffic = symbol_counts.iloc[:top_10_percent].sum()
                total_traffic = len(hft_patterns)
                results['hot_symbol_ratio'] = top_10_traffic / total_traffic
            
            # Check for market volatility (if volatility_factor exists)
            if 'volatility_factor' in hft_patterns.columns:
                volatility_factors = hft_patterns['volatility_factor'].value_counts(normalize=True)
                results['volatility_events'] = 1.0 - volatility_factors.get(1.0, 0.0)
            
            # Analyze access timing
            hft_patterns = hft_patterns.sort_values('timestamp')
            hft_patterns['time_diff'] = hft_patterns['timestamp'].diff()
            
            # Calculate access rates (ops/sec)
            mean_time_diff = hft_patterns['time_diff'].mean()
            results['average_ops_per_second'] = 1.0 / mean_time_diff if mean_time_diff > 0 else 0
            
            # Visualize HFT patterns
            plt.figure(figsize=(12, 6))
            
            # Plot symbol distribution
            plt.subplot(1, 2, 1)
            top_symbols = hft_patterns['key'].value_counts().head(20)
            sns.barplot(x=top_symbols.index, y=top_symbols.values)
            plt.title('Top 20 Symbols by Access Frequency')
            plt.xlabel('Symbol')
            plt.ylabel('Access Count')
            plt.xticks(rotation=90)
            
            # Plot access timing
            plt.subplot(1, 2, 2)
            sns.histplot(hft_patterns['time_diff'].dropna(), bins=50, kde=True)
            plt.title('Inter-Access Time Distribution')
            plt.xlabel('Time Between Accesses (seconds)')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/hft_validation.png")
            plt.close()
            
            # Validation criteria
            results['is_valid_hft'] = (
                0.85 <= results.get('hot_symbol_ratio', 0) <= 0.95 and  # Hot symbol concentration
                results.get('average_ops_per_second', 0) > 1000  # High access rate
            )
        else:
            results['is_valid_hft'] = False
            results['error'] = "No HFT patterns found"
            
        self.validation_results['hft'] = results
        return results
    
    def validate_gaming_patterns(self) -> Dict[str, Any]:
        """Validate gaming workload patterns"""
        if self.df is None:
            self.load_data()
            
        logger.info("Validating gaming patterns")
        results = {}
        
        # Check if this dataset has gaming patterns
        gaming_patterns = self.df[self.df['workload_type'] == 'gaming'] if 'workload_type' in self.df.columns else None
        
        if gaming_patterns is not None and len(gaming_patterns) > 0:
            # Check for player focus areas
            if 'in_focus_area' in gaming_patterns.columns:
                focus_area_ratio = gaming_patterns['in_focus_area'].mean()
                results['focus_area_ratio'] = focus_area_ratio
            
            # Check for object type distribution
            if 'object_type' in gaming_patterns.columns:
                object_types = gaming_patterns['object_type'].value_counts(normalize=True)
                results['object_type_distribution'] = dict(object_types)
            
            # Check player activity
            if 'player_id' in gaming_patterns.columns:
                player_activity = gaming_patterns.groupby('player_id').size()
                results['active_players'] = len(player_activity)
                results['avg_actions_per_player'] = player_activity.mean()
                results['max_actions_per_player'] = player_activity.max()
                results['min_actions_per_player'] = player_activity.min()
            
            # Visualize gaming patterns
            plt.figure(figsize=(12, 6))
            
            # Plot object type distribution
            plt.subplot(1, 2, 1)
            if 'object_type' in gaming_patterns.columns:
                sns.barplot(x=object_types.index, y=object_types.values)
                plt.title('Object Type Distribution')
                plt.xlabel('Object Type')
                plt.ylabel('Proportion')
                plt.xticks(rotation=45)
            
            # Plot player activity distribution
            plt.subplot(1, 2, 2)
            if 'player_id' in gaming_patterns.columns:
                sns.histplot(player_activity, bins=30, kde=True)
                plt.title('Player Activity Distribution')
                plt.xlabel('Actions per Player')
                plt.ylabel('Number of Players')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/gaming_validation.png")
            plt.close()
            
            # Validation criteria
            results['is_valid_gaming'] = (
                results.get('focus_area_ratio', 0) > 0.7 and  # Players mostly access their focus area
                results.get('active_players', 0) > 100  # Sufficient number of active players
            )
        else:
            results['is_valid_gaming'] = False
            results['error'] = "No gaming patterns found"
            
        self.validation_results['gaming'] = results
        return results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks and return results"""
        logger.info("Running all validations")
        
        self.validate_zipfian_distribution()
        self.validate_temporal_patterns()
        self.validate_ml_training_patterns()
        self.validate_hft_patterns()
        self.validate_gaming_patterns()
        
        # Calculate overall validation status
        valid_patterns = sum(1 for k, v in self.validation_results.items() 
                           if v.get('is_valid_' + k, False))
        total_patterns = len(self.validation_results)
        
        self.validation_results['overall'] = {
            'valid_patterns': valid_patterns,
            'total_patterns': total_patterns,
            'validation_score': valid_patterns / total_patterns if total_patterns > 0 else 0,
            'is_valid': valid_patterns == total_patterns and total_patterns > 0
        }
        
        # Save validation results
        self._save_validation_report()
        
        return self.validation_results
    
    def _save_validation_report(self) -> None:
        """Save validation results to file"""
        import json
        
        # Convert validation results to serializable format
        serializable_results = {}
        for category, results in self.validation_results.items():
            serializable_results[category] = {
                k: v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)
                for k, v in results.items()
            }
        
        # Save as JSON
        with open(f"{self.output_dir}/validation_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save as markdown report
        with open(f"{self.output_dir}/validation_report.md", 'w') as f:
            f.write("# Synthetic Data Validation Report\n\n")
            
            # Overall results
            overall = self.validation_results.get('overall', {})
            f.write("## Overall Results\n\n")
            f.write(f"- **Validation Score**: {overall.get('validation_score', 0):.2f}\n")
            f.write(f"- **Valid Patterns**: {overall.get('valid_patterns', 0)} / {overall.get('total_patterns', 0)}\n")
            f.write(f"- **Overall Status**: {'VALID' if overall.get('is_valid', False) else 'INVALID'}\n\n")
            
            # Individual pattern results
            for category, results in self.validation_results.items():
                if category == 'overall':
                    continue
                    
                f.write(f"## {category.capitalize()} Pattern Validation\n\n")
                f.write(f"- **Status**: {'VALID' if results.get('is_valid_' + category, False) else 'INVALID'}\n")
                
                # Add specific metrics for each pattern type
                if category == 'zipfian':
                    f.write(f"- **Power Law Exponent**: {results.get('power_law_exponent', 0):.2f}\n")
                    f.write(f"- **R-squared**: {results.get('power_law_r_squared', 0):.2f}\n")
                    f.write(f"- **Top 20% Traffic**: {results.get('top_20_percent_traffic', 0):.2f}\n")
                
                elif category == 'temporal':
                    f.write(f"- **Peak Hour**: {results.get('peak_hour', 'N/A')}\n")
                    f.write(f"- **Peak/Off-peak Ratio**: {results.get('peak_to_offpeak_ratio', 0):.2f}\n")
                    if 'weekday_to_weekend_ratio' in results:
                        f.write(f"- **Weekday/Weekend Ratio**: {results.get('weekday_to_weekend_ratio', 0):.2f}\n")
                
                elif category == 'ml_training':
                    if 'error' in results:
                        f.write(f"- **Error**: {results.get('error', 'Unknown error')}\n")
                    else:
                        f.write(f"- **Avg Batch Size**: {results.get('avg_batch_size', 0):.2f}\n")
                        f.write(f"- **Batch Size Variation**: {results.get('batch_size_cv', 0):.2f}\n")
                        if 'position_correlation' in results:
                            f.write(f"- **Sequential Access Correlation**: {results.get('position_correlation', 0):.2f}\n")
                
                elif category == 'hft':
                    if 'error' in results:
                        f.write(f"- **Error**: {results.get('error', 'Unknown error')}\n")
                    else:
                        f.write(f"- **Hot Symbol Ratio**: {results.get('hot_symbol_ratio', 0):.2f}\n")
                        f.write(f"- **Operations/Second**: {results.get('average_ops_per_second', 0):.2f}\n")
                        if 'volatility_events' in results:
                            f.write(f"- **Volatility Events**: {results.get('volatility_events', 0):.2f}\n")
                
                elif category == 'gaming':
                    if 'error' in results:
                        f.write(f"- **Error**: {results.get('error', 'Unknown error')}\n")
                    else:
                        f.write(f"- **Focus Area Ratio**: {results.get('focus_area_ratio', 0):.2f}\n")
                        f.write(f"- **Active Players**: {results.get('active_players', 0)}\n")
                        f.write(f"- **Avg Actions/Player**: {results.get('avg_actions_per_player', 0):.2f}\n")
                
                f.write("\n")
            
            # Add timestamp
            from datetime import datetime
            f.write(f"\n\n*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Validation report saved to {self.output_dir}/validation_report.md")


def quick_validate_dataset(data_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Quick validation of a dataset without creating visualizations
    
    Args:
        data_path: Path to dataset (.csv or .parquet)
        output_dir: Optional output directory for validation results
        
    Returns:
        Dictionary with validation results
    """
    if output_dir is None:
        output_dir = str(Path(data_path).parent / "validation")
    
    validator = SyntheticDataValidator(data_path, output_dir)
    return validator.run_all_validations()
