"""
Workload-Specific Pattern Generators

This module provides higher-level workload generators that combine multiple
access patterns to create realistic cache access scenarios for specific domains.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from .generators import (
    generate_zipfian_access_pattern,
    generate_temporal_access_pattern,
    generate_ml_training_pattern,
    generate_hft_pattern,
    generate_gaming_pattern,
    export_synthetic_data,
    validate_synthetic_patterns
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkloadGenerator:
    """Base class for workload generation"""
    
    def __init__(self, name: str, output_dir: str = "../../../data/synthetic"):
        self.name = name
        self.output_dir = output_dir
        self.access_patterns = []
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate the workload patterns - to be implemented by subclasses"""
        raise NotImplementedError
        
    def export(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Export the generated workload to files"""
        if not filename:
            filename = f"{self.output_dir}/{self.name.lower()}"
        
        if not self.access_patterns:
            logger.warning("No access patterns generated yet. Running generate() first.")
            self.generate()
            
        return export_synthetic_data(self.access_patterns, filename)
        
    def validate(self) -> Dict[str, Any]:
        """Validate the generated workload"""
        if not self.access_patterns:
            logger.warning("No access patterns generated yet. Running generate() first.")
            self.generate()
            
        df = pd.DataFrame(self.access_patterns)
        return validate_synthetic_patterns(df)


class WebServiceWorkload(WorkloadGenerator):
    """Web service workload with diurnal patterns and zipfian key distribution"""
    
    def __init__(self, 
                output_dir: str = "../../../data/synthetic",
                num_keys: int = 100000,
                duration_days: int = 7,
                requests_per_day: int = 1000000):
        super().__init__("WebService", output_dir)
        self.num_keys = num_keys
        self.duration_days = duration_days
        self.requests_per_day = requests_per_day
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate web service workload"""
        logger.info(f"Generating Web Service workload for {self.duration_days} days")
        
        # Zipfian base pattern (content popularity follows power law)
        zipfian_patterns = generate_zipfian_access_pattern(
            num_keys=self.num_keys,
            num_accesses=self.requests_per_day,
            alpha=1.1  # Slightly more skewed for web content
        )
        
        # Temporal patterns (daily and weekly cycles)
        temporal_patterns = generate_temporal_access_pattern(
            duration_hours=self.duration_days * 24,
            keys_per_hour=self.requests_per_day // 24,
            num_keys=self.num_keys
        )
        
        # Combine patterns
        all_patterns = zipfian_patterns + temporal_patterns
        
        # Add web-specific metadata
        for access in all_patterns:
            # Add content type
            if "key_" in access["key"]:
                key_id = int(access["key"].split("_")[1])
                # Classify content type
                if key_id % 10 == 0:
                    content_type = "image"
                elif key_id % 10 == 1:
                    content_type = "video"
                elif key_id % 10 in [2, 3]:
                    content_type = "user_profile"
                else:
                    content_type = "article"
                    
                access["metadata"] = access.get("metadata", {})
                access["metadata"]["content_type"] = content_type
        
        # Sort by timestamp
        all_patterns.sort(key=lambda x: x["timestamp"])
        
        # Store the patterns
        self.access_patterns = all_patterns
        logger.info(f"Generated {len(self.access_patterns)} web service access patterns")
        
        return self.access_patterns


class DatabaseWorkload(WorkloadGenerator):
    """Database workload with OLTP and OLAP patterns"""
    
    def __init__(self, 
                output_dir: str = "../../../data/synthetic",
                num_tables: int = 100,
                num_keys_per_table: int = 10000,
                duration_hours: int = 24,
                transactions_per_hour: int = 100000):
        super().__init__("Database", output_dir)
        self.num_tables = num_tables
        self.num_keys_per_table = num_keys_per_table
        self.duration_hours = duration_hours
        self.transactions_per_hour = transactions_per_hour
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate database workload with OLTP and OLAP patterns"""
        logger.info(f"Generating Database workload for {self.duration_hours} hours")
        
        all_patterns = []
        
        # OLTP: Many small transactional queries
        for hour in range(self.duration_hours):
            # Business hours have more OLTP traffic
            hour_of_day = hour % 24
            
            if 8 <= hour_of_day <= 18:  # Business hours
                oltp_factor = 1.5
            else:
                oltp_factor = 0.6
                
            num_transactions = int(self.transactions_per_hour * oltp_factor)
            
            # Generate OLTP patterns (focused on hot tables)
            hot_tables = np.random.choice(self.num_tables, size=int(self.num_tables * 0.2), replace=False)
            
            for tx in range(num_transactions):
                # Each transaction accesses 2-10 keys
                tx_size = np.random.randint(2, 11)
                base_timestamp = time.time() + hour * 3600 + (tx / num_transactions) * 3600
                
                # 80% of transactions hit hot tables
                if np.random.random() < 0.8:
                    table_id = np.random.choice(hot_tables)
                else:
                    table_id = np.random.randint(0, self.num_tables)
                
                # Generate keys for this transaction
                for i in range(tx_size):
                    # Within a transaction, keys are often related
                    if i == 0:
                        # First key is random within the table
                        key_id = np.random.randint(0, self.num_keys_per_table)
                    else:
                        # Subsequent keys are often nearby the first
                        key_id = min(max(0, key_id + np.random.randint(-100, 101)), 
                                    self.num_keys_per_table - 1)
                    
                    # Determine operation (75% reads, 25% writes)
                    operation = 'GET' if np.random.random() < 0.75 else 'PUT'
                    
                    # Create the access pattern
                    all_patterns.append({
                        'timestamp': base_timestamp + i * 0.001,  # 1ms between operations
                        'key': f"table_{table_id}_key_{key_id}",
                        'operation': operation,
                        'workload_type': 'db_oltp',
                        'metadata': {
                            'transaction_id': f"tx_{hour}_{tx}",
                            'table_id': int(table_id),
                            'key_id': int(key_id),
                            'is_hot_table': table_id in hot_tables,
                            'hour_of_day': hour_of_day
                        }
                    })
        
        # OLAP: Few large analytical queries (every hour)
        for hour in range(self.duration_hours):
            # Generate 1-5 OLAP queries per hour
            num_olap_queries = np.random.randint(1, 6)
            
            for query in range(num_olap_queries):
                # Each OLAP query scans large portions of tables
                olap_timestamp = time.time() + hour * 3600 + query * (3600 / num_olap_queries)
                
                # Select 1-3 tables to join
                num_tables = np.random.randint(1, 4)
                query_tables = np.random.choice(self.num_tables, size=num_tables, replace=False)
                
                # Scan significant portions of each table
                for table_id in query_tables:
                    # Scan 30-100% of the table
                    scan_percent = np.random.uniform(0.3, 1.0)
                    num_scanned = int(self.num_keys_per_table * scan_percent)
                    
                    # Generate sequential scan
                    start_key = np.random.randint(0, self.num_keys_per_table - num_scanned)
                    
                    # Add scan accesses - these will be sequential
                    scan_steps = min(1000, num_scanned)  # Cap to avoid too many records
                    step_size = max(1, num_scanned // scan_steps)
                    
                    for i in range(0, num_scanned, step_size):
                        key_id = start_key + i
                        scan_timestamp = olap_timestamp + (i / num_scanned) * 60  # 1 minute per scan
                        
                        # OLAP is almost all reads
                        all_patterns.append({
                            'timestamp': scan_timestamp,
                            'key': f"table_{table_id}_key_{key_id}",
                            'operation': 'GET',
                            'workload_type': 'db_olap',
                            'metadata': {
                                'query_id': f"olap_{hour}_{query}",
                                'table_id': int(table_id),
                                'key_id': int(key_id),
                                'scan_percent': scan_percent,
                                'hour_of_day': hour % 24
                            }
                        })
        
        # Sort by timestamp
        all_patterns.sort(key=lambda x: x["timestamp"])
        
        # Store the patterns
        self.access_patterns = all_patterns
        logger.info(f"Generated {len(self.access_patterns)} database access patterns")
        
        return self.access_patterns


class MLWorkload(WorkloadGenerator):
    """Machine Learning training and inference workload"""
    
    def __init__(self, 
                output_dir: str = "../../../data/synthetic",
                num_models: int = 10,
                dataset_size: int = 50000,
                batch_size: int = 256,
                num_epochs: int = 20,
                inference_requests_per_hour: int = 10000,
                duration_hours: int = 24):
        super().__init__("MLWorkload", output_dir)
        self.num_models = num_models
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.inference_requests = inference_requests_per_hour
        self.duration_hours = duration_hours
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate ML workload with training and inference patterns"""
        logger.info(f"Generating ML workload for {self.num_models} models")
        
        all_patterns = []
        
        # Training patterns for each model
        for model_id in range(self.num_models):
            # Generate training pattern for this model
            training_patterns = generate_ml_training_pattern(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                dataset_size=self.dataset_size
            )
            
            # Add model-specific metadata
            for access in training_patterns:
                access['metadata'] = access.get('metadata', {})
                access['metadata']['model_id'] = model_id
                access['metadata']['phase'] = 'training'
                
                # Rename key to include model
                access['key'] = f"model_{model_id}_{access['key']}"
            
            all_patterns.extend(training_patterns)
        
        # Inference patterns (more random access)
        base_time = time.time()
        
        for hour in range(self.duration_hours):
            # Diurnal pattern for inference
            hour_of_day = hour % 24
            if 8 <= hour_of_day <= 20:  # Daytime
                infer_factor = 1.5
            else:
                infer_factor = 0.5
                
            num_requests = int(self.inference_requests * infer_factor)
            
            for i in range(num_requests):
                timestamp = base_time + hour * 3600 + i * (3600 / num_requests)
                
                # Select random model (but some models are more popular)
                model_popularity = np.random.zipf(1.5, self.num_models)
                model_popularity = model_popularity / model_popularity.sum()
                model_id = np.random.choice(self.num_models, p=model_popularity)
                
                # For inference, we access a smaller set of keys
                # (weights, not training data)
                layer_id = np.random.randint(0, 10)  # 10 layers per model
                weight_id = np.random.randint(0, 1000)  # 1000 weights per layer
                
                all_patterns.append({
                    'timestamp': timestamp,
                    'key': f"model_{model_id}_weights_layer_{layer_id}_w_{weight_id}",
                    'operation': 'GET',
                    'workload_type': 'ml_inference',
                    'metadata': {
                        'model_id': int(model_id),
                        'phase': 'inference',
                        'layer_id': int(layer_id),
                        'hour_of_day': hour_of_day
                    }
                })
                
                # Sometimes we also access biases
                if np.random.random() < 0.3:
                    all_patterns.append({
                        'timestamp': timestamp + 0.0001,
                        'key': f"model_{model_id}_weights_layer_{layer_id}_b_{weight_id % 100}",
                        'operation': 'GET',
                        'workload_type': 'ml_inference',
                        'metadata': {
                            'model_id': int(model_id),
                            'phase': 'inference',
                            'layer_id': int(layer_id),
                            'hour_of_day': hour_of_day
                        }
                    })
        
        # Sort by timestamp
        all_patterns.sort(key=lambda x: x["timestamp"])
        
        # Store the patterns
        self.access_patterns = all_patterns
        logger.info(f"Generated {len(self.access_patterns)} ML workload access patterns")
        
        return self.access_patterns


class GamingWorkload(WorkloadGenerator):
    """Gaming workload with player sessions, world objects, and NPCs"""
    
    def __init__(self, 
                output_dir: str = "../../../data/synthetic",
                num_players: int = 1000,
                world_objects: int = 50000,
                npcs: int = 5000,
                duration_hours: int = 12):
        super().__init__("Gaming", output_dir)
        self.num_players = num_players
        self.world_objects = world_objects
        self.npcs = npcs
        self.duration_hours = duration_hours
        
    def generate(self) -> List[Dict[str, Any]]:
        """Generate gaming workload with player sessions"""
        logger.info(f"Generating Gaming workload with {self.num_players} players")
        
        # Generate base gaming pattern
        all_patterns = generate_gaming_pattern(
            num_players=self.num_players,
            num_game_objects=self.world_objects + self.npcs,
            duration_minutes=self.duration_hours * 60
        )
        
        # Add gaming-specific metadata
        for access in all_patterns:
            # Extract object ID
            if "game_obj_" in access["key"]:
                obj_id = int(access["key"].split("_")[-1])
                
                # Classify object type
                if obj_id < self.world_objects:
                    if obj_id < self.world_objects * 0.01:  # 1% are critical/popular objects
                        obj_type = "critical_location"
                    elif obj_id < self.world_objects * 0.2:  # 20% are interactive objects
                        obj_type = "interactive"
                    else:
                        obj_type = "static"
                else:
                    # This is an NPC
                    npc_id = obj_id - self.world_objects
                    if npc_id < self.npcs * 0.05:  # 5% are bosses
                        obj_type = "boss_npc"
                    elif npc_id < self.npcs * 0.2:  # 20% are quest givers
                        obj_type = "quest_npc"
                    else:
                        obj_type = "regular_npc"
                
                access["metadata"] = access.get("metadata", {})
                access["metadata"]["object_type"] = obj_type
        
        # Store the patterns
        self.access_patterns = all_patterns
        logger.info(f"Generated {len(self.access_patterns)} gaming access patterns")
        
        return self.access_patterns


def generate_combined_workload(
    output_dir: str = "../../../data/synthetic/combined",
    workload_mix: Dict[str, float] = None,
    duration_hours: int = 24
) -> pd.DataFrame:
    """
    Generate a combined workload with a mix of different patterns
    
    Args:
        output_dir: Directory to save output files
        workload_mix: Dictionary with workload types and their proportions
                     (default: equal mix of all workloads)
        duration_hours: Duration to simulate in hours
        
    Returns:
        DataFrame with combined workload
    """
    # Default to equal mix if not specified
    if workload_mix is None:
        workload_mix = {
            "web": 0.25,
            "database": 0.25,
            "ml": 0.25,
            "gaming": 0.25
        }
    
    logger.info(f"Generating combined workload with mix: {workload_mix}")
    
    all_patterns = []
    
    # Generate each workload type
    if workload_mix.get("web", 0) > 0:
        web_gen = WebServiceWorkload(
            output_dir=output_dir,
            duration_days=duration_hours // 24 + 1
        )
        web_patterns = web_gen.generate()
        # Take subset based on proportion
        sample_size = int(len(web_patterns) * workload_mix["web"])
        all_patterns.extend(np.random.choice(web_patterns, size=sample_size, replace=False))
    
    if workload_mix.get("database", 0) > 0:
        db_gen = DatabaseWorkload(
            output_dir=output_dir,
            duration_hours=duration_hours
        )
        db_patterns = db_gen.generate()
        # Take subset based on proportion
        sample_size = int(len(db_patterns) * workload_mix["database"])
        all_patterns.extend(np.random.choice(db_patterns, size=sample_size, replace=False))
    
    if workload_mix.get("ml", 0) > 0:
        ml_gen = MLWorkload(
            output_dir=output_dir,
            duration_hours=duration_hours
        )
        ml_patterns = ml_gen.generate()
        # Take subset based on proportion
        sample_size = int(len(ml_patterns) * workload_mix["ml"])
        all_patterns.extend(np.random.choice(ml_patterns, size=sample_size, replace=False))
    
    if workload_mix.get("gaming", 0) > 0:
        gaming_gen = GamingWorkload(
            output_dir=output_dir,
            duration_hours=duration_hours
        )
        gaming_patterns = gaming_gen.generate()
        # Take subset based on proportion
        sample_size = int(len(gaming_patterns) * workload_mix["gaming"])
        all_patterns.extend(np.random.choice(gaming_patterns, size=sample_size, replace=False))
    
    # Sort by timestamp
    all_patterns.sort(key=lambda x: x["timestamp"])
    
    # Export the combined workload
    df = export_synthetic_data(all_patterns, f"{output_dir}/combined_workload")
    logger.info(f"Generated combined workload with {len(all_patterns)} access patterns")
    
    return df
