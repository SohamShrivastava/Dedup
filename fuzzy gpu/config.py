
"""
Name: config.py
Handles configuration management, GPU environment setup, parameter validation,
and utility functions for the deduplication pipeline.
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
from dask.distributed import Client, LocalCluster


# ============================================================================
# CONSTANTS
# ============================================================================

SEED = 42
INDEX_COLUMN = "__index__"
SIGNATURE_COLUMN = "__signatures__"
CLUSTER_COLUMN = "__cluster__"

# Supported file formats
SUPPORTED_INPUT_FORMATS = {'.jsonl', '.parquet', '.json'}
SUPPORTED_OUTPUT_FORMATS = {'jsonl', 'parquet', 'json'}

# Hash function configurations
HASH_FUNCTIONS = {
    'xxh3': {'bits_supported': [16, 32, 64], 'default_bits': 32},
    'sha1': {'bits_supported': [32, 64], 'default_bits': 32},
    'murmur3': {'bits_supported': [32, 64], 'default_bits': 32}
}

# Default parameters
DEFAULT_CONFIG = {
    'threshold': 0.7,
    'num_perm': 250,
    'ngram_size': 1,
    'min_length': 0,
    'hash_bits': 32,
    'hash_func': 'xxh3',
    'n_workers': 1,
    'memory_limit': '8GB',
    'batch_size': 10000,
    'log_level': 'INFO'
}


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class DeduplicationConfig:
    """Configuration management for the deduplication pipeline."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with defaults and user overrides."""
        # Set defaults
        for key, value in DEFAULT_CONFIG.items():
            setattr(self, key, value)
        
        # Override with user parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Validate configuration
        self.validate()
        
        # Calculate LSH parameters if not provided
        if not hasattr(self, 'b') or not hasattr(self, 'r'):
            self.b, self.r = self.calculate_optimal_lsh_params()
    
    def validate(self):
        """Validate configuration parameters."""
        # Threshold validation
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        
        # Permutation validation
        if self.num_perm <= 0:
            raise ValueError(f"Number of permutations must be positive, got {self.num_perm}")
        
        # N-gram validation
        if self.ngram_size <= 0:
            raise ValueError(f"N-gram size must be positive, got {self.ngram_size}")
        
        # Hash function validation
        if self.hash_func not in HASH_FUNCTIONS:
            raise ValueError(f"Unsupported hash function: {self.hash_func}. "
                           f"Supported: {list(HASH_FUNCTIONS.keys())}")
        
        # Hash bits validation
        supported_bits = HASH_FUNCTIONS[self.hash_func]['bits_supported']
        if self.hash_bits not in supported_bits:
            raise ValueError(f"Hash function {self.hash_func} doesn't support {self.hash_bits} bits. "
                           f"Supported: {supported_bits}")
        
        # Worker validation
        if self.n_workers <= 0:
            raise ValueError(f"Number of workers must be positive, got {self.n_workers}")
        
        # Batch size validation
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
    
    def calculate_optimal_lsh_params(self) -> Tuple[int, int]:
        """
        Calculate optimal LSH parameters (b, r) for given threshold and number of permutations.
        
        Returns
        -------
        Tuple[int, int]
            Optimal (b, r) parameters where b * r = num_perm
        """
        threshold = self.threshold
        num_perm = self.num_perm
        
        # Find optimal b and r such that b * r = num_perm
        best_b, best_r = 1, num_perm
        best_error = float('inf')
        
        # Try all possible factorizations
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                
                # Calculate probability of collision for this (b, r)
                # P(collision) = 1 - (1 - s^r)^b, where s is similarity
                prob_at_threshold = 1 - (1 - threshold**r)**b
                
                # We want this probability to be close to 0.5 for optimal discrimination
                error = abs(prob_at_threshold - 0.5)
                
                if error < best_error:
                    best_error = error
                    best_b, best_r = b, r
        
        return best_b, best_r
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {key: getattr(self, key) for key in DEFAULT_CONFIG.keys()}
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "Deduplication Configuration:\n"
        config_str += f"  Threshold: {self.threshold}\n"
        config_str += f"  Permutations: {self.num_perm}\n"
        config_str += f"  LSH Parameters: b={self.b}, r={self.r}\n"
        config_str += f"  N-gram size: {self.ngram_size}\n"
        config_str += f"  Hash function: {self.hash_func} ({self.hash_bits} bits)\n"
        config_str += f"  Workers: {self.n_workers}\n"
        config_str += f"  Memory limit: {self.memory_limit}\n"
        config_str += f"  Batch size: {self.batch_size}"
        return config_str


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_gpu_environment() -> bool:
    """
    Setup GPU environment and verify RAPIDS availability.
    
    Returns
    -------
    bool
        True if GPU setup is successful
    """
    try:
        # Check if GPU is available
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count == 0:
            raise RuntimeError("No GPU devices found")
        
        # Set memory pool
        cp.cuda.MemoryPool().set_limit(size=None)  # Use all available memory
        
        # Verify cuDF import
        import cudf
        
        # Create a small test DataFrame to verify functionality
        test_df = cudf.DataFrame({'test': [1, 2, 3]})
        _ = test_df.sum()
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup GPU environment: {str(e)}")


def setup_dask_cluster(n_workers: int = 1, memory_limit: str = "8GB") -> Client:
    """
    Setup Dask cluster with GPU workers.
    
    Parameters
    ----------
    n_workers : int
        Number of workers
    memory_limit : str
        Memory limit per worker
        
    Returns
    -------
    Client
        Dask client instance
    """
    try:
        # Create local cluster with GPU support
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            silence_logs=logging.WARNING
        )
        
        # Create client
        client = Client(cluster)
        
        return client
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup Dask cluster: {str(e)}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level
    log_file : Optional[str]
        Optional log file path
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger('deduplicator')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# TIMER UTILITY
# ============================================================================

class Timer:
    """Timer utility for measuring execution time of different pipeline stages."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def start(self, name: str):
        """Start timing a named operation."""
        self.start_times[name] = time.time()
    
    def end(self, name: str) -> float:
        """
        End timing a named operation.
        
        Returns
        -------
        float
            Elapsed time in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        self.times[name] = elapsed
        del self.start_times[name]
        
        return elapsed
    
    def get_time(self, name: str) -> Optional[float]:
        """Get elapsed time for a named operation."""
        return self.times.get(name)
    
    def get_all_times(self) -> Dict[str, float]:
        """Get all recorded times."""
        return self.times.copy()
    
    def report(self, logger: logging.Logger):
        """Report all timing results."""
        if not self.times:
            logger.info("No timing data available")
            return
        
        logger.info("=== Timing Report ===")
        total_time = sum(self.times.values())
        
        for name, elapsed in sorted(self.times.items()):
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            logger.info(f"{name}: {elapsed:.2f}s ({percentage:.1f}%)")
        
        logger.info(f"Total time: {total_time:.2f}s")
    
    def reset(self):
        """Reset all timing data."""
        self.times.clear()
        self.start_times.clear()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_file_path(file_path: str, check_exists: bool = True) -> Path:
    """
    Validate file path and return Path object.
    
    Parameters
    ----------
    file_path : str
        File path to validate
    check_exists : bool
        Whether to check if file exists
        
    Returns
    -------
    Path
        Validated Path object
    """
    path = Path(file_path)
    
    if check_exists and not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return path


def validate_input_format(file_path: str) -> str:
    """
    Validate input file format.
    
    Parameters
    ----------
    file_path : str
        Input file path
        
    Returns
    -------
    str
        File extension
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported input format: {ext}. "
                        f"Supported formats: {SUPPORTED_INPUT_FORMATS}")
    
    return ext


def validate_output_format(output_format: str) -> str:
    """
    Validate output format.
    
    Parameters
    ----------
    output_format : str
        Output format
        
    Returns
    -------
    str
        Validated output format
    """
    fmt = output_format.lower()
    
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {fmt}. "
                        f"Supported formats: {SUPPORTED_OUTPUT_FORMATS}")
    
    return fmt


def get_memory_info() -> Dict[str, int]:
    """
    Get GPU memory information.
    
    Returns
    -------
    Dict[str, int]
        Memory information in bytes
    """
    try:
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        used_mem = total_mem - free_mem
        
        return {
            'total': total_mem,
            'used': used_mem,
            'free': free_mem
        }
    except Exception:
        return {'total': 0, 'used': 0, 'free': 0}


def format_memory(bytes_val: int) -> str:
    """Format memory value in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


# ============================================================================
# MAIN CONFIGURATION FACTORY
# ============================================================================

def create_config(**kwargs) -> DeduplicationConfig:
    """
    Factory function to create a validated configuration.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters
        
    Returns
    -------
    DeduplicationConfig
        Validated configuration object
    """
    return DeduplicationConfig(**kwargs)


if __name__ == "__main__":
    # Test configuration
    config = create_config(threshold=0.8, num_perm=128)
    print(config)
    
    # Test GPU setup
    try:
        setup_gpu_environment()
        print("GPU environment setup successful")
    except Exception as e:
        print(f"GPU setup failed: {e}")
    
    # Test timer
    with Timer() as timer:
        timer.start("test_operation")
        time.sleep(0.1)
        elapsed = timer.end("test_operation")
        print(f"Test operation took {elapsed:.3f} seconds")