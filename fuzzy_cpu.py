#!/usr/bin/env python3
"""
MinHash-based Text Deduplication Pipeline
Author: AI Assistant
Created: 2025-06-05

A comprehensive pipeline for deduplicating text datasets using MinHash LSH algorithm.
Supports multiple input/output formats and provides detailed logging and timing.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import time
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Constants
SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
INDEX_COLUMN = "__index__"
SIGNATURE_COLUMN = "__signatures__"
CLUSTER_COLUMN = "__cluster__"

# Set multiprocessing start method (only on Unix-like systems)
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    # fork not available on Windows
    pass


class Timer:
    """Timer utility for measuring execution time of different pipeline stages."""
    
    def __init__(self):
        self.times = {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        pass
    
    def start(self, name: str):
        """Start timing a named operation."""
        self.times[name] = {"start": time.time()}
    
    def end(self, name: str):
        """End timing a named operation."""
        if name in self.times:
            self.times[name]["end"] = time.time()
            self.times[name]["duration"] = self.times[name]["end"] - self.times[name]["start"]
    
    def report(self, logger: logging.Logger):
        """Report all timing results."""
        logger.info("=" * 60)
        logger.info("TIMING REPORT")
        logger.info("=" * 60)
        total_time = 0
        for name, timing in self.times.items():
            if "duration" in timing:
                duration = timing["duration"]
                total_time += duration
                logger.info(f"{name:<30}: {duration:.2f}s")
        logger.info("-" * 60)
        logger.info(f"{'Total Time':<30}: {total_time/2:.2f}s")
        logger.info("=" * 60)


class UnionFind:
    """Union-Find data structure for clustering similar documents."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """Union two elements by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    
    def reset(self):
        """Reset the Union-Find structure."""
        self.parent.clear()
        self.rank.clear()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def xxh3_32hash(data: bytes) -> np.uint32:
    """Simple hash function (placeholder - in production use xxhash library)."""
    return np.uint32(hash(data) & 0xFFFFFFFF)


def xxh3_16hash(data: bytes) -> np.uint16:
    """Simple 16-bit hash function (placeholder - in production use xxhash library)."""
    return np.uint16(hash(data) & 0xFFFF)


def sha1_hash(data: bytes, d: int = 32) -> Union[np.uint32, np.uint64]:
    """SHA1-based hash function."""
    hash_obj = hashlib.sha1(data)
    hash_bytes = hash_obj.digest()
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    if d == 16:
        return np.uint16(hash_int & 0xFFFF)
    elif d == 64:
        return np.uint64(hash_int)
    return np.uint32(hash_int)


def ngrams(tokens: List[str], n: int, min_length: int = 0) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < min_length:
        return []
    
    if len(tokens) < n:
        return [tuple(tokens)]
    
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def optimal_param(threshold: float, num_perm: int, 
                 false_positive_weight: float = 0.5,
                 false_negative_weight: float = 0.5) -> Tuple[int, int]:
    """
    Compute optimal LSH parameters (b, r) for given threshold and number of permutations.
    Based on the datasketch library implementation.
    """
    def false_positive_probability(threshold: float, b: int, r: int) -> float:
        return 1.0 - (1.0 - threshold ** r) ** b
    
    def false_negative_probability(threshold: float, b: int, r: int) -> float:
        return 1.0 - (1.0 - (1.0 - threshold) ** r) ** b
    
    min_error = float('inf')
    opt_b, opt_r = 1, num_perm
    
    for b in range(1, num_perm + 1):
        if num_perm % b == 0:
            r = num_perm // b
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = false_positive_weight * fp + false_negative_weight * fn
            
            if error < min_error:
                min_error = error
                opt_b, opt_r = b, r
    
    return opt_b, opt_r


def embed_func(content: str, idx: int, **kwargs) -> Dict[str, Any]:
    """
    Calculate MinHash signatures for the content.
    
    Parameters
    ----------
    content : str
        The text content to be hashed
    idx : int
        The index of the document
    **kwargs : dict
        Additional parameters for hashing
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing signatures and index
    """
    # Extract parameters
    num_perm = kwargs['num_perm']
    ngram_size = kwargs['ngram_size']
    min_length = kwargs['min_length']
    hash_ranges = kwargs['hash_ranges']
    permutations = kwargs['permutations']
    hash_func = kwargs['hash_func']
    dtype = kwargs['dtype']
    max_hash = kwargs['max_hash']
    modulo_prime = kwargs['modulo_prime']
    
    # Get permutation arrays
    a, b = permutations
    
    # Tokenize and create n-grams
    content = str(content)  # Ensure content is string
    tokens = NON_ALPHA.split(content.lower())
    tokens = [t for t in tokens if t.strip()]  # Remove empty tokens
    
    if len(tokens) < min_length:
        # Return empty signature for short documents
        empty_sigs = [b"" for _ in hash_ranges]
        return {SIGNATURE_COLUMN: empty_sigs, INDEX_COLUMN: idx}
    
    # Generate n-grams and convert to byte strings
    ngram_list = ngrams(tokens, ngram_size, min_length)
    if not ngram_list:
        empty_sigs = [b"" for _ in hash_ranges]
        return {SIGNATURE_COLUMN: empty_sigs, INDEX_COLUMN: idx}
    
    tokens_set: Set[bytes] = {
        bytes(" ".join(gram).lower(), "utf-8") for gram in ngram_list
    }
    
    if not tokens_set:
        empty_sigs = [b"" for _ in hash_ranges]
        return {SIGNATURE_COLUMN: empty_sigs, INDEX_COLUMN: idx}
    
    # Calculate hash values
    hash_values = np.array([hash_func(token) for token in tokens_set], dtype=dtype)
    hash_values = hash_values.reshape(len(hash_values), 1)
    
    # Apply permutations
    hash_values = (hash_values * a + b) % modulo_prime & max_hash
    
    # Calculate minimum hash values
    masks = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    min_hashes = np.vstack([hash_values, masks]).min(axis=0)
    
    # Create signature bands
    signatures = [bytes(min_hashes[start:end].byteswap().data) 
                 for start, end in hash_ranges]
    
    return {SIGNATURE_COLUMN: signatures, INDEX_COLUMN: idx}


def load_dataset_from_file(file_path: str, text_column: str) -> Dataset:
    """Load dataset from various file formats."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.jsonl':
        # Load JSONL file
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        if not data:
            raise ValueError("No valid JSON records found in file")
        
        df = pd.DataFrame(data)
        
    elif file_path.suffix.lower() == '.parquet':
        # Load Parquet file
        df = pd.read_parquet(file_path)
        
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Verify text column exists
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}")
    
    # Convert to HuggingFace Dataset
    return Dataset.from_pandas(df)


def save_dataset(dataset: Dataset, output_path: str, format: str):
    """Save dataset in specified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
    elif format.lower() == 'parquet':
        df = dataset.to_pandas()
        df.to_parquet(output_path, index=False)
        
    else:
        raise ValueError(f"Unsupported output format: {format}")


class MinHashDeduplicator:
    """Main class for MinHash-based deduplication."""
    
    def __init__(self, 
                 threshold: float = 0.7,
                 num_perm: int = 250,
                 ngram_size: int = 1,
                 min_length: int = 0,
                 hash_bits: int = 32,
                 hash_func: str = "xxh3",
                 num_proc: int = 1,
                 batch_size: int = 10000,
                 b: Optional[int] = None,
                 r: Optional[int] = None):
        """
        Initialize the MinHash deduplicator.
        
        Parameters
        ----------
        threshold : float
            Jaccard similarity threshold for considering documents as duplicates
        num_perm : int
            Number of permutations for MinHash
        ngram_size : int
            Size of n-grams for tokenization
        min_length : int
            Minimum document length in tokens
        hash_bits : int
            Number of bits for hash values (16, 32, or 64)
        hash_func : str
            Hash function to use ('xxh3' or 'sha1')
        num_proc : int
            Number of processes for parallel processing
        batch_size : int
            Batch size for processing
        b : Optional[int]
            Number of bands for LSH (if None, will be calculated)
        r : Optional[int]
            Number of rows per band for LSH (if None, will be calculated)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.min_length = min_length
        self.hash_bits = hash_bits
        self.hash_func = hash_func
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.b = b
        self.r = r
        
        # Initialize timer and logger first
        self.timer = Timer()
        self.logger = setup_logging()
        
        # Initialize hash configuration
        self.hash_config = {
            64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
            32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
            16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
        }
        
        self.dtype, self.max_hash, self.modulo_prime = self.hash_config.get(
            hash_bits, self.hash_config[32]
        )
        
        # Initialize hash function
        self._setup_hash_function()
        
        # Initialize LSH parameters
        self._setup_lsh_parameters()
        
        # Initialize permutations
        self.permutations = (
            RNG.randint(1, self.modulo_prime, size=(num_perm,), dtype=self.dtype),
            RNG.randint(0, self.modulo_prime, size=(num_perm,), dtype=self.dtype)
        )
        
        # Initialize Union-Find
        self.uf = UnionFind()
    
    def _setup_hash_function(self):
        """Setup the hash function based on configuration."""
        if self.hash_func == "sha1":
            self.hash_function = lambda x: sha1_hash(x, d=self.hash_bits)
        elif self.hash_func == "xxh3":
            if self.hash_bits == 16:
                self.hash_function = xxh3_16hash
            else:
                self.hash_function = xxh3_32hash
        else:
            raise ValueError(f"Unsupported hash function: {self.hash_func}")
    
    def _setup_lsh_parameters(self):
        """Setup LSH parameters (b, r)."""
        if self.b is not None and self.r is not None:
            if self.b * self.r != self.num_perm:
                raise ValueError(f"b * r must equal num_perm. Got {self.b} * {self.r} != {self.num_perm}")
        else:
            self.b, self.r = optimal_param(self.threshold, self.num_perm)
        
        self.hash_ranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.logger.info(f"LSH Parameters: b={self.b}, r={self.r}")
    
    def deduplicate(self, 
                   input_path: Optional[str] = None,
                   dataset: Optional[Dataset] = None,
                   text_column: str = "text",
                   output_path: str = "deduplicated_dataset",
                   output_format: str = "jsonl") -> Dataset:
        """
        Perform deduplication on the dataset.
        
        Parameters
        ----------
        input_path : Optional[str]
            Path to input file (JSONL, Parquet) or HuggingFace dataset name
        dataset : Optional[Dataset]
            Pre-loaded HuggingFace Dataset object
        text_column : str
            Name of the column containing text data
        output_path : str
            Path for saving deduplicated dataset
        output_format : str
            Output format ('jsonl' or 'parquet')
            
        Returns
        -------
        Dataset
            Deduplicated dataset
        """
        self.timer.start("Total")
        
        # Load dataset - Fixed logic here
        self.timer.start("Loading")
        if dataset is not None and hasattr(dataset, 'column_names'):
            # It's a proper Dataset object
            ds = dataset
            self.logger.info("Using provided dataset")
        elif input_path is not None:
            try:
                if "/" in input_path and not Path(input_path).exists():
                    # Assume it's a HuggingFace dataset
                    self.logger.info(f"Loading HuggingFace dataset: {input_path}")
                    ds = load_dataset(input_path, split="train")
                else:
                    # Load from file
                    self.logger.info(f"Loading dataset from file: {input_path}")
                    ds = load_dataset_from_file(input_path, text_column)
            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                raise
        elif isinstance(dataset, str):
            # Handle case where dataset parameter is actually a file path
            try:
                if "/" in dataset and not Path(dataset).exists():
                    # Assume it's a HuggingFace dataset
                    self.logger.info(f"Loading HuggingFace dataset: {dataset}")
                    ds = load_dataset(dataset, split="train")
                else:
                    # Load from file
                    self.logger.info(f"Loading dataset from file: {dataset}")
                    ds = load_dataset_from_file(dataset, text_column)
            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                raise
        else:
            raise ValueError("Either input_path or dataset must be provided")
        
        # Verify text column exists
        if text_column not in ds.column_names:
            raise ValueError(f"Text column '{text_column}' not found in dataset. "
                           f"Available columns: {list(ds.column_names)}")
        
        # Add index column
        if INDEX_COLUMN not in ds.column_names:
            ds = ds.add_column(INDEX_COLUMN, list(range(len(ds))))
        
        # Filter short documents
        original_size = len(ds)
        if self.min_length > 0:
            def filter_short_docs(example):
                text = str(example[text_column])
                tokens = NON_ALPHA.split(text.lower())
                tokens = [t for t in tokens if t.strip()]
                return len(tokens) >= self.min_length
            
            ds = ds.filter(
                filter_short_docs,
                num_proc=self.num_proc,
                desc="Filtering short documents..."
            )
        filtered_size = len(ds)
        
        self.timer.end("Loading")
        self.logger.info(f"Dataset loaded: {original_size} documents")
        self.logger.info(f"After filtering: {filtered_size} documents")
        
        # Generate MinHash signatures
        self.timer.start("MinHashing")
        embedded = ds.map(
            embed_func,
            fn_kwargs={
                "num_perm": self.num_perm,
                "hash_ranges": self.hash_ranges,
                "ngram_size": self.ngram_size,
                "min_length": self.min_length,
                "permutations": self.permutations,
                "hash_func": self.hash_function,
                "dtype": self.dtype,
                "max_hash": self.max_hash,
                "modulo_prime": self.modulo_prime,
            },
            input_columns=[text_column, INDEX_COLUMN],
            remove_columns=[col for col in ds.column_names if col not in [INDEX_COLUMN, text_column]],
            num_proc=self.num_proc,
            desc="Generating MinHash signatures..."
        )
        self.timer.end("MinHashing")
        self.logger.info(f"MinHash signatures generated for {len(embedded)} documents")
        
        # Perform LSH clustering
        self.timer.start("Clustering")
        hash_tables = [defaultdict(set) for _ in range(self.b)]
        
        # Build hash tables
        for idx, signatures in tqdm(
            zip(embedded[INDEX_COLUMN], embedded[SIGNATURE_COLUMN]),
            total=len(embedded),
            desc="Building hash tables..."
        ):
            for i, signature in enumerate(signatures):
                if signature:  # Skip empty signatures
                    hash_tables[i][signature].add(idx)
        
        # Find clusters and build edges
        edges = []
        self.uf.reset()
        
        for table in tqdm(hash_tables, desc="Clustering documents..."):
            for cluster in table.values():
                if len(cluster) <= 1:
                    continue
                cluster_list = list(cluster)
                root = min(cluster_list)
                for doc_id in cluster_list:
                    if doc_id != root:
                        edges.append((doc_id, root))
                        self.uf.union(doc_id, root)
        
        self.timer.end("Clustering")
        self.logger.info(f"Found {len(set(edges))} duplicate pairs")
        
        # Filter duplicates
        self.timer.start("Filtering")
        
        # Add cluster information to original dataset
        def add_cluster_id(example):
            return {CLUSTER_COLUMN: self.uf.find(example[INDEX_COLUMN])}
        
        ds_with_clusters = ds.map(
            add_cluster_id,
            num_proc=self.num_proc,
            desc="Assigning cluster IDs..."
        )
        
        # Keep only cluster representatives
        def is_cluster_representative(example):
            return example[CLUSTER_COLUMN] == example[INDEX_COLUMN]
        
        deduplicated = ds_with_clusters.filter(
            is_cluster_representative,
            num_proc=self.num_proc,
            desc="Filtering duplicates..."
        )
        
        # Remove auxiliary columns
        columns_to_remove = [col for col in [CLUSTER_COLUMN, INDEX_COLUMN] 
                           if col in deduplicated.column_names]
        if columns_to_remove:
            deduplicated = deduplicated.remove_columns(columns_to_remove)
        
        self.timer.end("Filtering")
        
        # Save results
        self.timer.start("Saving")
        save_dataset(deduplicated, output_path, output_format)
        self.timer.end("Saving")
        
        self.timer.end("Total")
        
        # Report results
        final_size = len(deduplicated)
        self.logger.info("=" * 60)
        self.logger.info("DEDUPLICATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Original dataset size: {original_size:,}")
        self.logger.info(f"After filtering short docs: {filtered_size:,}")
        self.logger.info(f"Final deduplicated size: {final_size:,}")
        self.logger.info(f"Documents removed: {original_size - final_size:,}")
        self.logger.info(f"Deduplication ratio: {(original_size - final_size) / original_size * 100:.2f}%")
        self.logger.info(f"Output saved to: {output_path}")
        
        # Report timing
        self.timer.report(self.logger)
        
        return deduplicated


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="MinHash-based text deduplication pipeline")
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", required=True, 
                       help="Input file path (JSONL/Parquet) or HuggingFace dataset name")
    parser.add_argument("--output", "-o", default="deduplicated_dataset.jsonl",
                       help="Output file path")
    parser.add_argument("--output-format", choices=["jsonl", "parquet"], default="jsonl",
                       help="Output format")
    parser.add_argument("--text-column", default="text",
                       help="Name of the column containing text data")
    
    # MinHash parameters
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Jaccard similarity threshold for duplicates")
    parser.add_argument("--num-perm", type=int, default=250,
                       help="Number of permutations for MinHash")
    parser.add_argument("--ngram-size", type=int, default=1,
                       help="Size of n-grams")
    parser.add_argument("--min-length", type=int, default=0,
                       help="Minimum document length in tokens")
    parser.add_argument("--hash-bits", type=int, choices=[16, 32, 64], default=32,
                       help="Number of bits for hash values")
    parser.add_argument("--hash-func", choices=["xxh3", "sha1"], default="xxh3",
                       help="Hash function to use")
    
    # LSH parameters
    parser.add_argument("--b", type=int, help="Number of bands for LSH")
    parser.add_argument("--r", type=int, help="Number of rows per band for LSH")
    
    # Processing parameters
    parser.add_argument("--num-proc", type=int, default=1,
                       help="Number of processes for parallel processing")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize deduplicator
    deduplicator = MinHashDeduplicator(
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        min_length=args.min_length,
        hash_bits=args.hash_bits,
        hash_func=args.hash_func,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        b=args.b,
        r=args.r
    )
    
    # Run deduplication
    deduplicator.deduplicate(
        input_path=args.input,
        text_column=args.text_column,
        output_path=args.output,
        output_format=args.output_format
    )

if __name__ == "__main__":
    main()