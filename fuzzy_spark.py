#!/usr/bin/env python3
"""
Spark-based MinHash Text Deduplication Pipeline (Complete Version)
Author: AI Assistant
Created: 2025-06-05

A comprehensive pipeline for deduplicating text datasets using MinHash LSH algorithm with Apache Spark.
Supports multiple input/output formats and provides detailed logging and timing.
"""

import argparse
import json
import logging
import os
import random
import re
import time
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from clustering import OptimizedClustering

import numpy as np
import pandas as pd

# Spark imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, udf, collect_list, struct, explode, broadcast, 
    row_number, monotonically_increasing_id, size as spark_size,
    regexp_replace, lower, split, filter as spark_filter, lit
)
from pyspark.sql.functions import col, coalesce, when, min as spark_min
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    ArrayType, BinaryType, LongType
)
from pyspark.sql.window import Window

# Constants
SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
INDEX_COLUMN = "__index__"
SIGNATURE_COLUMN = "__signatures__"
CLUSTER_COLUMN = "__cluster__"


class Timer:
    """Timer utility for measuring execution time of different pipeline stages."""
    
    def __init__(self):
        self.times = {}
        self.start_time = None
    
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


class SparkMinHashDeduplicator:
    """Spark-based MinHash deduplication pipeline."""
    
    def __init__(self,
                 threshold: float = 0.7,
                 num_perm: int = 250,
                 ngram_size: int = 1,
                 min_length: int = 0,
                 hash_bits: int = 32,
                 hash_func: str = "xxh3",
                 b: Optional[int] = None,
                 r: Optional[int] = None,
                 spark_config: Optional[Dict[str, str]] = None):
        """
        Initialize the Spark MinHash deduplicator.
        
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
        b : Optional[int]
            Number of bands for LSH (if None, will be calculated)
        r : Optional[int]
            Number of rows per band for LSH (if None, will be calculated)
        spark_config : Optional[Dict[str, str]]
            Additional Spark configuration options
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.min_length = min_length
        self.hash_bits = hash_bits
        self.hash_func = hash_func
        self.b = b
        self.r = r
        
        # Initialize timer and logger
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
        
        # Initialize LSH parameters
        self._setup_lsh_parameters()
        
        # Initialize permutations
        self.permutations = (
            RNG.randint(1, self.modulo_prime, size=(num_perm,), dtype=self.dtype),
            RNG.randint(0, self.modulo_prime, size=(num_perm,), dtype=self.dtype)
        )
        
        # Initialize Spark
        self._setup_spark(spark_config)
    
    def _setup_spark(self, spark_config: Optional[Dict[str, str]] = None):
        """Initialize Spark session with optimized configuration."""
        conf = SparkConf().setAppName("MinHashDeduplicator")
        
        # Default Spark configuration for deduplication workload
        default_config = {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.broadcastTimeout": "600",
            "spark.sql.shuffle.partitions": "200",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.execution.arrow.maxRecordsPerBatch": "10000"
        }
        
        # Apply default configuration
        for key, value in default_config.items():
            conf.set(key, value)
        
        # Apply user configuration
        if spark_config:
            for key, value in spark_config.items():
                conf.set(key, value)
        
        # Create Spark session
        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.logger.info(f"Spark session initialized with {self.spark.sparkContext.defaultParallelism} default parallelism")
    
    def _setup_lsh_parameters(self):
        """Setup LSH parameters (b, r)."""
        if self.b is not None and self.r is not None:
            if self.b * self.r != self.num_perm:
                raise ValueError(f"b * r must equal num_perm. Got {self.b} * {self.r} != {self.num_perm}")
        else:
            self.b, self.r = optimal_param(self.threshold, self.num_perm)
        
        self.hash_ranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.logger.info(f"LSH Parameters: b={self.b}, r={self.r}")
    
    def _load_data(self, input_path: str, text_column: str) -> DataFrame:
        """Load data into Spark DataFrame."""
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            if input_path_obj.suffix.lower() == '.jsonl':
                # Load JSONL file
                df = self.spark.read.option("multiline", "false").json(input_path)
            elif input_path_obj.suffix.lower() == '.parquet':
                # Load Parquet file
                df = self.spark.read.parquet(input_path)
            elif input_path_obj.suffix.lower() == '.json':
                # Load JSON file
                df = self.spark.read.option("multiLine", "true").json(input_path)
            else:
                # Try to read as text file and assume each line is a document
                df = self.spark.read.text(input_path)
                df = df.withColumnRenamed("value", text_column)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {input_path}: {str(e)}")
        
        # Verify text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset. "
                           f"Available columns: {df.columns}")
        
        df = df.withColumn(INDEX_COLUMN, monotonically_increasing_id())

        return df
    
    def _create_minhash_udf(self):
        """Create UDF for MinHash signature generation."""
        
        # Extract values that need to be serialized
        permutations_a = self.permutations[0]
        permutations_b = self.permutations[1]
        hash_ranges = self.hash_ranges
        ngram_size = self.ngram_size
        min_length = self.min_length
        b = self.b
        hash_func = self.hash_func
        hash_bits = self.hash_bits
        dtype = self.dtype
        max_hash = self.max_hash
        modulo_prime = self.modulo_prime
        
        # Broadcast the permutation arrays and other parameters
        permutations_a_bc = self.spark.sparkContext.broadcast(permutations_a)
        permutations_b_bc = self.spark.sparkContext.broadcast(permutations_b)
        hash_ranges_bc = self.spark.sparkContext.broadcast(hash_ranges)
        
        def minhash_func(text: str) -> List[bytes]:
            """Generate MinHash signatures for text."""
            if not text or not isinstance(text, str):
                return [b"" for _ in range(b)]
            
            # Get broadcasted values
            a = permutations_a_bc.value
            b_vals = permutations_b_bc.value
            hash_ranges_local = hash_ranges_bc.value
            
            # Get hash function
            if hash_func == "sha1":
                def hash_function(x):
                    hash_obj = hashlib.sha1(x)
                    hash_bytes = hash_obj.digest()
                    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
                    if hash_bits == 16:
                        return np.uint16(hash_int & 0xFFFF)
                    elif hash_bits == 64:
                        return np.uint64(hash_int)
                    return np.uint32(hash_int)
            elif hash_func == "xxh3":
                if hash_bits == 16:
                    def hash_function(x):
                        return np.uint16(hash(x) & 0xFFFF)
                else:
                    def hash_function(x):
                        return np.uint32(hash(x) & 0xFFFFFFFF)
            
            # Tokenize
            tokens = NON_ALPHA.split(text.lower())
            tokens = [t for t in tokens if t.strip()]
            
            if len(tokens) < min_length:
                return [b"" for _ in range(b)]
            
            # Generate n-grams
            if len(tokens) < ngram_size:
                ngram_list = [tuple(tokens)]
            else:
                ngram_list = [tuple(tokens[i:i+ngram_size]) 
                             for i in range(len(tokens) - ngram_size + 1)]
            
            if not ngram_list:
                return [b"" for _ in range(b)]
            
            # Convert to byte strings
            tokens_set = {bytes(" ".join(gram).lower(), "utf-8") for gram in ngram_list}
            
            if not tokens_set:
                return [b"" for _ in range(b)]
            
            # Calculate hash values
            hash_values = np.array([hash_function(token) for token in tokens_set], 
                                 dtype=dtype)
            
            if len(hash_values) == 0:
                return [b"" for _ in range(b)]
            
            hash_values = hash_values.reshape(len(hash_values), 1)
            
            # Apply permutations
            hash_values = (hash_values * a + b_vals) % modulo_prime & max_hash
            
            # Calculate minimum hash values
            masks = np.full(shape=len(a), dtype=dtype, fill_value=max_hash)
            if len(hash_values) > 0:
                min_hashes = np.minimum(hash_values.min(axis=0), masks)
            else:
                min_hashes = masks
            
            # Create signature bands
            signatures = []
            for start, end in hash_ranges_local:
                sig_bytes = bytes(min_hashes[start:end].byteswap().data)
                signatures.append(sig_bytes)
            
            return signatures
        
        return udf(minhash_func, ArrayType(BinaryType()))
    
    def _find_duplicate_pairs(self, df_with_sigs: DataFrame) -> DataFrame:
        """Find duplicate pairs using LSH."""
        # Create a DataFrame with signature bands
        band_data = []
        for i in range(self.b):
            band_df = df_with_sigs.select(
                col(INDEX_COLUMN),
                col(SIGNATURE_COLUMN)[i].alias("signature")
            ).withColumn("band_id", lit(i))
            band_data.append(band_df)
        
        # Union all bands
        df_exploded = band_data[0]
        for band_df in band_data[1:]:
            df_exploded = df_exploded.union(band_df)
        
        # Group by signature and band to find potential duplicates
        signature_groups = df_exploded.groupBy("band_id", "signature").agg(
            collect_list(INDEX_COLUMN).alias("doc_ids"),
            spark_size(collect_list(INDEX_COLUMN)).alias("group_size")
        ).filter(col("group_size") > 1)
        
        # Generate pairs from each group
        def generate_pairs(doc_ids):
            """Generate all pairs from a list of document IDs."""
            pairs = []
            if doc_ids and len(doc_ids) > 1:
                doc_list = sorted(set(doc_ids))  # Remove duplicates and sort
                for i in range(len(doc_list)):
                    for j in range(i + 1, len(doc_list)):
                        pairs.append((int(doc_list[i]), int(doc_list[j])))
            return pairs
        
        pairs_udf = udf(generate_pairs, ArrayType(StructType([
            StructField("doc1", LongType(), False),
            StructField("doc2", LongType(), False)
        ])))
        
        # Generate pairs and explode
        pairs_df = signature_groups.select(
            explode(pairs_udf(col("doc_ids"))).alias("pair")
        ).select(
            col("pair.doc1").alias("doc1"),
            col("pair.doc2").alias("doc2")
        ).distinct()
        
        return pairs_df
    
    def _find_clusters(self, duplicate_pairs: DataFrame) -> DataFrame:
        clustering = OptimizedClustering(self.spark, self.logger)
        return clustering.find_clusters(duplicate_pairs, method="graphframes")
        # """Simplified clustering approach that's more robust."""
        # if duplicate_pairs.count() == 0:
        #     return self.spark.createDataFrame([], schema=StructType([
        #         StructField("doc", LongType(), False),
        #         StructField("cluster", LongType(), False)
        #     ]))
        
        # # Collect all unique documents
        # all_docs = duplicate_pairs.select(col("doc1").alias("doc")).union(
        #     duplicate_pairs.select(col("doc2").alias("doc"))
        # ).distinct()
        
        # # Start with each document as its own cluster
        # clusters = all_docs.withColumn("cluster", col("doc"))
        
        # # Simple iterative approach
        # converged = False
        # iteration = 0
        # max_iterations = 20
        
        # while not converged and iteration < max_iterations:
        #     iteration += 1
        #     self.logger.info(f"Clustering iteration {iteration}")
            
        #     # Create mapping of doc -> min_cluster from pairs
        #     doc_to_min_cluster = duplicate_pairs.alias("pairs").join(
        #         clusters.alias("c1"), 
        #         col("pairs.doc1") == col("c1.doc")
        #     ).join(
        #         clusters.alias("c2"), 
        #         col("pairs.doc2") == col("c2.doc")
        #     ).select(
        #         col("pairs.doc1").alias("doc"),
        #         when(col("c1.cluster") < col("c2.cluster"), col("c1.cluster"))
        #         .otherwise(col("c2.cluster")).alias("min_cluster")
        #     ).union(
        #         duplicate_pairs.alias("pairs").join(
        #             clusters.alias("c1"), 
        #             col("pairs.doc1") == col("c1.doc")
        #         ).join(
        #             clusters.alias("c2"), 
        #             col("pairs.doc2") == col("c2.doc")
        #         ).select(
        #             col("pairs.doc2").alias("doc"),
        #             when(col("c1.cluster") < col("c2.cluster"), col("c1.cluster"))
        #             .otherwise(col("c2.cluster")).alias("min_cluster")
        #         )
        #     ).groupBy("doc").agg({"min_cluster": "min"}).withColumnRenamed("min(min_cluster)", "new_cluster")
            
        #     # Update clusters
        #     old_clusters = clusters
        #     clusters = clusters.join(
        #         doc_to_min_cluster,
        #         clusters.doc == doc_to_min_cluster.doc,
        #         "left"
        #     ).select(
        #         clusters.doc,
        #         coalesce(doc_to_min_cluster.new_cluster, clusters.cluster).alias("cluster")
        #     )
            
        #     # Check convergence
        #     changes = old_clusters.join(clusters, "doc").filter(
        #         old_clusters.cluster != clusters.cluster
        #     ).count()
            
        #     converged = (changes == 0)
        #     self.logger.info(f"Iteration {iteration}: {changes} cluster changes")
        
        # if not converged:
        #     self.logger.warning(f"Clustering did not converge after {max_iterations} iterations")
        
        # return clusters
    
    def deduplicate(self,
                   input_path: str,
                   text_column: str = "text",
                   output_path: str = "deduplicated_dataset",
                   output_format: str = "parquet") -> DataFrame:
        """
        Perform deduplication on the dataset.
        
        Parameters
        ----------
        input_path : str
            Path to input file
        text_column : str
            Name of the column containing text data
        output_path : str
            Path for saving deduplicated dataset
        output_format : str
            Output format ('parquet' or 'json')
            
        Returns
        -------
        DataFrame
            Deduplicated Spark DataFrame
        """
        self.timer.start("Total")
        
        # Load data
        self.timer.start("Loading")
        df = self._load_data(input_path, text_column)
        df.cache()  # Cache the initial DataFrame
        
        original_count = df.count()
        self.logger.info(f"Loaded {original_count:,} documents")
        self.timer.end("Loading")
        
        # Filter short documents
        self.timer.start("Filtering")
        if self.min_length > 0:
            # Create UDF for filtering
            min_length_local = self.min_length
            
            def filter_short_docs(text: str) -> bool:
                if not text or not isinstance(text, str):
                    return False
                tokens = NON_ALPHA.split(text.lower())
                tokens = [t for t in tokens if t.strip()]
                return len(tokens) >= min_length_local
            
            filter_udf = udf(filter_short_docs, "boolean")
            df = df.filter(filter_udf(col(text_column)))
            df.cache()
        
        filtered_count = df.count()
        self.logger.info(f"After filtering: {filtered_count:,} documents")
        self.timer.end("Filtering")
        
        if filtered_count == 0:
            self.logger.warning("No documents remaining after filtering")
            # Create empty output
            empty_df = df.limit(0).drop(INDEX_COLUMN)
            self._save_output(empty_df, output_path, output_format)
            self.timer.end("Total")
            return empty_df
        
        # Generate MinHash signatures
        self.timer.start("MinHashing")
        minhash_udf = self._create_minhash_udf()
        
        df_with_sigs = df.withColumn(SIGNATURE_COLUMN, minhash_udf(col(text_column)))
        df_with_sigs.cache()
        
        # Force computation to ensure caching
        sig_count = df_with_sigs.count()
        self.logger.info(f"Generated MinHash signatures for {sig_count:,} documents")
        self.timer.end("MinHashing")
        
        # Find duplicate pairs
        self.timer.start("Finding Pairs")
        duplicate_pairs = self._find_duplicate_pairs(df_with_sigs)
        duplicate_pairs.cache()
        
        pairs_count = duplicate_pairs.count()
        self.logger.info(f"Found {pairs_count:,} potential duplicate pairs")
        self.timer.end("Finding Pairs")
        
        if pairs_count == 0:
            self.logger.info("No duplicates found, returning original dataset")
            deduplicated = df.drop(INDEX_COLUMN)
        else:
            # Find clusters
            self.timer.start("Clustering")
            clusters = self._find_clusters(duplicate_pairs)
            clusters.cache()
            
            cluster_count = clusters.select("cluster").distinct().count()
            self.logger.info(f"Found {cluster_count:,} unique clusters")
            self.timer.end("Clustering")
            
            # Filter duplicates - keep one representative per cluster
            self.timer.start("Deduplicating")
            
            # Find cluster representatives (document with minimum index in each cluster)
            cluster_representatives = clusters.groupBy("cluster").agg(
                {"doc": "min"}
            ).withColumnRenamed("min(doc)", "representative")
            
            # Join with original data to get deduplicated dataset
            deduplicated = df.join(
                cluster_representatives,
                col(INDEX_COLUMN) == col("representative"),
                "inner"
            ).drop("representative", INDEX_COLUMN)
            
            self.timer.end("Deduplicating")
        
        # Save results
        self.timer.start("Saving")
        deduplicated.cache()
        final_count = deduplicated.count()
        
        self._save_output(deduplicated, output_path, output_format)
        
        self.timer.end("Saving")
        self.timer.end("Total")
        
        # Report results
        self.logger.info("=" * 60)
        self.logger.info("DEDUPLICATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Original dataset size: {original_count:,}")
        self.logger.info(f"After filtering short docs: {filtered_count:,}")
        self.logger.info(f"Final deduplicated size: {final_count:,}")
        self.logger.info(f"Documents removed: {original_count - final_count:,}")
        if original_count > 0:
            self.logger.info(f"Deduplication ratio: {(original_count - final_count) / original_count * 100:.2f}%")
        self.logger.info(f"Output saved to: {output_path}")
        
        # Report timing
        self.timer.report(self.logger)
        
        return deduplicated
    
    def _save_output(self, df: DataFrame, output_path: str, output_format: str):
        """Save output DataFrame to specified path and format."""
        try:
            # Create output directory if it doesn't exist
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format.lower() == "parquet":
                df.coalesce(1).write.mode("overwrite").parquet(output_path)
            elif output_format.lower() == "json":
                df.coalesce(1).write.mode("overwrite").json(output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        except Exception as e:
            self.logger.error(f"Failed to save output: {str(e)}")
            raise
    
    def stop(self):
        """Stop the Spark session."""
        if hasattr(self, 'spark'):
            self.spark.stop()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Spark-based MinHash text deduplication pipeline")
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", required=True,
                       help="Input file path (JSONL/Parquet/JSON)")
    parser.add_argument("--output", "-o", default="deduplicated_dataset",
                       help="Output directory path")
    parser.add_argument("--output-format", choices=["parquet", "json"], default="parquet",
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
    
    # Spark configuration
    parser.add_argument("--spark-master", default="local[*]",
                       help="Spark master URL")
    parser.add_argument("--spark-memory", default="4g",
                       help="Spark executor memory")
    parser.add_argument("--spark-cores", type=int, default=None,
                       help="Number of cores per executor")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Prepare Spark configuration
    spark_config = {
        "spark.master": args.spark_master,
        "spark.executor.memory": args.spark_memory,
    }
    
    if args.spark_cores:
        spark_config["spark.executor.cores"] = str(args.spark_cores)
    
    # Initialize deduplicator
    try:
        logger.info("Initializing MinHash deduplicator...")
        deduplicator = SparkMinHashDeduplicator(
            threshold=args.threshold,
            num_perm=args.num_perm,
            ngram_size=args.ngram_size,
            min_length=args.min_length,
            hash_bits=args.hash_bits,
            hash_func=args.hash_func,
            b=args.b,
            r=args.r,
            spark_config=spark_config
        )
        
        logger.info("Starting deduplication process...")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Text column: {args.text_column}")
        logger.info(f"Threshold: {args.threshold}")
        logger.info(f"Number of permutations: {args.num_perm}")
        logger.info(f"N-gram size: {args.ngram_size}")
        logger.info(f"Minimum length: {args.min_length}")
        logger.info(f"Hash bits: {args.hash_bits}")
        logger.info(f"Hash function: {args.hash_func}")
        
        # Run deduplication
        deduplicated_df = deduplicator.deduplicate(
            input_path=args.input,
            text_column=args.text_column,
            output_path=args.output,
            output_format=args.output_format
        )
        
        logger.info("Deduplication completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during deduplication: {str(e)}")
        raise
    finally:
        # Clean up Spark session
        if 'deduplicator' in locals():
            deduplicator.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
