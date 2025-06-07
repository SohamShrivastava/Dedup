#!/usr/bin/env python3
"""
Name: dedup.py
Main orchestration class for GPU-accelerated text deduplication pipeline.
Handles complete workflow from data loading to deduplication results.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cudf
import cupy as cp

from config import DeduplicationConfig, create_config
from data_handler import DataHandler
from minhash_engine import MinHashEngine
from lsh_clustering import LSHClusteringEngine


class TextDeduplicator:
    """Main text deduplication pipeline orchestrator."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize pipeline components
        self.data_handler = DataHandler(self.logger)
        self.minhash_engine = MinHashEngine(config, self.logger)
        self.clustering_engine = LSHClusteringEngine(config, self.logger)
        
        self.logger.info("GPU Text Deduplicator initialized")
        self._log_gpu_info()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('deduplication.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _log_gpu_info(self):
        """Log GPU information."""
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            current_device = cp.cuda.Device()
            memory_info = cp.cuda.MemoryInfo()
            
            self.logger.info(f"GPU Info: {gpu_count} device(s) available")
            self.logger.info(f"Using GPU {current_device.id}")
            self.logger.info(f"GPU Memory: {memory_info.total / (1024**3):.1f} GB total, "
                           f"{memory_info.free / (1024**3):.1f} GB free")
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")
    
    def run_deduplication(self, 
                         input_path: Optional[str] = None,
                         output_path: Optional[str] = None,
                         text_column: str = "text",
                         return_clustered: bool = False) -> cudf.DataFrame:
        """
        Run complete deduplication pipeline.
        
        Parameters
        ----------
        input_path : Optional[str]
            Path to input file
        output_path : Optional[str]
            Path to output file
        text_column : str
            Name of text column
        return_clustered : bool
            If True, return clustered data instead of deduplicated
            
        Returns
        -------
        cudf.DataFrame
            Processed DataFrame
        """
        start_time = time.time()
        self.logger.info("="*60)
        self.logger.info("STARTING TEXT DEDUPLICATION PIPELINE")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1/4: Loading and preparing data...")
            df = self.data_handler.load_and_prepare(
                input_path=input_path,
                text_column=text_column,
                min_length=self.config.min_text_length,
                max_length=self.config.max_text_length,
                clean_text=True
            )
            
            # Step 2: Generate MinHash signatures
            self.logger.info("Step 2/4: Generating MinHash signatures...")
            df = self.minhash_engine.generate_signatures(df, text_column)
            
            # Step 3: Perform clustering
            self.logger.info("Step 3/4: Performing LSH clustering...")
            if return_clustered:
                result_df = self.clustering_engine.cluster_documents(df, text_column)
            else:
                result_df = self.clustering_engine.deduplicate(df, text_column)
            
            # Step 4: Save results
            if output_path:
                self.logger.info("Step 4/4: Saving results...")
                output_format = Path(output_path).suffix[1:] or "parquet"
                self.data_handler.save_results(result_df, output_path, output_format)
            
            # Log final statistics
            self._log_final_stats(df, result_df, time.time() - start_time)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self._cleanup()
    
    def _log_final_stats(self, original_df: cudf.DataFrame, result_df: cudf.DataFrame, 
                        duration: float):
        """Log final pipeline statistics."""
        original_count = len(original_df)
        final_count = len(result_df)
        reduction_rate = (original_count - final_count) / original_count * 100
        
        self.logger.info("="*60)
        self.logger.info("DEDUPLICATION PIPELINE COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Original documents: {original_count:,}")
        self.logger.info(f"Final documents: {final_count:,}")
        self.logger.info(f"Reduction: {original_count - final_count:,} ({reduction_rate:.1f}%)")
        self.logger.info(f"Processing time: {duration:.2f} seconds")
        self.logger.info(f"Throughput: {original_count / duration:.0f} docs/second")
    
    def _cleanup(self):
        """Cleanup GPU memory."""
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            self.logger.warning(f"GPU cleanup warning: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated text deduplication using MinHash and LSH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input file path (JSONL, Parquet, or JSON)")
    parser.add_argument("--output", "-o", required=True,
                       help="Output file path")
    parser.add_argument("--text-column", default="text",
                       help="Name of the text column")
    
    # MinHash parameters
    parser.add_argument("--num-perm", type=int, default=128,
                       help="Number of MinHash permutations")
    parser.add_argument("--ngram-size", type=int, default=1,
                       help="N-gram size for tokenization")
    parser.add_argument("--hash-func", choices=['xxh3', 'murmur3', 'sha1'], 
                       default='xxh3', help="Hash function to use")
    parser.add_argument("--hash-bits", type=int, choices=[32, 64], default=32,
                       help="Number of hash bits")
    
    # LSH parameters
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Jaccard similarity threshold")
    parser.add_argument("--b", type=int, help="Number of LSH bands (auto-calculated if not set)")
    parser.add_argument("--r", type=int, help="Number of rows per band (auto-calculated if not set)")
    
    # Filtering parameters
    parser.add_argument("--min-length", type=int, default=0,
                       help="Minimum text length in characters")
    parser.add_argument("--max-length", type=int,
                       help="Maximum text length in characters")
    
    # Processing parameters
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Batch size for processing")
    
    # Output options
    parser.add_argument("--output-format", choices=['jsonl', 'parquet', 'json'],
                       help="Output format (auto-detected from extension if not set)")
    parser.add_argument("--return-clustered", action="store_true",
                       help="Return clustered data instead of deduplicated")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_config(
            num_perm=args.num_perm,
            ngram_size=args.ngram_size,
            hash_func=args.hash_func,
            hash_bits=args.hash_bits,
            threshold=args.threshold,
            b=args.b,
            r=args.r,
            min_text_length=args.min_length,
            max_text_length=args.max_length,
            batch_size=args.batch_size
        )
        
        # Initialize deduplicator
        deduplicator = TextDeduplicator(config)
        
        # Run deduplication
        result = deduplicator.run_deduplication(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            return_clustered=args.return_clustered
        )
        
        print(f"\nDeduplication completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Final document count: {len(result):,}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()