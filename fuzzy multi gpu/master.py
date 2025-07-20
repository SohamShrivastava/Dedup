"""
Master orchestrator for the complete fuzzy deduplication pipeline.
This file manages the lifecycle of: Dask cluster → MinHash → LSH → Bucketing → Connected Components → Deduplication → Removal
"""

import os
import sys
import argparse
import signal
import time
from typing import List, Union, Optional
import cudf
import dask_cudf
import dask

# Import from your existing files
from dask_manager import DaskClusterManager
from minhash_multi import MinHashGenerator
from lsh_multi import SimpleLSH
from buckets_edges import BucketsToEdges
from connected_components_multi import multi_gpu_connected_components


class FuzzyDeduplicationOrchestrator:
    def __init__(self):
        self.cluster_manager = None
        self.minhash_generator = None
        self.lsh_processor = None
        self.buckets_to_edges_processor = None
        self.scheduler_address = None
        self._minhash_initialized = False
        self._lsh_initialized = False
        self._buckets_to_edges_initialized = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Shutting down...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_cluster(self, 
                     cuda_devices: str = "0,1",
                     rmm_pool_size: float = 0.5,
                     enable_cudf_spill: bool = True,
                     local_directory: str = "/tmp/dask-scratch",
                     use_multi_gpu: bool = True,
                     device_memory_limit: str = "auto",
                     threads_per_worker: int = 1,
                     processes: bool = True) -> str:
        """Start Dask cluster using DaskClusterManager"""
        
        print("=" * 60)
        print("STEP 1: Starting Dask cluster...")
        print("=" * 60)
        
        self.cluster_manager = DaskClusterManager(
            cuda_visible_devices=cuda_devices,
            rmm_pool_size=rmm_pool_size,
            enable_cudf_spill=enable_cudf_spill,
            local_directory=local_directory,
            use_multi_gpu=use_multi_gpu,
            device_memory_limit=device_memory_limit,
            threads_per_worker=threads_per_worker,
            processes=processes,
        )
        dask.config.set({
        	'dataframe.shuffle.method': 'tasks',
        	'dataframe.optimization.split-large-chunks': False,
        	'dataframe.chunk-size': '1024MB',  # Larger chunks = fewer partitions
                'dataframe.query-planning': False,
        })        
        self.scheduler_address = self.cluster_manager.start_cluster()
        print(f"✓ Cluster started successfully at: {self.scheduler_address}")
        
        return self.scheduler_address
    
    def initialize_minhash_generator(self,
                                   seed: int = 42,
                                   num_hashes: int = 260,
                                   char_ngrams: int = 24,
                                   use_64bit_hash: bool = False,
                                   column_field: str = "text") -> None:
        """Initialize the MinHash generator"""
        
        if self._minhash_initialized:
            print("✓ MinHash generator already initialized, reusing...")
            return
        
        print("=" * 60)
        print("STEP 2: Initializing MinHash generator...")
        print("=" * 60)
        print(f"Seed: {seed}")
        print(f"Number of hashes: {num_hashes}")
        print(f"Character n-grams: {char_ngrams}")
        print(f"Use 64-bit hash: {use_64bit_hash}")
        print(f"Column field: {column_field}")
        
        self.minhash_generator = MinHashGenerator(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
            column_field=column_field
        )
        
        self._minhash_initialized = True
        print("✓ MinHash generator initialized successfully!")
    
    def compute_minhashes(self,
                         input_path: str,
                         output_path: str) -> None:
        """Compute MinHash signatures using the pre-initialized MinHashGenerator"""
        
        if not self._minhash_initialized or not self.minhash_generator:
            raise RuntimeError("MinHash generator not initialized. Call initialize_minhash_generator() first.")
        
        print("=" * 60)
        print("STEP 3: Computing MinHash signatures...")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        # Run the minhash computation
        self.minhash_generator(input_path, output_path)
        print(f"✓ MinHash signatures computed successfully!")
    
    def initialize_lsh_processor(self,
                               num_hashes: int = 260,
                               num_buckets: int = 13,
                               buckets_per_shuffle: int = 1,
                               id_field: str = "id",
                               minhash_field: str = "minhash_signature") -> None:
        """Initialize the LSH processor"""
        
        if self._lsh_initialized:
            print("✓ LSH processor already initialized, reusing...")
            return
        
        print("=" * 60)
        print("STEP 4: Initializing LSH processor...")
        print("=" * 60)
        print(f"Number of hashes: {num_hashes}")
        print(f"Number of buckets: {num_buckets}")
        print(f"Buckets per shuffle: {buckets_per_shuffle}")
        print(f"ID field: {id_field}")
        print(f"MinHash field: {minhash_field}")
        
        self.lsh_processor = SimpleLSH(
            num_hashes=num_hashes,
            num_buckets=num_buckets,
            buckets_per_shuffle=buckets_per_shuffle,
            id_field=id_field,
            minhash_field=minhash_field
        )
        
        self._lsh_initialized = True
        print("✓ LSH processor initialized successfully!")
    
    def compute_lsh_buckets(self,
                          input_path: str,
                          output_path: str) -> None:
        """Compute LSH buckets using the pre-initialized LSH processor"""
        
        if not self._lsh_initialized or not self.lsh_processor:
            raise RuntimeError("LSH processor not initialized. Call initialize_lsh_processor() first.")
        
        print("=" * 60)
        print("STEP 5: Computing LSH buckets...")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        # Run the LSH computation
        result_df = self.lsh_processor(input_path, output_path)
        print(f"✓ LSH buckets computed successfully!")
        print(f"✓ Result DataFrame shape: {result_df.compute().shape}")
    
    def initialize_buckets_to_edges_processor(self,
                                            id_field: str = "id",
                                            bucket_field: str = "lsh_bucket") -> None:
        """Initialize the BucketsToEdges processor"""
        
        if self._buckets_to_edges_initialized:
            print("✓ BucketsToEdges processor already initialized, reusing...")
            return
        
        print("=" * 60)
        print("STEP 6: Initializing BucketsToEdges processor...")
        print("=" * 60)
        print(f"ID field: {id_field}")
        print(f"Bucket field: {bucket_field}")
        
        self.buckets_to_edges_processor = BucketsToEdges(
            id_field=id_field,
            bucket_field=bucket_field
        )
        
        self._buckets_to_edges_initialized = True
        print("✓ BucketsToEdges processor initialized successfully!")
    
    def compute_edges(self,
                     input_path: str,
                     output_path: str) -> None:
        """Convert LSH buckets to edges using the pre-initialized BucketsToEdges processor"""
        
        if not self._buckets_to_edges_initialized or not self.buckets_to_edges_processor:
            raise RuntimeError("BucketsToEdges processor not initialized. Call initialize_buckets_to_edges_processor() first.")
        
        print("=" * 60)
        print("STEP 7: Converting buckets to edges...")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        # Run the buckets to edges computation
        result_df = self.buckets_to_edges_processor(input_path, output_path)
        print(f"✓ Edges computed successfully!")
        print(f"✓ Result DataFrame shape: {result_df.compute().shape}")
    
    def compute_connected_components(self,
                                   edges_path: str,
                                   output_path: str,
                                   left_id_col: str = "id_x",
                                   right_id_col: str = "id_y",
                                   jaccard_col: str = "jaccard",
                                   jaccard_threshold: float = 0.8) -> cudf.DataFrame:
        """Run connected components using the multi-GPU implementation"""
        
        print("=" * 60)
        print("STEP 8: Computing connected components...")
        print("=" * 60)
        print(f"Input: {edges_path}")
        print(f"Output: {output_path}")
        print(f"Left ID column: {left_id_col}")
        print(f"Right ID column: {right_id_col}")
        print(f"Jaccard column: {jaccard_col}")
        print(f"Jaccard threshold: {jaccard_threshold}")
        
        # Use the multi-GPU connected components function
        result_df = multi_gpu_connected_components(
            edges_path=edges_path,
            left_id_col=left_id_col,
            right_id_col=right_id_col,
            jaccard_col=jaccard_col,
            jaccard_threshold=jaccard_threshold,
            output_path=output_path
        )
        
        print(f"✓ Connected components computed successfully!")
        print(f"✓ Result DataFrame shape: {result_df.shape}")
        print(f"✓ Total nodes: {len(result_df)}")
        print(f"✓ Total groups: {result_df['group_id'].nunique()}")
        
        return result_df
    
    def cleanup(self):
        """Clean up resources"""
        print("=" * 60)
        print("CLEANUP: Shutting down resources...")
        print("=" * 60)
        
        if self.buckets_to_edges_processor:
            self.buckets_to_edges_processor = None
            self._buckets_to_edges_initialized = False
            print("✓ BucketsToEdges processor cleaned up")
        
        if self.lsh_processor:
            self.lsh_processor = None
            self._lsh_initialized = False
            print("✓ LSH processor cleaned up")
        
        if self.minhash_generator:
            self.minhash_generator = None
            self._minhash_initialized = False
            print("✓ MinHash generator cleaned up")
            
        if self.cluster_manager:
            self.cluster_manager.close()
            self.cluster_manager = None
            print("✓ Dask cluster closed")
            
        print("✓ Cleanup completed.")
    
    def run_full_pipeline(self,
                         input_path: str,
                         output_directory: str,
                         # Data processing args
                         data_column: str = "text",
                         id_field: str = "id",
                         # Cluster args
                         cuda_devices: str = "0,1",
                         rmm_pool_size: float = 0.5,
                         enable_cudf_spill: bool = True,
                         local_directory: str = "/tmp/dask-scratch",
                         use_multi_gpu: bool = True,
                         device_memory_limit: str = "auto",
                         threads_per_worker: int = 1,
                         processes: bool = True,
                         # MinHash args
                         seed: int = 42,
                         num_hashes: int = 260,
                         char_ngrams: int = 24,
                         use_64bit_hash: bool = False,
                         # LSH args
                         num_buckets: int = 13,
                         buckets_per_shuffle: int = 1,
                         # BucketsToEdges args
                         bucket_field: str = "lsh_bucket",
                         # Connected Components args
                         jaccard_threshold: float = 0.8,
                         run_connected_components: bool = True) -> Optional[cudf.DataFrame]:
        """Run the complete pipeline: cluster → minhash → LSH → buckets to edges → connected components"""
        
        start_time = time.time()
        
        try:
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            print("🚀 Starting Complete Fuzzy Deduplication Pipeline")
            print("=" * 80)
            
            # Create main output directory
            os.makedirs(output_directory, exist_ok=True)
            
            # Define intermediate file paths
            minhash_output_path = os.path.join(output_directory, "minhash_signatures.parquet")
            lsh_output_path = os.path.join(output_directory, "lsh_buckets.parquet")
            edges_output_path = os.path.join(output_directory, "edges.parquet")
            connected_components_output_path = os.path.join(output_directory, "connected_components.parquet")
            
            # Step 1: Start cluster
            self.start_cluster(
                cuda_devices=cuda_devices,
                rmm_pool_size=rmm_pool_size,
                enable_cudf_spill=enable_cudf_spill,
                local_directory=local_directory,
                use_multi_gpu=use_multi_gpu,
                device_memory_limit=device_memory_limit,
                threads_per_worker=threads_per_worker,
                processes=processes,
            )
            
            # Step 2: Initialize MinHash generator
            self.initialize_minhash_generator(
                seed=seed,
                num_hashes=num_hashes,
                char_ngrams=char_ngrams,
                use_64bit_hash=use_64bit_hash,
                column_field=data_column,
            )
            
            # Step 3: Compute MinHash signatures
            self.compute_minhashes(
                input_path=input_path,
                output_path=minhash_output_path,
            )
            
            # Step 4: Initialize LSH processor
            self.initialize_lsh_processor(
                num_hashes=num_hashes,
                num_buckets=num_buckets,
                buckets_per_shuffle=buckets_per_shuffle,
                id_field=id_field,
                minhash_field="minhash_signature"
            )
            
            # Step 5: Compute LSH buckets
            self.compute_lsh_buckets(
                input_path=minhash_output_path,
                output_path=lsh_output_path,
            )
            
            # Step 6: Initialize BucketsToEdges processor
            self.initialize_buckets_to_edges_processor(
                id_field=id_field,
                bucket_field=bucket_field
            )
            
            # Step 7: Convert buckets to edges
            self.compute_edges(
                input_path=lsh_output_path,
                output_path=edges_output_path,
            )
            
            # Step 8: Run connected components (optional)
            connected_components_result = None
            if run_connected_components:
                connected_components_result = self.compute_connected_components(
                    edges_path=edges_output_path,
                    output_path=connected_components_output_path,
                    left_id_col=f"{id_field}_x",
                    right_id_col=f"{id_field}_y",
                    jaccard_col="jaccard",
                    jaccard_threshold=jaccard_threshold
                )
            
            # Pipeline completed - Print summary
            total_time = time.time() - start_time
            print("=" * 80)
            print("🎉 COMPLETE FUZZY DEDUPLICATION PIPELINE COMPLETED!")
            print("=" * 80)
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Input: {input_path}")
            print(f"Output directory: {output_directory}")
            print(f"MinHash signatures: {minhash_output_path}")
            print(f"LSH buckets: {lsh_output_path}")
            print(f"Edges: {edges_output_path}")
            
            if run_connected_components and connected_components_result is not None:
                print(f"Connected components: {connected_components_output_path}")
                print(f"Total nodes processed: {len(connected_components_result)}")
                print(f"Total duplicate groups found: {connected_components_result['group_id'].nunique()}")
                
                # Calculate deduplication statistics
                single_node_groups = connected_components_result.groupby('group_id').size() == 1
                duplicate_groups = connected_components_result.groupby('group_id').size() > 1
                
                print(f"Single-node groups (no duplicates): {single_node_groups.sum()}")
                print(f"Multi-node groups (duplicates): {duplicate_groups.sum()}")
                
                # Calculate total duplicates that could be removed
                group_sizes = connected_components_result.groupby('group_id').size()
                duplicates_to_remove = (group_sizes - 1).sum()
                print(f"Total duplicates that can be removed: {duplicates_to_remove}")
                #print(f"Deduplication rate: {duplicates_to_remove/len(connected_components_result)*100:.2f}%")
            
            return connected_components_result
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Master Fuzzy Deduplication Pipeline")
    
    # Required arguments
    parser.add_argument("input_path", help="Path to input parquet/jsonl file or directory")
    parser.add_argument("output_directory", help="Directory to save all results")
    
    # Data processing configuration
    parser.add_argument("--data-column", default="text", help="Name of data column (default: text)")
    parser.add_argument("--id-field", default="id", help="Name of ID field (default: id)")
    
    # Cluster configuration
    parser.add_argument("--cuda-devices", default="0,1", help="CUDA devices to use (default: 0,1)")
    parser.add_argument("--rmm-pool-size", type=float, default=0.5, help="RMM pool size (default: 0.5)")
    parser.add_argument("--enable-cudf-spill", action="store_true", default=True, help="Enable cuDF spilling")
    parser.add_argument("--local-directory", default="/tmp/dask-scratch", help="Local directory for spilling")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Use single GPU instead of multi-GPU")
    parser.add_argument("--device-memory-limit", default="auto", help="Device memory limit (default: auto)")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per worker (default: 1)")
    parser.add_argument("--no-processes", action="store_true", help="Use threads instead of processes")
    
    # MinHash configuration
    parser.add_argument("--seed", type=int, default=42, help="Seed for minhash permutations (default: 42)")
    parser.add_argument("--num-hashes", type=int, default=260, help="Number of hash permutations (default: 260)")
    parser.add_argument("--char-ngrams", type=int, default=24, help="Character n-gram width (default: 24)")
    parser.add_argument("--use-64bit-hash", action="store_true", help="Use 64-bit hash instead of 32-bit")
    
    # LSH configuration
    parser.add_argument("--num-buckets", type=int, default=13, help="Number of LSH buckets (default: 13)")
    parser.add_argument("--buckets-per-shuffle", type=int, default=1, help="Number of buckets to shuffle concurrently (default: 1)")
    
    # BucketsToEdges configuration
    parser.add_argument("--bucket-field", default="_bucket_id", help="Name of the bucket field (default: _bucket_id)")
    
    # Connected Components configuration
    parser.add_argument("--jaccard-threshold", type=float, default=0.8, help="Jaccard threshold for connected components (default: 0.8)")
    parser.add_argument("--skip-connected-components", action="store_true", help="Skip connected components computation")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = FuzzyDeduplicationOrchestrator()
    
    # Run the pipeline
    result = orchestrator.run_full_pipeline(
        input_path=args.input_path,
        output_directory=args.output_directory,
        # Data processing args
        data_column=args.data_column,
        id_field=args.id_field,
        # Cluster args
        cuda_devices=args.cuda_devices,
        rmm_pool_size=args.rmm_pool_size,
        enable_cudf_spill=args.enable_cudf_spill,
        local_directory=args.local_directory,
        use_multi_gpu=not args.no_multi_gpu,
        device_memory_limit=args.device_memory_limit,
        threads_per_worker=args.threads_per_worker,
        processes=not args.no_processes,
        # MinHash args
        seed=args.seed,
        num_hashes=args.num_hashes,
        char_ngrams=args.char_ngrams,
        use_64bit_hash=args.use_64bit_hash,
        # LSH args
        num_buckets=args.num_buckets,
        buckets_per_shuffle=args.buckets_per_shuffle,
        # BucketsToEdges args
        bucket_field=args.bucket_field,
        # Connected Components args
        jaccard_threshold=args.jaccard_threshold,
        run_connected_components=not args.skip_connected_components,
    )


if __name__ == "__main__":
    main()
