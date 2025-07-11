
"""
Master orchestrator for the complete semantic deduplication pipeline with language splitting.
This file manages the lifecycle of: Dask cluster → Language Split → For each language: Embedding → Clustering → Deduplication → Removal
"""

import os
import sys
import argparse
import signal
import time
from typing import List, Union, Optional, Dict
import cudf

# Import from your existing files
from dask_manager import DaskClusterManager
from embedding_multi import EmbeddingCreator
from clustering_multi import ParquetEmbeddingClusterer
from semdedup_multi import GPUSemanticDeduplicator
from add_id_by_lang  import split_by_language_with_ids  # Import the language splitting function

class SemanticDeduplicationOrchestrator:
    def __init__(self):
        self.cluster_manager = None
        self.embedding_creator = None  # Will be initialized once and reused
        self.clusterer = None
        self.deduplicator = None
        self.scheduler_address = None
        self._embedding_model_loaded = False  # Track if model is loaded
        
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
        
        self.scheduler_address = self.cluster_manager.start_cluster()
        print(f"✓ Cluster started successfully at: {self.scheduler_address}")
        
        return self.scheduler_address
    
    def split_by_language(self,
                         input_path: str,
                         output_directory: str,
                         data_column: str = "data",
                         lang_column: str = "lang",
                         id_column: str = "id") -> Dict[str, str]:
        """Split input data by language and add sequential IDs"""
        
        print("=" * 60)
        print("STEP 2: Splitting data by language...")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output directory: {output_directory}")
        print(f"Data column: {data_column}")
        print(f"Language column: {lang_column}")
        print(f"ID column: {id_column}")
        
        # Create language split directory
        lang_split_dir = os.path.join(output_directory, "language_splits")
        os.makedirs(lang_split_dir, exist_ok=True)
        
        # Split by language
        language_files = split_by_language_with_ids(
            input_parquet_path=input_path,
            output_directory=lang_split_dir,
            data_column=data_column,
            lang_column=lang_column,
            id_column=id_column
        )
        
        print(f"✓ Data split into {len(language_files)} language files")
        return language_files
    
    def initialize_embedding_creator(self,
                                   model: str = "intfloat/multilingual-e5-large-instruct",
                                   batch_size: int = 1,
                                   pooling_strategy: str = "mean_pooling",
                                   input_column: str = "data",
                                   embedding_column: str = "embeddings",
                                   partition_size: str = "100MiB",
                                   max_seq_length: int = 512,
                                   embedding_max_mem_gb: int = 30) -> None:
        """Initialize the embedding creator once for all languages"""
        
        if self._embedding_model_loaded:
            print("✓ Embedding model already loaded, reusing...")
            return
            
        if not self.scheduler_address:
            raise RuntimeError("Cluster not started. Call start_cluster() first.")
        
        print("=" * 60)
        print("INITIALIZING EMBEDDING MODEL (ONE-TIME)")
        print("=" * 60)
        print(f"Model: {model}")
        print(f"Batch size: {batch_size}")
        print(f"Pooling strategy: {pooling_strategy}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Max memory per worker: {embedding_max_mem_gb}GB")
        
        self.embedding_creator = EmbeddingCreator(
            scheduler_address=self.scheduler_address,
            embedding_model_name_or_path=model,
            embedding_batch_size=batch_size,
            embedding_pooling_strategy=pooling_strategy,
            input_column=input_column,
            embedding_column=embedding_column,
            embedding_max_mem_gb=embedding_max_mem_gb,
            partition_size=partition_size,
            max_seq_length=max_seq_length,
        )
        
        self._embedding_model_loaded = True
        print("✓ Embedding model initialized successfully!")
    
    def create_embeddings(self,
                         input_path: Union[str, List[str]],
                         output_path: str,
                         language: str = "unknown") -> None:
        """Create embeddings using the pre-initialized EmbeddingCreator"""
        
        if not self._embedding_model_loaded or not self.embedding_creator:
            raise RuntimeError("Embedding creator not initialized. Call initialize_embedding_creator() first.")
        
        print(f"📊 Creating embeddings for {language}...")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        print("   Using pre-loaded model...")
        
        # Run the embedding creation using the existing creator
        self.embedding_creator(input_path, output_path)
        print(f"✓ Embeddings created successfully for {language}!")
    
    def run_clustering(self,
                      input_file: str,
                      output_dir: str,
                      language: str = "unknown",
                      n_clusters: int = 1000,
                      max_iter: int = 200,
                      id_column: str = "id",
                      embedding_column: str = "embeddings",
                      partition_size: str = "4GB",
                      random_state: int = 42) -> None:
        """Run clustering using ParquetEmbeddingClusterer"""
        
        print(f"🔍 Clustering embeddings for {language}...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_dir}")
        print(f"   Clusters: {n_clusters}")
        print(f"   Max iterations: {max_iter}")
        
        self.clusterer = ParquetEmbeddingClusterer(
            id_column=id_column,
            embedding_column=embedding_column,
            max_iter=max_iter,
            n_clusters=n_clusters,
            clustering_output_dir=output_dir,
            random_state=random_state,
            clustering_input_partition_size=partition_size,
        )
        
        # Run clustering
        clustered_df = self.clusterer(input_file)
        print(f"✓ Clustering completed successfully for {language}!")
        print(f"✓ Results saved to: {output_dir}")
    
    def run_deduplication(self,
                         clustering_results_dir: str,
                         eps: float,
                         language: str = "unknown",
                         output_dir: str = "./deduplication_results",
                         id_column: str = "id",
                         embedding_column: str = "embeddings",
                         which_to_keep: str = "hard",
                         sim_metric: str = "cosine",
                         batch_size: int = 1024,
                         compute_only: bool = False) -> Optional[str]:
        """Run semantic deduplication using GPUSemanticDeduplicator"""
        
        print(f"🔄 Running semantic deduplication for {language}...")
        print(f"   Clustering results: {clustering_results_dir}")
        print(f"   Epsilon: {eps}")
        print(f"   Output: {output_dir}")
        print(f"   Strategy: {which_to_keep}")
        print(f"   Metric: {sim_metric}")
        
        self.deduplicator = GPUSemanticDeduplicator(
            clustering_results_dir=clustering_results_dir,
            id_column=id_column,
            embedding_column=embedding_column,
            which_to_keep=which_to_keep,
            sim_metric=sim_metric,
            output_dir=output_dir,
            batched_cosine_similarity=batch_size,
        )
        
        # Step 1: Compute semantic match dataframes
        print(f"   Computing semantic match dataframes...")
        self.deduplicator.compute_semantic_match_dfs()
        
        if not compute_only:
            # Step 2: Extract documents to remove at given epsilon
            print(f"   Extracting documents to remove with epsilon={eps}...")
            documents_to_remove = self.deduplicator.extract_dedup_data(eps)
            
            duplicates_file = f"{output_dir}/unique_ids_{eps}.parquet"
            print(f"✓ Documents to remove saved at: {duplicates_file}")
            
            return duplicates_file
        else:
            print(f"✓ Semantic match computation complete for {language}.")
            return None
    
    def remove_duplicates(self,
                        original_path: str,
                        dedup_path: str,
                        output_path: str,
                        language: str = "unknown",
                        id_column: str = "id") -> None:
        """Remove duplicates from original dataset based on deduplication results"""
        
        print(f"🗑️  Removing duplicates for {language}...")
        print(f"   Original: {original_path}")
        print(f"   Duplicates: {dedup_path}")
        print(f"   Output: {output_path}")
        
        try:
            # Read original dataset (all columns)
            print("   📖 Reading original dataset...")
            original_df = cudf.read_parquet(original_path)
            
            # Extract IDs to remove (fault-tolerant approach)
            print("   📖 Extracting IDs to remove...")
            ids_to_remove = set()
            
            try:
                # Try to read the parquet file in chunks to handle schema issues
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(dedup_path)
                
                # Read each row group separately to handle schema inconsistencies
                for i in range(parquet_file.num_row_groups):
                    row_group = parquet_file.read_row_group(i, columns=[id_column])
                    ids_batch = row_group[id_column].to_pylist()
                    ids_to_remove.update(ids_batch)
                    
            except Exception as e:
                print(f"   ⚠️  Row group approach failed: {e}")
                print("   🔄 Trying alternative approach...")
                
                # Fallback: try reading entire file and extract IDs
                try:
                    # Use pyarrow to read just the ID column
                    import pyarrow.parquet as pq
                    table = pq.read_table(dedup_path, columns=[id_column])
                    ids_to_remove = set(table[id_column].to_pylist())
                except Exception as e2:
                    print(f"   ❌ Could not extract IDs: {e2}")
                    raise e2
            
            print(f"   🗑️  Found {len(ids_to_remove)} unique IDs to remove")
            
            # Convert to cuDF series for efficient filtering
            ids_to_remove_series = cudf.Series(list(ids_to_remove))
            
            # Filter using GPU anti-join (keep rows NOT in removal set)
            print("   🔧 Filtering dataset on GPU...")
            keep_mask = ~original_df[id_column].isin(ids_to_remove_series)
            cleaned_df = original_df[keep_mask]
            
            # Save results
            print("   💾 Saving cleaned dataset...")
            cleaned_df.to_parquet(output_path, index=False)
            
            # Print summary
            original_count = len(original_df)
            cleaned_count = len(cleaned_df)
            removed_count = original_count - cleaned_count
            
            print(f"   📊 Summary for {language}:")
            print(f"      Original: {original_count:,} documents")
            print(f"      Removed:  {removed_count:,} duplicates")
            print(f"      Final:    {cleaned_count:,} documents")
            print(f"      Reduction: {(removed_count/original_count)*100:.2f}%")
            print(f"✓ Cleaned dataset saved for {language}: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error in remove_duplicates for {language}: {str(e)}")
            raise e
    
    def process_single_language(self,
                               language: str,
                               input_file: str,
                               base_output_dir: str,
                               eps: float,
                               # Clustering args
                               n_clusters: int = 1000,
                               max_iter: int = 200,
                               id_column: str = "id",
                               embedding_column: str = "embeddings",
                               clustering_partition_size: str = "4GB",
                               random_state: int = 42,
                               # Deduplication args
                               which_to_keep: str = "hard",
                               sim_metric: str = "cosine",
                               dedup_batch_size: int = 1024,
                               compute_only: bool = False) -> Optional[str]:
        """Process a single language through the complete pipeline (embeddings already initialized)"""
        
        print(f"\n🌍 Processing language: {language.upper()}")
        print("=" * 60)
        
        # Create language-specific directories
        lang_dir = os.path.join(base_output_dir, f"{language}_results")
        os.makedirs(lang_dir, exist_ok=True)
        
        embeddings_output = os.path.join(lang_dir, f"{language}_embeddings.parquet")
        clustering_output = os.path.join(lang_dir, f"{language}_clustering")
        deduplication_output = os.path.join(lang_dir, f"{language}_deduplication")
        cleaned_output = os.path.join(base_output_dir, f"{language}_cleaned_eps_{eps}.parquet")
        
        try:
            # Step 1: Create embeddings (using pre-initialized model)
            self.create_embeddings(
                input_path=input_file,
                output_path=embeddings_output,
                language=language
            )
            
            # Step 2: Run clustering
            self.run_clustering(
                input_file=embeddings_output,
                output_dir=clustering_output,
                language=language,
                n_clusters=n_clusters,
                max_iter=max_iter,
                id_column=id_column,
                embedding_column=embedding_column,
                partition_size=clustering_partition_size,
                random_state=random_state,
            )
            
            # Step 3: Run deduplication
            duplicates_file = self.run_deduplication(
                clustering_results_dir=clustering_output,
                eps=eps,
                language=language,
                output_dir=deduplication_output,
                id_column=id_column,
                embedding_column=embedding_column,
                which_to_keep=which_to_keep,
                sim_metric=sim_metric,
                batch_size=dedup_batch_size,
                compute_only=compute_only,
            )
            
            # Step 4: Remove duplicates (if not compute_only)
            if not compute_only and duplicates_file:
                self.remove_duplicates(
                    original_path=input_file,
                    dedup_path=duplicates_file,
                    output_path=cleaned_output,
                    language=language,
                    id_column=id_column,
                )
                
                print(f"✅ {language.upper()} processing completed successfully!")
                return cleaned_output
            else:
                print(f"✅ {language.upper()} semantic match computation completed!")
                return None
                
        except Exception as e:
            print(f"❌ Error processing {language}: {str(e)}")
            print(f"⚠️  Continuing with next language...")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        print("=" * 60)
        print("CLEANUP: Shutting down resources...")
        print("=" * 60)
        
        if self.embedding_creator:
            self.embedding_creator.close()
            self.embedding_creator = None
            self._embedding_model_loaded = False
            print("✓ Embedding creator closed")
            
        if self.cluster_manager:
            self.cluster_manager.close()
            self.cluster_manager = None
            print("✓ Dask cluster closed")
            
        print("✓ Cleanup completed.")
    
    def run_full_pipeline_with_languages(self,
                                       input_path: str,
                                       output_directory: str,
                                       # Language splitting args
                                       data_column: str = "data",
                                       lang_column: str = "lang",
                                       id_column: str = "id",
                                       # Cluster args
                                       cuda_devices: str = "0,1",
                                       rmm_pool_size: float = 0.5,
                                       enable_cudf_spill: bool = True,
                                       local_directory: str = "/tmp/dask-scratch",
                                       use_multi_gpu: bool = True,
                                       device_memory_limit: str = "auto",
                                       threads_per_worker: int = 1,
                                       processes: bool = True,
                                       # Embedding args
                                       model: str = "intfloat/multilingual-e5-large-instruct",
                                       batch_size: int = 1,
                                       pooling_strategy: str = "mean_pooling",
                                       partition_size: str = "100MiB",
                                       max_seq_length: int = 512,
                                       embedding_max_mem_gb: int = 30,
                                       # Clustering args
                                       n_clusters: int = 1000,
                                       max_iter: int = 200,
                                       clustering_partition_size: str = "4GB",
                                       random_state: int = 42,
                                       # Deduplication args
                                       eps: float = 0.8,
                                       which_to_keep: str = "hard",
                                       sim_metric: str = "cosine",
                                       dedup_batch_size: int = 1024,
                                       compute_only: bool = False) -> None:
        """Run the complete pipeline with language splitting: cluster → split → initialize model once → for each language: embed → cluster → dedup → remove"""
        
        start_time = time.time()
        
        try:
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            print("🚀 Starting Complete Semantic Deduplication Pipeline with Language Splitting")
            print("=" * 80)
            
            # Create main output directory
            os.makedirs(output_directory, exist_ok=True)
            
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
            
            # Step 2: Split data by language
            language_files = self.split_by_language(
                input_path=input_path,
                output_directory=output_directory,
                data_column=data_column,
                lang_column=lang_column,
                id_column=id_column,
            )
            
            # Step 3: Initialize embedding model ONCE for all languages
            self.initialize_embedding_creator(
                model=model,
                batch_size=batch_size,
                pooling_strategy=pooling_strategy,
                input_column=data_column,
                embedding_column="embeddings",
                partition_size=partition_size,
                max_seq_length=max_seq_length,
                embedding_max_mem_gb=embedding_max_mem_gb,
            )
            
            # Step 4: Process each language using the same model
            print("=" * 60)
            print(f"STEP 4: Processing {len(language_files)} languages with shared model...")
            print("=" * 60)
            
            successful_languages = []
            failed_languages = []
            
            for i, (language, input_file) in enumerate(language_files.items(), 1):
                print(f"\n[{i}/{len(language_files)}] Processing {language.upper()}...")
                
                cleaned_output = self.process_single_language(
                    language=language,
                    input_file=input_file,
                    base_output_dir=output_directory,
                    eps=eps,
                    # Clustering args
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    id_column=id_column,
                    embedding_column="embeddings",
                    clustering_partition_size=clustering_partition_size,
                    random_state=random_state,
                    # Deduplication args
                    which_to_keep=which_to_keep,
                    sim_metric=sim_metric,
                    dedup_batch_size=dedup_batch_size,
                    compute_only=compute_only,
                )
                
                if cleaned_output:
                    successful_languages.append((language, cleaned_output))
                else:
                    failed_languages.append(language)
            
            # Pipeline completed - Print final summary
            total_time = time.time() - start_time
            print("=" * 80)
            print("🎉 COMPLETE PIPELINE FINISHED!")
            print("=" * 80)
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Total languages processed: {len(language_files)}")
            print(f"Successful: {len(successful_languages)}")
            print(f"Failed: {len(failed_languages)}")
            
            if successful_languages:
                print("\n✅ Successfully processed languages:")
                for lang, output_path in successful_languages:
                    print(f"   {lang}: {output_path}")
            
            if failed_languages:
                print("\n❌ Failed languages:")
                for lang in failed_languages:
                    print(f"   {lang}")
            
            print(f"\nAll results saved in: {output_directory}")
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Master Semantic Deduplication Pipeline with Language Splitting")
    
    # Required arguments
    parser.add_argument("input_path", help="Path to input parquet file")
    parser.add_argument("output_directory", help="Directory to save all results")
    
    # Language splitting configuration
    parser.add_argument("--data-column", default="data", help="Name of data column (default: data)")
    parser.add_argument("--lang-column", default="lang", help="Name of language column (default: lang)")
    parser.add_argument("--id-column", default="id", help="Name of ID column to add (default: id)")
    
    # Cluster configuration
    parser.add_argument("--cuda-devices", default="0,1", help="CUDA devices to use (default: 0,1)")
    parser.add_argument("--rmm-pool-size", type=float, default=0.5, help="RMM pool size (default: 0.5)")
    parser.add_argument("--enable-cudf-spill", action="store_true", default=True, help="Enable cuDF spilling")
    parser.add_argument("--local-directory", default="/tmp/dask-scratch", help="Local directory for spilling")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Use single GPU instead of multi-GPU")
    parser.add_argument("--device-memory-limit", default="auto", help="Device memory limit (default: auto)")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per worker (default: 1)")
    parser.add_argument("--no-processes", action="store_true", help="Use threads instead of processes")
    
    # Embedding configuration
    parser.add_argument("--model", default="intfloat/multilingual-e5-large-instruct", help="Model name or path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for embedding inference (default: 1)")
    parser.add_argument("--pooling-strategy", choices=["mean_pooling", "last_token"], default="mean_pooling", help="Pooling strategy")
    parser.add_argument("--partition-size", default="100MiB", help="Partition size for embedding processing")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--embedding-max-mem-gb", type=int, default=30, help="Maximum memory per embedding worker (GB)")
    
    # Clustering configuration
    parser.add_argument("--n-clusters", type=int, default=1000, help="Number of clusters (default: 1000)")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations for clustering (default: 200)")
    parser.add_argument("--clustering-partition-size", default="4GB", help="Partition size for clustering (default: 4GB)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for clustering (default: 42)")
    
    # Deduplication configuration
    parser.add_argument("--eps", type=float, default=0.8, help="Epsilon threshold for deduplication (default: 0.8)")
    parser.add_argument("--which-to-keep", choices=["hard", "easy", "random"], default="hard", 
                       help="Strategy for keeping duplicates (default: hard)")
    parser.add_argument("--sim-metric", choices=["cosine", "l2"], default="cosine", 
                       help="Similarity metric (default: cosine)")
    parser.add_argument("--dedup-batch-size", type=int, default=1024, 
                       help="Batch size for deduplication similarity computations (default: 1024)")
    parser.add_argument("--compute-only", action="store_true", 
                       help="Only compute semantic matches without extracting duplicates")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = SemanticDeduplicationOrchestrator()
    
    # Run the full pipeline with language splitting
    orchestrator.run_full_pipeline_with_languages(
        input_path=args.input_path,
        output_directory=args.output_directory,
        # Language splitting args
        data_column=args.data_column,
        lang_column=args.lang_column,
        id_column=args.id_column,
        # Cluster args
        cuda_devices=args.cuda_devices,
        rmm_pool_size=args.rmm_pool_size,
        enable_cudf_spill=args.enable_cudf_spill,
        local_directory=args.local_directory,
        use_multi_gpu=not args.no_multi_gpu,
        device_memory_limit=args.device_memory_limit,
        threads_per_worker=args.threads_per_worker,
        processes=not args.no_processes,
        # Embedding args
        model=args.model,
        batch_size=args.batch_size,
        pooling_strategy=args.pooling_strategy,
        partition_size=args.partition_size,
        max_seq_length=args.max_seq_length,
        embedding_max_mem_gb=args.embedding_max_mem_gb,
        # Clustering args
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        clustering_partition_size=args.clustering_partition_size,
        random_state=args.random_state,
        # Deduplication args
        eps=args.eps,
        which_to_keep=args.which_to_keep,
        sim_metric=args.sim_metric,
        dedup_batch_size=args.dedup_batch_size,
        compute_only=args.compute_only,
    )


if __name__ == "__main__":
    main()