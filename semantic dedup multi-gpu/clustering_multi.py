import argparse
import logging
import os
import shutil
import sys
import time
from typing import Optional, Union

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import torch
from cuml.dask.cluster import KMeans
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar


class ParquetEmbeddingClusterer:
    def __init__(
        self,
        id_column: str = "id",
        embedding_column: str = "embeddings", 
        max_iter: int = 100,
        n_clusters: int = 1000,
        clustering_output_dir: str = "./clustering_results",
        random_state: int = 1234,
        clustering_input_partition_size: str = "2gb",
        num_gpus: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        GPU-accelerated K-means clustering for embeddings from parquet files.
        
        Args:
            id_column: Column name for document IDs
            embedding_column: Column name containing embeddings
            max_iter: Maximum iterations for K-means clustering
            n_clusters: Number of clusters to create
            clustering_output_dir: Directory to save clustering results
            random_state: Random seed for reproducibility
            clustering_input_partition_size: Partition size for processing
            num_gpus: Number of GPUs available for clustering
            logger: Logger instance for output
        """
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.random_state = random_state
        self.clustering_input_partition_size = clustering_input_partition_size
        self.num_gpus = num_gpus
        
        # Constants for distance columns (using NVIDIA naming)
        self.L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
        self.COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"
        
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        # Create output directory
        os.makedirs(self.clustering_output_dir, exist_ok=True)
        if os.path.exists(self.clustering_output_dir) and os.listdir(self.clustering_output_dir):
            self.logger.warning(f"Clustering output directory {self.clustering_output_dir} already exists and will be overwritten")

    def _normalize_embeddings_col_in_df(self, df: cudf.DataFrame, embedding_col: str) -> cudf.DataFrame:
        """Normalize embeddings to unit vectors using NVIDIA implementation."""
        tensor = torch.Tensor(self._get_array_from_df(df, embedding_col))
        normalized_tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
        df[embedding_col] = create_list_series_from_1d_or_2d_ar(cp.asarray(normalized_tensor), index=df.index)
        return df

    def _get_array_from_df(self, df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
        """Extract embeddings as cupy array using NVIDIA implementation."""
        return df[embedding_col].list.leaves.values.reshape(len(df), -1)  # noqa: PD011

    def _add_l2_cosine_dist_to_centroid(
        self, 
        df: cudf.DataFrame, 
        embedding_col: str,
        centroids: cp.ndarray
    ) -> cudf.DataFrame:
        """
        Add L2 and cosine distance to assigned centroid using NVIDIA implementation.
        Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
        Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
        """
        normalized_embeddings = self._get_array_from_df(df, embedding_col)
        centroids_ar = centroids[df["nearest_cent"].values]  # noqa: PD011
        dist_to_cents = cp.sqrt(cp.sum((normalized_embeddings - centroids_ar) ** 2, axis=1))
        df[self.L2_DIST_TO_CENT_COL] = dist_to_cents
        del centroids_ar
        centroids_norm = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_ar = centroids_norm[df["nearest_cent"].values]  # noqa: PD011
        # We normalize the centroids as well
        cosine_similarities = cp.sum(normalized_embeddings * centroids_ar, axis=1)
        df[self.COSINE_DIST_TO_CENT_COL] = 1 - cosine_similarities
        return df

    def _validate_input(self, ddf: dask_cudf.DataFrame):
        """Validate that required columns exist."""
        columns = ddf.columns.tolist()
        if self.id_column not in columns:
            raise ValueError(f"ID column '{self.id_column}' not found. Available columns: {columns}")
        if self.embedding_column not in columns:
            raise ValueError(f"Embedding column '{self.embedding_column}' not found. Available columns: {columns}")

    def __call__(self, embeddings_input: Union[str, dask_cudf.DataFrame]) -> dask_cudf.DataFrame:
        """
        Perform K-means clustering on embeddings.
        
        Args:
            embeddings_input: Path to parquet file or dask_cudf.DataFrame with embeddings
            
        Returns:
            dask_cudf.DataFrame with clustering results partitioned by nearest_cent
        """
        start_time = time.time()
        
        # Load data with only required columns
        if isinstance(embeddings_input, str):
            self.logger.info(f"Loading embeddings from: {embeddings_input}")
            # Read only the required columns from parquet
            embeddings_df = dask_cudf.read_parquet(
                embeddings_input,
                columns=[self.id_column, self.embedding_column],
                blocksize="1GB"
            )
        else:
            embeddings_df = embeddings_input
            # Keep only required columns
            embeddings_df = embeddings_df[[self.id_column, self.embedding_column]]
            
        # Validate input
        self._validate_input(embeddings_df)
        
        self.logger.info(f"Starting clustering with {self.n_clusters} clusters")
        
        # Persist to avoid re-computation
        embeddings_df = embeddings_df.persist()

        # Repartition based on num_gpus
        current_npartitions = embeddings_df.npartitions
        if current_npartitions < self.num_gpus:
            self.logger.info(f"Current partitions ({current_npartitions}) < num_gpus ({self.num_gpus}), repartitioning to {self.num_gpus}")
            embeddings_df = embeddings_df.repartition(npartitions=self.num_gpus)
        elif self.clustering_input_partition_size is not None:
            embeddings_df = embeddings_df.repartition(partition_size=self.clustering_input_partition_size)

        # Optimize to ensure consistent partition counts
        embeddings_df = embeddings_df.optimize()

        # Normalize embeddings before clustering
        embeddings_df = embeddings_df.map_partitions(
            self._normalize_embeddings_col_in_df,
            embedding_col=self.embedding_column,
            meta=embeddings_df._meta.copy(),
        )

        # Convert to cupy array for clustering
        cupy_normalized_darr = embeddings_df.map_partitions(
            self._get_array_from_df, 
            self.embedding_column, 
            meta=cp.ndarray([1, 1])
        )

        # Compute chunk sizes with error handling
        try:
            cupy_normalized_darr.compute_chunk_sizes()
        except Exception:
            try:
                import dask
                # Workaround for cudf task fusion error
                with dask.config.set({"optimization.fuse.active": False}):
                    cupy_normalized_darr.compute_chunk_sizes()
            except Exception as inner_e:
                raise RuntimeError(
                    "Unable to compute chunk sizes for the embeddings array. "
                    "Please raise an issue or check your data format."
                ) from inner_e

        # Check if we have enough data points
        dataset_length = len(cupy_normalized_darr)
        if dataset_length < self.n_clusters:
            raise ValueError(
                f"Number of clusters ({self.n_clusters}) is greater than the number of documents ({dataset_length}). "
                f"Please reduce n_clusters to be less than or equal to {dataset_length}."
            )

        # Perform K-means clustering
        self.logger.info("Starting K-means fitting...")
        kmeans_start = time.time()
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=1,
        )
        
        kmeans.fit(cupy_normalized_darr)
        self.logger.info(f"K-means fit completed in {time.time() - kmeans_start:.2f}s")

        # Predict nearest centroids
        self.logger.info("Computing nearest centroids...")
        predict_start = time.time()
        nearest_cents = kmeans.predict(cupy_normalized_darr)
        self.logger.info(f"K-means predict completed in {time.time() - predict_start:.2f}s")

        # Add nearest centroid column
        embeddings_df["nearest_cent"] = nearest_cents.astype(np.int32)
        del nearest_cents

        # Add distance columns
        self.logger.info("Computing distances to centroids...")
        distance_start = time.time()
        
        meta_df_with_distances = embeddings_df._meta.copy()
        meta_df_with_distances[self.L2_DIST_TO_CENT_COL] = cp.zeros(1)
        meta_df_with_distances[self.COSINE_DIST_TO_CENT_COL] = cp.zeros(1)
        
        embeddings_df = embeddings_df.map_partitions(
            self._add_l2_cosine_dist_to_centroid,
            embedding_col=self.embedding_column,
            centroids=kmeans.cluster_centers_,
            meta=meta_df_with_distances,
        )
        
        embeddings_df = embeddings_df.reset_index(drop=True)
        self.logger.info(f"Distance computation completed in {time.time() - distance_start:.2f}s")

        # Save centroids
        centroids = kmeans.cluster_centers_
        kmeans_centroids_file = os.path.join(self.clustering_output_dir, "kmeans_centroids.npy")
        np.save(kmeans_centroids_file, centroids)
        self.logger.info(f"Centroids saved to: {kmeans_centroids_file}")

        # Save embeddings partitioned by nearest center
        clustering_output_dir = os.path.join(self.clustering_output_dir, "embs_by_nearest_center")
        if os.path.exists(clustering_output_dir):
            self.logger.warning(f"Output directory {clustering_output_dir} already exists and will be overwritten")
            shutil.rmtree(clustering_output_dir)

        self.logger.info(f"Saving clustered embeddings to: {clustering_output_dir}")
        save_start = time.time()
        
        embeddings_df.to_parquet(
            clustering_output_dir,
            index=False,
            partition_on="nearest_cent",
            write_index=False,
            compression="snappy"
        )
        
        self.logger.info(f"Clustering results saved in {time.time() - save_start:.2f}s")
        
        del embeddings_df
        del centroids

        # Read back partitioned by cluster for downstream processing
        fps = [os.path.join(clustering_output_dir, f"nearest_cent={i}") for i in range(self.n_clusters)]
        clustered_df = dd.from_map(cudf.read_parquet, fps)
        
        total_time = time.time() - start_time
        self.logger.info(f"Total clustering completed in {total_time:.2f}s")
        
        return clustered_df


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated K-means clustering for embeddings from parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        help="Path to input parquet file containing embeddings"
    )
    
    # Clustering parameters
    parser.add_argument(
        "--n-clusters", "-k",
        type=int,
        default=1000,
        help="Number of clusters to create (default: 1000)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for K-means clustering (default: 100)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )
    
    # Data parameters
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name for document IDs (default: id)"
    )
    
    parser.add_argument(
        "--embedding-column",
        type=str,
        default="embeddings",
        help="Column name containing embeddings (default: embeddings)"
    )
    
    # Performance parameters
    parser.add_argument(
        "--partition-size",
        type=str,
        default="2gb",
        help="Partition size for processing (default: 2gb)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs available for clustering (default: 2)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./clustering_results",
        help="Directory to save clustering results (default: ./clustering_results)"
    )
    
    # Logging parameters
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: log to stdout only)"
    )
    
    # Validation flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running clustering"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Validate n_clusters
    if args.n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got: {args.n_clusters}")
    
    # Validate max_iter
    if args.max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got: {args.max_iter}")
    
    # Validate num_gpus
    if args.num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got: {args.num_gpus}")
    
    # Validate partition size format
    partition_size = args.partition_size.lower()
    if not any(partition_size.endswith(suffix) for suffix in ['b', 'kb', 'mb', 'gb', 'tb']):
        raise ValueError(f"Invalid partition size format: {args.partition_size}")


def main():
    """Main CLI function."""
    args = parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Setup logging
        logger = setup_logging(args.log_level, args.log_file)
        
        # Print configuration
        logger.info("GPU Embedding Clustering Configuration:")
        logger.info(f"  Input file: {args.input_file}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Number of clusters: {args.n_clusters}")
        logger.info(f"  Max iterations: {args.max_iter}")
        logger.info(f"  ID column: {args.id_column}")
        logger.info(f"  Embedding column: {args.embedding_column}")
        logger.info(f"  Partition size: {args.partition_size}")
        logger.info(f"  Number of GPUs: {args.num_gpus}")
        logger.info(f"  Random state: {args.random_state}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            return 0
        
        # Initialize clusterer
        clusterer = ParquetEmbeddingClusterer(
            id_column=args.id_column,
            embedding_column=args.embedding_column,
            max_iter=args.max_iter,
            n_clusters=args.n_clusters,
            clustering_output_dir=args.output_dir,
            random_state=args.random_state,
            clustering_input_partition_size=args.partition_size,
            num_gpus=args.num_gpus,
            logger=logger,
        )
        
        # Run clustering
        logger.info("Starting clustering process...")
        clustered_df = clusterer(args.input_file)
        
        logger.info("Clustering completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Centroids saved to: {os.path.join(args.output_dir, 'kmeans_centroids.npy')}")
        logger.info(f"Clustered embeddings saved to: {os.path.join(args.output_dir, 'embs_by_nearest_center')}")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())