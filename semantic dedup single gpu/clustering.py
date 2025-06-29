import logging
import os
import shutil
import time
from typing import Optional, Union

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np

#Replaced cuml.dask.cluster.KMeans(multi-gpu) with cuml.cluster.KMeans (single GPU version)
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
        keep_all_columns: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        GPU-accelerated K-means clustering for embeddings from parquet files.
        Works on single GPU environments like Google Colab.
        
        Args:
            id_column: Column name for document IDs
            embedding_column: Column name containing embeddings
            max_iter: Maximum iterations for K-means clustering
            n_clusters: Number of clusters to create
            clustering_output_dir: Directory to save clustering results
            random_state: Random seed for reproducibility
            clustering_input_partition_size: Partition size for processing
            keep_all_columns: Whether to keep all columns or just ID and embeddings
            logger: Logger instance for output
        """
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.random_state = random_state
        self.clustering_input_partition_size = clustering_input_partition_size
        self.keep_all_columns = keep_all_columns
        
        # Constants for distance columns
        self.L2_DIST_TO_CENT_COL = "l2_dist_to_centroid"
        self.COSINE_DIST_TO_CENT_COL = "cosine_dist_to_centroid"
        
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
        """Normalize embeddings to unit vectors."""
        embeddings_array = self._get_array_from_df(df, embedding_col)
        
        # L2 normalize
        norms = cp.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-12)  # Avoid division by zero
        normalized_embeddings = embeddings_array / norms
        
        # Convert back to list format
        df_copy = df.copy()
        df_copy[embedding_col] = cudf.Series(normalized_embeddings.tolist())
        return df_copy

    def _get_array_from_df(self, df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
        """Extract embeddings as cupy array."""
        if len(df) == 0:
            return cp.array([]).reshape(0, -1)
        
        # Handle both list column and array formats
        if hasattr(df[embedding_col].iloc[0], '__len__'):
            # List format - convert to array
            embeddings_list = df[embedding_col].to_arrow().to_pylist()
            return cp.array(embeddings_list)
        else:
            # Already array format
            embeddings = df[embedding_col].list.leaves
            n_docs = len(df)
            embedding_dim = len(df[embedding_col].iloc[0])
            return embeddings.values.reshape(n_docs, embedding_dim)

    def _single_gpu_kmeans_fit_predict(self, embeddings_array: cp.ndarray) -> tuple:
        """Perform K-means clustering on a single GPU using cuML's single GPU implementation."""
        from cuml.cluster import KMeans  # Single GPU KMeans
        
        self.logger.info(f"Fitting single GPU K-means on {embeddings_array.shape[0]} samples with {embeddings_array.shape[1]} dimensions")
        
        # Initialize single GPU KMeans
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=1,
            init='k-means++',
            verbose=True
        )
        
        # Fit and predict in one go
        labels = kmeans.fit_predict(embeddings_array)
        centroids = kmeans.cluster_centers_
        
        return labels, centroids

    def _process_large_dataset_in_batches(self, ddf: dask_cudf.DataFrame) -> tuple:
        """Process large datasets in batches for single GPU clustering."""
        self.logger.info("Processing dataset in batches for single GPU clustering")
        
        # First pass: collect all embeddings to fit KMeans
        all_embeddings = []
        batch_size = 50000  # Adjust based on GPU memory
        
        self.logger.info("Collecting embeddings for clustering...")
        for i, partition in enumerate(ddf.to_delayed()):
            partition_df = partition.compute()
            if len(partition_df) == 0:
                continue
                
            embeddings = self._get_array_from_df(partition_df, self.embedding_column)
            all_embeddings.append(embeddings)
            
            # If we have enough samples, fit the model
            total_samples = sum(arr.shape[0] for arr in all_embeddings)
            if total_samples >= batch_size or i == len(ddf.to_delayed()) - 1:
                break
        
        # Concatenate embeddings and fit KMeans
        if all_embeddings:
            combined_embeddings = cp.vstack(all_embeddings)
            self.logger.info(f"Fitting KMeans on {combined_embeddings.shape[0]} samples")
            
            from cuml.cluster import KMeans
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=1,
                init='k-means++',
                verbose=True
            )
            kmeans.fit(combined_embeddings)
            centroids = kmeans.cluster_centers_
            
            return kmeans, centroids
        else:
            raise ValueError("No embeddings found in dataset")

    def _add_l2_cosine_dist_to_centroid(
        self, 
        df: cudf.DataFrame, 
        embedding_col: str,
        centroids: cp.ndarray
    ) -> cudf.DataFrame:
        """Add L2 and cosine distance to assigned centroid."""
        if len(df) == 0:
            return df
            
        embeddings = self._get_array_from_df(df, embedding_col)
        nearest_cents = df["nearest_cent"].values
        
        # Get centroids for each point
        assigned_centroids = centroids[nearest_cents]
        
        # Calculate L2 distance
        l2_distances = cp.linalg.norm(embeddings - assigned_centroids, axis=1)
        
        # Calculate cosine distance (1 - cosine similarity)
        # Embeddings are already normalized, so cosine similarity is just dot product
        cosine_similarities = cp.sum(embeddings * assigned_centroids, axis=1)
        cosine_distances = 1.0 - cosine_similarities
        
        # Add to dataframe
        df_copy = df.copy()
        df_copy[self.L2_DIST_TO_CENT_COL] = cudf.Series(l2_distances)
        df_copy[self.COSINE_DIST_TO_CENT_COL] = cudf.Series(cosine_distances)
        
        return df_copy

    def _predict_partition(self, df: cudf.DataFrame, kmeans_model, embedding_col: str) -> cudf.DataFrame:
        """Predict cluster labels for a partition."""
        if len(df) == 0:
            df_copy = df.copy()
            df_copy["nearest_cent"] = cudf.Series([], dtype=cp.int32)
            return df_copy
            
        embeddings = self._get_array_from_df(df, embedding_col)
        labels = kmeans_model.predict(embeddings)
        
        df_copy = df.copy()
        df_copy["nearest_cent"] = cudf.Series(labels.astype(cp.int32))
        return df_copy

    def _validate_input(self, ddf: dask_cudf.DataFrame):
        """Validate that required columns exist."""
        columns = ddf.columns.tolist()
        if self.id_column not in columns:
            raise ValueError(f"ID column '{self.id_column}' not found. Available columns: {columns}")
        if self.embedding_column not in columns:
            raise ValueError(f"Embedding column '{self.embedding_column}' not found. Available columns: {columns}")

    def __call__(self, embeddings_input: Union[str, dask_cudf.DataFrame]) -> dask_cudf.DataFrame:
        """
        Perform K-means clustering on embeddings using single GPU.
        
        Args:
            embeddings_input: Path to parquet file or dask_cudf.DataFrame with embeddings
            
        Returns:
            dask_cudf.DataFrame with clustering results partitioned by nearest_cent
        """
        start_time = time.time()
        
        # Load data if path is provided
        if isinstance(embeddings_input, str):
            self.logger.info(f"Loading embeddings from: {embeddings_input}")
            embeddings_df = dask_cudf.read_parquet(embeddings_input, blocksize="1GB")
        else:
            embeddings_df = embeddings_input
            
        # Validate input
        self._validate_input(embeddings_df)
        
        if self.embedding_column not in embeddings_df.columns:
            raise ValueError(f'Expected embedding column "{self.embedding_column}" not found in dataset. Available columns: {embeddings_df.columns}')

        self.logger.info(f"Starting single GPU clustering with {self.n_clusters} clusters")
        
        # Keep only required columns if specified
        if not self.keep_all_columns:
            embeddings_df = embeddings_df[[self.id_column, self.embedding_column]]
            embeddings_df = embeddings_df.persist()
        else:
            self.logger.warning("Since all columns are being kept, we will not persist the embeddings_df which may result in slowdown")

        # Repartition if specified
        if self.clustering_input_partition_size is not None:
            embeddings_df = embeddings_df.repartition(partition_size=self.clustering_input_partition_size)

        # Optimize to ensure consistent partition counts
        embeddings_df = embeddings_df.optimize()

        # Normalize embeddings before clustering
        embeddings_df = embeddings_df.map_partitions(
            self._normalize_embeddings_col_in_df,
            embedding_col=self.embedding_column,
            meta=embeddings_df._meta.copy(),
        )

        # Check dataset size
        dataset_length = len(embeddings_df)
        if dataset_length < self.n_clusters:
            raise ValueError(
                f"Number of clusters ({self.n_clusters}) is greater than the number of documents ({dataset_length}). "
                f"Please reduce n_clusters to be less than or equal to {dataset_length}."
            )

        # Fit KMeans model on a sample of the data
        self.logger.info("Fitting K-means model...")
        kmeans_start = time.time()
        
        kmeans_model, centroids = self._process_large_dataset_in_batches(embeddings_df)
        
        self.logger.info(f"K-means fit completed in {time.time() - kmeans_start:.2f}s")

        # Predict cluster labels for all partitions
        self.logger.info("Predicting cluster labels for all data...")
        predict_start = time.time()
        
        meta_with_labels = embeddings_df._meta.copy()
        meta_with_labels["nearest_cent"] = cp.zeros(1, dtype=cp.int32)
        
        embeddings_df = embeddings_df.map_partitions(
            self._predict_partition,
            kmeans_model=kmeans_model,
            embedding_col=self.embedding_column,
            meta=meta_with_labels,
        )
        
        self.logger.info(f"Prediction completed in {time.time() - predict_start:.2f}s")

        # Add distance columns
        self.logger.info("Computing distances to centroids...")
        distance_start = time.time()
        
        meta_df_with_distances = embeddings_df._meta.copy()
        meta_df_with_distances[self.L2_DIST_TO_CENT_COL] = cp.zeros(1)
        meta_df_with_distances[self.COSINE_DIST_TO_CENT_COL] = cp.zeros(1)
        
        embeddings_df = embeddings_df.map_partitions(
            self._add_l2_cosine_dist_to_centroid,
            embedding_col=self.embedding_column,
            centroids=centroids,
            meta=meta_df_with_distances,
        )
        
        embeddings_df = embeddings_df.reset_index(drop=True)
        self.logger.info(f"Distance computation completed in {time.time() - distance_start:.2f}s")

        # Save centroids
        kmeans_centroids_file = os.path.join(self.clustering_output_dir, "kmeans_centroids.npy")
        cp.save(kmeans_centroids_file, centroids)
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
        )
        
        self.logger.info(f"Clustering results saved in {time.time() - save_start:.2f}s")
        
        del embeddings_df

        # Read back partitioned by cluster for downstream processing
        fps = [os.path.join(clustering_output_dir, f"nearest_cent={i}") for i in range(self.n_clusters)]
        clustered_df = dd.from_map(cudf.read_parquet, fps)
        
        total_time = time.time() - start_time
        self.logger.info(f"Total clustering completed in {total_time:.2f}s")
        
        return clustered_df


# Example usage for Google Colab
if __name__ == "__main__":
    # Initialize the clustering model - works perfectly on single GPU
    clusterer = ParquetEmbeddingClusterer(
        n_clusters=500,  # Adjust based on your dataset size
        max_iter=100,
        clustering_output_dir="./clustering_output",
        id_column="id",
        embedding_column="embeddings",
        keep_all_columns=False
    )
    
    # Cluster embeddings from parquet file
    # clustered_df = clusterer("path/to/embeddings.parquet")
    
    print("Clusterer ready for single GPU environments like Google Colab!")