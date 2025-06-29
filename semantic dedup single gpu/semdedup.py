import logging
import os
import shutil
import time
from typing import Optional, Literal
import json

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import dask.bag as db
import pandas as pd
import torch
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar

# Constants matching NeMo Curator
L2_DIST_TO_CENT_COL = "l2_dist_to_centroid"  
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_centroid"  

class GPUSemanticDeduplicator:
    def __init__(
        self,
        clustering_results_dir: str,
        id_column: str = "id",
        embedding_column: str = "embeddings",
        which_to_keep: Literal["hard", "easy", "random"] = "hard",
        sim_metric: Literal["cosine", "l2"] = "cosine",
        output_dir: str = "./deduplication_results",
        batched_cosine_similarity: int = 1024,  # Following NeMo Curator naming
        logger: Optional[logging.Logger] = None,
    ):
        """
        GPU-accelerated semantic deduplication following NeMo Curator approach.
        
        Args:
            clustering_results_dir: Directory containing clustering results
            id_column: Column name for document IDs  
            embedding_column: Column name containing embeddings
            which_to_keep: Strategy for keeping duplicates ("hard", "easy", "random")
                - "hard": Keep outliers (farthest from centroid)
                - "easy": Keep representatives (closest to centroid)
                - "random": Keep random documents
            sim_metric: Similarity metric ("cosine" or "l2") 
            output_dir: Directory to save deduplication results
            batched_cosine_similarity: Batch size for similarity computations
            logger: Logger instance
        """
        self.clustering_results_dir = clustering_results_dir
        self.emb_by_clust_dir = os.path.join(clustering_results_dir, "embs_by_nearest_center")
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.which_to_keep = which_to_keep
        self.sim_metric = sim_metric
        self.output_dir = output_dir
        self.batched_cosine_similarity = batched_cosine_similarity
        
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.semdedup_pruning_tables_dir = os.path.join(self.output_dir, "semdedup_pruning_tables")
        
        # Auto-detect number of clusters
        self.n_clusters = self._detect_clusters()
        self.logger.info(f"Detected {self.n_clusters} clusters")
        
        # Track computation state
        self.computed_semantic_match_dfs = False
        
        # Validate inputs
        self._validate_inputs()
        
    def _detect_clusters(self) -> int:
        """Auto-detect number of clusters from directory structure."""
        if not os.path.exists(self.emb_by_clust_dir):
            raise ValueError(f"Clustering results directory not found: {self.emb_by_clust_dir}")
            
        cluster_dirs = [d for d in os.listdir(self.emb_by_clust_dir) if d.startswith("nearest_cent=")]
        if not cluster_dirs:
            raise ValueError(f"No cluster directories found in {self.emb_by_clust_dir}")
            
        # Extract cluster numbers
        cluster_nums = []
        for d in cluster_dirs:
            try:
                num = int(d.split("=")[1])
                cluster_nums.append(num)
            except ValueError:
                continue
                
        return max(cluster_nums) + 1 if cluster_nums else 0
        
    def _validate_inputs(self):
        """Validate that clustering results exist."""
        if not os.path.exists(self.emb_by_clust_dir):
            raise ValueError(f"Clustering results directory not found: {self.emb_by_clust_dir}")
            
        existing_clusters = [d for d in os.listdir(self.emb_by_clust_dir) if d.startswith("nearest_cent=")]
        
        if len(existing_clusters) == 0:
            raise ValueError(f"No cluster directories found in {self.emb_by_clust_dir}")
            
        self.logger.info(f"Found {len(existing_clusters)} cluster directories")

    def _get_array_from_df(self, df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
        """Extract embeddings array from DataFrame, following NeMo Curator approach."""
        return df[embedding_col].list.leaves.values.reshape(len(df), -1)

    def _pairwise_cosine_similarity(
        self, 
        cluster_reps: torch.Tensor,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Compute pairwise cosine similarity between cluster items following NeMo Curator approach.
        Returns max similarity and corresponding indices for each document.
        """
        # Move to device
        cluster_reps = cluster_reps.to(device)
        # Compute pairwise cosine similarity
        pairwise_sim_matrix = torch.mm(cluster_reps, cluster_reps.T)
        del cluster_reps
        
        # Get upper triangular matrix (exclude diagonal)
        if pairwise_sim_matrix.shape[0] != pairwise_sim_matrix.shape[1]:
            msg = "Pairwise similarity matrix is not square"
            raise ValueError(msg)
        
        triu_sim_mat = torch.triu(pairwise_sim_matrix, diagonal=1)
        
        # Get max similarity and indices for each row
        max_values_and_indices = torch.max(triu_sim_mat, dim=0)
        max_similarity = max_values_and_indices[0]
        max_indices = max_values_and_indices[1]
        return cp.asarray(max_similarity, dtype=cp.float32), cp.asarray(max_indices)

    def _pairwise_cosine_similarity_batched(
        self,
        cluster_reps: torch.Tensor,
        device: Literal["cuda", "cpu"] = "cuda", 
        batch_size: int = 1024,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Batched version following NeMo Curator approach.
        Memory requirements are O(N*B) instead of O(N^2).
        """
        cluster_reps = cluster_reps.to(device)
        max_similarity = torch.zeros(cluster_reps.shape[0], dtype=torch.float32, device=device)
        max_indices = torch.zeros(cluster_reps.shape[0], dtype=torch.int64, device=device)
        
        for start_idx in range(0, cluster_reps.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, cluster_reps.shape[0])
            batch = cluster_reps[start_idx:end_idx]
            pairwise_sim_matrix = torch.mm(cluster_reps, batch.T)
            triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1 - start_idx)
            del batch, pairwise_sim_matrix
            
            max_values_and_indices = torch.max(triu_sim_matrix, dim=0)
            max_similarity[start_idx:end_idx] = max_values_and_indices[0]
            max_indices[start_idx:end_idx] = max_values_and_indices[1]

        return cp.asarray(max_similarity), cp.asarray(max_indices)

    def _get_semantic_matches_per_cluster(self, cluster_id: int) -> None:
        """
        Get the semantic matches for a single cluster following NeMo Curator approach.
        Reads cluster embeddings and computes pairwise cosine similarity.
        """
        if self.sim_metric == "cosine":
            distance_col = COSINE_DIST_TO_CENT_COL
        elif self.sim_metric == "l2":
            distance_col = L2_DIST_TO_CENT_COL
        else:
            msg = f"Invalid similarity metric: {self.sim_metric}. Only cosine and l2 are supported."
            raise ValueError(msg)

        cluster_dir = os.path.join(self.emb_by_clust_dir, f"nearest_cent={cluster_id}")
        if not os.path.exists(cluster_dir):
            return

        try:
            cluster_df = cudf.read_parquet(
                cluster_dir,
                columns=[self.embedding_column, self.id_column, distance_col],
            )
            
            output_df_file_path = os.path.join(self.semdedup_pruning_tables_dir, f"cluster_{cluster_id}.parquet")
            
            if len(cluster_df) == 1:
                # Single document cluster
                cluster_df["id"] = cluster_df[self.id_column]
                cluster_df["max_id"] = cluster_df[self.id_column]
                cluster_df["cosine_sim_score"] = [0.0]
                cluster_df = cluster_df[["id", "max_id", "cosine_sim_score"]]
                cluster_df.to_parquet(output_df_file_path)
                return

            # Sort based on strategy
            if self.which_to_keep == "hard":
                cluster_df = cluster_df.sort_values(by=[distance_col, self.id_column], ascending=False, ignore_index=True)
            elif self.which_to_keep == "easy":
                cluster_df = cluster_df.sort_values(by=[distance_col, self.id_column], ascending=True, ignore_index=True)
            elif self.which_to_keep == "random":
                cluster_df = cluster_df.sample(frac=1, random_state=42, ignore_index=True)

            # Extract embeddings and compute similarities
            cluster_embeddings = torch.as_tensor(self._get_array_from_df(cluster_df, self.embedding_column), device="cuda")
            ids = cluster_df[self.id_column]
            
            if cluster_embeddings.shape[0] != len(ids):
                msg = "Cluster embeddings and IDs have different lengths"
                raise ValueError(msg)

            # Compute pairwise similarities
            if self.batched_cosine_similarity > 0:
                max_similarity, max_indices = self._pairwise_cosine_similarity_batched(
                    cluster_embeddings, "cuda", self.batched_cosine_similarity
                )
            else:
                max_similarity, max_indices = self._pairwise_cosine_similarity(cluster_embeddings, "cuda")
            
            # Create output dataframe following NeMo Curator format
            max_indices_id = ids.iloc[max_indices].reset_index(drop=True)
            points_to_remove_df = cudf.DataFrame(
                {
                    "id": ids,
                    "max_id": max_indices_id,
                    "cosine_sim_score": max_similarity,
                }
            )
            points_to_remove_df.to_parquet(output_df_file_path)
            
        except Exception as e:
            self.logger.error(f"Error processing cluster {cluster_id}: {str(e)}")

    def compute_semantic_match_dfs(self) -> None:
        """Compute similarity tables for all clusters following NeMo Curator approach."""
        if os.path.exists(self.semdedup_pruning_tables_dir):
            self.logger.info(f"Removing existing directory {self.semdedup_pruning_tables_dir}")
            shutil.rmtree(self.semdedup_pruning_tables_dir)
        os.makedirs(self.semdedup_pruning_tables_dir, exist_ok=True)
        
        t0 = time.time()
        
        # Process clusters in parallel using Dask, following NeMo Curator approach
        tasks = db.from_sequence(list(range(self.n_clusters)), npartitions=self.n_clusters).map(
            lambda cluster_id: self._get_semantic_matches_per_cluster(cluster_id)
        )
        tasks.compute()
        
        self.logger.info(f"Time taken for Computing Semantic Matches : {time.time() - t0}")
        self.computed_semantic_match_dfs = True

    def _prune_single_cluster(
        self,
        cluster_id: int, 
        eps: float,
    ) -> cudf.DataFrame:
        """
        Process data for a single cluster, applying pruning based on epsilon.
        Following NeMo Curator approach exactly.
        
        Args:
            cluster_id: The specific cluster ID to process
            eps: Epsilon value for pruning
            
        Returns:
            cudf.DataFrame: DataFrame of documents to be removed (duplicates)
        """
        cluster_dir = os.path.join(self.emb_by_clust_dir, f"nearest_cent={cluster_id}")
        
        if not os.path.exists(cluster_dir):
            return cudf.DataFrame(columns=[self.id_column, COSINE_DIST_TO_CENT_COL, "cluster"]).astype({
                self.id_column: "object",
                COSINE_DIST_TO_CENT_COL: "float32", 
                "cluster": "int32"
            })
        
        try:
            # Read cluster data with required columns
            df_cluster = cudf.read_parquet(
                cluster_dir, 
                columns=[self.id_column, COSINE_DIST_TO_CENT_COL]
            ).assign(cluster=cluster_id)

            pruning_table_fname = os.path.join(self.semdedup_pruning_tables_dir, f"cluster_{cluster_id}.parquet")
            
            if not os.path.exists(pruning_table_fname):
                # No pruning table means no similarities computed
                return cudf.DataFrame(columns=df_cluster.columns).astype(df_cluster.dtypes)
            
            # Read pruning table
            pruning_table = cudf.read_parquet(pruning_table_fname, columns=["id", "cosine_sim_score"])
            
            # If the pruning table only has one row, no duplicates to remove
            if len(pruning_table) == 1:
                return cudf.DataFrame(columns=df_cluster.columns).astype(df_cluster.dtypes)
            
            # Keep only records that are very similar i.e cosine_sim_score >= 1 - eps
            # These are the documents to be REMOVED (duplicates)
            pruning_table = pruning_table[pruning_table["cosine_sim_score"] >= 1 - eps][["id"]]
            
            # Return documents to be removed
            return df_cluster.merge(pruning_table.rename(columns={"id": self.id_column}), on=self.id_column, how="inner")
            
        except Exception as e:
            self.logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
            return cudf.DataFrame(columns=[self.id_column, COSINE_DIST_TO_CENT_COL, "cluster"]).astype({
                self.id_column: "object",
                COSINE_DIST_TO_CENT_COL: "float32",
                "cluster": "int32"
            })

    def extract_dedup_data(self, eps_to_extract: float) -> dask_cudf.DataFrame:
        """
        Extract similar records that are within epsilon threshold. 
        These records can be removed from the dataset (duplicates).
        Following NeMo Curator approach exactly.
        
        Args:
            eps_to_extract: Epsilon threshold for extracting deduplicated data
            
        Returns:
            dask_cudf.DataFrame: Dataset containing list of ids that can be removed
        """
        if not self.computed_semantic_match_dfs:
            msg = "Run compute_semantic_match_dfs before calling extract_dedup_data"
            self.logger.error(msg)
            raise ValueError(msg)
            
        if not isinstance(eps_to_extract, float):
            msg = "eps_to_extract must be a float"
            self.logger.error(msg)
            raise TypeError(msg)
            
        output_parquet_path = os.path.join(self.output_dir, f"unique_ids_{eps_to_extract}.parquet")

        t0 = time.time()
        
        # Use Dask to process clusters in parallel following NeMo Curator approach
        results_df = dd.from_map(
            self._prune_single_cluster,
            range(self.n_clusters),
            eps=eps_to_extract,
        )
        results_df.to_parquet(output_parquet_path, index=False, ignore_index=True)
        
        self.logger.info(
            f"Time taken for Extracting Pruned Data : {time.time() - t0} and output written at {output_parquet_path}"
        )

        # Write out summary file
        output_summary_file = os.path.join(self.output_dir, f"dedup_summary_{eps_to_extract}.csv")
        self._write_pruned_summary_file(
            eps=eps_to_extract,
            filtered_unique_ids_path=output_parquet_path,
            output_summary_file=output_summary_file,
        )
        
        return dd.read_parquet(output_parquet_path, blocksize="1gb")

    def _write_pruned_summary_file(
        self,
        eps: float,
        filtered_unique_ids_path: str,
        output_summary_file: str,
    ) -> None:
        """Write summary file following NeMo Curator approach."""
        removed = len(dd.read_parquet(filtered_unique_ids_path))
        total = len(dd.read_parquet(self.emb_by_clust_dir))
        kept = total - removed

        self.logger.info(f"DONE saving {kept} out of {total}. Removed: {removed}. Epsilon: {eps:.4f}")
        result_dict = {
            "eps": [eps],
            "kept": [kept],
            "removed": [removed],
            "total": [total],
        }
        df = pd.DataFrame(result_dict)
        df.to_csv(output_summary_file, index=False)


# Example usage following NeMo Curator pattern
if __name__ == "__main__":
    # Initialize deduplicator
    deduplicator = GPUSemanticDeduplicator(
        clustering_results_dir="./clustering_results",
        id_column="id",
        embedding_column="embeddings",
        which_to_keep="hard",
        sim_metric="cosine",
        output_dir="./clustering_results",
        batched_cosine_similarity=1024,
    )
    
    # Step 1: Compute semantic match dataframes
    deduplicator.compute_semantic_match_dfs()
    
    # Step 2: Extract documents to remove at given epsilon
    eps_to_extract = 0.8
    documents_to_remove = deduplicator.extract_dedup_data(eps_to_extract)
    
    print(f"Documents to remove saved at: ./clustering_results/unique_ids_{eps_to_extract}.parquet")
    print("Summary available at: ./clustering_results/dedup_summary_{}.csv".format(eps_to_extract))