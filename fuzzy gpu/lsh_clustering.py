#!/usr/bin/env python3
"""
Name: lsh_clustering.py
GPU-accelerated LSH clustering engine for MinHash-based deduplication.
Handles LSH bucket creation, edge list generation, and connected components clustering.
"""

import logging
from typing import List, Set, Tuple, Optional

import cudf
import cupy as cp
import cugraph
from numba import cuda, types

from config import INDEX_COLUMN, SIGNATURE_COLUMN, CLUSTER_COLUMN, DeduplicationConfig


# ============================================================================
# LSH BUCKET GENERATION
# ============================================================================

@cuda.jit
def create_lsh_buckets_kernel(signatures, buckets, b, r, num_docs, num_perm):
    """CUDA kernel for creating LSH buckets from MinHash signatures."""
    doc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if doc_idx >= num_docs:
        return
    
    # Generate hash for each band
    for band_idx in range(b):
        hash_val = types.uint64(0)
        
        # Hash the r values in this band
        for i in range(r):
            perm_idx = band_idx * r + i
            if perm_idx < num_perm:
                sig_val = signatures[doc_idx, perm_idx]
                # Simple polynomial rolling hash
                hash_val = hash_val * 31 + sig_val
        
        buckets[doc_idx * b + band_idx] = hash_val


class LSHBucketer:
    """GPU-accelerated LSH bucket generator."""
    
    def __init__(self, config: DeduplicationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"LSH bucketer initialized with b={config.b}, r={config.r}")
    
    def create_buckets(self, signatures: cp.ndarray) -> cudf.DataFrame:
        """
        Create LSH buckets from MinHash signatures.
        
        Parameters
        ----------
        signatures : cp.ndarray
            MinHash signatures matrix (num_docs x num_perm)
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with columns: doc_id, band_id, bucket_hash
        """
        num_docs, num_perm = signatures.shape
        
        self.logger.info(f"Creating LSH buckets for {num_docs} documents")
        
        # Output array for bucket hashes
        buckets = cp.zeros(num_docs * self.config.b, dtype=cp.uint64)
        
        # Launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (num_docs + threads_per_block - 1) // threads_per_block
        
        create_lsh_buckets_kernel[blocks_per_grid, threads_per_block](
            signatures, buckets, self.config.b, self.config.r, num_docs, num_perm
        )
        
        # Convert to DataFrame format
        doc_ids = cp.repeat(cp.arange(num_docs), self.config.b)
        band_ids = cp.tile(cp.arange(self.config.b), num_docs)
        
        bucket_df = cudf.DataFrame({
            'doc_id': doc_ids,
            'band_id': band_ids,
            'bucket_hash': buckets
        })
        
        self.logger.info(f"Created {len(bucket_df)} LSH buckets")
        return bucket_df


# ============================================================================
# EDGE LIST GENERATION
# ============================================================================

class EdgeGenerator:
    """Generate edges between documents in same LSH buckets."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_edges(self, bucket_df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Generate edges between documents that share LSH buckets.
        
        Parameters
        ----------
        bucket_df : cudf.DataFrame
            LSH bucket DataFrame with columns: doc_id, band_id, bucket_hash
            
        Returns
        -------
        cudf.DataFrame
            Edge list with columns: src, dst
        """
        self.logger.info("Generating edges from LSH buckets")
        
        # Group by bucket_hash to find documents in same buckets
        bucket_groups = bucket_df.groupby('bucket_hash')['doc_id'].apply(list).reset_index()
        
        # Filter out buckets with only one document
        bucket_groups = bucket_groups[bucket_groups['doc_id'].list.len() > 1]
        
        if len(bucket_groups) == 0:
            return cudf.DataFrame({'src': [], 'dst': []})
        
        # Generate all pairs within each bucket
        edges = []
        for bucket_docs in bucket_groups['doc_id'].to_pandas():
            docs = list(bucket_docs)
            # Generate all pairs
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    edges.append((docs[i], docs[j]))
        
        if not edges:
            return cudf.DataFrame({'src': [], 'dst': []})
        
        # Convert to DataFrame and remove duplicates
        edge_df = cudf.DataFrame(edges, columns=['src', 'dst'])
        edge_df = edge_df.drop_duplicates()
        
        self.logger.info(f"Generated {len(edge_df)} unique edges")
        return edge_df


# ============================================================================
# CONNECTED COMPONENTS CLUSTERING
# ============================================================================

class ConnectedComponentsClustering:
    """GPU-accelerated connected components clustering using cuGraph."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def cluster(self, edge_df: cudf.DataFrame, num_nodes: int) -> cudf.DataFrame:
        """
        Find connected components in the graph.
        
        Parameters
        ----------
        edge_df : cudf.DataFrame
            Edge list with columns: src, dst
        num_nodes : int
            Total number of nodes (documents)
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with columns: vertex, labels (cluster_id)
        """
        self.logger.info(f"Running connected components on {len(edge_df)} edges, {num_nodes} nodes")
        
        if len(edge_df) == 0:
            # No edges, each node is its own cluster
            return cudf.DataFrame({
                'vertex': range(num_nodes),
                'labels': range(num_nodes)
            })
        
        # Create graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')
        
        # Run connected components
        result = cugraph.connected_components(G)
        
        # Ensure all nodes are included (isolated nodes get their own cluster)
        all_nodes = cudf.DataFrame({'vertex': range(num_nodes)})
        result = all_nodes.merge(result, on='vertex', how='left')
        
        # Fill missing labels (isolated nodes) with unique cluster IDs
        max_label = result['labels'].max()
        if cp.isnan(max_label):
            max_label = -1
        
        missing_mask = result['labels'].isna()
        num_missing = missing_mask.sum()
        
        if num_missing > 0:
            new_labels = cudf.Series(range(int(max_label) + 1, int(max_label) + 1 + num_missing))
            result.loc[missing_mask, 'labels'] = new_labels.values
        
        self.logger.info(f"Found {result['labels'].nunique()} clusters")
        return result


# ============================================================================
# CLUSTER REPRESENTATIVE SELECTION
# ============================================================================

class RepresentativeSelector:
    """Select representative documents for each cluster."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def select_representatives(self, df: cudf.DataFrame, cluster_df: cudf.DataFrame, 
                            text_column: str = 'text') -> cudf.DataFrame:
        """
        Select one representative document per cluster.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Original DataFrame with documents
        cluster_df : cudf.DataFrame
            Clustering results with columns: vertex, labels
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with representative documents only
        """
        self.logger.info("Selecting cluster representatives")
        
        # Add cluster labels to original DataFrame
        df_with_clusters = df.copy()
        df_with_clusters[CLUSTER_COLUMN] = cluster_df['labels']
        
        # For each cluster, select the document with the shortest text as representative
        # (you can modify this selection criteria as needed)
        if text_column in df_with_clusters.columns:
            # Add text length column
            df_with_clusters['_text_length'] = df_with_clusters[text_column].str.len()
            
            # Select representative (shortest text per cluster)
            representatives = df_with_clusters.loc[
                df_with_clusters.groupby(CLUSTER_COLUMN)['_text_length'].idxmin()
            ]
            
            # Remove temporary column
            representatives = representatives.drop(columns=['_text_length'])
        else:
            # If no text column, just take the first document in each cluster
            representatives = df_with_clusters.groupby(CLUSTER_COLUMN).first().reset_index()
        
        self.logger.info(f"Selected {len(representatives)} representatives from "
                        f"{df_with_clusters[CLUSTER_COLUMN].nunique()} clusters")
        
        return representatives


# ============================================================================
# MAIN LSH CLUSTERING ENGINE
# ============================================================================

class LSHClusteringEngine:
    """Main LSH clustering engine combining all components."""
    
    def __init__(self, config: DeduplicationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.bucketer = LSHBucketer(config, logger)
        self.edge_generator = EdgeGenerator(logger)
        self.clusterer = ConnectedComponentsClustering(logger)
        self.representative_selector = RepresentativeSelector(logger)
    
    def cluster_documents(self, df: cudf.DataFrame, text_column: str = 'text') -> cudf.DataFrame:
        """
        Perform complete LSH clustering pipeline.
        
        Parameters
        ----------
        df : cudf.DataFrame
            DataFrame with MinHash signatures
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with cluster labels added
        """
        self.logger.info(f"Starting LSH clustering for {len(df)} documents")
        
        # Extract signatures
        signatures_list = df[SIGNATURE_COLUMN].to_pandas().tolist()
        signatures_matrix = cp.array(signatures_list)
        
        # Create LSH buckets
        bucket_df = self.bucketer.create_buckets(signatures_matrix)
        
        # Generate edges
        edge_df = self.edge_generator.generate_edges(bucket_df)
        
        # Perform clustering
        cluster_result = self.clusterer.cluster(edge_df, len(df))
        
        # Add cluster labels to original DataFrame
        result_df = df.copy()
        result_df[CLUSTER_COLUMN] = cluster_result['labels']
        
        self.logger.info(f"LSH clustering completed. Found {result_df[CLUSTER_COLUMN].nunique()} clusters")
        
        return result_df
    
    def deduplicate(self, df: cudf.DataFrame, text_column: str = 'text') -> cudf.DataFrame:
        """
        Perform deduplication by clustering and selecting representatives.
        
        Parameters
        ----------
        df : cudf.DataFrame
            DataFrame with MinHash signatures
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with only representative documents
        """
        self.logger.info(f"Starting deduplication for {len(df)} documents")
        
        # Perform clustering
        clustered_df = self.cluster_documents(df, text_column)
        
        # Extract cluster results for representative selection
        cluster_df = cudf.DataFrame({
            'vertex': clustered_df[INDEX_COLUMN],
            'labels': clustered_df[CLUSTER_COLUMN]
        })
        
        # Select representatives
        representatives = self.representative_selector.select_representatives(
            clustered_df, cluster_df, text_column
        )
        
        original_count = len(df)
        final_count = len(representatives)
        dedup_rate = (original_count - final_count) / original_count * 100
        
        self.logger.info(f"Deduplication completed: {original_count} -> {final_count} "
                        f"documents ({dedup_rate:.1f}% reduction)")
        
        return representatives
    
    def get_cluster_statistics(self, df: cudf.DataFrame) -> dict:
        """
        Get clustering statistics.
        
        Parameters
        ----------
        df : cudf.DataFrame
            DataFrame with cluster labels
            
        Returns
        -------
        dict
            Clustering statistics
        """
        if CLUSTER_COLUMN not in df.columns:
            return {'error': 'No cluster information found'}
        
        cluster_sizes = df[CLUSTER_COLUMN].value_counts()
        
        stats = {
            'total_documents': len(df),
            'total_clusters': df[CLUSTER_COLUMN].nunique(),
            'singleton_clusters': (cluster_sizes == 1).sum(),
            'largest_cluster_size': int(cluster_sizes.max()),
            'average_cluster_size': float(cluster_sizes.mean()),
            'duplicate_documents': len(df) - df[CLUSTER_COLUMN].nunique(),
            'deduplication_rate': (len(df) - df[CLUSTER_COLUMN].nunique()) / len(df) * 100
        }
        
        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_lsh_memory(num_docs: int, b: int) -> int:
    """
    Estimate memory usage for LSH bucketing.
    
    Parameters
    ----------
    num_docs : int
        Number of documents
    b : int
        Number of bands
        
    Returns
    -------
    int
        Estimated memory usage in bytes
    """
    # Each document generates b buckets, each bucket is 8 bytes (uint64)
    bucket_memory = num_docs * b * 8
    
    # Additional memory for doc_ids and band_ids
    id_memory = num_docs * b * 4 * 2  # Two int32 arrays
    
    return bucket_memory + id_memory


def create_lsh_engine(config: DeduplicationConfig, logger: Optional[logging.Logger] = None) -> LSHClusteringEngine:
    """
    Factory function to create LSH clustering engine.
    
    Parameters
    ----------
    config : DeduplicationConfig
        Configuration object
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    LSHClusteringEngine
        Configured LSH clustering engine
    """
    return LSHClusteringEngine(config, logger)