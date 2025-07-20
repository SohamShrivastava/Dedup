import os
import time
import tempfile
import shutil
from typing import Optional

import cudf
import cugraph.dask as dcg
import cugraph.dask.comms.comms as cugraph_comms
import dask_cudf
import numpy as np
import pandas as pd
import cupy as cp
from cugraph import MultiGraph
from dask.utils import M


def multi_gpu_connected_components(
    edges_path: str,
    left_id_col: str = "id_x",
    right_id_col: str = "id_y",
    jaccard_col: str = "jaccard",
    jaccard_threshold: float = 0.8,
    output_path: Optional[str] = None
):
    """
    Multi-GPU connected components for large datasets using cuGraph + Dask.
    
    This replicates the NVIDIA ConnectedComponents workflow but with a simple interface.
    Assumes Dask cluster is already running.
    
    Args:
        edges_path: Path to parquet file with edge data
        left_id_col: Name of left ID column 
        right_id_col: Name of right ID column
        jaccard_col: Name of jaccard similarity column
        jaccard_threshold: Minimum jaccard score to consider as duplicate
        output_path: Optional path to save results
        
    Returns:
        cudf.DataFrame with columns: [node_id, group_id] (similar to simple version)
    """
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp(prefix="multi_gpu_cc_")
    
    try:
        print("Starting multi-GPU connected components workflow...")
        
        # Step 1: Write dedup parsed id (extract unique nodes and assign UIDs)
        print("Step 1: Creating unique node mappings...")
        deduped_parsed_id_path = _write_dedup_parsed_id(
            edges_path, left_id_col, right_id_col, cache_dir
        )
        
        # Step 2: Encode jaccard pairs (map original IDs to UIDs)
        print("Step 2: Encoding jaccard pairs...")
        encoded_jaccard_pair_path = _write_encoded_jaccard_pair(
            edges_path, deduped_parsed_id_path, left_id_col, right_id_col, 
            jaccard_col, cache_dir
        )
        
        # Step 3: Dedup encoded jaccard pairs (remove duplicates, apply threshold)
        print("Step 3: Deduplicating encoded pairs...")
        deduped_encoded_jaccard_path = _write_dedup_encoded_jaccard_pair(
            encoded_jaccard_pair_path, left_id_col, right_id_col, 
            jaccard_col, jaccard_threshold, cache_dir
        )
        
        # Step 4: Run connected components
        print("Step 4: Running connected components...")
        result_df = _run_connected_components(
            deduped_encoded_jaccard_path, deduped_parsed_id_path,
            left_id_col, right_id_col, cache_dir
        )
        
        # Convert back to original ID format and rename columns to match simple version
        final_result = _convert_to_simple_format(result_df, left_id_col)
        
        if output_path:
            print(f"Saving to {output_path}")
            final_result.to_parquet(output_path)
            
        print("Multi-GPU connected components completed!")
        return final_result
        
    finally:
        # Cleanup temp directory
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def _write_dedup_parsed_id(edges_path: str, left_id_col: str, right_id_col: str, cache_dir: str) -> str:
    """Extract unique node IDs and assign sequential UIDs"""
    dedup_parsed_id_path = f"{cache_dir}/dedup_parsed_id.parquet"
    
    # Read only the ID columns
    ddf = dask_cudf.read_parquet(
        edges_path,
        columns=[left_id_col, right_id_col],
        blocksize="512MB",
        aggregate_files=True,
    )
    
    # Get unique IDs from both columns
    unique_docs = ddf.map_partitions(_get_unique_ids_per_partition, 
                                   left_id_col=left_id_col, right_id_col=right_id_col)
    unique_docs = unique_docs.drop_duplicates(
        split_out=max(ddf.npartitions // 4, 1)
    )
    
    # Assign sequential UIDs
    unique_docs["uid"] = np.uint64(1)
    unique_docs["uid"] = unique_docs["uid"].cumsum()
    unique_docs["uid"] = unique_docs["uid"] - 1
    
    unique_docs.to_parquet(dedup_parsed_id_path, write_index=False, overwrite=True)
    return dedup_parsed_id_path


def _write_encoded_jaccard_pair(edges_path: str, dedup_parsed_id_path: str, 
                               left_id_col: str, right_id_col: str, jaccard_col: str,
                               cache_dir: str) -> str:
    """Map original IDs to UIDs in the jaccard pairs"""
    output_path = f"{cache_dir}/encoded_jaccard_pair/"
    
    # Load ID mappings
    ddf_id = dask_cudf.read_parquet(dedup_parsed_id_path, blocksize="2GB", aggregate_files=True)
    
    # Load jaccard pairs
    ddf = dask_cudf.read_parquet(edges_path, blocksize="1GB", aggregate_files=True)
    
    # Set index for efficient merging
    base_id_col = left_id_col.replace("_x", "").replace("_y", "")  # Remove _x/_y suffix
    ddf_id = ddf_id.set_index(base_id_col)
    
    # Map both left and right IDs to UIDs
    for tag in ["x", "y"]:
        pair_id = f"{base_id_col}_{tag}"
        ddf = ddf.merge(
            ddf_id,
            left_on=pair_id,
            right_index=True,
            how="inner",
            broadcast=True,
        )
        ddf = ddf.drop(columns=pair_id)
        ddf = ddf.rename(columns={"uid": f"{base_id_col}_{tag}"})
    
    # Keep only the encoded columns and jaccard
    ddf = ddf[[left_id_col, right_id_col, jaccard_col]]
    ddf.to_parquet(output_path, write_index=False, overwrite=True)
    
    return output_path


def _write_dedup_encoded_jaccard_pair(encoded_jaccard_pair_path: str, left_id_col: str, 
                                    right_id_col: str, jaccard_col: str, 
                                    jaccard_threshold: float, cache_dir: str) -> str:
    """Apply threshold, sort IDs, and remove duplicates"""
    output_path = f"{cache_dir}/final_dedup_encoded_jaccard_pair.parquet"
    
    ddf = dask_cudf.read_parquet(encoded_jaccard_pair_path, blocksize="512MB", aggregate_files=True)
    
    meta = {
        left_id_col: "uint64",
        right_id_col: "uint64",
        jaccard_col: "float32",
    }
    
    # Sort IDs within each pair
    ddf = ddf.map_partitions(
        _sort_ids,
        id_columns=[left_id_col, right_id_col],
        meta=meta,
    )
    
    # Apply threshold (convert to binary)
    ddf = ddf.map_partitions(
        _thresholding,
        threshold=jaccard_threshold,
        column_to_threshold=jaccard_col,
        meta=meta,
    )
    
    # Remove duplicates within partitions
    ddf = ddf.map_partitions(
        M.drop_duplicates,
        meta=ddf._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
    )
    
    # Shuffle and remove duplicates across partitions
    ddf = ddf.shuffle(
        [left_id_col, right_id_col],
        ignore_index=True,
        shuffle_method="tasks",
    )
    ddf = ddf.map_partitions(
        M.drop_duplicates,
        meta=ddf._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
    )
    
    ddf.to_parquet(output_path, write_index=False, overwrite=True)
    return output_path


def _run_connected_components(deduped_encoded_jaccard_path: str, deduped_parsed_id_path: str,
                            left_id_col: str, right_id_col: str, cache_dir: str):
    """Run the actual connected components algorithm"""
    
    # Initialize cuGraph communications
    cugraph_comms.initialize(p2p=False)
    
    try:
        # Load filtered edges (only jaccard=1 after thresholding)
        df = dask_cudf.read_parquet(deduped_encoded_jaccard_path, blocksize="1GB", aggregate_files=True)
        df = df[df[df.columns[2]] == 1].reset_index(drop=True)  # jaccard column is index 2
        
        # Load node labels
        labels_df = dask_cudf.read_parquet(deduped_parsed_id_path)
        num_nodes = len(labels_df)
        
        # Add self-edges (each node connected to itself)
        self_edge_df = labels_df[["uid"]].rename(columns={"uid": left_id_col})
        self_edge_df[right_id_col] = self_edge_df[left_id_col]
        
        # Combine edges with self-edges
        df = df[[left_id_col, right_id_col]].astype(np.int64)
        df = dask_cudf.concat([df, self_edge_df])
        
        # Create multigraph and run connected components
        multigraph = MultiGraph(directed=False)
        multigraph.from_dask_cudf_edgelist(df, source=left_id_col, destination=right_id_col, renumber=False)
        result = dcg.weakly_connected_components(multigraph)
        del multigraph
        
        # Get component statistics
        max_partitions = min(32, result.npartitions)
        n_components = len(result[["labels"]].drop_duplicates(split_out=max_partitions))
        num_labels = len(result)
        
        # Merge results with original labels
        labels_df = labels_df.merge(result, left_on=["uid"], right_on=["vertex"], how="inner")
        base_id_col = left_id_col.replace("_x", "").replace("_y", "")
        labels_df = labels_df[[base_id_col, "labels"]]
        labels_df = labels_df.rename(columns={"labels": "group"})
        labels_df = labels_df.persist()
        
        print(f"Found {n_components} connected components")
        print(f"Docs removed: {num_labels - n_components}")
        print(f"Total nodes: {num_nodes}")
        
        if num_nodes != len(labels_df):
            raise ValueError(f"Node count mismatch: {num_nodes} vs {len(labels_df)}")
        
        # Ensure all docs in same group are in same partition
        labels_df = labels_df.shuffle(on=["group"], ignore_index=True)
        
        return labels_df
        
    finally:
        cugraph_comms.destroy()


def _convert_to_simple_format(result_df, left_id_col: str):
    """Convert to the same format as simple_connected_components"""
    base_id_col = left_id_col.replace("_x", "").replace("_y", "")
    result_df = result_df.rename(columns={
        base_id_col: "id", 
        "group": "group_id"
    })
    return result_df.compute()  # Convert to cudf DataFrame


# Helper functions (static methods from original code)

def _sort_ids(df: cudf.DataFrame, id_columns: list[str]) -> cudf.DataFrame:
    """Sort IDs within each pair to ensure consistent ordering"""
    x = df[id_columns].values
    x = cp.sort(x, axis=1)
    for i, id_column in enumerate(id_columns):
        df[id_column] = x[:, i]
        df[id_column] = df[id_column].astype("uint64")
    return df


def _thresholding(df: cudf.DataFrame, threshold: float, column_to_threshold: str) -> cudf.DataFrame:
    """Convert jaccard scores to binary based on threshold"""
    mask = df[column_to_threshold] > threshold
    df.loc[mask, column_to_threshold] = np.int8(1)
    df.loc[~mask, column_to_threshold] = np.int8(0)
    return df


def _get_unique_ids_per_partition(df: cudf.DataFrame, left_id_col: str, right_id_col: str) -> cudf.DataFrame:
    """Extract unique IDs from both columns in a partition"""
    base_id_col = left_id_col.replace("_x", "").replace("_y", "")
    
    unique_df_ls = []
    for col in [left_id_col, right_id_col]:
        subset_df = df[[col]].drop_duplicates(ignore_index=True)
        subset_df = subset_df.rename(columns={col: base_id_col})
        unique_df_ls.append(subset_df)
    
    unique_df = cudf.concat(unique_df_ls, ignore_index=True)
    return unique_df.drop_duplicates(ignore_index=True)


# Usage example
if __name__ == "__main__":
    # Make sure to start Dask cluster first:
    # from dask_cuda import LocalCUDACluster
    # from dask.distributed import Client
    # cluster = LocalCUDACluster()
    # client = Client(cluster)
    
    result_df = multi_gpu_connected_components(
        edges_path="your_edges_file.parquet",
        left_id_col="id_x",
        right_id_col="id_y", 
        jaccard_col="jaccard",
        jaccard_threshold=0.8,
        output_path="/content/output_cc.parquet"
    )
    
    print("\nSample results:")
    print(result_df.head())
    print(f"\nTotal nodes: {len(result_df)}")
    print(f"Total groups: {result_df['group_id'].nunique()}")