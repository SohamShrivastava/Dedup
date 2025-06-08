import cudf
import cugraph
import pandas as pd

def simple_connected_components(
    edges_path: str,
    left_id_col: str = "id_x",
    right_id_col: str = "id_y", 
    jaccard_col: str = "jaccard",
    jaccard_threshold: float = 0.8,
    output_path: str = None
):
    """
    Simple connected components for small datasets using single-GPU cuGraph.
    
    Args:
        edges_path: Path to parquet file with edge data
        left_id_col: Name of left ID column 
        right_id_col: Name of right ID column
        jaccard_col: Name of jaccard similarity column
        jaccard_threshold: Minimum jaccard score to consider as duplicate
        output_path: Optional path to save results
        
    Returns:
        cuDF DataFrame with columns: [node_id, group_id]
    """
    print("Loading edges...")
    # Read edges
    try:
        edges_df = cudf.read_parquet(edges_path)
    except:
        # Fallback to pandas if cudf fails
        edges_df = pd.read_parquet(edges_path)
        edges_df = cudf.from_pandas(edges_df)
    
    print(f"Loaded {len(edges_df)} edges")
    print(f"Columns: {list(edges_df.columns)}")
    
    # Filter by jaccard threshold if column exists
    if jaccard_col in edges_df.columns:
        print(f"Filtering by {jaccard_col} >= {jaccard_threshold}")
        edges_df = edges_df[edges_df[jaccard_col] >= jaccard_threshold]
        print(f"After filtering: {len(edges_df)} edges")
    
    if len(edges_df) == 0:
        print("No edges after filtering!")
        return cudf.DataFrame({"node_id": [], "group_id": []})
    
    # Create graph
    print("Creating graph...")
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        edges_df, 
        source=left_id_col, 
        destination=right_id_col
    )
    
    # Run connected components
    print("Running connected components...")
    cc_result = cugraph.connected_components(G)
    
    # Rename columns for consistency
    cc_result = cc_result.rename(columns={
        "vertex": "node_id",
        "labels": "group_id"
    })
    
    print(f"Found {cc_result['group_id'].nunique()} connected components")
    print(f"Result shape: {cc_result.shape}")
    
    if output_path:
        print(f"Saving to {output_path}")
        cc_result.to_parquet(output_path)
    
    return cc_result

# Usage
if __name__ == "__main__":
    result_df = simple_connected_components(
        edges_path="your_edges_file.parquet",
        left_id_col="id_x",
        right_id_col="id_y",
        jaccard_col="jaccard",
        jaccard_threshold=0.8,
        output_path="/content/output_cc.parquet"
    )
    
    print("\nSample results:")
    print(result_df.head())
    
    # Show some statistics
    print(f"\nTotal nodes: {len(result_df)}")
    print(f"Total groups: {result_df['group_id'].nunique()}")
    
    # Show group sizes
    group_sizes = result_df.groupby('group_id').size().sort_values(ascending=False)
    print(f"\nLargest groups:")
    print(group_sizes.head())