import cudf
import dask_cudf


def add_id_column(input_parquet_path: str, output_parquet_path: str = None, id_column: str = "id"):
    """
    Add an ID column to a parquet file.
    
    Args:
        input_parquet_path: Path to input parquet file
        output_parquet_path: Path to save output parquet (if None, overwrites input)
        id_column: Name of the ID column to add
        
    Returns:
        str: Path to the output parquet file
    """
    # Read parquet file
    df = dask_cudf.read_parquet(input_parquet_path)
    
    # Add ID column (0-based index)
    df[id_column] = df.index
    
    # Reorder columns to put ID first
    cols = [id_column] + [col for col in df.columns if col != id_column]
    df = df[cols]
    
    # Set output path
    if output_parquet_path is None:
        output_parquet_path = input_parquet_path
    
    # Save to parquet
    df.to_parquet(output_parquet_path, write_index=False)
    
    print(f"Added '{id_column}' column to {output_parquet_path}")
    return output_parquet_path


# Example usage
if __name__ == "__main__":
    # Add ID column and save to new file
    add_id_column("input.parquet", "output_with_id.parquet")
    
    # Or overwrite the original file
    # add_id_column("input.parquet")