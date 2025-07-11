import cudf
import dask_cudf
import argparse
import os
from pathlib import Path

def split_by_language_with_ids(
    input_parquet_path: str, 
    output_directory: str, 
    data_column: str = "data", 
    lang_column: str = "lang",
    id_column: str = "id"
):
    """
    Split a parquet file by language, add sequential IDs for each language,
    and save each language as a separate parquet file.
    MAINTAINS ORIGINAL FILE ORDERING.
    
    Args:
        input_parquet_path: Path to input parquet file
        output_directory: Directory to save language-specific parquet files
        data_column: Name of the data column
        lang_column: Name of the language column
        id_column: Name of the ID column to add
        
    Returns:
        dict: Dictionary mapping language codes to output file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read parquet file
    print(f"Reading parquet file: {input_parquet_path}")
    df = dask_cudf.read_parquet(input_parquet_path)
    
    # Convert to cudf for easier grouping operations
    df_cudf = df.compute()
    
    # PRESERVE ORIGINAL ORDER: Add original index before any operations
    df_cudf = df_cudf.reset_index().rename(columns={'index': 'original_order'})
    
    # Get unique languages
    unique_languages = df_cudf[lang_column].unique().to_pandas()
    print(f"Found {len(unique_languages)} unique languages: {sorted(unique_languages)}")
    
    output_files = {}
    
    # Process each language
    for lang in unique_languages:
        print(f"Processing language: {lang}")
        
        # Filter data for current language
        lang_data = df_cudf[df_cudf[lang_column] == lang].copy()
        
        # MAINTAIN ORIGINAL ORDER: Sort by original order, not by data content
        lang_data = lang_data.sort_values(by='original_order').reset_index(drop=True)
        
        # Add sequential ID column (0-based) - this will now respect original ordering
        lang_data[id_column] = range(len(lang_data))
        
        # Remove the original_order column as it's no longer needed
        lang_data = lang_data.drop('original_order', axis=1)
        
        # Reorder columns to put ID first
        cols = [id_column] + [col for col in lang_data.columns if col != id_column]
        lang_data = lang_data[cols]
        
        # Create output filename
        output_filename = f"{lang}.parquet"
        output_path = os.path.join(output_directory, output_filename)
        
        # Convert back to dask_cudf for saving
        lang_data_dask = dask_cudf.from_cudf(lang_data, npartitions=1)
        
        # Save to parquet
        lang_data_dask.to_parquet(output_path, write_index=False)
        
        output_files[lang] = output_path
        print(f"  Saved {len(lang_data)} rows to: {output_path}")
    
    print(f"\nProcessing complete! Created {len(output_files)} language-specific files in: {output_directory}")
    
    # Print summary
    print("\nSummary:")
    total_rows = 0
    for lang, path in sorted(output_files.items()):
        # Read back to get actual row count
        temp_df = dask_cudf.read_parquet(path)
        row_count = len(temp_df)
        total_rows += row_count
        print(f"  {lang}: {row_count} rows")
    
    print(f"  Total: {total_rows} rows")
    
    return output_files

def create_embedding_script(output_directory: str, embedding_script_path: str = None):
    """
    Create a batch script to run embeddings for all language files.
    
    Args:
        output_directory: Directory containing language-specific parquet files
        embedding_script_path: Path to save the embedding batch script
    """
    if embedding_script_path is None:
        embedding_script_path = os.path.join(output_directory, "run_embeddings.sh")
    
    # Get all parquet files in the directory
    parquet_files = [f for f in os.listdir(output_directory) if f.endswith('.parquet')]
    
    with open(embedding_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Batch script to create embeddings for all languages\n\n")
        
        for parquet_file in sorted(parquet_files):
            lang = parquet_file.replace('.parquet', '')
            input_path = os.path.join(output_directory, parquet_file)
            output_path = os.path.join(output_directory, f"{lang}_embeddings.parquet")
            
            f.write(f"echo 'Processing {lang}...'\n")
            f.write(f"python embedding_creator.py '{input_path}' '{output_path}' --input-column data\n")
            f.write(f"echo 'Completed {lang}'\n\n")
    
    # Make script executable
    os.chmod(embedding_script_path, 0o755)
    print(f"Created embedding batch script: {embedding_script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split parquet file by language with sequential IDs")
    parser.add_argument("input", help="Input parquet file path")
    parser.add_argument("output_dir", help="Output directory for language-specific files")
    parser.add_argument("--data-column", default="data", help="Name of data column (default: data)")
    parser.add_argument("--lang-column", default="lang", help="Name of language column (default: lang)")
    parser.add_argument("--id-column", default="id", help="Name of ID column to add (default: id)")
    parser.add_argument("--create-batch-script", action="store_true", help="Create batch script for embeddings")
    
    args = parser.parse_args()
    
    # Split by language and add IDs
    output_files = split_by_language_with_ids(
        args.input, 
        args.output_dir, 
        args.data_column, 
        args.lang_column,
        args.id_column
    )
    
    # Optionally create batch script
    if args.create_batch_script:
        create_embedding_script(args.output_dir)