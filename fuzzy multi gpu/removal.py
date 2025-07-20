import dask.dataframe as dd
import warnings
import argparse

def deduplicate_groups(duplicates: dd.DataFrame, group_field: str | None, perform_shuffle: bool) -> dd.DataFrame:
    if group_field is None:
        return duplicates
    if perform_shuffle:  # noqa: SIM108
        # Redistribute data across partitions so that all duplicates are in same partition
        duplicates_shuffled = duplicates.shuffle(on=[group_field], ignore_index=True)
    else:
        duplicates_shuffled = duplicates
    return (
        duplicates_shuffled
        # For each partition, keep only the duplicated rows (excluding first occurrence)
        .map_partitions(lambda x: x[x[group_field].duplicated(keep="first")]).drop(columns=group_field)
    )

def left_anti_join(
    left: dd.DataFrame,
    right: dd.DataFrame,
    left_on: str | list[str],
    right_on: str | list[str],
) -> dd.DataFrame:
    if left_on == right_on:
        msg = "left_on and right_on cannot be the same"
        raise ValueError(msg)
    merge = left.merge(
        right=right,
        how="left",
        broadcast=True,  # Broadcast smaller DataFrame to all partitions
        left_on=left_on,
        right_on=right_on,
    )
    # This effectively removes all rows that were not in duplicates_to_remove
    return merge[merge[right_on].isna()].drop(columns=[right_on])

def remove_duplicates(
    left: dd.DataFrame,
    duplicates: dd.DataFrame,
    id_field: str,
    group_field: str | None = None,
    perform_shuffle: bool = False,
) -> dd.DataFrame:
    left_npartitions = left.optimize().npartitions
    right_npartitions = duplicates.optimize().npartitions
    if left_npartitions < right_npartitions:
        msg = (
            f"The number of partitions in `dataset` ({left_npartitions}) is less than "
            f"the number of partitions in the duplicates ({right_npartitions}). "
            "This may lead to a shuffle join. Repartitioning right dataset to match left partitions."
            "To control this behavior, call identify_duplicates and removal as two separate steps"
        )
        warnings.warn(msg, stacklevel=2)
        duplicates = duplicates.repartition(npartitions=left_npartitions)
    # Create a new column name for temporary ID storage during merge
    new_id_field = f"{id_field}_new"
    duplicates_to_remove = (
        deduplicate_groups(duplicates, group_field, perform_shuffle)
        # Rename the ID field to avoid conflicts in the upcoming merge
        .rename(columns={id_field: new_id_field})[[new_id_field]]
    )
    return left_anti_join(
        left=left,
        right=duplicates_to_remove,
        left_on=id_field,
        right_on=new_id_field,
    )

def deduplicate_parquet_data(input_path: str, duplicates_path: str, output_path: str, id_field: str, group_field: str | None = None, perform_shuffle: bool = False):
    """
    Deduplicate parquet data using the provided deduplication functions.
    
    Args:
        input_path: Path to directory containing parquet files with your main data
        duplicates_path: Path to parquet file containing duplicates to remove
        output_path: Path where deduplicated data should be saved
        id_field: The field name to use for matching IDs
        group_field: The field name to use for grouping duplicates
        perform_shuffle: Whether to perform shuffle for group-based deduplication
    """
    
    # Load your main dataset from parquet files
    print("Loading main dataset...")
    main_data = dd.read_parquet(input_path)
    print(f"Main dataset loaded: {len(main_data)} rows, {main_data.npartitions} partitions")
    
    # Load duplicates data from parquet
    print("Loading duplicates data...")
    duplicates_data = dd.read_parquet(duplicates_path)
    
    # Check if we need to rename node_id to id (only if node_id exists and id_field is 'id')
    if 'node_id' in duplicates_data.columns and id_field == 'id':
        duplicates_data = duplicates_data.rename(columns={'node_id': 'id'})
    
    print(f"Duplicates loaded: {len(duplicates_data)} rows")
    
    # Perform deduplication
    print("Performing deduplication...")
    deduplicated_data = remove_duplicates(
        left=main_data,
        duplicates=duplicates_data,
        id_field=id_field,
        group_field=group_field,
        perform_shuffle=perform_shuffle
    )
    
    # Save the result
    print(f"Saving deduplicated data to {output_path}...")
    deduplicated_data.to_parquet(output_path)
    
    print("Deduplication complete!")
    print(f"Original records: {len(main_data)}")
    print(f"Records after deduplication: {len(deduplicated_data)}")
    print(f"Records removed: {len(main_data) - len(deduplicated_data)}")

def main():
    parser = argparse.ArgumentParser(description="Deduplicate parquet data")
    parser.add_argument("--input", required=True, help="Path to directory containing parquet files with your main data")
    parser.add_argument("--duplicates", required=True, help="Path to parquet file containing duplicates to remove")
    parser.add_argument("--output", required=True, help="Path where deduplicated data should be saved")
    parser.add_argument("--id-field", required=True, help="The field name to use for matching IDs")
    parser.add_argument("--group-field", help="The field name to use for grouping duplicates")
    parser.add_argument("--shuffle", action="store_true", help="Whether to perform shuffle for group-based deduplication")
    
    args = parser.parse_args()
    
    deduplicate_parquet_data(
        input_path=args.input,
        duplicates_path=args.duplicates,
        output_path=args.output,
        id_field=args.id_field,
        group_field=args.group_field,
        perform_shuffle=args.shuffle
    )

if __name__ == "__main__":
    main()

# Alternative: If you want to work with the data directly in memory
def deduplicate_in_memory_example():
    """
    Example of how to use the deduplication functions directly
    """
    # Load your data
    main_data = dd.read_parquet("/path/to/your/parquet/directory")
    duplicates_data = dd.read_parquet("/path/to/your/duplicates.parquet")
    
    # Rename node_id to id to match
    duplicates_data = duplicates_data.rename(columns={'node_id': 'id'})
    
    # Remove duplicates
    result = remove_duplicates(
        left=main_data,
        duplicates=duplicates_data,
        id_field='id',
        group_field='group_id',
        perform_shuffle=True
    )
    
    # Compute result (this will actually execute the computation)
    result = result.compute()  # Be careful with this if data is very large
    
    return result