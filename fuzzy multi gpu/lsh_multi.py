import math
import os
import warnings
import numpy as np
import cudf
import dask_cudf


def check_empty_buckets(bucket_path: str) -> bool:
    """
    Inspects parquet metadata of the buckets dataset to check if it's an empty dataset.
    """
    import pyarrow.dataset as ds
    dataset = ds.dataset(bucket_path, format="parquet")
    for fragment in dataset.get_fragments():  # noqa: SIM110
        if fragment.metadata.num_rows > 0:
            return False
    return True


class SimpleLSH:
    """
    Performs LSH on MinHash signatures from a parquet file
    """
    def __init__(
        self,
        num_hashes: int,
        num_buckets: int,
        buckets_per_shuffle: int = 1,
        id_field: str = "id",
        minhash_field: str = "minhash_signature",
    ):
        """
        Parameters
        ----------
        num_hashes: Length of minhash signature
        num_buckets: Number of bands/buckets to create from the minhash signature
        buckets_per_shuffle: Number of bands/buckets to shuffle concurrently
        id_field: Column name for document ID
        minhash_field: Column name for minhash signature
        """
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.id_field = id_field
        self.minhash_field = minhash_field
        self.buckets_per_shuffle = buckets_per_shuffle
        self.bucket_ranges = self._generate_bucket_ranges(num_buckets, num_hashes)
    
    def _generate_bucket_ranges(self, num_buckets: int, num_hashes: int) -> list[list[int]]:
        """
        Generates bucket ranges for minhash signatures
        """
        minhashes_per_bucket = num_hashes // num_buckets
        return [
            list(range(bucket * minhashes_per_bucket, (bucket + 1) * minhashes_per_bucket))
            for bucket in range(num_buckets)
        ]
    
    def minhash_to_buckets(self, df: cudf.DataFrame, bucket_ranges: list[list[int]]) -> cudf.DataFrame:
        """
        Convert minhash signatures to bucket IDs
        """
        df2 = df[[self.id_field]]
        for i, h in enumerate(bucket_ranges):
            indices = cudf.Series([h]).repeat(len(df2))
            df2[f"_bucket_{i}"] = f"b{i}_" + df[self.minhash_field].list.take(indices).hash_values(method="md5")
        return df2
    
    def _get_meta(self, df: dask_cudf.DataFrame) -> cudf.DataFrame:
        """
        Generate metadata for map_partitions
        """
        meta = df._meta_nonempty[[self.id_field]]
        meta[self.minhash_field] = [np.ones(self.num_hashes)] * len(meta)
        return self.minhash_to_buckets(meta, self.bucket_ranges)
    
    def _write_buckets(self, df: dask_cudf.DataFrame, write_path: str, overwrite: bool = True):
        """
        Write buckets to parquet file
        """
        if overwrite and os.path.exists(write_path):
            warnings.warn(f"Output path {write_path} already exists and will be overwritten", stacklevel=2)
        
        df.to_parquet(
            write_path,
            write_index=False,
            overwrite=overwrite,
            append=not overwrite,
            ignore_divisions=True if not overwrite else False,
        )
    
    def lsh(self, input_parquet: str, output_parquet: str) -> dask_cudf.DataFrame:
        """
        Perform LSH on minhash signatures from parquet file
        
        Parameters
        ----------
        input_parquet: Path to input parquet file with id and minhash columns
        output_parquet: Path to output parquet file for buckets
        
        Returns
        -------
        dask_cudf.DataFrame: DataFrame with document pairs in same buckets
        """
        # Read input parquet
        df = dask_cudf.read_parquet(input_parquet)
        
        # Convert minhashes to buckets
        meta = self._get_meta(df)
        df = df.map_partitions(
            self.minhash_to_buckets,
            bucket_ranges=self.bucket_ranges,
            meta=meta,
        )
        
        wrote_buckets = False
        are_buckets_empty = True
        
        # Process buckets in batches
        for i in range(0, self.num_buckets, self.buckets_per_shuffle):
            bucket_columns = [
                f"_bucket_{j}" 
                for j in range(i, min(self.num_buckets, i + self.buckets_per_shuffle))
            ]
            
            # Melt to long format
            df_batch = df.melt(
                id_vars=[self.id_field],
                value_name="_bucket_id",
                value_vars=bucket_columns,
            )[[self.id_field, "_bucket_id"]]
            
            # Shuffle and filter for duplicates
            # Use a more conservative partition count to avoid integer overflow
            target_partitions = min(df_batch.npartitions, 64)  # Cap at 64 partitions
            df_batch = df_batch.shuffle(
                on=["_bucket_id"],
                ignore_index=True,
                npartitions=target_partitions,
            ).map_partitions(lambda x: x[x["_bucket_id"].duplicated(keep=False)])
            
            df_batch = df_batch.reset_index(drop=True)
            
            # Write batch to parquet
            self._write_buckets(df_batch, output_parquet, overwrite=not wrote_buckets)
            wrote_buckets = True
            
            # Check if buckets written so far are empty using NVIDIA's method
            if are_buckets_empty:
                are_buckets_empty = check_empty_buckets(output_parquet)
        
        # If no duplicates found, clean up
        if are_buckets_empty:
            if os.path.exists(output_parquet):
                import shutil
                shutil.rmtree(output_parquet)
            
            # Create empty dataframe with correct schema
            empty_df = cudf.DataFrame({
                self.id_field: cudf.Series([], dtype='object'),
                '_bucket_id': cudf.Series([], dtype='object')
            })
            empty_ddf = dask_cudf.from_cudf(empty_df, npartitions=1)
            self._write_buckets(empty_ddf, output_parquet, overwrite=True)
        
        # Return the result dataframe
        return dask_cudf.read_parquet(output_parquet, split_row_groups=False)
    
    def __call__(self, input_parquet: str, output_parquet: str) -> dask_cudf.DataFrame:
        """
        Main entry point for LSH processing
        
        Parameters
        ----------
        input_parquet: Path to input parquet file with minhash signatures
        output_parquet: Path to output parquet file for duplicate buckets
        
        Returns
        -------
        dask_cudf.DataFrame: DataFrame containing document pairs in same buckets
        """
        return self.lsh(input_parquet, output_parquet)

# Usage example:
# lsh = SimpleLSH(num_hashes=128, num_buckets=32, id_field="doc_id", minhash_field="minhash")
# result_df = lsh("input_minhashes.parquet", "output_buckets.parquet")