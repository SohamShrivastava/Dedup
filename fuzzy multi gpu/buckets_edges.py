import dask_cudf
import cudf
import pandas as pd
import numpy as np
import pyarrow as pa
from itertools import pairwise
import os
import time
import warnings

class BucketsToEdges:
    """
    Maps buckets generated from LSH into an edgelist that
    can be processed further by Connected Components to find duplicate
    documents
    """
    def __init__(self, 
                 id_field: str = "id",
                 bucket_field: str = "lsh_bucket"):
        """
        Parameters
        ----------
        id_field: str
          Name of the id field of documents in buckets_df
        bucket_field: str
          Column denoting bucket ID
        """
        self.id_field = id_field
        self.output_ids = [f"{self.id_field}_x", f"{self.id_field}_y"]
        self.bucket_field = bucket_field

    def buckets_to_edges(self, buckets_df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Convert LSH buckets to edges for a single partition using cuDF
        """
        # Group by bucket and get list of document IDs, then sort
        grouped_buckets = buckets_df.groupby(self.bucket_field)[self.id_field].agg(list).list.sort_values()
        bucket_docs = grouped_buckets.to_arrow().to_pylist()
        
        edges = []
        # Create pairs of all documents within a bucket since they are near duplicates
        # Effectively create an edge list of all near duplicate documents
        for bucket_doc in bucket_docs:
            edges.extend(pairwise(bucket_doc))
        
        # Convert to pandas DataFrame first, then to PyArrow, then to cuDF
        edges = pd.DataFrame(edges, columns=self.output_ids)
        edges = pa.Table.from_pandas(edges)
        result_df = cudf.DataFrame.from_arrow(edges)
        del edges
        
        # Remove duplicates and add jaccard score
        result_df = result_df.drop_duplicates(self.output_ids).reset_index(drop=True)
        result_df["jaccard"] = np.float32(1.0)
        
        return result_df

    def __call__(self, input_parquet_path: str, output_parquet_path: str) -> dask_cudf.DataFrame:
        """
        Main call function that converts buckets to edges
        
        Parameters
        ----------
        input_parquet_path: str
            Path to input parquet file with columns [id_field, bucket_field]
        output_parquet_path: str
            Path to save output parquet file with edges
        
        Returns
        -------
        dask_cudf.DataFrame: Edge list with columns [id_x, id_y, jaccard]
        """
        
        print("Starting conversion of LSH Buckets to Graph Edgelist")
        print(f"Reading input from: {input_parquet_path}")
        
        # Read input parquet file
        buckets_df = dask_cudf.read_parquet(input_parquet_path, split_row_groups=False)
        
        # Validate input columns
        if self.id_field not in buckets_df.columns:
            raise ValueError(f"ID column '{self.id_field}' not found in DataFrame")
        if self.bucket_field not in buckets_df.columns:
            raise ValueError(f"Bucket column '{self.bucket_field}' not found in DataFrame")
        
        # Define output meta (schema) for dask-cudf
        meta = [(output_id, buckets_df[self.id_field].dtype) for output_id in self.output_ids]
        meta.append(("jaccard", np.float32))
        
        # Apply buckets_to_edges to each partition
        edges_df = buckets_df.map_partitions(self.buckets_to_edges, meta=meta)
        
        # Save to output parquet file
        if os.path.exists(output_parquet_path):
            warnings.warn(f"Output path {output_parquet_path} already exists and will be overwritten", stacklevel=2)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_parquet_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        t0 = time.time()
        edges_df.to_parquet(output_parquet_path, write_index=False, overwrite=True)
        
        print(f"Time taken for Converted Buckets To Edgelist = {time.time() - t0}s and output written at {output_parquet_path}")
        
        # Return dask-cudf DataFrame reading from saved file
        return dask_cudf.read_parquet(output_parquet_path, split_row_groups=False)