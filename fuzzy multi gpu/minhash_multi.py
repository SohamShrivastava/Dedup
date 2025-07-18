import os
import argparse
import numpy as np
import cudf
import dask_cudf
from pathlib import Path
from typing import List, Union

BIT_WIDTH_32 = 32
BIT_WIDTH_64 = 64

class MinHashGenerator:
    """
    Simplified MinHash signature generator for JSONL and Parquet files
    """
    
    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        column_field: str = "text"
    ):
        """
        Parameters
        ----------
        seed: Seed for minhash permutations
        num_hashes: Length of minhash signature (No. of minhash permutations)
        char_ngrams: Width of text window (in characters) while computing minhashes
        use_64bit_hash: Whether to use a 64 bit hash function
        column_field: Column name to compute minhashes on
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngrams
        self.column_field = column_field
        
        self.seeds = self._generate_hash_permutation_seeds(
            bit_width=64 if use_64bit_hash else 32,
            n_permutations=self.num_hashes,
            seed=seed,
        )
        
        self.minhash_method = self._minhash64 if use_64bit_hash else self._minhash32
    
    def _generate_hash_permutation_seeds(self, bit_width: int, n_permutations: int = 260, seed: int = 0) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)
        
        if bit_width == BIT_WIDTH_32:
            mersenne_prime = np.uint32((1 << 31) - 1)
            dtype = np.uint32
        elif bit_width == BIT_WIDTH_64:
            mersenne_prime = np.uint64((1 << 61) - 1)
            dtype = np.uint64
        else:
            raise ValueError("Unsupported bit width. Use either 32 or 64.")
        
        return np.array(
            [
                (
                    gen.randint(1, mersenne_prime, dtype=dtype),
                    gen.randint(0, mersenne_prime, dtype=dtype),
                )
                for _ in range(n_permutations)
            ],
            dtype=dtype,
        )
    
    def _minhash32(self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")
        
        seeds_a = cudf.Series(seeds[:, 0], dtype="uint32")
        seeds_b = cudf.Series(seeds[:, 1], dtype="uint32")
        
        return ser.str.minhash(a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram)
    
    def _minhash64(self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")
        
        seeds_a = cudf.Series(seeds[:, 0], dtype="uint64")
        seeds_b = cudf.Series(seeds[:, 1], dtype="uint64")
        
        return ser.str.minhash64(a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram)
    
    def _get_input_files(self, input_path: str) -> List[str]:
        """
        Get list of input files from either a single file or directory
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            # Single file input
            return [str(input_path)]
        elif input_path.is_dir():
            # Directory input - find all .jsonl and .parquet files
            files = []
            for pattern in ['*.jsonl', '*.parquet']:
                files.extend(input_path.glob(pattern))
            
            if not files:
                raise ValueError(f"No .jsonl or .parquet files found in directory: {input_path}")
            
            return [str(f) for f in sorted(files)]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    
    def _load_data(self, input_files: List[str]) -> dask_cudf.DataFrame:
        """
        Load data from JSONL or Parquet files
        """
        if len(input_files) == 1:
            # Single file
            input_file = Path(input_files[0])
            
            if input_file.suffix.lower() == '.jsonl':
                df = dask_cudf.read_json(str(input_file), lines=True, blocksize="256MB")
            elif input_file.suffix.lower() == '.parquet':
                df = dask_cudf.read_parquet(str(input_file), blocksize="256MB")
            else:
                raise ValueError(f"Unsupported file format: {input_file.suffix}. Use .jsonl or .parquet")
        else:
            # Multiple files - need to separate by type
            jsonl_files = [f for f in input_files if Path(f).suffix.lower() == '.jsonl']
            parquet_files = [f for f in input_files if Path(f).suffix.lower() == '.parquet']
            
            dfs = []
            
            if jsonl_files:
                print(f"Loading {len(jsonl_files)} JSONL files...")
                for file in jsonl_files:
                    df_jsonl = dask_cudf.read_json(file, lines=True, blocksize="256MB")
                    dfs.append(df_jsonl)
            
            if parquet_files:
                print(f"Loading {len(parquet_files)} Parquet files...")
                # Can read multiple parquet files at once
                df_parquet = dask_cudf.read_parquet(parquet_files, blocksize="256MB")
                dfs.append(df_parquet)
            
            if not dfs:
                raise ValueError("No valid files found")
            
            # Concatenate all dataframes without resetting index
            df = dask_cudf.concat(dfs, ignore_index=False)
        
        # Validate required columns exist
        if self.column_field not in df.columns:
            raise ValueError(f"Column field '{self.column_field}' not found in data")
        
        if 'id' not in df.columns:
            raise ValueError("'id' column not found in input data")
        
        return df
    
    def __call__(self, input_path: str, output_path: str):
        """
        Generate MinHash signatures for input data and save to Parquet
        
        Parameters
        ----------
        input_path: Path to input file, or directory containing JSONL/Parquet files
        output_path: Path to output Parquet file
        """
        print(f"Scanning input path: {input_path}")
        input_files = self._get_input_files(input_path)
        print(f"Found {len(input_files)} file(s) to process")
        
        print(f"Loading data...")
        df = self._load_data(input_files)
        
        print(f"Computing MinHash signatures...")
        # Compute minhash signatures while preserving the original dataframe structure
        def compute_minhash_for_partition(partition):
            # Create a copy to avoid modifying original
            result_partition = partition[['id']].copy()
            # Compute minhash signatures for this partition
            minhash_sigs = self.minhash_method(
                partition[self.column_field], 
                self.seeds, 
                self.char_ngram
            )
            result_partition["minhash_signature"] = minhash_sigs
            return result_partition
        
        result = df.map_partitions(compute_minhash_for_partition, meta={'id': 'object', 'minhash_signature': 'object'})
                
        print(f"Saving results to {output_path}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Parquet
        result.to_parquet(output_path, write_index=False, overwrite=True)
        
        print(f"MinHash signatures saved to {output_path}")


# Example usage:
# Single file input:
# generator = MinHashGenerator(column_field="text")
# generator("input.jsonl", "output.parquet")
# 
# Directory input:
# generator = MinHashGenerator(num_hashes=128, use_64bit_hash=True, column_field="content")  
# generator("/path/to/data_directory/", "output.parquet")
#
# Mixed directory with both .jsonl and .parquet files:
# generator = MinHashGenerator(column_field="text")
# generator("/path/to/mixed_files/", "combined_output.parquet")