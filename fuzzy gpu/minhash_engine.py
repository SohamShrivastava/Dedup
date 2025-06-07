#!/usr/bin/env python3
"""
Name: minhash_engine.py
GPU-accelerated MinHash signature generation engine for text deduplication.
Supports multiple hash functions, n-gram tokenization, and batch processing.
"""

import logging
import struct
from typing import List, Optional, Tuple, Union, Set
import warnings

import cudf
import cupy as cp
import numpy as np
from numba import cuda, types
from numba.cuda import random

from config import (
    SEED, INDEX_COLUMN, SIGNATURE_COLUMN,
    HASH_FUNCTIONS, DeduplicationConfig
)

# ============================================================================
# HASH FUNCTION IMPLEMENTATIONS
# ============================================================================

@cuda.jit(device=True)
def xxh3_hash_device(data: types.uint64, seed: types.uint64) -> types.uint64:
    """Device function for XXH3 hash (simplified version)."""
    # Simplified XXH3-like hash for GPU
    # This is a fast approximation, not the full XXH3 algorithm
    prime1 = types.uint64(0x9E3779B185EBCA87)
    prime2 = types.uint64(0xC2B2AE3D27D4EB4F)
    prime3 = types.uint64(0x165667B19E3779F9)
    
    h = seed + prime1
    h ^= data
    h *= prime2
    h = ((h << 13) | (h >> 51)) * prime3
    h ^= h >> 32
    h *= prime1
    h ^= h >> 29
    
    return h


@cuda.jit(device=True)
def murmur3_hash_device(data: types.uint64, seed: types.uint32) -> types.uint32:
    """Device function for MurmurHash3."""
    c1 = types.uint32(0xcc9e2d51)
    c2 = types.uint32(0x1b873593)
    r1 = types.uint32(15)
    r2 = types.uint32(13)
    m = types.uint32(5)
    n = types.uint32(0xe6546b64)
    
    # Process as 32-bit chunks
    k = types.uint32(data & 0xFFFFFFFF)
    k *= c1
    k = ((k << r1) | (k >> (32 - r1)))
    k *= c2
    
    hash_val = seed
    hash_val ^= k
    hash_val = ((hash_val << r2) | (hash_val >> (32 - r2)))
    hash_val = hash_val * m + n
    
    # Finalization
    hash_val ^= types.uint32(4)  # length approximation
    hash_val ^= hash_val >> 16
    hash_val *= types.uint32(0x85ebca6b)
    hash_val ^= hash_val >> 13
    hash_val *= types.uint32(0xc2b2ae35)
    hash_val ^= hash_val >> 16
    
    return hash_val


@cuda.jit(device=True)
def simple_hash_device(data: types.uint64, seed: types.uint64) -> types.uint64:
    """Simple hash function for fallback (based on FNV)."""
    fnv_prime = types.uint64(0x100000001b3)
    fnv_offset = types.uint64(0xcbf29ce484222325)
    
    hash_val = fnv_offset ^ seed
    
    # Process 8 bytes
    for i in range(8):
        byte_val = (data >> (i * 8)) & 0xFF
        hash_val ^= byte_val
        hash_val *= fnv_prime
    
    return hash_val


# ============================================================================
# N-GRAM TOKENIZATION
# ============================================================================

class NGramTokenizer:
    """GPU-accelerated n-gram tokenizer."""
    
    def __init__(self, ngram_size: int = 1, logger: Optional[logging.Logger] = None):
        self.ngram_size = ngram_size
        self.logger = logger or logging.getLogger(__name__)
    
    def tokenize_batch(self, texts: cudf.Series) -> List[List[str]]:
        """
        Tokenize a batch of texts into n-grams.
        
        Parameters
        ----------
        texts : cudf.Series
            Batch of texts to tokenize
            
        Returns
        -------
        List[List[str]]
            List of n-gram lists for each text
        """
        if self.ngram_size == 1:
            return self._word_tokenize_batch(texts)
        else:
            return self._ngram_tokenize_batch(texts)
    
    def _word_tokenize_batch(self, texts: cudf.Series) -> List[List[str]]:
        """Fast word tokenization using cuDF string operations."""
        # Convert to host for processing
        texts_host = texts.to_pandas()
        
        # Simple whitespace tokenization
        tokens_batch = []
        for text in texts_host:
            if pd.isna(text):
                tokens_batch.append([])
            else:
                # Simple split on whitespace and punctuation
                tokens = str(text).lower().split()
                # Remove empty tokens
                tokens = [t for t in tokens if t.strip()]
                tokens_batch.append(tokens)
        
        return tokens_batch
    
    def _ngram_tokenize_batch(self, texts: cudf.Series) -> List[List[str]]:
        """Generate character n-grams for a batch of texts."""
        texts_host = texts.to_pandas()
        
        ngrams_batch = []
        for text in texts_host:
            if pd.isna(text):
                ngrams_batch.append([])
            else:
                text_str = str(text).lower()
                ngrams = []
                
                # Generate character n-grams
                for i in range(len(text_str) - self.ngram_size + 1):
                    ngram = text_str[i:i + self.ngram_size]
                    # Skip n-grams that are all whitespace
                    if ngram.strip():
                        ngrams.append(ngram)
                
                ngrams_batch.append(ngrams)
        
        return ngrams_batch


# ============================================================================
# MINHASH SIGNATURE GENERATION
# ============================================================================

@cuda.jit
def compute_minhash_signatures_kernel(
    token_hashes,      # Input: token hashes for all documents
    doc_boundaries,    # Input: document boundary indices
    permutation_a,     # Input: permutation parameters a
    permutation_b,     # Input: permutation parameters b
    signatures,        # Output: minhash signatures
    num_perm,         # Number of permutations
    hash_bits         # Number of hash bits (32 or 64)
):
    """CUDA kernel for computing MinHash signatures."""
    doc_idx = cuda.blockIdx.x
    perm_idx = cuda.threadIdx.x
    
    if doc_idx >= len(doc_boundaries) - 1 or perm_idx >= num_perm:
        return
    
    # Get document boundaries
    start_idx = doc_boundaries[doc_idx]
    end_idx = doc_boundaries[doc_idx + 1]
    
    if start_idx >= end_idx:
        signatures[doc_idx, perm_idx] = 0  # Empty document
        return
    
    # Initialize minimum hash value
    if hash_bits == 32:
        min_hash = types.uint32(0xFFFFFFFF)
    else:
        min_hash = types.uint64(0xFFFFFFFFFFFFFFFF)
    
    # Compute minimum hash for this permutation
    for token_idx in range(start_idx, end_idx):
        token_hash = token_hashes[token_idx]
        
        # Apply permutation: (a * hash + b) mod prime
        if hash_bits == 32:
            prime = types.uint32(0xFFFFFFFB)  # Large prime for 32-bit
            perm_hash = types.uint32((permutation_a[perm_idx] * token_hash + permutation_b[perm_idx]) % prime)
        else:
            prime = types.uint64(0xFFFFFFFFFFFFFFC5)  # Large prime for 64-bit
            perm_hash = types.uint64((permutation_a[perm_idx] * token_hash + permutation_b[perm_idx]) % prime)
        
        if perm_hash < min_hash:
            min_hash = perm_hash
    
    signatures[doc_idx, perm_idx] = min_hash


class MinHashEngine:
    """GPU-accelerated MinHash signature generation engine."""
    
    def __init__(self, config: DeduplicationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tokenizer
        self.tokenizer = NGramTokenizer(config.ngram_size, logger)
        
        # Generate permutation parameters
        self.permutation_a, self.permutation_b = self._generate_permutations()
        
        # Choose hash function
        self.hash_func = self._get_hash_function()
        
        self.logger.info(f"MinHash engine initialized with {config.num_perm} permutations, "
                        f"{config.hash_func} hash function ({config.hash_bits} bits)")
    
    def _generate_permutations(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """Generate random permutation parameters."""
        # Set random seed for reproducibility
        cp.random.seed(SEED)
        
        if self.config.hash_bits == 32:
            max_val = 2**31 - 1
            dtype = cp.uint32
        else:
            max_val = 2**63 - 1
            dtype = cp.uint64
        
        # Generate random coefficients for permutation functions
        # Using the form: (a * x + b) mod prime
        a = cp.random.randint(1, max_val, size=self.config.num_perm, dtype=dtype)
        b = cp.random.randint(0, max_val, size=self.config.num_perm, dtype=dtype)
        
        return a, b
    
    def _get_hash_function(self):
        """Get the appropriate hash function."""
        hash_name = self.config.hash_func.lower()
        
        if hash_name == 'xxh3':
            return self._hash_xxh3
        elif hash_name == 'murmur3':
            return self._hash_murmur3
        elif hash_name == 'sha1':
            return self._hash_simple  # Use simple hash as SHA1 approximation
        else:
            self.logger.warning(f"Unknown hash function {hash_name}, using simple hash")
            return self._hash_simple
    
    def _hash_xxh3(self, token: str) -> int:
        """Compute XXH3 hash of a token."""
        # Convert string to bytes and then to uint64 for hashing
        token_bytes = token.encode('utf-8')
        # Pad or truncate to 8 bytes
        if len(token_bytes) < 8:
            token_bytes = token_bytes.ljust(8, b'\x00')
        else:
            token_bytes = token_bytes[:8]
        
        # Convert to uint64
        token_int = struct.unpack('<Q', token_bytes)[0]
        
        # Simple hash (approximation of XXH3)
        return self._simple_hash_cpu(token_int, SEED)
    
    def _hash_murmur3(self, token: str) -> int:
        """Compute MurmurHash3 of a token."""
        token_bytes = token.encode('utf-8')
        if len(token_bytes) < 8:
            token_bytes = token_bytes.ljust(8, b'\x00')
        else:
            token_bytes = token_bytes[:8]
        
        token_int = struct.unpack('<Q', token_bytes)[0]
        return self._murmur3_cpu(token_int, SEED)
    
    def _hash_simple(self, token: str) -> int:
        """Compute simple hash of a token."""
        token_bytes = token.encode('utf-8')
        if len(token_bytes) < 8:
            token_bytes = token_bytes.ljust(8, b'\x00')
        else:
            token_bytes = token_bytes[:8]
        
        token_int = struct.unpack('<Q', token_bytes)[0]
        return self._simple_hash_cpu(token_int, SEED)
    
    def _simple_hash_cpu(self, data: int, seed: int) -> int:
        """CPU version of simple hash."""
        fnv_prime = 0x100000001b3
        fnv_offset = 0xcbf29ce484222325
        
        hash_val = fnv_offset ^ seed
        
        for i in range(8):
            byte_val = (data >> (i * 8)) & 0xFF
            hash_val ^= byte_val
            hash_val = (hash_val * fnv_prime) & ((1 << 64) - 1)
        
        if self.config.hash_bits == 32:
            return hash_val & 0xFFFFFFFF
        return hash_val
    
    def _murmur3_cpu(self, data: int, seed: int) -> int:
        """CPU version of MurmurHash3."""
        c1 = 0xcc9e2d51
        c2 = 0x1b873593
        r1 = 15
        r2 = 13
        m = 5
        n = 0xe6546b64
        
        k = data & 0xFFFFFFFF
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << r1) | (k >> (32 - r1))) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        
        hash_val = seed
        hash_val ^= k
        hash_val = ((hash_val << r2) | (hash_val >> (32 - r2))) & 0xFFFFFFFF
        hash_val = (hash_val * m + n) & 0xFFFFFFFF
        
        # Finalization
        hash_val ^= 4
        hash_val ^= hash_val >> 16
        hash_val = (hash_val * 0x85ebca6b) & 0xFFFFFFFF
        hash_val ^= hash_val >> 13
        hash_val = (hash_val * 0xc2b2ae35) & 0xFFFFFFFF
        hash_val ^= hash_val >> 16
        
        if self.config.hash_bits == 64:
            # Extend to 64 bits by combining with upper bits
            upper = self._simple_hash_cpu(data >> 32, seed + 1)
            return (upper << 32) | hash_val
        
        return hash_val
    
    def generate_signatures(self, df: cudf.DataFrame, text_column: str) -> cudf.DataFrame:
        """
        Generate MinHash signatures for all documents.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame with documents
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with added signature column
        """
        self.logger.info(f"Generating MinHash signatures for {len(df)} documents")
        
        # Process in batches to manage GPU memory
        batch_size = min(self.config.batch_size, len(df))
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        all_signatures = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            batch_df = df.iloc[start_idx:end_idx]
            batch_signatures = self._process_batch(batch_df[text_column])
            all_signatures.extend(batch_signatures)
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {batch_idx + 1}/{num_batches} batches")
        
        # Add signatures to DataFrame
        df = df.copy()
        df[SIGNATURE_COLUMN] = all_signatures
        
        self.logger.info("MinHash signature generation completed")
        return df
    
    def _process_batch(self, texts: cudf.Series) -> List[List[int]]:
        """Process a batch of texts to generate signatures."""
        # Tokenize texts
        tokens_batch = self.tokenizer.tokenize_batch(texts)
        
        # Hash tokens and prepare for GPU processing
        all_token_hashes = []
        doc_boundaries = [0]
        
        for tokens in tokens_batch:
            # Hash all tokens in this document
            token_hashes = [self.hash_func(token) for token in tokens]
            all_token_hashes.extend(token_hashes)
            doc_boundaries.append(len(all_token_hashes))
        
        if not all_token_hashes:
            # Return empty signatures for empty batch
            return [[] for _ in range(len(texts))]
        
        # Convert to GPU arrays
        if self.config.hash_bits == 32:
            dtype = cp.uint32
        else:
            dtype = cp.uint64
        
        token_hashes_gpu = cp.array(all_token_hashes, dtype=dtype)
        doc_boundaries_gpu = cp.array(doc_boundaries, dtype=cp.int32)
        
        # Prepare output array
        signatures_gpu = cp.zeros((len(texts), self.config.num_perm), dtype=dtype)
        
        # Launch CUDA kernel
        threads_per_block = min(self.config.num_perm, 1024)
        blocks_per_grid = len(texts)
        
        compute_minhash_signatures_kernel[blocks_per_grid, threads_per_block](
            token_hashes_gpu,
            doc_boundaries_gpu,
            self.permutation_a,
            self.permutation_b,
            signatures_gpu,
            self.config.num_perm,
            self.config.hash_bits
        )
        
        # Convert back to CPU and return as lists
        signatures_cpu = signatures_gpu.get()
        return [list(sig) for sig in signatures_cpu]
    
    def compute_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Compute Jaccard similarity between two MinHash signatures.
        
        Parameters
        ----------
        sig1, sig2 : List[int]
            MinHash signatures to compare
            
        Returns
        -------
        float
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        
        if not sig1 or not sig2:
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def get_signature_stats(self, signatures: List[List[int]]) -> dict:
        """
        Get statistics about the generated signatures.
        
        Parameters
        ----------
        signatures : List[List[int]]
            List of MinHash signatures
            
        Returns
        -------
        dict
            Statistics about the signatures
        """
        if not signatures:
            return {'total_signatures': 0}
        
        # Convert to numpy for easier computation
        sig_array = np.array(signatures)
        
        stats = {
            'total_signatures': len(signatures),
            'signature_length': len(signatures[0]) if signatures[0] else 0,
            'unique_signatures': len(set(tuple(sig) for sig in signatures)),
            'duplicate_rate': 1 - len(set(tuple(sig) for sig in signatures)) / len(signatures)
        }
        
        # Hash value distribution stats
        if sig_array.size > 0:
            stats.update({
                'min_hash_value': int(sig_array.min()),
                'max_hash_value': int(sig_array.max()),
                'mean_hash_value': float(sig_array.mean()),
                'std_hash_value': float(sig_array.std())
            })
        
        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_signatures(signatures: List[List[int]], expected_length: int) -> bool:
    """
    Validate that all signatures have the expected length.
    
    Parameters
    ----------
    signatures : List[List[int]]
        List of signatures to validate
    expected_length : int
        Expected signature length
        
    Returns
    -------
    bool
        True if all signatures are valid
    """
    if not signatures:
        return True
    
    return all(len(sig) == expected_length for sig in signatures)


def signatures_to_matrix(signatures: List[List[int]]) -> cp.ndarray:
    """
    Convert list of signatures to CuPy matrix.
    
    Parameters
    ----------
    signatures : List[List[int]]
        List of signatures
        
    Returns
    -------
    cp.ndarray
        Signature matrix on GPU
    """
    if not signatures:
        return cp.array([])
    
    return cp.array(signatures)


def estimate_memory_usage(num_docs: int, num_perm: int, hash_bits: int) -> int:
    """
    Estimate memory usage for signature generation.
    
    Parameters
    ----------
    num_docs : int
        Number of documents
    num_perm : int
        Number of permutations
    hash_bits : int
        Number of hash bits
        
    Returns
    -------
    int
        Estimated memory usage in bytes
    """
    bytes_per_signature = (hash_bits // 8) * num_perm
    total_signature_memory = num_docs * bytes_per_signature
    
    # Add overhead for permutation parameters and intermediate data
    overhead = num_perm * (hash_bits // 8) * 2  # for a and b arrays
    
    return total_signature_memory + overhead


if __name__ == "__main__":
    # Test the MinHash engine
    import logging
    from config import create_config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create test configuration
    config = create_config(
        num_perm=128,
        ngram_size=1,
        hash_func='xxh3',
        hash_bits=32
    )
    
    # Create test data
    test_texts = cudf.Series([
        "This is a test document for MinHash.",
        "This is another test document for MinHash.",
        "Completely different content here.",
        "This is a test document for MinHash.",  # Duplicate
        ""  # Empty document
    ])
    
    test_df = cudf.DataFrame({
        'text': test_texts,
        '__index__': range(len(test_texts))
    })
    
    # Test MinHash engine
    engine = MinHashEngine(config, logger)
    
    # Generate signatures
    result_df = engine.generate_signatures(test_df, 'text')
    
    # Print results
    print(f"Generated signatures for {len(result_df)} documents")
    signatures = result_df[SIGNATURE_COLUMN].to_pandas().tolist()
    
    # Test similarity computation
    if len(signatures) >= 2:
        sim = engine.compute_similarity(signatures[0], signatures[1])
        print(f"Similarity between doc 0 and 1: {sim:.3f}")
        
        sim = engine.compute_similarity(signatures[0], signatures[3])
        print(f"Similarity between doc 0 and 3 (duplicate): {sim:.3f}")
    
    # Get signature statistics
    stats = engine.get_signature_stats(signatures)
    print("Signature statistics:", stats)
    
    # Memory usage estimation
    memory_usage = estimate_memory_usage(len(test_df), config.num_perm, config.hash_bits)
    print(f"Estimated memory usage: {memory_usage / (1024*1024):.2f} MB")