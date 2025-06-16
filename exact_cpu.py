import argparse
import hashlib
import os
import time
import logging
from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Literal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def hash_text(text: str, method: Literal["md5", "sha256", "xxh3"] = "md5") -> str:
    if method == "md5":
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    elif method == "sha256":
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    else:
        raise ValueError("Unsupported hash method")

def add_hash(example, method):
    example["__hash"] = hash_text(example["text"], method)
    return example

def deduplicate_dataset(dataset, method="md5", num_proc=1):
    # Processing phase - adding hashes
    processing_start = time.time()
    dataset = dataset.map(lambda ex: add_hash(ex, method), num_proc=num_proc)
    processing_time = time.time() - processing_start
    
    # Filtering phase - removing duplicates
    filtering_start = time.time()
    seen_hashes = set()
    def is_unique(example):
        h = example["__hash"]
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    dataset = dataset.filter(is_unique, num_proc=1)
    filtering_time = time.time() - filtering_start
    
    return dataset, processing_time, filtering_time

def load_multiple_files(input_path, format, column):
    """Load and concatenate multiple files from a directory or single file"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        if format == "jsonl":
            dataset = load_dataset("json", data_files=str(input_path), split="train")
        else:  # parquet
            dataset = load_dataset("parquet", data_files=str(input_path), split="train")
        
        # Rename column if needed
        if column != "text" and column in dataset.column_names:
            dataset = dataset.rename_column(column, "text")
        
        return dataset
    
    elif input_path.is_dir():
        # Multiple files in directory
        if format == "jsonl":
            file_pattern = "*.jsonl"
            dataset_type = "json"
        else:  # parquet
            file_pattern = "*.parquet"
            dataset_type = "parquet"
        
        # Find all matching files
        file_paths = list(input_path.glob(file_pattern))
        
        if not file_paths:
            raise ValueError(f"No {format} files found in directory: {input_path}")
        
        datasets = []
        for file_path in file_paths:
            try:
                dataset = load_dataset(dataset_type, data_files=str(file_path), split="train")
                
                # Rename column if needed
                if column != "text" and column in dataset.column_names:
                    dataset = dataset.rename_column(column, "text")
                
                datasets.append(dataset)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        combined_dataset = concatenate_datasets(datasets)
        
        return combined_dataset
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def save_dataset(dataset, output_path, format):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        dataset.to_json(str(output_path / "deduplicated.jsonl"), orient="records", lines=True)
    elif format == "parquet":
        dataset.to_parquet(str(output_path / "deduplicated.parquet"))
    else:
        raise ValueError("Unsupported format")

def run_pipeline(input_path, column, output_path, format="jsonl", method="md5", num_proc=1):
    total_start = time.time()
    
    # Loading phase
    loading_start = time.time()
    data_files = {"train": input_path} if os.path.isdir(input_path) else input_path
    dataset = load_dataset("json" if format == "jsonl" else "parquet", data_files=data_files, split="train")
    dataset = dataset.rename_column(column, "text") if column != "text" else dataset
    loading_time = time.time() - loading_start
    
    original_count = len(dataset)
    
    # Deduplication (processing + filtering)
    dataset, processing_time, filtering_time = deduplicate_dataset(dataset, method=method, num_proc=num_proc)
    
    deduplicated_count = len(dataset)
    
    # Saving phase
    saving_start = time.time()
    save_dataset(dataset, output_path, format=format)
    saving_time = time.time() - saving_start
    
    total_time = time.time() - total_start
    
    # Log results in the requested format
    logger.info(f"Loading                       : {loading_time:.2f}s")
    logger.info(f"Processing                    : {processing_time:.2f}s")
    logger.info(f"Filtering                     : {filtering_time:.2f}s")
    logger.info(f"Saving                        : {saving_time:.2f}s")
    logger.info(f"Total                         : {total_time:.2f}s")
    logger.info(f"Before                        : {original_count}")
    logger.info(f"After                         : {deduplicated_count}")

def main():
    parser = argparse.ArgumentParser(description="Exact Deduplication Pipeline")
    parser.add_argument("--input-path", required=True, help="Path to input dataset file or directory")
    parser.add_argument("--output-path", required=True, help="Path to output deduplicated dataset")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default="parquet", help="Dataset format")
    parser.add_argument("--column", type=str, default="text", help="Column to deduplicate")
    parser.add_argument("--hash-func", choices=["md5", "sha256"], default="md5", help="Hash function")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input_path,
        column=args.column,
        output_path=args.output_path,
        format=args.format,
        method=args.hash_func,
        num_proc=args.num_proc,
    )

if __name__ == "__main__":
    main()


'''
```bash
#input
python exact_cpu.py \
    --input-path "sample_dataset/sample.parquet" \
    --output-path "sample_dataset/dedup_output.parquet" \
    --format parquet \
    --column text \
    --hash-func md5 \
    --num-proc 4

#output
INFO     Loading                       : 1.57s
INFO     Processing                    : 40.34s
INFO     Filtering                     : 17.09s
INFO     Saving                        : 44.83s
INFO     Total                         : 103.94s
INFO     Before                        : 5000000
INFO     After                         : 2999835
```
'''
