import argparse
import hashlib
import os
import time
from datasets import load_dataset, DatasetDict
from typing import Literal
from pathlib import Path

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
    print("Hashing dataset...")
    dataset = dataset.map(lambda ex: add_hash(ex, method), num_proc=num_proc)

    print("Removing duplicates...")
    seen_hashes = set()
    def is_unique(example):
        h = example["__hash"]
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    dataset = dataset.filter(is_unique, num_proc=1)
    return dataset

def print_stats(name, dataset):
    print(f"{name}: {len(dataset):,} rows")

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
    start = time.time()

    print(f"Loading dataset from {input_path}...")
    data_files = {"train": input_path} if os.path.isdir(input_path) else input_path
    dataset = load_dataset("json" if format == "jsonl" else "parquet", data_files=data_files, split="train")

    dataset = dataset.rename_column(column, "text") if column != "text" else dataset

    print_stats("Original", dataset)

    dataset = deduplicate_dataset(dataset, method=method, num_proc=num_proc)

    print_stats("Deduplicated", dataset)

    save_dataset(dataset, output_path, format=format)

    print(f"\nFinished in {time.time() - start:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Exact Deduplication Pipeline")
    parser.add_argument("--input-path", required=True, help="Path to input dataset file or directory")
    parser.add_argument("--output-path", required=True, help="Path to output deduplicated dataset")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl", help="Dataset format")
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
