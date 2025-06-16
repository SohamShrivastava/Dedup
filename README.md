### Dedup

# Exact Deduplication (CPU)

This script performs **exact deduplication** over a column (default `text`) of a large dataset using parallel hashing and filtering. It is optimized to run with high `--num-proc` values on CPU clusters.

# Script: `exact_cpu.py`

## Features
- Loads `.parquet` or `.jsonl` files
- Hashes selected column contents
- Removes **exact duplicates**
- Writes deduplicated dataset to disk in `.parquet` format

---

# Usage

```bash
#input(on a sample dataset with exact duplicates present)
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

