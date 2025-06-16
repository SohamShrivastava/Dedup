###Dedup

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
python3 /path/to/exact_cpu.py \
  --input-path "/path/to/input/*.parquet" \
  --output-path /path/to/output_directory \
  --format parquet \
  --column text \
  --num-proc 10

Stats from a sample run:
Original: 30,662,700 rows
Hashing dataset... 
Map (num_proc=200): 30662700/30662700 [1:15:16, 6788.88 examples/s]
Removing duplicates... 
Filter: 30662700/30662700 [21:13 , 24076.81 examples/s]
Deduplicated: 30,662,700 rows (No exact duplicates found)
Creating parquet from Arrow format...
Finished in 8796.27 seconds
``` 

