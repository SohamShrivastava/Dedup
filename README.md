### Dedup

# Exact Deduplication (CPU)

This script performs **exact deduplication** over a column (default `text`) of a large dataset using parallel hashing and filtering. It is optimized to run with high `--num-proc` values on CPU clusters.
(To do: Add a pre processing filter which will pre-process text before it is run for deduplication)

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

```bash
#input(on around 250gb of codeparrot dataset)
python3 exact_cpu.py \
  --input-path "/mnt/CFS2/Codegen/tmp_300/*.parquet" \
  --output-path /mnt/CFS2/Codegen/Dedup/output_files \
  --format parquet \
  --column content \
  --num-proc 36

#output
INFO     Loading                       : 152.49s
INFO     Processing                    : 7130.26s
INFO     Filtering                     : 1.24s
INFO     Saving                        : 3343.76s
INFO     Total                         : 10627.92s
INFO     Before                        : 30662700
INFO     After                         : 30662700

No exact duplicates found, in this case we can do a modification that the data is not saved again, as it is same as the input data thus saving us time. 
```
-------------------------------------------------------
# Script: `exact_spark.py`

## Features
- Loads `.parquet` or `.jsonl` files using apache spark for faster deduplication
- Hashes selected column contents
- Removes **exact duplicates** 
- Writes deduplicated dataset to disk in `.parquet` format

---

# Usage

```bash
#input(on a sample dataset with exact duplicates present)
python exact_spark.py \
    --input-path "sample_dataset/sample.parquet" \
    --output-path "sample_dataset/dedup_output_spark.parquet" \
    --dedup-column "text" \
    --executor-memory "4g" \
    --driver-memory "4g" \
    --executor-cores "1" \
    --shuffle-partitions "300"

#output
INFO     Loading                       : 0.41s                                  
INFO     Processing                    : 0.01s
INFO     Saving                        : 7.82s
INFO     Total                         : 14.92s
INFO     Before                        : 5000000
INFO     After                         : 2999835
```

```bash
add for 250gb code parrot data here
```





