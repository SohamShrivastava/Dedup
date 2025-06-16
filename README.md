# Dedup
python3 /mnt/CFS2/Codegen/Dedup/exact_cpu.py \
  --input-path "/mnt/CFS2/Codegen/tmp_300/*.parquet" \
  --output-path /mnt/CFS2/Codegen/Dedup/output_files \
  --format parquet \
  --column content \
  --num-proc 200
Loading dataset from /mnt/CFS2/Codegen/tmp_300/*.parquet...
Resolving data files: 0.2s
Downloading data: 0.2s
Generating train split: 30662700 examples [14:30, 35214.78 examples/s]
Loading dataset shards: 480/480 (0s)
Original: 30,662,700 rows
Hashing dataset...
Map (num_proc=200): 30662700/30662700 [1:15:16, 6788.88 examples/s]
Removing duplicates...
Filter: 30662700/30662700 [21:13 , 24076.81 examples/s]
Deduplicated: 30,662,700 rows(No exact duplicates found)
Creating parquet from Arrow format: 30663/30663 [35:20 , 14.46ba/s]

Finished in 8796.27 seconds
