import argparse
import time
import os
import glob
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def timed_step(name, func):
    print(f"\n Starting: {name}")
    start = time.time()
    result = func()
    end = time.time()
    duration = end - start
    print(f"Finished: {name} in {duration:.2f} seconds")
    return result, duration

def create_spark_session(app_name="ExactDeduplication", conf=None):
    builder = SparkSession.builder.appName(app_name)
    
    # Try local mode, but allow override for memory issues
    master = os.environ.get('SPARK_MASTER', 'local[*]')
    builder = builder.master(master)
    
    if conf:
        for key, val in conf.items():
            # Skip memory configs that might conflict in local mode
            if key in ['spark.executor.memory', 'spark.driver.memory'] and os.environ.get('JAVA_OPTIONS'):
                logger.warning(f"Skipping {key} due to JAVA_OPTIONS conflict")
                continue
            builder = builder.config(key, val)
    
    return builder.getOrCreate()

def deduplicate_dataset(input_path, output_path, dedup_column=None, spark_conf=None, max_files=None):
    spark = create_spark_session(conf=spark_conf)
    
    # Track timing and counts   
    timings = {}
    counts = {}

    def load_data():
        input_path_obj = Path(input_path)
        
        # Check if it's a glob pattern
        if '*' in input_path or '?' in input_path:
            # Handle glob pattern
            matching_files = glob.glob(input_path)
            if not matching_files:
                raise ValueError(f"No files found matching pattern: {input_path}")
            
            # Sort files for consistent ordering
            matching_files.sort()
            
            # Apply max_files limit
            total_files = len(matching_files)
            if max_files and max_files < total_files:
                matching_files = matching_files[:max_files]
                print(f"Loading {len(matching_files)} out of {total_files} files matching pattern")
            else:
                print(f"Found {len(matching_files)} files matching pattern")
            
            # Determine file type from first file
            first_file = matching_files[0]
            if first_file.endswith(('.jsonl', '.json')):
                return spark.read.json(*matching_files)
            elif first_file.endswith('.parquet'):
                return spark.read.parquet(*matching_files)
            else:
                raise ValueError("Unsupported file format in glob pattern: use .jsonl or .parquet")
        
        elif input_path_obj.is_file():
            # Single file - original logic
            if input_path.endswith(".jsonl") or input_path.endswith(".json"):
                return spark.read.json(input_path)
            elif input_path.endswith(".parquet"):
                return spark.read.parquet(input_path)
            else:
                raise ValueError("Unsupported file format: use .jsonl or .parquet")
        
        elif input_path_obj.is_dir():
            # Directory with multiple files
            jsonl_files = list(input_path_obj.glob("*.jsonl")) + list(input_path_obj.glob("*.json"))
            parquet_files = list(input_path_obj.glob("*.parquet"))
            
            if jsonl_files and parquet_files:
                raise ValueError("Directory contains both JSON/JSONL and Parquet files. Please use a directory with only one file type.")
            
            if jsonl_files:
                # Sort files for consistent ordering
                jsonl_files.sort()
                total_files = len(jsonl_files)
                if max_files and max_files < total_files:
                    jsonl_files = jsonl_files[:max_files]
                    print(f"Loading {len(jsonl_files)} out of {total_files} JSON/JSONL files from directory")
                else:
                    print(f"Found {len(jsonl_files)} JSON/JSONL files in directory")
                file_paths = [str(f) for f in jsonl_files]
                return spark.read.json(*file_paths)
            
            elif parquet_files:
                # Sort files for consistent ordering
                parquet_files.sort()
                total_files = len(parquet_files)
                if max_files and max_files < total_files:
                    parquet_files = parquet_files[:max_files]
                    print(f"Loading {len(parquet_files)} out of {total_files} Parquet files from directory")
                else:
                    print(f"Found {len(parquet_files)} Parquet files in directory")
                file_paths = [str(f) for f in parquet_files]
                return spark.read.parquet(*file_paths)
            
            else:
                raise ValueError("No supported files (.jsonl, .json, .parquet) found in directory")
        
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def deduplicate():
        if dedup_column:
            return df.dropDuplicates([dedup_column])
        return df.dropDuplicates()

    def save_output():
        if output_path.endswith(".jsonl") or output_path.endswith(".json"):
            dedup_df.write.mode("overwrite").json(output_path)
        elif output_path.endswith(".parquet"):
            dedup_df.write.mode("overwrite").parquet(output_path)
        else:
            raise ValueError("Output format must be .jsonl or .parquet")

    # Execute pipeline with timing
    total_start = time.time()
    
    df, loading_time = timed_step("Step 1: Load input", load_data)
    timings['Loading'] = loading_time
    
    original_count = df.count()
    counts['Before'] = original_count
    print(f"Input dataset count: {original_count:,}")

    dedup_df, processing_time = timed_step("Step 2: Deduplicate records", deduplicate)
    timings['Processing'] = processing_time
    
    deduplicated_count = dedup_df.count()
    counts['After'] = deduplicated_count
    print(f"Deduplicated dataset count: {deduplicated_count:,}")

    _, saving_time = timed_step("Step 3: Save deduplicated output", save_output)
    timings['Saving'] = saving_time
    
    total_time = time.time() - total_start
    timings['Total'] = total_time
    
    spark.stop()
    
    # Print summary
    print("\n" + "="*50)
    logger.info(f"Loading                       : {timings['Loading']:.2f}s")
    logger.info(f"Processing                    : {timings['Processing']:.2f}s")
    logger.info(f"Saving                        : {timings['Saving']:.2f}s")
    logger.info(f"Total                         : {timings['Total']:.2f}s")
    logger.info(f"Before                        : {counts['Before']}")
    logger.info(f"After                         : {counts['After']}")

# === CLI support ===
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Exact deduplication using PySpark")
    parser.add_argument("--input-path", required=True, help="Path to input file (.jsonl or .parquet), directory containing multiple files, or glob pattern (e.g., /path/to/files/*.parquet)")
    parser.add_argument("--output-path", required=True, help="Path to output file (.jsonl or .parquet)")
    parser.add_argument("--dedup-column", default=None, help="Column to deduplicate on (default: all columns)")
    parser.add_argument("--executor-memory", default="8g", help="Spark executor memory")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory")
    parser.add_argument("--executor-cores", default="4", help="Spark executor cores")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to load from directory (default: load all files)")
    parser.add_argument("--shuffle-partitions", default="200", help="Number of shuffle partitions")
    parser.add_argument("--batch-size", type=int, default=4096, help="Parquet batch size for columnar reader")
    parser.add_argument("--disable-vectorized", action="store_true", help="Disable vectorized Parquet reader")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    spark_config = {
        "spark.executor.memory": args.executor_memory,
        "spark.driver.memory": args.driver_memory,
        "spark.executor.cores": args.executor_cores,
        "spark.sql.shuffle.partitions": args.shuffle_partitions,
        # Parquet optimizations for large files
        "spark.sql.parquet.columnarReaderBatchSize": args.batch_size,
        "spark.sql.parquet.enableVectorizedReader": "false" if args.disable_vectorized else "true",
        # Additional memory and performance settings
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
    }
    deduplicate_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        dedup_column=args.dedup_column,
        spark_conf=spark_config,
        max_files=args.max_files,
    )
