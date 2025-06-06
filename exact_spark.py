import argparse
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def timed_step(name, func):
    print(f"\n⏳ Starting: {name}")
    start = time.time()
    result = func()
    end = time.time()
    print(f"✅ Finished: {name} in {end - start:.2f} seconds")
    return result


def create_spark_session(app_name="ExactDeduplication", conf=None):
    builder = SparkSession.builder.appName(app_name)
    if conf:
        for key, val in conf.items():
            builder = builder.config(key, val)
    return builder.getOrCreate()


def deduplicate_dataset(input_path, output_path, dedup_column=None, spark_conf=None):
    spark = create_spark_session(conf=spark_conf)

    def load_data():
        if input_path.endswith(".jsonl") or input_path.endswith(".json"):
            return spark.read.json(input_path)
        elif input_path.endswith(".parquet"):
            return spark.read.parquet(input_path)
        else:
            raise ValueError("Unsupported file format: use .jsonl or .parquet")

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

    df = timed_step("Step 1: Load input", load_data)
    print(f"📊 Input dataset count: {df.count():,}")

    dedup_df = timed_step("Step 2: Deduplicate records", deduplicate)
    print(f"📉 Deduplicated dataset count: {dedup_df.count():,}")

    timed_step("Step 3: Save deduplicated output", save_output)
    spark.stop()


# === CLI support ===
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Exact deduplication using PySpark")
    parser.add_argument("--input-path", required=True, help="Path to input file (.jsonl or .parquet)")
    parser.add_argument("--output-path", required=True, help="Path to output file (.jsonl or .parquet)")
    parser.add_argument("--dedup-column", default=None, help="Column to deduplicate on (default: all columns)")
    parser.add_argument("--executor-memory", default="8g", help="Spark executor memory")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory")
    parser.add_argument("--executor-cores", default="4", help="Spark executor cores")
    parser.add_argument("--shuffle-partitions", default="200", help="Number of shuffle partitions")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_args()
    spark_config = {
        "spark.executor.memory": args.executor_memory,
        "spark.driver.memory": args.driver_memory,
        "spark.executor.cores": args.executor_cores,
        "spark.sql.shuffle.partitions": args.shuffle_partitions,
    }
    deduplicate_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        dedup_column=args.dedup_column,
        spark_conf=spark_config,
    )
