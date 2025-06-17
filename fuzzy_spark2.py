# Local MinHashLSH Deduplication Pipeline
# Optimized for TB-scale data deduplication

import argparse
import math
import re
import sys
import time
import warnings
import logging
from itertools import tee
from typing import List
from typing import Set
from typing import Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    import numpy.typing as npt
    import pyspark
    import xxhash
    from graphframes import GraphFrame  # type: ignore
    from pyspark import SparkConf
    from pyspark.sql import DataFrame
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import BooleanType
    from scipy.integrate import quad as integrate

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
DTYPE = np.uint32
MAX_HASH = 4_294_967_295  # maximum 32-bit unsigned integer
MOD_PRIME = 4_294_967_291  # maximum 32-bit prime number


def setup_logging():
    """Setup Python logging to work with Spark"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.

    Examples
    --------
    >>> generate_edges([1, 2, 3])
    [(2, 1), (3, 1)]
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]

# region: Hashing
def ngrams(sequence: List[str], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=1))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=5))
    []
    >>> list(ngrams(["a", "b"], 3, min_length=1))
    [('a', 'b')]
    """
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)

def ngram_hashes(content: str, n: int, min_length: int = 5) -> Set[int]:
    """
    Return the ngrams in hash values. This function fuses few steps together for performance reasons.

    Parameters
    ----------
    content : str
        The content of the document.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    Set[int]
        The set of ngrams in hash values.

    Examples
    --------
    >>> sorted(list(ngrams("a b c d", 2, min_length=1)))
    [145323813, 433422276, 459146835]
    >>> list(ngrams("a b c d", 2, min_length=5))
    []
    >>> list(ngrams("a b", 3, min_length=1))
    [433422276]
    """
    tokens: List[str] = NON_ALPHA.split(content.lower())
    ng: set[bytes] = {bytes(" ".join(t).lower(), "utf-8") for t in ngrams(tokens, n, min_length)}
    return {xxhash.xxh32_intdigest(n) for n in ng}

def ngrams_length_check(content: str, n: int, min_length: int = 5) -> bool:
    """
    Return the ngrams in hash values. This function fuses few steps together for performance reasons.

    Parameters
    ----------
    content : str
        The content of the document.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
        bool
        True if at least one ngram meets the `min_length` requirement, otherwise False.

    Examples
    --------
    >>> ngrams_length_check("a b c d", 2, min_length=1)
    True
    >>> ngrams_length_check("a b c d", 2, min_length=5)
    False
    >>> ngrams_length_check("a b", 3, min_length=1)
    True
    """
    tokens: List[str] = NON_ALPHA.split(content.lower())
    return len(tokens) >= min_length

def generate_hash_values(
    content: str,
    idx: int,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: Tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]],
) -> List[Tuple[int, bytes, int]]:
    """
    Generate the MinHashLSH values for a given document.

    Parameters
    ----------
    content : str
        The content of the document.
    idx : int
        The index of the document.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of the n-grams.
    min_length : int
        The minimum number of tokens in a document.
    hashranges : list
        The ranges of offsets for each hash value.
    permutations : Tuple[np.ndarray, np.ndarray]
        The permutations for the hash values.

    Returns
    -------
    List[Tuple[int, bytes, int]]
        The list of (band_idx, hash value, idx) for the document.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> PERMUTATIONS = (
    ...     RNG.randint(1, MOD_PRIME, size=(num_perm,), dtype=DTYPE),
    ...     RNG.randint(0, MOD_PRIME, size=(num_perm,), dtype=DTYPE),
    ... )
    >>> res = generate_hash_values(content, idx, num_perm, ngram_size, 0, hashranges, PERMUTATIONS)
    >>> len(res)
    10
    >>> sum(len(h) for _, h, _ in res) == len(res) * 25 * np.dtype(DTYPE).itemsize
    True
    """
    a, b = permutations
    hashes = np.array(list(ngram_hashes(content, ngram_size, min_length)), dtype=DTYPE)
    p_hashes = ((np.outer(hashes, a) + b) % MOD_PRIME) & MAX_HASH
    min_hashes = np.vstack([p_hashes, np.full(num_perm, MAX_HASH, dtype=DTYPE)]).min(axis=0)
    return [(band_idx, min_hashes[start:end].data.tobytes(), idx) for band_idx, (start, end) in enumerate(hashranges)]
# endregion
# region: MinHashLSH
def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """
    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(area, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(area, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt
# endregion
# region: IO
def partitioned_save(df: DataFrame, chunk_size: int, max_partitions: int, output: str):
    """
    Save a Spark DataFrame to a local directory in batches of `chunk_size` rows. PySpark natively does not support this
    functionality, so this workaround is necessary.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The Spark DataFrame to save.
    chunk_size : int
        The number of rows per batch.
    max_partitions : int
        The maximum number of partitions.
    output : str
        The local output directory.

    Raises
    ------
    RuntimeError
        If the save fails.
    """

    total_rows = df.count()
    partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))

    (
        df.repartition(partitions)
        .withColumn("__pid__", F.spark_partition_id())
        .write.partitionBy("__pid__")
        .parquet(output, mode="overwrite", compression="snappy")
    )


# endregion


if __name__ == "__main__":  # pragma: no cover
    # Setup logging first
    logger = setup_logging()
    
    # region: Argument Parsing
    parser = argparse.ArgumentParser(description="Intra-dataset near-deduplicating with PySpark - Optimized for TB-scale data")
    
    # Data processing arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input parquet file or directory of parquet files",
    )
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum token length of document to be considered. Short ones will be removed",
    )
    parser.add_argument("--num_perm", type=int, default=250, help="Number of permutations")
    parser.add_argument("--b", type=int, default=None, help="Number of bands")
    parser.add_argument("--r", type=int, default=None, help="Number of rows per band")
    parser.add_argument("--column", "-c", type=str, default="content", help="Column to deduplicate on")
    parser.add_argument("--index", type=str, default=None, help="Column to index on")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Local output directory of parquet files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    
    # Essential Spark configuration arguments for TB-scale deduplication
    spark_group = parser.add_argument_group("Spark Configuration", "Essential Spark settings for TB-scale deduplication")
    
    spark_group.add_argument(
        "--spark_master",
        type=str,
        default="local[*]",
        help="Spark master URL (default: local[*]). Examples: local[4], spark://host:port, yarn"
    )
    
    spark_group.add_argument(
        "--driver_memory",
        type=str,
        default="8g",
        help="Amount of memory for driver process (default: 8g). For TB data: 16g-64g recommended"
    )
    
    spark_group.add_argument(
        "--executor_memory",
        type=str,
        default="8g",
        help="Amount of memory per executor (default: 8g). For TB data: 16g-64g recommended"
    )
    
    spark_group.add_argument(
        "--num_executors",
        type=int,
        default=None,
        help="Number of executors. Critical for TB-scale: 8-50+ depending on cluster size"
    )
    
    spark_group.add_argument(
        "--executor_cores",
        type=int,
        default=4,
        help="Cores per executor (default: 4). For TB data: 4-8 cores optimal"
    )
    
    spark_group.add_argument(
        "--max_result_size",
        type=str,
        default="4g",
        help="Limit of serialized results (default: 4g). For TB data: 8g-16g recommended"
    )
    
    spark_group.add_argument(
        "--default_parallelism",
        type=int,
        default=None,
        help="Default RDD partitions. For TB data: 2-4x total cores (e.g., 96-200)"
    )
    
    spark_group.add_argument(
        "--shuffle_partitions",
        type=int,
        default=8192,
        help="Shuffle partitions (default: 8192). For TB data: 8192-20000 recommended"
    )
    
    spark_group.add_argument(
        "--broadcast_threshold",
        type=str,
        default="50mb",
        help="Broadcast join threshold (default: 50mb). For TB data: 50mb-200mb"
    )
    
    spark_group.add_argument(
        "--memory_fraction",
        type=float,
        default=0.8,
        help="Heap fraction for execution/storage (default: 0.8). 0.7-0.9 for TB data"
    )
    
    spark_group.add_argument(
        "--extra_spark_conf",
        type=str,
        action="append",
        help="Additional Spark config in key=value format. Can be used multiple times"
    )
    
    args = parser.parse_args()
    # endregion

    # region: Spark Configuration
    conf = SparkConf().set("spark.app.name", "MinHashLSH-TB-Dedup")
    
    # Core settings for TB-scale processing
    conf.set("spark.master", args.spark_master)
    conf.set("spark.driver.memory", args.driver_memory)
    conf.set("spark.executor.memory", args.executor_memory)
    conf.set("spark.executor.cores", str(args.executor_cores))
    conf.set("spark.driver.maxResultSize", args.max_result_size)
    
    # Set number of executors if specified (critical for TB processing)
    if args.num_executors is not None:
        conf.set("spark.executor.instances", str(args.num_executors))
        
    # Memory and performance settings optimized for large-scale deduplication
    conf.set("spark.storage.memoryFraction", str(args.memory_fraction))
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    # Parallelism settings
    if args.default_parallelism is not None:
        conf.set("spark.default.parallelism", str(args.default_parallelism))
    else:
        # Conservative default for TB processing
        import multiprocessing
        default_parallelism = max(200, multiprocessing.cpu_count() * 4)
        conf.set("spark.default.parallelism", str(default_parallelism))
    
    # SQL settings optimized for deduplication workloads
    conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    conf.set("spark.sql.autoBroadcastJoinThreshold", args.broadcast_threshold)
    
    # Essential optimizations for TB-scale data
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256mb")
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.broadcastTimeout", "3600")
    
    # GraphFrames package
    conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
    
    # Network and shuffle optimizations for large data
    conf.set("spark.network.timeout", "800s")
    conf.set("spark.shuffle.file.buffer", "1m")
    conf.set("spark.shuffle.unsafe.file.output.buffer", "5m")
    conf.set("spark.rpc.askTimeout", "600s")
    conf.set("spark.rpc.lookupTimeout", "600s")
    
    # Logging configuration to reduce noise
    conf.set("spark.sql.adaptive.logLevel", "WARN")
    
    # Add any extra configuration provided by user
    if args.extra_spark_conf:
        for config in args.extra_spark_conf:
            if "=" in config:
                key, value = config.split("=", 1)
                conf.set(key.strip(), value.strip())
            else:
                logger.warning(f"Invalid configuration format '{config}'. Expected key=value format.")
    
    # Create Spark session
    spark = SparkSession.Builder().config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(args.checkpoint_dir)
    
    # Set Spark logging level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    # endregion

    # region: Global Variables
    FINAL_SIZE: int = 0
    MAX_WRITE_CHUNK_SIZE: int = 200_000
    MAX_WRITE_PARTITIONS: int = 2048

    B, R = args.b, args.r
    if B is None or R is None:
        B, R = optimal_param(args.threshold, args.num_perm)

    HASH_RANGES: List[Tuple[int, int]] = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS: Tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]] = (
        RNG.randint(1, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
        RNG.randint(0, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
    )
    # endregion

    start_time: float = time.time()
    index_column = args.index or "__id__"

    # Log configuration
    logger.info("=" * 120)
    logger.info("TB-SCALE DEDUPLICATION - SPARK CONFIGURATION")
    logger.info("=" * 120)
    logger.info(f"Spark Master:           {args.spark_master}")
    logger.info(f"Driver Memory:          {args.driver_memory}")
    logger.info(f"Executor Memory:        {args.executor_memory}")
    logger.info(f"Executor Cores:         {args.executor_cores}")
    if args.num_executors:
        logger.info(f"Number of Executors:    {args.num_executors}")
    logger.info(f"Default Parallelism:    {sc.defaultParallelism}")
    logger.info(f"Shuffle Partitions:     {args.shuffle_partitions}")
    logger.info(f"Broadcast Threshold:    {args.broadcast_threshold}")
    logger.info(f"Memory Fraction:        {args.memory_fraction}")
    logger.info("=" * 120)

    # region: Data Loading
    # persist justification: this data will be needed when removing duplicates
    df: DataFrame = (
        spark.read.option("mergeSchema", "true")
        .parquet(args.input)
        .filter(
            udf(ngrams_length_check, BooleanType())(F.col(args.column), F.lit(args.ngram_size), F.lit(args.min_length))
        )
        .withColumn("__id__", F.monotonically_increasing_id())
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    # persist trigger
    DATA_SIZE: int = df.count()
    logger.info("-" * 120)
    logger.info("DEDUPLICATION PARAMETERS")
    logger.info("-" * 120)
    logger.info(f"Using B={B}, R={R}")
    logger.info(f"Loaded documents: {DATA_SIZE:,}")
    logger.info(f"Input path: {args.input}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info(f"N-gram size: {args.ngram_size}")
    logger.info(f"Min token length: {args.min_length}")
    logger.info(f"Number of permutations: {args.num_perm}")
    logger.info(f"Deduplication column: {args.column}")
    logger.info("-" * 120)
    logger.info("DATA SCHEMA")
    logger.info("-" * 120)
    for col, dtype in df.dtypes:
        logger.info(f"{col:<64}: {dtype}")
    logger.info("-" * 120)

    if DATA_SIZE == 0:
        logger.info("No data found after filtering.")
        spark.stop()
        sys.exit(0)
    # endregion

    # region: MinHash
    logger.info("Starting MinHash computation...")
    edges: pyspark.RDD = (
        df.select(index_column, args.column)
        .rdd.flatMap(
            lambda x: generate_hash_values(
                content=x[1],  # args.column
                idx=x[0],  # __id__
                num_perm=args.num_perm,
                ngram_size=args.ngram_size,
                min_length=args.min_length,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
            )
        )  # (band_idx, band hash value, idx)
        .groupBy(lambda x: (x[0], x[1]))  # group by (band_idx, band hash value)
        .flatMap(lambda x: generate_edges([ele[2] for ele in x[1]]))
        .distinct()
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    
    logger.info("MinHash computation completed.")
    # endregion

    # region: Connected Components
    if edges.isEmpty():
        logger.info("No potential duplicates found. Saving original data...")
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
        df.unpersist()
        edges.unpersist()

        logger.info("-" * 120)
        logger.info("DEDUPLICATION COMPLETED - NO DUPLICATES FOUND")
        logger.info("-" * 120)
        logger.info(f"Input documents:        {DATA_SIZE:,}")
        logger.info(f"Output documents:       {DATA_SIZE:,}")
        logger.info(f"Duplicates removed:     0")
        logger.info(f"Output location:        {args.output}")
        logger.info(f"Total processing time:  {time.time() - start_time:.2f}s")
        logger.info("-" * 120)

        spark.stop()
        sys.exit(0)

    logger.info("Computing connected components for duplicate clusters...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edges_df: DataFrame = (
            spark.createDataFrame(edges, schema=["src", "dst"])
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        edges_count = edges_df.count()
        logger.info(f"Generated {edges_count:,} edges for duplicate detection")
        
        vertices_df: DataFrame = (
            edges_df.select(F.col("src").alias("id"))
            .union(edges_df.select(F.col("dst").alias("id")))
            .distinct()
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        vertices_count = vertices_df.count()
        logger.info(f"Found {vertices_count:,} unique vertices")
        
        assignment: DataFrame = (
            GraphFrame(vertices_df, edges_df).connectedComponents().persist(pyspark.StorageLevel.DISK_ONLY)
        )
        assignment_count = assignment.count()
        logger.info(f"Connected components analysis completed: {assignment_count:,} components")
        
        edges_df.unpersist()
        vertices_df.unpersist()
    # endregion

    # region: Merge Results
    logger.info("Merging duplicate assignments with original data...")
    df = df.join(
        assignment.select(F.col("id").alias(index_column), F.col("component").alias("__component__")),
        on=index_column,
        how="left",
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    assignment.unpersist()
    
    merge_count = df.count()
    logger.info(f"Merged {merge_count:,} records with component assignments")
    # endregion

    logger.info("Filtering duplicates - keeping one representative per cluster...")
    df = (
        df.filter(F.col("__component__").isNull() | (F.col("__component__") == F.col(index_column)))
        .drop("__component__")
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    FINAL_SIZE = df.count()

    # region: Output
    logger.info("Saving deduplicated data...")
    partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
    df.unpersist()
    edges.unpersist()

    # endregion
    total_time = time.time() - start_time
    duplicates_removed = DATA_SIZE - FINAL_SIZE
    
    logger.info("-" * 120)
    logger.info("TB-SCALE DEDUPLICATION COMPLETED")
    logger.info("=" * 120)
    logger.info(f"Input documents:        {DATA_SIZE:,}")
    logger.info(f"Output documents:       {FINAL_SIZE:,}")
    logger.info(f"Duplicates removed:     {duplicates_removed:,}")
    logger.info(f"Deduplication rate:     {(duplicates_removed/max(1, DATA_SIZE)*100):.2f}%")
    logger.info(f"Documents retained:     {(FINAL_SIZE/max(1, DATA_SIZE)*100):.2f}%")
    logger.info(f"Output location:        {args.output}")
    logger.info(f"Total processing time:  {total_time:.2f}s")
    logger.info(f"Processing rate:        {DATA_SIZE/max(1, total_time):.0f} docs/sec")
    logger.info("=" * 120)
    
    # Properly close Spark session
    spark.stop()