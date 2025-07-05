########################################## FUZZY CPU ##################################################
import argparse
import time
import os
import glob
import logging
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Dict
from datasketch import MinHash, MinHashLSH
import tokenize
from io import BytesIO
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Tokenizes the given code text using Python's tokenizer
def extract_code_tokens(text: str) -> List[str]:
    try:
        tokens = []
        g = tokenize.tokenize(BytesIO(text.encode('utf-8')).readline)
        for toknum, tokval, _, _, _ in g:
            if toknum == tokenize.ENDMARKER:
                break
            if tokval.strip():
                tokens.append(tokval)
        return tokens
    except:
        return []


# measures time
def timed_step(name: str, log_key: str, func, log_dict: dict):
    start = time.time()
    result = func()
    end = time.time()
    duration = end - start
    log_dict[log_key] = duration
    logger.info(f"{name:<30}: {duration:.2f}s")
    return result



# loads input records from JSONL or Parquet files
def load_input_files(input_path: str, text_column: str, max_files: int = None, min_length: int = 0) -> List[Dict]:
    input_path = Path(input_path)
    records = []

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(list(input_path.glob("*.jsonl")) + list(input_path.glob("*.parquet")))
    else:
        files = sorted(glob.glob(input_path))

    if max_files:
        files = files[:max_files]

    for file in tqdm(files, desc="Reading input files"):
        if str(file).endswith(".jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if len(obj.get(text_column, "")) >= min_length:
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
        elif str(file).endswith(".parquet"):
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                if len(str(row.get(text_column, ""))) >= min_length:
                    records.append(row.to_dict())
        else:
            raise ValueError("Unsupported file format")
    return records


# computes MinHash signature for one document using token shingles
def minhash_worker(args):
    idx, content, num_perm, ngram_size = args
    mh = MinHash(num_perm=num_perm)
    tokens = extract_code_tokens(content)  # Tokenizing content
    shingles = set(" ".join(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1))
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return idx, mh


# Main Deduplication Function 
def fuzzy_deduplicate(data: List[Dict], text_column: str, threshold: float = 0.85, num_perm: int = 64, ngram_size: int = 5, num_cpus: int = None, timings: dict = None) -> List[Dict]:
    num_workers = num_cpus or cpu_count()

    logger.info(f"Generating MinHash signatures using {num_workers} CPU cores...")
    args_list = [(i, item[text_column], num_perm, ngram_size) for i, item in enumerate(data)]

    # compute MinHash signatures
    def compute_signatures():
        with Pool(processes=num_workers) as pool:
            return list(tqdm(pool.imap(minhash_worker, args_list, chunksize=128), total=len(data)))

    results = timed_step("Processing (Signatures)", "Processing", compute_signatures, timings)

    # insert signatures into LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = [None] * len(data)

    logger.info("Inserting signatures into LSH...")
    for idx, mh in results:
        lsh.insert(str(idx), mh)
        signatures[idx] = mh

    # query LSH and group near-duplicates using Union-Find
    logger.info("Finding duplicates with optimized LSH querying...")

    def filter_lsh():
        parent = list(range(len(data)))

        # Union-Find helper functions
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv

        processed = set()

        #perform LSH-based grouping
        for i in tqdm(range(len(signatures)), desc="Querying LSH"):
            if i in processed:
                continue

            matches = lsh.query(signatures[i])
            match_indices = [int(m) for m in matches if int(m) != i]

            if match_indices:
                for j in match_indices:
                    if find(i) != find(j):
                        union(i, j)
                processed.update(match_indices)
            processed.add(i)

        # collect unique representatives
        seen = set()
        output = []
        for i in range(len(data)):
            rep = find(i)
            if rep not in seen:
                seen.add(rep)
                output.append(data[i])

        return output

    deduped = timed_step("Filtering (LSH)", "Filtering", filter_lsh, timings)
    return deduped


# Save output in jsonl or parquet
def save_output(records: List[Dict], output_path: str):
    if output_path.endswith(".jsonl") or output_path.endswith(".json"):
        with open(output_path, "w") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif output_path.endswith(".parquet"):
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported output format")


# Argument Parser 
def parse_args():
    parser = argparse.ArgumentParser(description="CPU fuzzy deduplication pipeline")
    parser.add_argument("--input", required=True, help="Input path (dir, file, or glob)")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .parquet)")
    parser.add_argument("--text-column", default="content", help="Column for deduplication")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum content length")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard threshold for MinHash LSH")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of input files")
    parser.add_argument("--num-cpus", type=int, default=None, help="Number of CPU cores to use")
    return parser.parse_args()


#  Main Pipeline
def main():
    args = parse_args()
    timings = {}

    #Load data from disk
    data = timed_step("Loading", "Loading", lambda: load_input_files(args.input, args.text_column, args.max_files, args.min_length), timings)

    #Run MinHash + LSH-based deduplication
    deduped = fuzzy_deduplicate(data, args.text_column, threshold=args.threshold, num_cpus=args.num_cpus, timings=timings)

    #Save final deduplicated output
    timed_step("Saving", "Saving", lambda: save_output(deduped, args.output), timings)

    total_time = sum(timings.values())

    
    logger.info("\n" + "=" * 20 + " FINAL SUMMARY " + "=" * 20)
    logger.info(f"{'Loading':<30}: {timings.get('Loading', 0):.2f}s")
    logger.info(f"{'Processing (Signatures)':<30}: {timings.get('Processing', 0):.2f}s")
    logger.info(f"{'Filtering (LSH)':<30}: {timings.get('Filtering', 0):.2f}s")
    logger.info(f"{'Saving':<30}: {timings.get('Saving', 0):.2f}s")
    logger.info(f"{'Total':<30}: {total_time:.2f}s")
    logger.info(f"{'Before':<30}: {len(data)}")
    logger.info(f"{'After':<30}: {len(deduped)}")


if __name__ == "__main__":
    main()
