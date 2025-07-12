# Semantic Deduplication Pipeline

A high-performance, GPU-accelerated semantic deduplication system for large-scale text datasets. This pipeline uses advanced embedding models, clustering algorithms, and similarity computations to identify and remove semantically similar documents while preserving dataset quality.

## 🚀 Features

- **Multi-GPU Support**: Leverages CUDA-enabled GPUs for fast processing
- **Language-Aware Processing**: Automatically splits data by language and processes each separately
- **Scalable Architecture**: Built on Dask for distributed computing
- **Memory Efficient**: Optimized memory usage with spilling and partitioning
- **Flexible Configuration**: Extensive command-line options for customization
- **Robust Error Handling**: Graceful handling of failures and resource cleanup

## 🏗️ Architecture

The pipeline consists of several interconnected components:

```
Input Data → Split data according to language → Start Dask Cluster -> Embeddings → K-Means Clustering → Semantic Dedup → Deduplicated Dataset Output -> Close Dask Cluster
```

### Core Components

1. **Dask Cluster Manager**: Manages distributed GPU workers
2. **Language Splitter**: Separates documents by language for optimal processing
3. **Embedding Creator**: Generates semantic embeddings using transformer models
4. **Clustering Engine**: Groups similar embeddings for efficient deduplication
5. **Semantic Deduplicator**: Identifies and removes duplicate content according to similarity metric, saving the final deduplicated dataset 

## 📋 Requirements
- Python 3.8+
- CUDA 11.0+
- cuDF and cuML libraries
- Dask and Dask-CUDA
- PyTorch or TensorFlow
- Transformers library

### Installation

```bash
# Install RAPIDS (cuDF, cuML, etc.)
conda install -c rapidsai -c nvidia -c conda-forge \
    cudf cuml dask-cuda

# Install additional dependencies
pip install transformers torch dask distributed pyarrow
```

## 🛠️ Usage

### Two Pipeline Modes

#### 1. Multi-Language Pipeline
Automatically splits data by language and processes each separately. Here input data needs to have a column tag which tell what the corresonping language is, and then does deduplication accordingly for all languages. The splitting is according to the columns not, language is not auto detectd.

```bash
python master_lang_split.py input.parquet ./output_results \
    --data-column text \
    --lang-column language \
    --eps 0.3 \
    --n-clusters 1000 \
    --batch-size 32 \
    --model "intfloat/multilingual-e5-large-instruct"
```

#### 2. Standard Pipeline( Recemmended if input data contains only one language) 
Processes entire dataset as single unit:

```bash
python master.py input.parquet ./output_results \
    --data-column text \
    --eps 0.3 \
    --n-clusters 1000 \
    --batch-size 32 \
    --model "intfloat/multilingual-e5-large-instruct"
```

## ⚙️ Configuration Options

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-column` | Name of text column to process | `data` |
| `--lang-column` | Language column (lang pipeline only) | `lang` |
| `--eps` | Similarity Threshold: 1-eps | `0.8` |
| `--n-clusters` | Number of clusters for grouping | `1000` |
| `--batch-size` | Embedding batch size | `1` |

### GPU Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cuda-devices` | GPU devices to use | `0,1` |
| `--rmm-pool-size` | GPU memory pool fraction | `0.5` |
| `--device-memory-limit` | Per-GPU memory limit | `auto` |
| `--no-multi-gpu` | Use single GPU only | `False` |

### Embedding Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Embedding model name/path | `intfloat/multilingual-e5-large-instruct` |
| `--max-seq-length` | Maximum token sequence length | `512` |
| `--pooling-strategy` | Embedding pooling method | `mean_pooling` |
| `--embedding-max-mem-gb` | Max memory per embedding worker | `30` |

### Clustering & Deduplication

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-iter` | Maximum clustering iterations | `200` |
| `--which-to-keep` | Duplicate retention strategy | `hard` |
| `--sim-metric` | Similarity metric | `cosine` |
| `--dedup-batch-size` | Deduplication batch size | `1024` |
| `--compute-only` | Only compute similarities | `False` |


For other arguments, and their description refer to the code, or get help using the python command.

## 📊 Input Data Format

Your input Parquet file should contain:

```
Required columns:
- text/data column: The actual text content
- language column: Language code (for language-aware pipeline)

Optional columns:
- Any additional metadata columns (preserved in output)
```

Example structure:
```
| text                    | language      |metadata  |
|-------------------------|---------------|----------|
| "This is sample text"   | english       | value1   |
| "यह सैंपल टेक्स्ट है "         | hindi         | value2   |
```

## 🔧 Advanced Usage

### Memory Optimization

For large datasets, tune these parameters:

```bash
--rmm-pool-size 0.7 \
--embedding-max-mem-gb 40 \
--partition-size 200MiB \
--clustering-partition-size 8GB
```

### High-Quality Deduplication

For stricter deduplication:

```bash
--eps 0.9 \
--which-to-keep hard \
--n-clusters 2000 \
--max-iter 300
```

### Fast Processing

For quicker results with less precision:

```bash
--eps 0.7 \
--which-to-keep easy \
--n-clusters 500 \
--max-iter 100 \
--batch-size 64
```

### Compute-Only Mode

To only compute similarities without removing duplicates:

```bash
--compute-only
```

## 📈 Performance Tuning

### GPU Memory Management
- Adjust `--rmm-pool-size` based on GPU memory
- Use `--enable-cudf-spill` for out-of-memory scenarios
- Monitor GPU utilization with `nvidia-smi`

### Batch Size Optimization
- Larger batch sizes = faster processing but more memory
- Start with small batches and increase gradually
- Monitor memory usage during embedding creation

### Clustering Optimization
- More clusters = finer granularity but slower processing
- Balance between accuracy and speed
- Use domain knowledge to set appropriate cluster counts

## 🗂️ Output Structure

The pipeline creates organized output directories:

```
output_results/
├── language_splits/                    # Language-separated data (Contains input data for each language)
│   ├── english.parquet
│   ├── hindi.parquet
│   └── ...
├── English_results/                     # English processing results
│   ├── English_clustering/              # Folder contaning files wherein each file stores the id's stored in the cluster
│   ├── English_deduplication/           # Deduplication summary: Entire summary plus the documents id's to be removed(basically id's which are considered duplicates
│   └── English_embeddings/              # Embeddings results
├── Hindi_results/                       # Hindi processing results
│   ├── Hindi_clustering/
│   ├── Hindi_deduplication/
│   └── Hindi_embeddings/
├── English_cleaned_eps_0.1.parquet      # Final deduplicated dataset (eps_0.1 indicates the eps used for dedup)
├── Hindi_cleaned_eps_0.1.parquet        # Final deduplicated dataset (eps_0.1 indicates the eps used for dedup)
└── ...
```

## 🚨 Error Handling

The pipeline includes robust error handling:

- **Graceful shutdown**: Ctrl+C safely terminates processing
- **Resource cleanup**: Automatic cleanup of GPU memory and clusters
- **Fault tolerance**: Continues processing other languages if one fails
- **Detailed logging**: Comprehensive progress and error reporting

## 📊 Monitoring Progress

The pipeline provides detailed progress information:

```
🚀 Starting Complete Semantic Deduplication Pipeline with Language Splitting
============================================================
STEP 1: Starting Dask cluster...
✓ Cluster started successfully at: tcp://127.0.0.1:8786

STEP 2: Splitting data by language...
✓ Data split into 15 language files

INITIALIZING EMBEDDING MODEL (ONE-TIME)
✓ Embedding model initialized successfully!

[1/15] Processing ENGLISH...
📊 Creating embeddings for english...
✓ Embeddings created successfully for english!
🔍 Clustering embeddings for english...
✓ Clustering completed successfully for english!
🔄 Running semantic deduplication for english...
✓ Documents to remove saved at: ./output/english_deduplication/unique_ids_0.8.parquet
🗑️ Removing duplicates for english...
   📊 Summary for english:
      Original: 1,000,000 documents
      Removed:  150,000 duplicates
      Final:    850,000 documents
      Reduction: 15.00%
✅ ENGLISH processing completed successfully!
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce `--rmm-pool-size`
   - Decrease `--batch-size`
   - Enable `--enable-cudf-spill`

2. **Slow Processing**
   - Increase `--batch-size`
   - Reduce `--n-clusters`
   - Use fewer iterations (`--max-iter`)

3. **Poor Deduplication Quality**
   - Adjust `--eps` threshold
   - Try different `--which-to-keep` strategies
   - Increase `--n-clusters`

4. **Memory Errors**
   - Reduce `--partition-size`
   - Lower `--embedding-max-mem-gb`
   - Use `--no-multi-gpu` for single GPU

## Acknowledgments
- Built on RAPIDS ecosystem (cuDF, cuML, Dask-CUDA)
- Utilizes Hugging Face transformers
- Heavily inspired by the Nvidia-Nemo Curator code for semantic deduplication. The core idea was picked up from Nvidia-Nemo Curator and used to provide a similar deduplication tool-kit with certain fine tuning.
