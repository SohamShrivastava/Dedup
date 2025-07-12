import sys
import os
from typing import List, Optional, Union
import argparse

import torch
import dask
import dask_cudf
from dataclasses import dataclass
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from distributed import Client
import numpy as np

@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    max_seq_length: int = 512
    pooling_strategy: str = "mean_pooling"

    def __post_init__(self):
        if self.pooling_strategy not in ["mean_pooling", "last_token"]:
            raise ValueError("Pooling_strategy must be either 'mean_pooling' or 'last_token'")


class EmbeddingPytorchModel(nn.Module):
    def __init__(self, config: EmbeddingConfig, device: str = "cuda") -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        # Load model WITHOUT device_map - let each worker handle its own device
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Move entire model to the specified device
        self.model = self.model.to(device)

    def feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device as model
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.autocast(device_type=self.device.split(':')[0]):  # 'cuda' from 'cuda:0'
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Ensure all batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        if self.config.pooling_strategy == "mean_pooling":
            return self._mean_pooling(feature, batch["attention_mask"])
        else:
            return self._get_last_token(feature, batch["attention_mask"])

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

    def _get_last_token(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_indices = last_token_indices.to(torch.long)
        batch_size = attention_mask.size(0)
        batch_indices = torch.arange(batch_size, device=attention_mask.device)
        last_token_embeddings = token_embeddings[batch_indices, last_token_indices]
        return F.normalize(last_token_embeddings, dim=1)


class EmbeddingCrossFitModel(HFModel):
    def __init__(self, config: EmbeddingConfig, max_mem_gb: int | None = None):
        self.config = config
        super().__init__(self.config.model_name_or_path, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda") -> EmbeddingPytorchModel:
        """Load model on the specified device - each worker gets its own copy"""
        print(f"Loading model on device: {device}")
        
        # Create model instance for this specific device
        model = EmbeddingPytorchModel(self.config, device=device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model

    def max_seq_length(self) -> int:
        return self.config.max_seq_length

    def load_config(self) -> AutoConfig:
        return AutoConfig.from_pretrained(self.config.model_name_or_path)

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        
        # If no padding token is set, default to eos_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Fallback in case eos_token is also None
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        return tokenizer

class EmbeddingCreator:
    def __init__(
        self,
        scheduler_address: str,
        embedding_model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_batch_size: int = 2,
        embedding_pooling_strategy: str = "mean_pooling",
        input_column: str = "text",
        embedding_column: str = "embeddings",
        embedding_max_mem_gb: int | None = 30,
        partition_size: str = "50MiB",
        max_seq_length: int = 512,
    ):
        self.scheduler_address = scheduler_address
        self.embeddings_config = EmbeddingConfig(
            model_name_or_path=embedding_model_name_or_path,
            pooling_strategy=embedding_pooling_strategy,
            max_seq_length=max_seq_length,
        )
        self.batch_size = embedding_batch_size
        self.input_column = input_column
        self.embedding_column = embedding_column
        self.model = EmbeddingCrossFitModel(self.embeddings_config, max_mem_gb=embedding_max_mem_gb)
        self.partition_size = partition_size
        
        # Connect to existing cluster
        print(f"Connecting to existing Dask cluster at: {self.scheduler_address}")
        self.client = Client(self.scheduler_address)
        
        # Print cluster information
        print(f"Connected to cluster successfully!")
        print(f"Dashboard available at: {self.client.dashboard_link}")
        
        workers = self.client.scheduler_info()['workers']
        print(f"Number of workers: {len(workers)}")
        
        # Show which GPU each worker is using
        for worker_id, worker_info in workers.items():
            resources = worker_info.get('resources', {})
            print(f"Worker {worker_id}: {resources}")
    
    def read_parquet(
        self,
        input_files: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> dask_cudf.DataFrame:
        """Read parquet with optimized partitioning"""
        ddf = dask_cudf.read_parquet(
            input_files,
            columns=columns,
            blocksize=self.partition_size,
            **kwargs
        )
        
        # Ensure proper partition distribution across workers
        workers = self.client.scheduler_info()['workers']
        num_workers = len(workers)
        current_partitions = ddf.npartitions
        
        # Ensure we have enough partitions for all workers
        min_partitions = max(num_workers, current_partitions)
        
        if current_partitions < min_partitions:
            print(f"Repartitioning from {current_partitions} to {min_partitions} partitions")
            ddf = ddf.repartition(npartitions=min_partitions)
        
        return ddf
    
    def write_parquet(
        self,
        ddf: dask_cudf.DataFrame,
        output_path: str,
        **kwargs
    ) -> None:
        ddf.to_parquet(output_path, **kwargs)

    def create_embeddings(self, ddf: dask_cudf.DataFrame, input_column: str = "text") -> dask_cudf.DataFrame:
        """Create embeddings with each worker using its own GPU"""
        print(f"Processing {len(ddf)} rows across {ddf.npartitions} partitions")
        
        workers = self.client.scheduler_info()['workers']
        print(f"Each worker will load model on its assigned GPU")
        print(f"Available workers: {len(workers)}")
        
        # Show partition distribution
        partition_sizes = ddf.map_partitions(len).compute()
        print(f"Partition sizes: {partition_sizes}")
        print(f"Total rows: {sum(partition_sizes)}")
        
        # Create the processing pipeline
        # Each worker will automatically get its own model instance
        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[input_column],
                tokenizer_type="default",
                max_length=self.embeddings_config.max_seq_length,
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.embedding_column,
            ),
            keep_cols=ddf.columns.tolist(),
        )
        
        return pipe(ddf) 

    def __call__(self, input_path: Union[str, List[str]], output_path: str, **kwargs) -> dask_cudf.DataFrame:
        print(f"Reading data from: {input_path}")
        ddf = self.read_parquet(input_path, **kwargs)
        
        print(f"Data shape: {ddf.shape}")
        print(f"Number of partitions: {ddf.npartitions}")
        
        print(f"Creating embeddings for column: {self.input_column}")
        embedding_ddf = self.create_embeddings(ddf, self.input_column)
        
        print(f"Writing results to: {output_path}")
        self.write_parquet(embedding_ddf, output_path)
        
        return embedding_ddf
    
    def close(self):
        if self.client:
            self.client.close()
            print("Dask client closed.")


def main():
    parser = argparse.ArgumentParser(description="Embedding Creator (connects to existing cluster)")
    
    # Required arguments
    parser.add_argument("scheduler_address", help="Address of existing Dask scheduler")
    parser.add_argument("input_path", help="Path to input parquet file(s)")
    parser.add_argument("output_path", help="Path to save output parquet")
    
    # Optional arguments
    parser.add_argument("--model", default="intfloat/multilingual-e5-large-instruct", help="Model name or path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--pooling-strategy", choices=["mean_pooling", "last_token"], default="mean_pooling", help="Pooling strategy")
    parser.add_argument("--input-column", default="text", help="Input column name")
    parser.add_argument("--embedding-column", default="embeddings", help="Embedding column name")
    parser.add_argument("--partition-size", default="100MiB", help="Partition size for better GPU distribution")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length for tokenization")
    
    args = parser.parse_args()
    
    creator = EmbeddingCreator(
        scheduler_address=args.scheduler_address,
        embedding_model_name_or_path=args.model,
        embedding_batch_size=args.batch_size,
        embedding_pooling_strategy=args.pooling_strategy,
        input_column=args.input_column,
        embedding_column=args.embedding_column,
        partition_size=args.partition_size,
        max_seq_length=args.max_seq_length,
    )

    creator(args.input_path, args.output_path)
    print("Embeddings created successfully!")


if __name__ == "__main__":
    main()