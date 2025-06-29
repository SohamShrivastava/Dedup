import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Union

import cudf
import dask_cudf
import torch
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    max_seq_length: Optional[int] = None
    pooling_strategy: str = "mean_pooling"  # Options: "mean_pooling" or "last_token"

    def __post_init__(self):
        if self.max_seq_length is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.max_seq_length = tokenizer.model_max_length
            
            # Guard against Hugging Face bug which sets max_seq_length to max(int)
            if self.max_seq_length > 1e5:
                config = AutoConfig.from_pretrained(self.model_name_or_path)
                self.max_seq_length = config.max_position_embeddings
                
        if self.pooling_strategy not in ["mean_pooling", "last_token"]:
            raise ValueError("pooling_strategy must be either 'mean_pooling' or 'last_token'")


class EmbeddingPytorchModel(nn.Module):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path, 
            force_download=False
        )

    def feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=input_ids.device.type):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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
    def __init__(self, config: EmbeddingConfig, max_mem_gb: Optional[int] = None):
        self.config = config
        super().__init__(self.config.model_name_or_path, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda") -> EmbeddingPytorchModel:
        model = EmbeddingPytorchModel(self.config)
        model = model.to(device)
        model.eval()
        return model

    def max_seq_length(self) -> int:
        return self.config.max_seq_length

    def load_config(self) -> AutoConfig:
        return AutoConfig.from_pretrained(self.config.model_name_or_path)

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.config.model_name_or_path)


class ParquetEmbeddingCreator:
    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,  # Reduced for 15GB GPU
        max_mem_gb: int = 10,  # Conservative for 15GB GPU
        pooling_strategy: str = "mean_pooling",
        id_column: str = "id",
        text_column: str = "text",
        embedding_column: str = "embeddings",
        output_dir: str = "./output_embeddings",
        save_parquet: bool = True,
    ):
        """
        GPU-optimized embedding creator for parquet files.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            batch_size: Batch size for processing (reduced for 15GB GPU)
            max_mem_gb: Maximum memory usage in GB (conservative for 15GB GPU)
            pooling_strategy: "mean_pooling" or "last_token"
            id_column: Name of ID column in input parquet
            text_column: Name of text column to embed
            embedding_column: Name of output embedding column
            output_dir: Directory to save output parquet
            save_parquet: Whether to save as parquet file
        """
        self.embedding_config = EmbeddingConfig(
            model_name_or_path=model_name_or_path,
            pooling_strategy=pooling_strategy
        )
        self.batch_size = batch_size
        self.max_mem_gb = max_mem_gb
        self.id_column = id_column
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.output_dir = output_dir
        self.save_parquet = save_parquet
        
        # Initialize model
        self.model = EmbeddingCrossFitModel(self.embedding_config, max_mem_gb=max_mem_gb)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_embeddings(self, ddf: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """Create embeddings for the text column."""
        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[self.text_column],
                tokenizer_type="default",
                max_length=self.embedding_config.max_seq_length,
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.embedding_column,
            ),
            keep_cols=[self.id_column, self.text_column],  # Keep only ID and text
        )
        return pipe(ddf)
    
    def _validate_input(self, ddf: dask_cudf.DataFrame):
        """Validate that required columns exist."""
        columns = ddf.columns.tolist()
        if self.id_column not in columns:
            raise ValueError(f"ID column '{self.id_column}' not found in parquet. Available columns: {columns}")
        if self.text_column not in columns:
            raise ValueError(f"Text column '{self.text_column}' not found in parquet. Available columns: {columns}")
    
    def __call__(self, parquet_path: str) -> dask_cudf.DataFrame:
        """
        Process parquet file and create embeddings.
        
        Args:
            parquet_path: Path to input parquet file
            
        Returns:
            dask_cudf.DataFrame with ID and embeddings columns
        """
        start_time = time.time()
        
        # Load parquet file
        self.logger.info(f"Loading parquet file: {parquet_path}")
        ddf = dask_cudf.read_parquet(parquet_path, blocksize="1GB")
        
        # Validate input
        self._validate_input(ddf)
        self.logger.info(f"Input shape: {len(ddf)} rows")
        
        # Create embeddings
        self.logger.info("Creating embeddings...")
        embedding_ddf = self._create_embeddings(ddf)
        
        # Select only ID and embeddings columns for output
        output_ddf = embedding_ddf[[self.id_column, self.embedding_column]]
        
        # Handle categorical columns for parquet compatibility
        for col in output_ddf.columns:
            if output_ddf[col].dtype.name == "category":
                output_ddf[col] = output_ddf[col].astype("str")
        
        # Save to parquet if requested
        if self.save_parquet:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, "embeddings.parquet")
            self.logger.info(f"Saving embeddings to: {output_path}")
            output_ddf.to_parquet(output_path, write_index=False)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Embedding creation completed in {elapsed_time:.2f} seconds")
        
        return output_ddf


# Example usage:
if __name__ == "__main__":
    # Initialize the embedding creator
    creator = ParquetEmbeddingCreator(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,  # Optimized for 15GB GPU
        max_mem_gb=10,  # Conservative memory usage
        id_column="id",
        text_column="text",
        output_dir="./embeddings_output"
    )
    
    # Process parquet file
    # result_df = creator("path/to/your/input.parquet")
    
    # The result contains ID and embeddings columns, ready for clustering
    # print(result_df.head())