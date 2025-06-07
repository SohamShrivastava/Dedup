#!/usr/bin/env python3
"""
Name: data_handler.py
Handles data loading, saving, preprocessing, and validation for the deduplication pipeline.
Supports JSONL, Parquet, and JSON formats with GPU-accelerated operations using cuDF.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cudf
import cupy as cp
import pandas as pd
from datasets import Dataset, load_dataset

from config import (
    INDEX_COLUMN, SIGNATURE_COLUMN, CLUSTER_COLUMN,
    SUPPORTED_INPUT_FORMATS, SUPPORTED_OUTPUT_FORMATS,
    validate_file_path, validate_input_format, validate_output_format
)


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """GPU-accelerated data loader for various file formats."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_from_file(self, file_path: str, text_column: str) -> cudf.DataFrame:
        """
        Load dataset from file into GPU memory.
        
        Parameters
        ----------
        file_path : str
            Path to input file
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            Loaded DataFrame in GPU memory
        """
        path = validate_file_path(file_path, check_exists=True)
        ext = validate_input_format(str(path))
        
        self.logger.info(f"Loading dataset from {path} (format: {ext})")
        
        try:
            if ext == '.jsonl':
                df = self._load_jsonl(path)
            elif ext == '.parquet':
                df = self._load_parquet(path)
            elif ext == '.json':
                df = self._load_json(path)
            else:
                raise ValueError(f"Unsupported format: {ext}")
            
            # Validate and prepare DataFrame
            df = self._prepare_dataframe(df, text_column)
            
            self.logger.info(f"Loaded {len(df)} documents")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def load_from_huggingface(self, dataset_name: str, text_column: str, 
                            split: str = "train", streaming: bool = False) -> cudf.DataFrame:
        """
        Load dataset from Hugging Face Hub.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset on Hugging Face Hub
        text_column : str
            Name of the text column
        split : str
            Dataset split to load
        streaming : bool
            Whether to use streaming mode
            
        Returns
        -------
        cudf.DataFrame
            Loaded DataFrame in GPU memory
        """
        self.logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        
        try:
            # Load dataset using datasets library
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            # If streaming, we need to iterate and collect
            if streaming:
                data = []
                for item in dataset:
                    data.append(item)
                df = pd.DataFrame(data)
            else:
                df = dataset.to_pandas()
            
            # Convert to cuDF
            df = cudf.from_pandas(df)
            
            # Validate and prepare DataFrame
            df = self._prepare_dataframe(df, text_column)
            
            self.logger.info(f"Loaded {len(df)} documents from Hugging Face")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face dataset: {str(e)}")
            raise
    
    def load_from_dataframe(self, df: Union[pd.DataFrame, cudf.DataFrame], 
                          text_column: str) -> cudf.DataFrame:
        """
        Load dataset from existing DataFrame.
        
        Parameters
        ----------
        df : Union[pd.DataFrame, cudf.DataFrame]
            Input DataFrame
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            Prepared DataFrame in GPU memory
        """
        self.logger.info(f"Loading from existing DataFrame with {len(df)} rows")
        
        try:
            # Convert to cuDF if needed
            if isinstance(df, pd.DataFrame):
                df = cudf.from_pandas(df)
            elif not isinstance(df, cudf.DataFrame):
                raise ValueError("Input must be pandas or cuDF DataFrame")
            
            # Validate and prepare DataFrame
            df = self._prepare_dataframe(df, text_column)
            
            self.logger.info(f"Prepared {len(df)} documents")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load from DataFrame: {str(e)}")
            raise
    
    def _load_jsonl(self, path: Path) -> cudf.DataFrame:
        """Load JSONL file using cuDF."""
        return cudf.read_json(path, lines=True)
    
    def _load_parquet(self, path: Path) -> cudf.DataFrame:
        """Load Parquet file using cuDF."""
        return cudf.read_parquet(path)
    
    def _load_json(self, path: Path) -> cudf.DataFrame:
        """Load JSON file using cuDF."""
        return cudf.read_json(path)
    
    def _prepare_dataframe(self, df: cudf.DataFrame, text_column: str) -> cudf.DataFrame:
        """
        Prepare DataFrame for processing.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
            
        Returns
        -------
        cudf.DataFrame
            Prepared DataFrame
        """
        # Validate text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found. "
                           f"Available columns: {list(df.columns)}")
        
        # Remove any existing processing columns
        cols_to_remove = [col for col in [INDEX_COLUMN, SIGNATURE_COLUMN, CLUSTER_COLUMN] 
                         if col in df.columns]
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            self.logger.info(f"Removed existing processing columns: {cols_to_remove}")
        
        # Add row index column
        df = df.reset_index(drop=True)
        df[INDEX_COLUMN] = cudf.Series(range(len(df)))
        
        # Handle missing values in text column
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        
        if len(df) < initial_count:
            dropped = initial_count - len(df)
            self.logger.warning(f"Dropped {dropped} rows with missing text")
        
        # Convert text column to string if needed
        if df[text_column].dtype != 'object':
            df[text_column] = df[text_column].astype(str)
        
        return df


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Text preprocessing utilities for deduplication."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def filter_by_length(self, df: cudf.DataFrame, text_column: str, 
                        min_length: int = 0, max_length: Optional[int] = None) -> cudf.DataFrame:
        """
        Filter documents by text length.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        min_length : int
            Minimum text length in characters
        max_length : Optional[int]
            Maximum text length in characters
            
        Returns
        -------
        cudf.DataFrame
            Filtered DataFrame
        """
        initial_count = len(df)
        
        if min_length > 0:
            # Calculate text lengths using cuDF string operations
            text_lengths = df[text_column].str.len()
            df = df[text_lengths >= min_length]
            
        if max_length is not None:
            # Calculate text lengths using cuDF string operations
            text_lengths = df[text_column].str.len()
            df = df[text_lengths <= max_length]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} documents by length")
        
        return df
    
    def filter_by_token_count(self, df: cudf.DataFrame, text_column: str, 
                            min_tokens: int = 0, max_tokens: Optional[int] = None) -> cudf.DataFrame:
        """
        Filter documents by token count (approximate using whitespace).
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        min_tokens : int
            Minimum token count
        max_tokens : Optional[int]
            Maximum token count
            
        Returns
        -------
        cudf.DataFrame
            Filtered DataFrame
        """
        initial_count = len(df)
        
        if min_tokens > 0 or max_tokens is not None:
            # Approximate token count using whitespace splits
            # This is a rough approximation but much faster than proper tokenization
            token_counts = df[text_column].str.count(' ') + 1
            
            if min_tokens > 0:
                df = df[token_counts >= min_tokens]
            
            if max_tokens is not None:
                df = df[token_counts <= max_tokens]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} documents by token count")
        
        return df
    
    def clean_text(self, df: cudf.DataFrame, text_column: str, 
                  remove_extra_whitespace: bool = True,
                  remove_empty_lines: bool = True) -> cudf.DataFrame:
        """
        Clean text data using cuDF string operations.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        remove_extra_whitespace : bool
            Whether to remove extra whitespace
        remove_empty_lines : bool
            Whether to remove empty lines
            
        Returns
        -------
        cudf.DataFrame
            DataFrame with cleaned text
        """
        text_series = df[text_column]
        
        if remove_extra_whitespace:
            # Remove extra whitespace using cuDF string operations
            text_series = text_series.str.replace(r'\s+', ' ', regex=True)
            text_series = text_series.str.strip()
        
        if remove_empty_lines:
            # Remove empty lines
            text_series = text_series.str.replace(r'\n\s*\n', '\n', regex=True)
        
        # Update the DataFrame
        df = df.copy()
        df[text_column] = text_series
        
        return df


# ============================================================================
# DATA SAVING
# ============================================================================

class DataSaver:
    """GPU-accelerated data saver for various file formats."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def save_to_file(self, df: cudf.DataFrame, output_path: str, 
                    output_format: str, **kwargs) -> None:
        """
        Save DataFrame to file.
        
        Parameters
        ----------
        df : cudf.DataFrame
            DataFrame to save
        output_path : str
            Output file path
        output_format : str
            Output format ('jsonl', 'parquet', 'json')
        **kwargs
            Additional parameters for saving
        """
        output_path = Path(output_path)
        output_format = validate_output_format(output_format)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving {len(df)} documents to {output_path} (format: {output_format})")
        
        try:
            # Remove processing columns before saving
            df_to_save = self._prepare_for_saving(df)
            
            if output_format == 'jsonl':
                self._save_jsonl(df_to_save, output_path, **kwargs)
            elif output_format == 'parquet':
                self._save_parquet(df_to_save, output_path, **kwargs)
            elif output_format == 'json':
                self._save_json(df_to_save, output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            self.logger.info(f"Successfully saved dataset to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {str(e)}")
            raise
    
    def _save_jsonl(self, df: cudf.DataFrame, path: Path, **kwargs) -> None:
        """Save as JSONL using cuDF."""
        df.to_json(path, orient='records', lines=True, **kwargs)
    
    def _save_parquet(self, df: cudf.DataFrame, path: Path, **kwargs) -> None:
        """Save as Parquet using cuDF."""
        df.to_parquet(path, **kwargs)
    
    def _save_json(self, df: cudf.DataFrame, path: Path, **kwargs) -> None:
        """Save as JSON using cuDF."""
        df.to_json(path, orient='records', **kwargs)
    
    def _prepare_for_saving(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Prepare DataFrame for saving by removing processing columns.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
            
        Returns
        -------
        cudf.DataFrame
            Cleaned DataFrame
        """
        # Remove processing columns
        cols_to_remove = [col for col in [INDEX_COLUMN, SIGNATURE_COLUMN, CLUSTER_COLUMN] 
                         if col in df.columns]
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            self.logger.debug(f"Removed processing columns: {cols_to_remove}")
        
        return df


# ============================================================================
# MAIN DATA HANDLER CLASS
# ============================================================================

class DataHandler:
    """Main data handler combining loading, preprocessing, and saving."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.loader = DataLoader(logger)
        self.preprocessor = DataPreprocessor(logger)
        self.saver = DataSaver(logger)
    
    def load_and_prepare(self, 
                        input_path: Optional[str] = None,
                        dataset: Optional[Union[cudf.DataFrame, pd.DataFrame]] = None,
                        hf_dataset: Optional[str] = None,
                        text_column: str = "text",
                        min_length: int = 0,
                        max_length: Optional[int] = None,
                        min_tokens: int = 0,
                        max_tokens: Optional[int] = None,
                        clean_text: bool = True) -> cudf.DataFrame:
        """
        Load and prepare dataset for deduplication.
        
        Parameters
        ----------
        input_path : Optional[str]
            Path to input file
        dataset : Optional[Union[cudf.DataFrame, pd.DataFrame]]
            Pre-loaded DataFrame
        hf_dataset : Optional[str]
            Hugging Face dataset name
        text_column : str
            Name of the text column
        min_length : int
            Minimum text length in characters
        max_length : Optional[int]
            Maximum text length in characters
        min_tokens : int
            Minimum token count
        max_tokens : Optional[int]
            Maximum token count
        clean_text : bool
            Whether to clean text
            
        Returns
        -------
        cudf.DataFrame
            Prepared DataFrame
        """
        # Load data
        if input_path:
            df = self.loader.load_from_file(input_path, text_column)
        elif dataset is not None:
            df = self.loader.load_from_dataframe(dataset, text_column)
        elif hf_dataset:
            df = self.loader.load_from_huggingface(hf_dataset, text_column)
        else:
            raise ValueError("Must provide either input_path, dataset, or hf_dataset")
        
        # Preprocess data
        if clean_text:
            df = self.preprocessor.clean_text(df, text_column)
        
        if min_length > 0 or max_length is not None:
            df = self.preprocessor.filter_by_length(df, text_column, min_length, max_length)
        
        if min_tokens > 0 or max_tokens is not None:
            df = self.preprocessor.filter_by_token_count(df, text_column, min_tokens, max_tokens)
        
        return df
    
    def save_results(self, df: cudf.DataFrame, output_path: str, 
                    output_format: str = "parquet") -> None:
        """
        Save deduplication results.
        
        Parameters
        ----------
        df : cudf.DataFrame
            Results DataFrame
        output_path : str
            Output file path
        output_format : str
            Output format
        """
        self.saver.save_to_file(df, output_path, output_format)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dataset_info(df: cudf.DataFrame, text_column: str) -> dict:
    """
    Get information about the dataset.
    
    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
        
    Returns
    -------
    dict
        Dataset information
    """
    info = {
        'total_documents': len(df),
        'columns': list(df.columns),
        'text_column': text_column,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Text statistics
    if text_column in df.columns:
        text_lengths = df[text_column].str.len()
        info.update({
            'avg_text_length': float(text_lengths.mean()),
            'min_text_length': int(text_lengths.min()),
            'max_text_length': int(text_lengths.max()),
            'total_characters': int(text_lengths.sum())
        })
    
    return info


if __name__ == "__main__":
    # Test the data handler
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create test data
    test_data = cudf.DataFrame({
        'text': ['This is a test document.', 'Another test document.', 'Short.', ''],
        'id': [1, 2, 3, 4]
    })
    
    # Test data handler
    handler = DataHandler(logger)
    
    # Test loading from DataFrame
    df = handler.load_and_prepare(dataset=test_data, text_column='text', min_length=10)
    print(f"Loaded {len(df)} documents after filtering")
    
    # Test dataset info
    info = get_dataset_info(df, 'text')
    print("Dataset info:", info)