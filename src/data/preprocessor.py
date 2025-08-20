"""
Simple data preprocessing with file-based storage.

This module provides simple preprocessing of event logs with direct storage
in the processed directory for reuse.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from .loader import load_event_log
from .prefixer import generate_prefixes
from .encoders import Vocabulary


class SimplePreprocessor:
    """Simple preprocessor that saves/loads from processed directory."""
    
    def __init__(self, raw_dir: Path, processed_dir: Path, config: Dict[str, Any]):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
    
    def _get_processed_paths(self, dataset_name: str) -> Tuple[Path, Path, Path]:
        """Get paths for processed files."""
        base_path = self.processed_dir / dataset_name
        prefixes_file = base_path.with_suffix('.prefixes.pkl')
        labels_file = base_path.with_suffix('.labels.pkl')
        vocabulary_file = base_path.with_suffix('.vocabulary.pkl')
        return prefixes_file, labels_file, vocabulary_file
    
    def is_processed(self, dataset_name: str) -> bool:
        """Check if dataset is already processed."""
        prefixes_file, labels_file, vocabulary_file = self._get_processed_paths(dataset_name)
        return all(f.exists() for f in [prefixes_file, labels_file, vocabulary_file])
    
    def load_processed_data(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load processed data from files."""
        prefixes_file, labels_file, vocabulary_file = self._get_processed_paths(dataset_name)
        
        with open(prefixes_file, 'rb') as f:
            prefixes_df = pickle.load(f)
        with open(labels_file, 'rb') as f:
            labels_series = pickle.load(f)
        with open(vocabulary_file, 'rb') as f:
            vocabulary = pickle.load(f)
        
        return prefixes_df, labels_series, vocabulary
    
    def save_processed_data(self, dataset_name: str, prefixes_df: pd.DataFrame, 
                           labels_series: pd.Series, vocabulary: Vocabulary):
        """Save processed data to files."""
        prefixes_file, labels_file, vocabulary_file = self._get_processed_paths(dataset_name)
        
        with open(prefixes_file, 'wb') as f:
            pickle.dump(prefixes_df, f)
        with open(labels_file, 'wb') as f:
            pickle.dump(labels_series, f)
        with open(vocabulary_file, 'wb') as f:
            pickle.dump(vocabulary, f)
    
    def preprocess_dataset(self, dataset_name: str, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """
        Preprocess a dataset with simple file-based storage.
        
        Args:
            dataset_name: Name of the dataset to preprocess
            force_reprocess: Force reprocessing even if processed data exists
            
        Returns:
            Tuple of (prefixes_df, labels_series, vocabulary)
        """
        # Check if we can use existing processed data
        if not force_reprocess and self.is_processed(dataset_name):
            print(f"Using existing processed data for {dataset_name}")
            return self.load_processed_data(dataset_name)
        
        # Find raw file
        csv_candidate = self.raw_dir / f"{dataset_name}.csv"
        xes_candidate = self.raw_dir / f"{dataset_name}.xes"
        
        if csv_candidate.exists():
            raw_file_path = csv_candidate
        elif xes_candidate.exists():
            raw_file_path = xes_candidate
        else:
            raise FileNotFoundError(
                f"Neither {csv_candidate.name} nor {xes_candidate.name} exist in {self.raw_dir}"
            )
        
        # Preprocess from scratch
        print(f"Preprocessing {dataset_name} from raw data...")
        event_log = load_event_log(str(raw_file_path))
        
        # Generate prefixes
        prefixes_df, labels_series = generate_prefixes(
            event_log_dataframe=event_log,
            end_of_case_token=self.config["data"]["end_of_case_token"],
            max_prefix_length=self.config["data"]["max_prefix_length"],
            attribute_mode=self.config["data"]["attribute_mode"]
        )
        
        # Build vocabulary
        all_activity_tokens = prefixes_df["prefix_activities"].explode().tolist()
        all_activity_tokens.append(self.config["data"]["end_of_case_token"])
        vocabulary = Vocabulary(all_activity_tokens)
        
        # Save the results
        self.save_processed_data(dataset_name, prefixes_df, labels_series, vocabulary)
        
        print(f"Processed {dataset_name} and saved to {self.processed_dir}")
        return prefixes_df, labels_series, vocabulary
    
    def preprocess_multiple_datasets(self, dataset_names: list, force_reprocess: bool = False) -> Dict[str, Tuple[pd.DataFrame, pd.Series, Vocabulary]]:
        """
        Preprocess multiple datasets efficiently.
        
        Args:
            dataset_names: List of dataset names to preprocess
            force_reprocess: Force reprocessing even if processed data exists
            
        Returns:
            Dictionary mapping dataset names to their preprocessed data
        """
        results = {}
        
        for dataset_name in dataset_names:
            try:
                prefixes_df, labels_series, vocabulary = self.preprocess_dataset(
                    dataset_name, force_reprocess
                )
                results[dataset_name] = (prefixes_df, labels_series, vocabulary)
            except Exception as e:
                print(f"Error preprocessing {dataset_name}: {e}")
                continue
        
        return results
    
    def clear_processed_data(self, dataset_name: Optional[str] = None):
        """Clear processed data for specific dataset or all datasets."""
        if dataset_name is None:
            # Clear all processed data
            for file_path in self.processed_dir.glob("*.pkl"):
                file_path.unlink()
            print("Cleared all processed data")
        else:
            # Clear processed data for specific dataset
            prefixes_file, labels_file, vocabulary_file = self._get_processed_paths(dataset_name)
            for file_path in [prefixes_file, labels_file, vocabulary_file]:
                if file_path.exists():
                    file_path.unlink()
            print(f"Cleared processed data for {dataset_name}")
    
    def get_processed_info(self) -> Dict[str, Any]:
        """Get information about processed datasets."""
        processed_files = list(self.processed_dir.glob("*.pkl"))
        dataset_names = set()
        
        for file_path in processed_files:
            # Extract dataset name from filename (before first dot)
            dataset_name = file_path.stem.split('.')[0]
            dataset_names.add(dataset_name)
        
        return {
            "processed_datasets": sorted(list(dataset_names)),
            "processed_dir": str(self.processed_dir),
            "total_files": len(processed_files)
        }
