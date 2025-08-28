"""
Data preprocessing based on the original ProcessTransformer implementation.
Provides standardized preprocessing for all PBPM tasks with canonical 5-fold case-based cross-validation.
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from .loader import load_event_log
from .encoders import Vocabulary
from ..utils.cross_validation import CanonicalCrossValidation


class CanonicalLogsProcessor:
    """
    Canonical logs processor that creates standardized 5-fold case-based splits
    following Rama-Maneiro et al. (2021) methodology.
    """
    
    def __init__(self, name: str, filepath: str, columns: List[str], 
                 dir_path: str = "./data/processed", pool: int = 1):
        """Initialize canonical logs processor.
        
        Args:
            name: Dataset name
            filepath: Path to raw logs dataset
            columns: List of column names
            dir_path: Path to directory for saving processed data
            pool: Number of CPUs for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = Path(dir_path)
        self._pool = pool
        
        # Create processed directory
        processed_dir = self._dir_path / self._name / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir = processed_dir
    
    def _load_and_standardize_df(self, sort_temporally: bool = False) -> pd.DataFrame:
        """Load and standardize the raw event log."""
        df = load_event_log(self._filepath)
        
        # Standardize column names to match original ProcessTransformer format
        df = df.rename(columns={"case:concept:name": "Case ID", 
                               "concept:name": "Activity", 
                               "time:timestamp": "Complete Timestamp"})
        
        # Apply the same preprocessing as original ProcessTransformer
        df["Activity"] = df["Activity"].str.lower()
        df["Activity"] = df["Activity"].str.replace(" ", "-")
        
        # Handle timestamp
        if df["Complete Timestamp"].dtype == 'object':
            df["Complete Timestamp"] = df["Complete Timestamp"].str.replace("/", "-")
            df["Complete Timestamp"] = pd.to_datetime(df["Complete Timestamp"], 
                                                     dayfirst=True).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            df["Complete Timestamp"] = df["Complete Timestamp"].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        if sort_temporally:
            df.sort_values(by=["Complete Timestamp"], inplace=True)
        
        return df
    
    def _extract_vocabularies(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Extract vocabularies for activities and create metadata."""
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["Activity"].unique())
        keys.extend(activities)
        val = range(len(keys))

        x_word_dict = dict(zip(keys, val))
        y_word_dict = dict(zip(activities, range(len(activities))))

        # Save metadata
        metadata = {
            "x_word_dict": x_word_dict,
            "y_word_dict": y_word_dict,
            "vocab_size": len(x_word_dict),
            "num_activities": len(activities),
            "activities": activities
        }
        
        with open(self._processed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {"x_word_dict": x_word_dict, "y_word_dict": y_word_dict}
    
    def _process_next_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for next activity prediction task."""
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "next_act"])
        idx = 0
        
        for case in df["Case ID"].unique():
            case_activities = df[df["Case ID"] == case]["Activity"].tolist()
            
            for i in range(len(case_activities) - 1):
                if i == 0:
                    prefix = case_activities[0]
                else:
                    prefix = " ".join(case_activities[:i+1])
                
                next_act = case_activities[i+1]
                
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_act"] = next_act
                idx += 1
        
        return processed_df
    
    def _process_next_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for next time prediction task."""
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "next_time", 
                                           "recent_time", "latest_time", "time_passed"])
        idx = 0
        
        for case in df["Case ID"].unique():
            case_data = df[df["Case ID"] == case].sort_values("Complete Timestamp")
            case_activities = case_data["Activity"].tolist()
            timestamps = pd.to_datetime(case_data["Complete Timestamp"]).tolist()
            
            for i in range(len(case_activities) - 1):
                if i == 0:
                    prefix = case_activities[0]
                else:
                    prefix = " ".join(case_activities[:i+1])
                
                # Calculate time features
                current_time = timestamps[i]
                next_time = timestamps[i+1]
                time_diff = (next_time - current_time).total_seconds() / 3600  # hours
                
                # Recent and latest times (simplified for now)
                recent_time = 0  # Placeholder
                latest_time = 0  # Placeholder
                time_passed = 0  # Placeholder
                
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_time"] = time_diff
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "time_passed"] = time_passed
                idx += 1
        
        return processed_df
    
    def _process_remaining_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for remaining time prediction task."""
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "remaining_time_days",
                                           "recent_time", "latest_time", "time_passed"])
        idx = 0
        
        for case in df["Case ID"].unique():
            case_data = df[df["Case ID"] == case].sort_values("Complete Timestamp")
            case_activities = case_data["Activity"].tolist()
            timestamps = pd.to_datetime(case_data["Complete Timestamp"]).tolist()
            
            case_end_time = timestamps[-1]
            
            for i in range(len(case_activities) - 1):
                if i == 0:
                    prefix = case_activities[0]
                else:
                    prefix = " ".join(case_activities[:i+1])
                
                # Calculate remaining time
                current_time = timestamps[i]
                remaining_time = (case_end_time - current_time).total_seconds() / (24 * 3600)  # days
                
                # Recent and latest times (simplified for now)
                recent_time = 0  # Placeholder
                latest_time = 0  # Placeholder
                time_passed = 0  # Placeholder
                
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "remaining_time_days"] = remaining_time
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "time_passed"] = time_passed
                idx += 1
        
        return processed_df
    
    def process_dataset(self, random_state: int = 42) -> Dict[str, Any]:
        """
        Process the dataset and create canonical 5-fold case-based splits.
        
        Args:
            random_state: Random seed for reproducible splits
            
        Returns:
            Dictionary with processing information
        """
        print(f"Processing dataset: {self._name}")
        
        # Load and standardize data
        df = self._load_and_standardize_df(sort_temporally=False)
        print(f"Loaded {len(df)} events from {df['Case ID'].nunique()} cases")
        
        # Extract vocabularies
        vocabularies = self._extract_vocabularies(df)
        print(f"Vocabulary size: {len(vocabularies['x_word_dict'])}")
        print(f"Number of activities: {len(vocabularies['y_word_dict'])}")
        
        # Process for all tasks
        tasks_data = {}
        
        # Next Activity
        next_activity_df = self._process_next_activity(df)
        tasks_data['next_activity'] = next_activity_df
        print(f"Next activity: {len(next_activity_df)} samples")
        
        # Next Time
        next_time_df = self._process_next_time(df)
        tasks_data['next_time'] = next_time_df
        print(f"Next time: {len(next_time_df)} samples")
        
        # Remaining Time
        remaining_time_df = self._process_remaining_time(df)
        tasks_data['remaining_time'] = remaining_time_df
        print(f"Remaining time: {len(remaining_time_df)} samples")
        
        # Create canonical splits for each task
        cv = CanonicalCrossValidation(n_folds=5, random_state=random_state)
        
        for task_name, task_df in tasks_data.items():
            print(f"\nCreating canonical splits for {task_name}...")
            splits_info = cv.create_case_based_splits(task_df, self._name, self._dir_path, task_name)
            
            # Save task-specific metadata
            task_metadata = {
                "task": task_name,
                "splits_info": splits_info,
                "vocabularies": vocabularies
            }
            
            with open(self._processed_dir / f"{task_name}_metadata.json", "w") as f:
                json.dump(task_metadata, f, indent=2)
        
        return {
            "dataset_name": self._name,
            "total_cases": df["Case ID"].nunique(),
            "total_events": len(df),
            "tasks": list(tasks_data.keys()),
            "vocabularies": vocabularies
        }


class SimplePreprocessor:
    """Simple preprocessor that uses the canonical ProcessTransformer pipeline."""
    
    def __init__(self, raw_dir: Path, processed_dir: Path, config: Dict[str, Any]):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
    
    def is_processed(self, dataset_name: str) -> bool:
        """Check if dataset is already processed with canonical splits."""
        processed_path = self.processed_dir / dataset_name / "processed"
        if not processed_path.exists():
            return False
            
        # Check for canonical splits
        splits_dir = processed_path / "splits"
        if not splits_dir.exists():
            return False
        
        # Check if all required files exist
        required_files = [
            processed_path / "metadata.json",
            splits_dir / "splits_metadata.json",
            splits_dir / "case_assignments.json"
        ]
        
        # Check if all folds exist
        for fold_idx in range(5):
            fold_dir = splits_dir / f"fold_{fold_idx}"
            if not (fold_dir / "train.csv").exists() or not (fold_dir / "val.csv").exists():
                return False
        
        return all(f.exists() for f in required_files)
    
    def preprocess_dataset(self, dataset_name: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Preprocess a dataset using the canonical ProcessTransformer pipeline.
        
        Args:
            dataset_name: Name of the dataset to preprocess
            force_reprocess: Force reprocessing even if processed data exists
            
        Returns:
            Dictionary with processing information
        """
        # Check if we can use existing processed data
        if not force_reprocess and self.is_processed(dataset_name):
            print(f"Using existing canonical processed data for {dataset_name}")
            return self._load_processing_info(dataset_name)
        
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
        
        # Use canonical processor
        print(f"Preprocessing {dataset_name} using canonical ProcessTransformer pipeline...")
        
        # Determine column names based on file format
        if "Case ID" in pd.read_csv(raw_file_path).columns:
            columns = ["Case ID", "Activity", "Complete Timestamp"]
        else:
            columns = ["case:concept:name", "concept:name", "time:timestamp"]
        
        processor = CanonicalLogsProcessor(
            name=dataset_name,
            filepath=str(raw_file_path),
            columns=columns,
            dir_path=str(self.processed_dir),
            pool=1
        )
        
        # Process dataset with canonical splits
        processing_info = processor.process_dataset(random_state=self.config.get('seed', 42))
        
        return processing_info
    
    def _load_processing_info(self, dataset_name: str) -> Dict[str, Any]:
        """Load processing information for a dataset."""
        processed_path = self.processed_dir / dataset_name / "processed"
        
        with open(processed_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        return {
            "dataset_name": dataset_name,
            "vocabularies": {
                "x_word_dict": metadata["x_word_dict"],
                "y_word_dict": metadata["y_word_dict"]
            },
            "vocab_size": metadata["vocab_size"],
            "num_activities": metadata["num_activities"]
        }
    
    def get_processed_info(self) -> Dict[str, Any]:
        """Get information about processed datasets."""
        processed_datasets = []
        total_files = 0
        
        for item in self.processed_dir.iterdir():
            if item.is_dir() and self.is_processed(item.name):
                processed_datasets.append(item.name)
                processed_path = item / "processed"
                total_files += len(list(processed_path.glob("*.csv"))) + len(list(processed_path.glob("*.json")))
        
        return {
            "processed_dir": str(self.processed_dir),
            "processed_datasets": processed_datasets,
            "total_files": total_files
        }
    
    def clear_processed_data(self, dataset_name: Optional[str] = None):
        """Clear processed data for a specific dataset or all datasets."""
        if dataset_name:
            dataset_path = self.processed_dir / dataset_name
            if dataset_path.exists():
                import shutil
                shutil.rmtree(dataset_path)
                print(f"Cleared processed data for {dataset_name}")
        else:
            import shutil
            shutil.rmtree(self.processed_dir)
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            print("Cleared all processed data")
