import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .encoders import Vocabulary
from ..utils.cross_validation import CanonicalCrossValidation


class CanonicalLogsDataLoader:
    """
    Canonical logs data loader that works with 5-fold case-based splits.
    
    This loader provides access to the canonical splits created by the CanonicalLogsProcessor
    and ensures consistent data loading across all models and experiments.
    """
    
    def __init__(self, name: str, dir_path: str = "./data/processed"):
        """Initialize canonical logs data loader.
        
        Args:
            name: Dataset name
            dir_path: Path to processed data directory
        """
        self._name = name
        self._dir_path = Path(dir_path)
        self._processed_dir = self._dir_path / name / "processed"
        self._splits_dir = self._dir_path / name / "splits"
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        metadata_file = self._processed_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.x_word_dict = self.metadata["x_word_dict"]
        self.y_word_dict = self.metadata["y_word_dict"]
        self.vocab_size = self.metadata["vocab_size"]
        self.num_activities = self.metadata["num_activities"]
    
    def load_fold_data(self, task: str, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and validation data for a specific task and fold.
        
        Args:
            task: Task name (next_activity, suffix, next_time, remaining_time)
            fold_idx: Fold index (0-4)
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if task not in ["next_activity", "suffix", "next_time", "remaining_time"]:
            raise ValueError("Invalid task. Must be one of: next_activity, suffix, next_time, remaining_time")
        
        if not 0 <= fold_idx <= 4:
            raise ValueError("Invalid fold index. Must be between 0 and 4")
        
        # Use task-specific subdirectory
        fold_dir = self._splits_dir / task / f"fold_{fold_idx}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold {fold_idx} not found for dataset {self._name} and task {task}")
        
        train_df = pd.read_csv(fold_dir / "train.csv")
        val_df = pd.read_csv(fold_dir / "val.csv")
        
        return train_df, val_df
    
    def get_splits_info(self, task: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the canonical splits."""
        if task:
            metadata_file = self._splits_dir / task / f"splits_metadata_{task}.json"
        else:
            metadata_file = self._splits_dir / "splits_metadata.json"
        if not metadata_file.exists():
            return {}
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def get_max_case_length(self, df: pd.DataFrame) -> int:
        """Compute the maximum number of activities per prefix in a DataFrame.
        Expects a column 'prefix' where each entry is a space-separated string
        of activity tokens.
        """
        max_length = 0
        for prefix in df["prefix"].values:
            length = len(prefix.split())
            max_length = max(max_length, length)
        return max_length
    
    def prepare_next_activity_data(self, df: pd.DataFrame, 
                                  max_case_length: int, 
                                  shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for next activity prediction task."""
        x = df["prefix"].values
        y = df["next_act"].values
        
        if shuffle:
            x, y = utils.shuffle(x, y)

        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])

        # Tokenize labels
        token_y = []
        for _y in y:
            token_y.append(self.y_word_dict[_y])

        # Pad sequences
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')

        token_x = np.array(token_x, dtype=np.int32)
        token_y = np.array(token_y, dtype=np.int32)

        return token_x, token_y
    
    def prepare_next_time_data(self, df: pd.DataFrame, 
                              max_case_length: int,
                              time_scaler: Optional[preprocessing.StandardScaler] = None, 
                              y_scaler: Optional[preprocessing.StandardScaler] = None, 
                              shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, preprocessing.StandardScaler, preprocessing.StandardScaler]:
        """Prepare data for next time prediction task."""
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["next_time"].values.astype(np.float32)
        
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])

        # Scale time features
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)            

        # Scale target
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        # Pad sequences
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')
        
        token_x = np.array(token_x, dtype=np.int32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler, y_scaler
    
    def prepare_remaining_time_data(self, df: pd.DataFrame, 
                                   max_case_length: int,
                                   time_scaler: Optional[preprocessing.StandardScaler] = None, 
                                   y_scaler: Optional[preprocessing.StandardScaler] = None, 
                                   shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, preprocessing.StandardScaler, preprocessing.StandardScaler]:
        """Prepare data for remaining time prediction task."""
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["remaining_time_days"].values.astype(np.float32)

        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.x_word_dict[s] for s in _x.split()])

        # Scale time features
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)            

        # Scale target
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        # Pad sequences
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')
        
        token_x = np.array(token_x, dtype=np.int32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        return token_x, time_x, y, time_scaler, y_scaler
    
    def is_processed(self) -> bool:
        """Dataset-level check: metadata and global case assignments exist.
        Task-specific fold files are validated when loading per task.
        """
        if not self._processed_dir.exists():
            return False
        if not self._splits_dir.exists():
            return False
        required_files = [
            self._processed_dir / "metadata.json",
            self._splits_dir / "case_assignments.json"
        ]
        return all(f.exists() for f in required_files)


# Legacy loader for backward compatibility (deprecated)
class LogsDataLoader:
    """Legacy data loader - deprecated. Use CanonicalLogsDataLoader instead."""
    
    def __init__(self, name: str, dir_path: str = "./data/processed"):
        """Initialize legacy loader (deprecated)."""
        import warnings
        warnings.warn("LogsDataLoader is deprecated. Use CanonicalLogsDataLoader instead.", 
                     DeprecationWarning, stacklevel=2)
        
        self._dir_path = f"{dir_path}/{name}/processed"
        self._name = name

    def prepare_data_next_activity(self, df: pd.DataFrame, 
        x_word_dict: Dict[str, int], y_word_dict: Dict[str, int], 
        max_case_length: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for next activity prediction task (legacy)."""
        x = df["prefix"].values
        y = df["next_act"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)

        token_x = np.array(token_x, dtype=np.int32)
        token_y = np.array(token_y, dtype=np.int32)

        return token_x, token_y

    def prepare_data_next_time(self, df: pd.DataFrame, 
        x_word_dict: Dict[str, int], max_case_length: int, 
        time_scaler: Optional[preprocessing.StandardScaler] = None, 
        y_scaler: Optional[preprocessing.StandardScaler] = None, 
        shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, preprocessing.StandardScaler, preprocessing.StandardScaler]:
        """Prepare data for next time prediction task (legacy)."""
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["next_time"].values.astype(np.float32)
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)            

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        
        token_x = np.array(token_x, dtype=np.int32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler, y_scaler

    def prepare_data_remaining_time(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
        max_case_length: int, time_scaler: Optional[preprocessing.StandardScaler] = None, 
        y_scaler: Optional[preprocessing.StandardScaler] = None, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, preprocessing.StandardScaler, preprocessing.StandardScaler]:
        """Prepare data for remaining time prediction task (legacy)."""
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["remaining_time_days"].values.astype(np.float32)

        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)            

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        
        token_x = np.array(token_x, dtype=np.int32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        return token_x, time_x, y, time_scaler, y_scaler

    def get_max_case_length(self, train_x: np.ndarray) -> int:
        """Get maximum case length from training data (legacy)."""
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int], int, int, int]:
        """Load processed data for a specific task (legacy)."""
        if task not in ["next_activity", "next_time", "remaining_time"]:
            raise ValueError("Invalid task. Must be one of: next_activity, next_time, remaining_time")

        train_df = pd.read_csv(f"{self._dir_path}/{task}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task}_test.csv")

        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, test_df, x_word_dict, y_word_dict, 
                max_case_length, vocab_size, total_classes)

    def is_processed(self) -> bool:
        """Check if dataset is already processed (legacy)."""
        required_files = [
            f"{self._dir_path}/next_activity_train.csv",
            f"{self._dir_path}/next_activity_test.csv", 
            f"{self._dir_path}/next_time_train.csv",
            f"{self._dir_path}/next_time_test.csv",
            f"{self._dir_path}/remaining_time_train.csv",
            f"{self._dir_path}/remaining_time_test.csv",
            f"{self._dir_path}/metadata.json"
        ]
        return all(os.path.exists(f) for f in required_files)


def load_event_log(file_path: str) -> pd.DataFrame:
    """
    Load an event log from a CSV or XES file into a pandas DataFrame.
    Returns:
        pandas DataFrame with the event log
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        
        # Handle different column name formats
        column_mapping = {}
        
        # Map case ID column
        if "case:concept:name" in df.columns:
            column_mapping["case:concept:name"] = "case:concept:name"
        elif "case_id" in df.columns:
            column_mapping["case:concept:name"] = "case_id"
        elif "event_log_case_id" in df.columns:
            column_mapping["case:concept:name"] = "event_log_case_id"
        elif "Case ID" in df.columns:
            column_mapping["case:concept:name"] = "Case ID"
        else:
            raise ValueError("No case ID column found. Expected: case:concept:name, case_id, event_log_case_id, or Case ID")
            
        # Map activity column
        if "concept:name" in df.columns:
            column_mapping["concept:name"] = "concept:name"
        elif "activity" in df.columns:
            column_mapping["concept:name"] = "activity"
        elif "event_log_activity" in df.columns:
            column_mapping["concept:name"] = "event_log_activity"
        elif "Activity" in df.columns:
            column_mapping["concept:name"] = "Activity"
        else:
            raise ValueError("No activity column found. Expected: concept:name, activity, event_log_activity, or Activity")
            
        # Map timestamp column
        if "time:timestamp" in df.columns:
            column_mapping["time:timestamp"] = "time:timestamp"
        elif "timestamp" in df.columns:
            column_mapping["time:timestamp"] = "timestamp"
        elif "event_log_timestamp" in df.columns:
            column_mapping["time:timestamp"] = "event_log_timestamp"
        elif "Complete Timestamp" in df.columns:
            column_mapping["time:timestamp"] = "Complete Timestamp"
        else:
            raise ValueError("No timestamp column found. Expected: time:timestamp, timestamp, event_log_timestamp, or Complete Timestamp")
        
        # Rename columns to standard format
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Parse timestamps
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        
        return df
    elif file_path.endswith(".xes"):
        # For XES files, we'll need to implement XES parsing
        # For now, raise an error
        raise NotImplementedError("XES file support not yet implemented")
    else:
        raise ValueError("Unsupported file format. Only CSV files are currently supported")


def get_available_datasets(processed_dir: str = "data/processed") -> list:
    """
    Get list of available datasets in the processed directory.
    Args:
        processed_dir: Directory containing processed data
    Returns:
        List of dataset names
    """
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        return []
    
    # Look for directories that contain processed data with canonical splits
    available_datasets = []
    for item in processed_path.iterdir():
        if item.is_dir():
            # Check if it has canonical splits
            loader = CanonicalLogsDataLoader(item.name, processed_dir)
            if loader.is_processed():
                available_datasets.append(item.name)
    
    return available_datasets
