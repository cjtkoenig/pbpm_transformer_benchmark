"""
Data preprocessing based on the original ProcessTransformer implementation.
Provides standardized preprocessing for all PBPM tasks with adapter support.
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


class LogsDataProcessor:
    def __init__(self, name: str, filepath: str, columns: List[str], 
                 dir_path: str = "./data/processed", pool: int = 1):
        """Provides support for processing raw logs.
        
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path: str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

    def _load_df(self, sort_temporally: bool = False) -> pd.DataFrame:
        """Load and standardize the raw event log."""
        df = load_event_log(self._filepath)
        
        # Standardize column names to match original ProcessTransformer format
        if "Case ID" in self._org_columns:
            df = df.rename(columns={"case:concept:name": "Case ID", 
                                   "concept:name": "Activity", 
                                   "time:timestamp": "Complete Timestamp"})
        
        # Apply the same preprocessing as original ProcessTransformer
        df["Activity"] = df["Activity"].str.lower()
        df["Activity"] = df["Activity"].str.replace(" ", "-")
        
        # Handle timestamp - convert to string format if it's not already
        if df["Complete Timestamp"].dtype == 'object':
            df["Complete Timestamp"] = df["Complete Timestamp"].str.replace("/", "-")
            df["Complete Timestamp"] = pd.to_datetime(df["Complete Timestamp"], 
                                                     dayfirst=True).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            # If it's already a datetime, convert to string format
            df["Complete Timestamp"] = df["Complete Timestamp"].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        if sort_temporally:
            df.sort_values(by=["Complete Timestamp"], inplace=True)
        
        return df

    def _extract_logs_metadata(self, df: pd.DataFrame):
        """Extract and save metadata including vocabularies."""
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["Activity"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict": dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function for next activity processing."""
        case_id, case_name = "Case ID", "Activity"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))        
                next_act = act[i+1]
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_act"] = next_act
                idx = idx + 1
        return processed_df

    def _process_next_activity(self, df: pd.DataFrame, train_list: List[str], test_list: List[str]):
        """Process data for next activity prediction task."""
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/next_activity_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/next_activity_test.csv", index=False)

    def _next_time_helper_func(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function for next time processing."""
        case_id = "Case ID"
        event_name = "Activity"
        event_time = "Complete Timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            next_time = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time
                if i+1 < len(time):
                    next_time = datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "next_time"] = next_time_days
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time", "next_time"]]
        return processed_df_time

    def _process_next_time(self, df: pd.DataFrame, train_list: List[str], test_list: List[str]):
        """Process data for next time prediction task."""
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_time_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/next_time_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/next_time_test.csv", index=False)

    def _remaining_time_helper_func(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function for remaining time processing."""
        case_id = "Case ID"
        event_name = "Activity"
        event_time = "Complete Timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed", 
                "recent_time", "latest_time", "next_act", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                        datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                ttc = str(ttc.days)  

                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k", 
            "time_passed", "recent_time", "latest_time", "remaining_time_days"]]
        return processed_df_remaining_time

    def _process_remaining_time(self, df: pd.DataFrame, train_list: List[str], test_list: List[str]):
        """Process data for remaining time prediction task."""
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._remaining_time_helper_func, df_split))
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/remaining_time_train.csv", index=False)
        test_remaining_time.to_csv(f"{self._dir_path}/remaining_time_test.csv", index=False)

    def process_logs(self, task: str, sort_temporally: bool = False, 
                    train_test_ratio: float = 0.80):
        """Process raw logs for a specific task."""
        df = self._load_df(sort_temporally)
        self._extract_logs_metadata(df)
        train_test_ratio = int(abs(df["Case ID"].nunique() * train_test_ratio))
        train_list = df["Case ID"].unique()[:train_test_ratio]
        test_list = df["Case ID"].unique()[train_test_ratio:]
        
        if task == "next_activity":
            self._process_next_activity(df, train_list, test_list)
        elif task == "next_time":
            self._process_next_time(df, train_list, test_list)
        elif task == "remaining_time":
            self._process_remaining_time(df, train_list, test_list)
        else:
            raise ValueError("Invalid task. Must be one of: next_activity, next_time, remaining_time")


class SimplePreprocessor:
    """Simple preprocessor that uses the original ProcessTransformer pipeline."""
    
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
        """Check if dataset is already processed using ProcessTransformer format."""
        processed_path = self.processed_dir / dataset_name / "processed"
        if not processed_path.exists():
            return False
            
        required_files = [
            processed_path / "next_activity_train.csv",
            processed_path / "next_activity_test.csv", 
            processed_path / "next_time_train.csv",
            processed_path / "next_time_test.csv",
            processed_path / "remaining_time_train.csv",
            processed_path / "remaining_time_test.csv",
            processed_path / "metadata.json"
        ]
        return all(f.exists() for f in required_files)
    
    def load_processed_data(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load processed data in the original format for backward compatibility."""
        # For backward compatibility, we'll create the old format from the new format
        from .loader import LogsDataLoader
        
        data_loader = LogsDataLoader(dataset_name, str(self.processed_dir))
        
        # Load next activity data as default
        train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("next_activity")
        
        # Combine train and test for the old format
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create vocabulary from the metadata
        vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create labels series
        labels_series = combined_df["next_act"]
        
        return combined_df, labels_series, vocabulary
    
    def save_processed_data(self, dataset_name: str, prefixes_df: pd.DataFrame, 
                           labels_series: pd.Series, vocabulary: Vocabulary):
        """Save processed data in the original format for backward compatibility."""
        prefixes_file, labels_file, vocabulary_file = self._get_processed_paths(dataset_name)
        
        # Save in pickle format for backward compatibility
        import pickle
        with open(prefixes_file, 'wb') as f:
            pickle.dump(prefixes_df, f)
        with open(labels_file, 'wb') as f:
            pickle.dump(labels_series, f)
        with open(vocabulary_file, 'wb') as f:
            pickle.dump(vocabulary, f)
    
    def preprocess_dataset(self, dataset_name: str, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """
        Preprocess a dataset using the original ProcessTransformer pipeline.
        
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
        
        # Use ProcessTransformer processor
        print(f"Preprocessing {dataset_name} using ProcessTransformer pipeline...")
        
        # Determine column names based on file format
        if "Case ID" in pd.read_csv(raw_file_path).columns:
            columns = ["Case ID", "Activity", "Complete Timestamp"]
        else:
            columns = ["case:concept:name", "concept:name", "time:timestamp"]
        
        processor = LogsDataProcessor(
            name=dataset_name,
            filepath=str(raw_file_path),
            columns=columns,
            dir_path=str(self.processed_dir),
            pool=1
        )
        
        # Process for all tasks
        processor.process_logs(task="next_activity", sort_temporally=False)
        processor.process_logs(task="next_time", sort_temporally=False)
        processor.process_logs(task="remaining_time", sort_temporally=False)
        
        # Return data in the old format for backward compatibility
        return self.load_processed_data(dataset_name)
    
    def get_processed_info(self) -> Dict[str, Any]:
        """Get information about processed datasets."""
        processed_datasets = []
        total_files = 0
        
        for item in self.processed_dir.iterdir():
            if item.is_dir() and (item / "processed").exists():
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
