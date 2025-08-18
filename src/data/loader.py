import pandas as pandas_lib
from pathlib import Path
import os

def load_event_log(file_path: str):
    """
    Load an event log from a CSV file into a pandas DataFrame.
    Returns:
        pandas DataFrame with the event log
    """
    if not file_path.endswith(".csv"):
        raise ValueError("Only CSV files are supported")
    df = pandas_lib.read_csv(file_path)
    
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
    df["time:timestamp"] = pandas_lib.to_datetime(df["time:timestamp"])
    
    return df

def get_available_datasets(processed_dir: str = "data/processed"):
    """
    Get list of available datasets in the processed directory.
    Args:
        processed_dir: Directory containing processed CSV files
    Returns:
        List of dataset names (without .csv extension)
    """
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        return []
    
    csv_files = list(processed_path.glob("*.csv"))
    return [file.stem for file in csv_files]
