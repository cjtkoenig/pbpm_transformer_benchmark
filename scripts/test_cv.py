#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.cross_validation import CanonicalCrossValidation
from src.data.loader import CanonicalLogsDataLoader
import pandas as pd

# Test the canonical cross-validation
def test_cv():
    print("Testing canonical cross-validation...")
    
    # Initialize CV
    cv = CanonicalCrossValidation(n_folds=5, random_state=42)
    
    # Check if splits exist
    processed_dir = Path("../data/processed")
    dataset_name = "Helpdesk"
    
    # Try to load splits info
    splits_info = cv.get_splits_info(dataset_name, processed_dir)
    print(f"Splits info keys: {list(splits_info.keys()) if splits_info else 'No splits found'}")
    
    # Try to load fold data with task name
    try:
        train_df, val_df = cv.load_fold_data(dataset_name, 0, processed_dir, "next_activity")
        print(f"Successfully loaded fold 0: {len(train_df)} train, {len(val_df)} val")
    except Exception as e:
        print(f"Error loading fold 0: {e}")
    
    # Try to load with canonical loader
    try:
        loader = CanonicalLogsDataLoader(dataset_name, str(processed_dir))
        train_df, val_df = loader.load_fold_data("next_activity", 0)
        print(f"Successfully loaded with canonical loader: {len(train_df)} train, {len(val_df)} val")
    except Exception as e:
        print(f"Error with canonical loader: {e}")

if __name__ == "__main__":
    test_cv()
