import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import KFold
import hashlib
import pickle
from sklearn.preprocessing import StandardScaler


class CanonicalCrossValidation:
    """
    Canonical 5-fold case-based cross-validation following Rama-Maneiro et al. (2021) methodology.
    
    This implementation ensures:
    - Case-based splits (all prefixes from same case in same fold)
    - 5-fold CV with reproducible splits
    - Persisted splits for consistent evaluation across models
    - No data leakage between train/validation sets
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """Initialize canonical cross-validation.
        
        Args:
            n_folds: Number of folds (default: 5)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        
    def create_case_based_splits(self, df: pd.DataFrame, dataset_name: str, 
                                processed_dir: Path, task_name: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Create and persist canonical 5-fold case-based splits.
        
        Args:
            df: DataFrame with processed data (must contain 'case_id' column)
            dataset_name: Name of the dataset
            processed_dir: Directory to save splits
            
        Returns:
            Dictionary with split information and metadata
        """
        if 'case_id' not in df.columns:
            raise ValueError("DataFrame must contain 'case_id' column for case-based splitting")
        
        # Get unique cases
        case_ids = df['case_id'].unique()
        print(f"Creating canonical splits for {dataset_name}: {len(case_ids)} unique cases")
        
        # Create splits directory
        base_splits_dir = processed_dir / dataset_name / "splits"
        base_splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Use task-specific subdirectory if task_name is provided
        if task_name:
            splits_dir = base_splits_dir / task_name
        else:
            splits_dir = base_splits_dir
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if splits already exist and are valid
        if not force and self._splits_exist_and_valid(splits_dir, case_ids, task_name):
            print(f"Using existing canonical splits for {dataset_name}")
            # Even if splits exist, we may need to augment CSVs with newly available optional columns
            splits_metadata = self._load_splits_metadata(splits_dir, task_name)
            try:
                self._augment_split_files_if_needed(df, splits_dir, task_name, splits_metadata)
            except Exception as e:
                # Do not fail the run if augmentation fails; keep existing splits
                print(f"Warning: failed to augment existing splits with new columns: {e}")
            return splits_metadata
        
        # Create new splits
        print(f"Creating new canonical splits for {dataset_name}")
        
        # Shuffle cases with fixed seed for reproducibility
        np.random.seed(self.random_state)
        shuffled_cases = np.random.permutation(case_ids)
        
        # Create 5-fold splits
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        splits_info = {
            'dataset_name': dataset_name,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'total_cases': len(case_ids),
            'total_samples': len(df),
            'case_ids_hash': self._hash_case_ids(case_ids),
            'folds': {}
        }
        
        for fold_idx, (train_case_idx, val_case_idx) in enumerate(kf.split(shuffled_cases)):
            train_cases = shuffled_cases[train_case_idx]
            val_cases = shuffled_cases[val_case_idx]
            
            # Get data for this fold
            train_mask = df['case_id'].isin(train_cases)
            val_mask = df['case_id'].isin(val_cases)
            
            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()
            
            # Save fold data
            fold_dir = splits_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)
            
            train_df.to_csv(fold_dir / "train.csv", index=False)
            val_df.to_csv(fold_dir / "val.csv", index=False)

            # Persist scalers for time-based tasks (fit on train only)
            if task_name in ("next_time", "remaining_time"):
                try:
                    # Feature scaler
                    time_scaler = StandardScaler()
                    train_time_x = train_df[["recent_time", "latest_time", "time_passed"]].values.astype(float)
                    time_scaler.fit(train_time_x)
                    with open(fold_dir / f"time_scaler_{task_name}.pkl", "wb") as f:
                        pickle.dump(time_scaler, f)
                    # Target scaler
                    y_col = "next_time" if task_name == "next_time" else "remaining_time_days"
                    y_scaler = StandardScaler()
                    y_scaler.fit(train_df[[y_col]].values.astype(float))
                    with open(fold_dir / f"y_scaler_{task_name}.pkl", "wb") as f:
                        pickle.dump(y_scaler, f)
                except Exception as e:
                    print(f"Warning: failed to persist scalers for {task_name} fold {fold_idx}: {e}")
            
            # Store fold information
            splits_info['folds'][f'fold_{fold_idx}'] = {
                'train_cases': len(train_cases),
                'val_cases': len(val_cases),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'train_case_ids': train_cases.tolist(),
                'val_case_ids': val_cases.tolist()
            }
            
            print(f"Fold {fold_idx + 1}: {len(train_cases)} train cases ({len(train_df)} samples), "
                  f"{len(val_cases)} val cases ({len(val_df)} samples)")
        
        # Save splits metadata (task-specific if task_name is provided)
        if task_name:
            metadata_file = splits_dir / f"splits_metadata_{task_name}.json"
        else:
            metadata_file = splits_dir / "splits_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(splits_info, f, indent=2)
        
        # Save case assignments for verification (global, task-independent)
        case_assignments = {}
        for fold_idx in range(self.n_folds):
            fold_info = splits_info['folds'][f'fold_{fold_idx}']
            for case_id in fold_info['train_case_ids']:
                case_assignments[case_id] = {'fold': fold_idx, 'split': 'train'}
            for case_id in fold_info['val_case_ids']:
                case_assignments[case_id] = {'fold': fold_idx, 'split': 'val'}
        
        # Save global case assignments (task-independent)
        with open(base_splits_dir / "case_assignments.json", 'w') as f:
            json.dump(case_assignments, f, indent=2)
        
        return splits_info
    
    def _splits_exist_and_valid(self, splits_dir: Path, case_ids: np.ndarray, task_name: Optional[str] = None) -> bool:
        """Check if canonical splits exist and are valid."""
        # Use task-specific metadata file if task_name is provided
        if task_name:
            metadata_file = splits_dir / f"splits_metadata_{task_name}.json"
        else:
            metadata_file = splits_dir / "splits_metadata.json"
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if case IDs hash matches
            stored_hash = metadata.get('case_ids_hash')
            current_hash = self._hash_case_ids(case_ids)
            
            if stored_hash != current_hash:
                print(f"Case IDs hash mismatch - recreating splits")
                return False
            
            # Check if all fold files exist
            for fold_idx in range(self.n_folds):
                fold_dir = splits_dir / f"fold_{fold_idx}"
                if not (fold_dir / "train.csv").exists() or not (fold_dir / "val.csv").exists():
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validating splits: {e}")
            return False
    
    def _load_splits_metadata(self, splits_dir: Path, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Load splits metadata from disk."""
        # Use task-specific metadata file if task_name is provided
        if task_name:
            metadata_file = splits_dir / f"splits_metadata_{task_name}.json"
        else:
            metadata_file = splits_dir / "splits_metadata.json"
        
        with open(metadata_file, 'r') as f:
            return json.load(f)

    def _augment_split_files_if_needed(self, task_df: pd.DataFrame, splits_dir: Path, task_name: Optional[str], splits_metadata: Dict[str, Any]) -> None:
        """Augment existing split CSVs with newly available optional columns without changing case assignments.
        Currently supports next_activity extended attributes: resource_prefix and delta_t_prefix.
        """
        if task_name != "next_activity":
            return
        if task_df is None or task_df.empty:
            return
        desired_cols = set(task_df.columns)
        extended_cols = [c for c in ["resource_prefix", "delta_t_prefix"] if c in desired_cols]
        if not extended_cols:
            return
        # Check whether existing split files already contain these columns
        any_missing = False
        for fold_idx in range(self.n_folds):
            fold_dir = splits_dir / f"fold_{fold_idx}"
            train_path = fold_dir / "train.csv"
            if not train_path.exists():
                continue
            try:
                existing_cols = set(pd.read_csv(train_path, nrows=0).columns)
            except Exception:
                continue
            for c in extended_cols:
                if c not in existing_cols:
                    any_missing = True
                    break
            if any_missing:
                break
        if not any_missing:
            return
        print(f"Augmenting existing canonical splits to include extended columns: {extended_cols}")
        # Reconstruct per-fold files using stored case assignments in splits_metadata
        folds_meta = (splits_metadata or {}).get("folds", {})
        for fold_name, fold_info in folds_meta.items():
            try:
                fold_idx = int(str(fold_name).split("_")[-1])
            except Exception:
                continue
            train_cases = set(fold_info.get("train_case_ids", []) or [])
            val_cases = set(fold_info.get("val_case_ids", []) or [])
            train_df = task_df[task_df["case_id"].isin(train_cases)].copy()
            val_df = task_df[task_df["case_id"].isin(val_cases)].copy()
            fold_dir = splits_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)
            train_df.to_csv(fold_dir / "train.csv", index=False)
            val_df.to_csv(fold_dir / "val.csv", index=False)

    def _hash_case_ids(self, case_ids: np.ndarray) -> str:
        """Create hash of case IDs for validation."""
        sorted_ids = sorted(case_ids)
        return hashlib.md5(str(sorted_ids).encode()).hexdigest()
    
    def load_fold_data(self, dataset_name: str, fold_idx: int, 
                      processed_dir: Path, task_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and validation data for a specific fold.
        
        Args:
            dataset_name: Name of the dataset
            fold_idx: Fold index (0-4)
            processed_dir: Directory containing processed data
            
        Returns:
            Tuple of (train_df, val_df)
        """
        base_splits_dir = processed_dir / dataset_name / "splits"
        if task_name:
            splits_dir = base_splits_dir / task_name
        else:
            splits_dir = base_splits_dir
        fold_dir = splits_dir / f"fold_{fold_idx}"
        
        if not fold_dir.exists():
            raise ValueError(f"Fold {fold_idx} not found for dataset {dataset_name}")
        
        train_df = pd.read_csv(fold_dir / "train.csv")
        val_df = pd.read_csv(fold_dir / "val.csv")
        
        return train_df, val_df
    
    def get_splits_info(self, dataset_name: str, processed_dir: Path, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about existing splits."""
        base_splits_dir = processed_dir / dataset_name / "splits"
        if task_name:
            splits_dir = base_splits_dir / task_name
            metadata_file = splits_dir / f"splits_metadata_{task_name}.json"
        else:
            splits_dir = base_splits_dir
            metadata_file = splits_dir / "splits_metadata.json"
        
        if not metadata_file.exists():
            return {}
        
        with open(metadata_file, 'r') as f:
            return json.load(f)


def run_canonical_cross_validation(task_class, config: Dict[str, Any], 
                                  df: pd.DataFrame, dataset_name: str,
                                  processed_dir: Path, task_name: str) -> Dict[str, Any]:
    """
    Run canonical cross-validation for a given task.
    
    Args:
        task_class: Task class to instantiate
        config: Configuration dictionary
        df: DataFrame with all data
        dataset_name: Name of the dataset
        processed_dir: Directory containing processed data
        
    Returns:
        Dictionary with CV results including fold results and aggregated metrics
    """
    cv = CanonicalCrossValidation(
        n_folds=config['cv'].get('n_folds', 5),
        random_state=config.get('seed', 42)
    )
    
    # Create or load canonical splits
    splits_info = cv.create_case_based_splits(df, dataset_name, processed_dir, task_name)
    
    fold_results = []
    
    for fold_idx in range(splits_info['n_folds']):
        print(f"\n=== Fold {fold_idx + 1}/{splits_info['n_folds']} ===")
        
        # Load fold data
        train_df, val_df = cv.load_fold_data(dataset_name, fold_idx, processed_dir, task_name)
        
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Create task instance
        task = task_class(config)
        task.current_dataset = dataset_name
        
        # Train and evaluate
        fold_result = task.train_and_evaluate_fold(
            train_df, val_df, fold_idx
        )
        
        fold_results.append(fold_result)
        print(f"Fold {fold_idx + 1} results: {fold_result['metrics']}")
    
    # Aggregate results
    cv_results = aggregate_cv_results(fold_results)
    
    return {
        'fold_results': fold_results,
        'cv_summary': cv_results,
        'splits_info': splits_info,
        'dataset_name': dataset_name
    }


def aggregate_cv_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from all CV folds.
    
    Args:
        fold_results: List of results from each fold
    Returns:
        Dictionary with aggregated metrics (mean, std, min, max)
    """
    if not fold_results:
        return {}
    
    # Extract metrics from all folds
    all_metrics = [fold['metrics'] for fold in fold_results]
    
    # Aggregate each metric
    aggregated = {}
    for metric_name in all_metrics[0].keys():
        if isinstance(all_metrics[0][metric_name], (int, float)):
            values = [fold[metric_name] for fold in all_metrics]
            aggregated[f"{metric_name}_mean"] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_min"] = np.min(values)
            aggregated[f"{metric_name}_max"] = np.max(values)
            aggregated[f"{metric_name}_values"] = values  # Keep individual fold values
        else:
            # For non-numeric metrics, just store all values
            aggregated[f"{metric_name}_all"] = [fold[metric_name] for fold in all_metrics]
    
    return aggregated
