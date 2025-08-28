import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split


class ProcessMiningCrossValidation:
    """Cross-validation for process mining tasks with case-based splitting to avoid data leakage."""
    
    def __init__(self, n_folds: int = 5, stratify: Optional[str] = None, 
                 random_state: int = 42):
        """Initialize cross-validation.
        Args:
            n_folds: Number of folds for cross-validation
            stratify: Column name to stratify by (currently not used - kept for compatibility)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.stratify = stratify
        self.random_state = random_state
        
    def split_by_cases(self, prefixes_df: pd.DataFrame, 
                      labels_series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data by case IDs to avoid data leakage.
        
        This ensures that all prefixes from the same case are either in train or validation,
        preventing data leakage between related prefixes.
        
        Args:
            prefixes_df: DataFrame with prefix data (must contain 'case_id' column)
            labels_series: Series with labels
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        if 'case_id' not in prefixes_df.columns:
            raise ValueError("DataFrame must contain 'case_id' column for case-based splitting")

        case_ids = prefixes_df['case_id'].unique()
        print(f"Found {len(case_ids)} unique cases for cross-validation")

        # Use simple k-fold split on cases (no stratification for now)
        # This ensures each case appears in exactly one validation fold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        splits = []
        for fold_idx, (train_case_idx, val_case_idx) in enumerate(kf.split(case_ids)):
            train_cases = case_ids[train_case_idx]
            val_cases = case_ids[val_case_idx]
            
            # Get indices for all prefixes belonging to these cases
            train_indices = prefixes_df[prefixes_df['case_id'].isin(train_cases)].index.values
            val_indices = prefixes_df[prefixes_df['case_id'].isin(val_cases)].index.values
            
            print(f"Fold {fold_idx + 1}: {len(train_cases)} train cases ({len(train_indices)} prefixes), "
                  f"{len(val_cases)} val cases ({len(val_indices)} prefixes)")
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    def split_by_prefixes(self, prefixes_df: pd.DataFrame, 
                         labels_series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split data by individual prefixes (standard CV - NOT recommended for process mining).
        
        WARNING: This can cause data leakage as prefixes from the same case may appear
        in both train and validation sets.
        
        Args:
            prefixes_df: DataFrame with prefix data
            labels_series: Series with labels
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(prefixes_df)
        indices = np.arange(n_samples)
        
        if self.stratify:
            # Stratified split based on labels
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                random_state=self.random_state)
            return list(skf.split(indices, labels_series.values))
        else:
            # Simple k-fold split
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            return list(kf.split(indices))


def run_cross_validation(task_class, config: Dict[str, Any], 
                        prefixes_df: pd.DataFrame, labels_series: pd.Series,
                        vocabulary, cv_config: Dict[str, Any], 
                        dataset_name: str = None) -> Dict[str, Any]:
    """
    Run cross-validation for a given task.
    
    Args:
        task_class: Task class to instantiate
        config: Configuration dictionary
        prefixes_df: DataFrame with prefix data
        labels_series: Series with labels
        vocabulary: Vocabulary object
        cv_config: Cross-validation configuration
        dataset_name: Name of the dataset being processed
    Returns:
        Dictionary with CV results including fold results and aggregated metrics
    """
    cv = ProcessMiningCrossValidation(
        n_folds=cv_config.get('n_folds', 5),
        stratify=cv_config.get('stratify'),
        random_state=config.get('seed', 42)
    )
    
    # Choose split method based on task
    if cv_config.get('split_by_cases', True):
        print(f"Using case-based cross-validation for {dataset_name}")
        splits = cv.split_by_cases(prefixes_df, labels_series)
    else:
        print(f"Using prefix-based cross-validation for {dataset_name}")
        splits = cv.split_by_prefixes(prefixes_df, labels_series)
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/{len(splits)} ===")
        
        # Split data
        train_prefixes = prefixes_df.iloc[train_indices].reset_index(drop=True)
        train_labels = labels_series.iloc[train_indices].reset_index(drop=True)
        val_prefixes = prefixes_df.iloc[val_indices].reset_index(drop=True)
        val_labels = labels_series.iloc[val_indices].reset_index(drop=True)
        
        print(f"Train samples: {len(train_prefixes)}, Validation samples: {len(val_prefixes)}")
        
        # Create task instance
        task = task_class(config)
        task.vocabulary = vocabulary
        
        # Set the dataset name
        if dataset_name:
            task.current_dataset = dataset_name
        
        # Train and evaluate
        fold_result = task.train_and_evaluate_fold(
            train_prefixes, train_labels, val_prefixes, val_labels, fold_idx
        )
        
        fold_results.append(fold_result)
        print(f"Fold {fold_idx + 1} results: {fold_result['metrics']}")
    
    # Aggregate results
    cv_results = aggregate_cv_results(fold_results)
    
    return {
        'fold_results': fold_results,
        'cv_summary': cv_results,
        'cv_config': cv_config,
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
