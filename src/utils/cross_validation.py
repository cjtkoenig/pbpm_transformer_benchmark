import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split


class ProcessMiningCrossValidation:
    """Cross-validation for process mining tasks with support for stratification."""
    
    def __init__(self, n_folds: int = 5, stratify: Optional[str] = None, 
                 random_state: int = 42):
        """Initialize cross-validation.
        Args:
            n_folds: Number of folds for cross-validation
            stratify: Column name to stratify by (e.g., 'case_id' for case-level CV)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.stratify = stratify
        self.random_state = random_state
        
    def split_by_cases(self, prefixes_df: pd.DataFrame, 
                      labels_series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data by case IDs to avoid data leakage.
        Args:
            prefixes_df: DataFrame with prefix data
            labels_series: Series with labels
        Returns:
            List of (train_indices, val_indices) tuples
        """

        case_ids = prefixes_df['case_id'].unique()

        if self.stratify:
            # Stratified split based on case-level labels
            case_labels = []
            for case_id in case_ids:
                case_data = prefixes_df[prefixes_df['case_id'] == case_id]
                case_label = case_data['prefix_length'].mean()  # Use prefix length as stratification
                case_labels.append(case_label)
            
            # Create stratified k-fold split
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                random_state=self.random_state)
            
            splits = []
            for train_case_idx, val_case_idx in skf.split(case_ids, case_labels):
                train_cases = case_ids[train_case_idx]
                val_cases = case_ids[val_case_idx]
                
                # Get indices for all prefixes belonging to these cases
                train_indices = prefixes_df[prefixes_df['case_id'].isin(train_cases)].index.values
                val_indices = prefixes_df[prefixes_df['case_id'].isin(val_cases)].index.values
                
                splits.append((train_indices, val_indices))
            
            return splits
        else:
            # Simple k-fold split
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            splits = []
            for train_case_idx, val_case_idx in kf.split(case_ids):
                train_cases = case_ids[train_case_idx]
                val_cases = case_ids[val_case_idx]
                
                # Get indices for all prefixes belonging to these cases
                train_indices = prefixes_df[prefixes_df['case_id'].isin(train_cases)].index.values
                val_indices = prefixes_df[prefixes_df['case_id'].isin(val_cases)].index.values
                
                splits.append((train_indices, val_indices))
            
            return splits
    
    def split_by_prefixes(self, prefixes_df: pd.DataFrame, 
                         labels_series: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split data by individual prefixes (standard CV).
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
                        vocabulary, cv_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run cross-validation for a given task.
    Args:
        task_class: Task class to instantiate
        config: Configuration dictionary
        prefixes_df: DataFrame with prefix data
        labels_series: Series with labels
        vocabulary: Vocabulary object
        cv_config: Cross-validation configuration
    Returns:
        Dictionary with CV results
    """
    cv = ProcessMiningCrossValidation(
        n_folds=cv_config.get('n_folds', 5),
        stratify=cv_config.get('stratify'),
        random_state=config.get('seed', 42)
    )
    
    # Choose split method based on task
    if cv_config.get('split_by_cases', True):
        splits = cv.split_by_cases(prefixes_df, labels_series)
    else:
        splits = cv.split_by_prefixes(prefixes_df, labels_series)
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/{len(splits)} ===")
        
        # Split data
        train_prefixes = prefixes_df.iloc[train_indices].reset_index(drop=True)
        train_labels = labels_series.iloc[train_indices].reset_index(drop=True)
        val_prefixes = prefixes_df.iloc[val_indices].reset_index(drop=True)
        val_labels = labels_series.iloc[val_indices].reset_index(drop=True)
        
        # Create task instance
        task = task_class(config)
        task.vocabulary = vocabulary
        
        # Train and evaluate
        fold_result = task.train_and_evaluate_fold(
            train_prefixes, train_labels, val_prefixes, val_labels, fold_idx
        )
        
        fold_results.append(fold_result)
        print(f"Fold {fold_idx + 1} - {fold_result['metrics']}")
    
    # Aggregate results
    cv_results = aggregate_cv_results(fold_results)
    
    return {
        'fold_results': fold_results,
        'cv_summary': cv_results,
        'cv_config': cv_config
    }


def aggregate_cv_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from all CV folds.
    Args:
        fold_results: List of results from each fold
    Returns:
        Dictionary with aggregated metrics
    """
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
        else:
            # For non-numeric metrics, just store all values
            aggregated[f"{metric_name}_all"] = [fold[metric_name] for fold in all_metrics]
    
    return aggregated
