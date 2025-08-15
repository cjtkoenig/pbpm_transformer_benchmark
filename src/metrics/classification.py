"""Classification metrics for predictive process monitoring."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import classification_report
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import torch


def accuracy_score(y_true: List[int], y_pred: List[int], 
                  num_classes: Optional[int] = None, 
                  ignore_index: int = 0) -> float:
    """Calculate accuracy score for classification tasks. Returns: Accuracy score between 0 and 1"""
    if num_classes is not None:
        # Use torchmetrics for better handling of multi-class
        metric = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)
        return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()
    else:
        # Use sklearn for simple cases
        return sk_accuracy_score(y_true, y_pred)


def f1_score(y_true: List[int], y_pred: List[int], 
             average: str = 'weighted',
             num_classes: Optional[int] = None,
             ignore_index: int = 0) -> float:
    """Calculate F1 score for classification tasks. Returns: F1 score between 0 and 1"""
    if num_classes is not None:
        # Use torchmetrics for better handling of multi-class
        metric = MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average=average)
        return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()
    else:
        # Use sklearn for simple cases
        return sk_f1_score(y_true, y_pred, average=average, zero_division=0)


def classification_report_metrics(y_true: List[int], y_pred: List[int], 
                                 target_names: Optional[List[str]] = None,
                                 zero_division: int = 0) -> Dict[str, Any]:
    """Generate comprehensive classification report. Returns: Dictionary with precision, recall, f1-score, and support for each class"""
    return classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )


def confusion_matrix_metrics(y_true: List[int], y_pred: List[int], 
                           num_classes: int) -> Dict[str, Any]:
    """Calculate confusion matrix and derived metrics. Returns: Dictionary with confusion matrix and per-class metrics"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[f"class_{i}"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": cm[i, :].sum()
        }
    
    return {
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "overall_accuracy": cm.diagonal().sum() / cm.sum()
    }
