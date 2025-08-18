import numpy as np
from typing import List, Dict, Any
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
import torch


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    metric = MeanAbsoluteError()
    return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()


def mean_squared_error(y_true: List[float], y_pred: List[float], 
                      squared: bool = True) -> float:
    """
    Calculate Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

    Args:
        y_true: True values
        y_pred: Predicted values
        squared: If True, return MSE. If False, return RMSE

    Returns:
        MSE or RMSE value
    """
    metric = MeanSquaredError(squared=squared)
    return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate R² score (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score between -∞ and 1 (1 is perfect prediction)
    """
    metric = R2Score()
    return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, MSE, RMSE, and R² scores
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred, squared=True),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred)
    }


def time_prediction_metrics(y_true: List[float], y_pred: List[float],
                           time_unit: str = "hours") -> Dict[str, Any]:
    """
    Specialized metrics for time prediction tasks.
    
    Args:
        y_true: True time values
        y_pred: Predicted time values
        time_unit: Unit of time for reporting
        
    Returns:
        Dictionary with time-specific metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic regression metrics (using torchmetrics)
    metrics = regression_metrics(y_true, y_pred)
    
    # Time-specific metrics
    errors = np.abs(y_true - y_pred)
    relative_errors = errors / (np.abs(y_true) + 1e-8)  # Avoid division by zero
    
    time_metrics = {
        f"mae_{time_unit}": metrics["mae"],
        f"rmse_{time_unit}": metrics["rmse"],
        "mean_relative_error": np.mean(relative_errors),
        "median_relative_error": np.median(relative_errors),
        "max_error": np.max(errors),
        "min_error": np.min(errors),
        "std_error": np.std(errors),
        "r2_score": metrics["r2"]
    }
    
    # Add percentiles
    for percentile in [25, 50, 75, 90, 95]:
        time_metrics[f"error_p{percentile}"] = np.percentile(errors, percentile)
    
    return time_metrics
