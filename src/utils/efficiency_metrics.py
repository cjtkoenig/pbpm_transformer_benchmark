"""
Efficiency metrics tracking for benchmark evaluation.
Tracks training time, inference time, memory usage, and model size.
"""

import time
import psutil
import os
import torch
import tensorflow as tf
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

class EfficiencyTracker:
    """
    Track efficiency metrics during model training and inference.
    """
    
    def __init__(self):
        self.metrics = {
            'training_time': 0.0,
            'inference_time': 0.0,
            'memory_peak': 0.0,
            'model_size_mb': 0.0,
            'parameters_count': 0,
            'gpu_memory_peak': 0.0
        }
        self.start_time = None
        self.start_memory = None
        
    def start_training_tracking(self):
        """Start tracking training metrics."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('GPU'):
            # TensorFlow GPU tracking
            pass
    
    def end_training_tracking(self, model):
        """End training tracking and calculate metrics."""
        if self.start_time is None:
            return
        
        # Calculate training time
        self.metrics['training_time'] = time.time() - self.start_time
        
        # Calculate memory usage
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.metrics['memory_peak'] = end_memory - self.start_memory
        
        # Calculate model size
        self.metrics['model_size_mb'] = self._calculate_model_size(model)
        self.metrics['parameters_count'] = self._count_parameters(model)
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            self.metrics['gpu_memory_peak'] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        self.start_time = None
        self.start_memory = None
    
    def measure_inference_time(self, model, test_data, num_runs: int = 100):
        """Measure inference time on test data."""
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(test_data)
                
                # Measure inference time
                start_time = time.time()
                for _ in range(num_runs):
                    _ = model(test_data)
                end_time = time.time()
                
                self.metrics['inference_time'] = (end_time - start_time) / num_runs
        
        elif hasattr(model, 'predict'):
            # TensorFlow/Keras model
            # Warmup
            for _ in range(10):
                _ = model.predict(test_data, verbose=0)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(num_runs):
                _ = model.predict(test_data, verbose=0)
            end_time = time.time()
            
            self.metrics['inference_time'] = (end_time - start_time) / num_runs
    
    def _calculate_model_size(self, model) -> float:
        """Calculate model size in MB."""
        if isinstance(model, torch.nn.Module):
            # PyTorch model
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
        
        elif hasattr(model, 'get_weights'):
            # TensorFlow/Keras model
            total_size = 0
            for weight in model.get_weights():
                total_size += weight.nbytes
            
            return total_size / 1024 / 1024
        
        return 0.0
    
    def _count_parameters(self, model) -> int:
        """Count total number of parameters."""
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
        
        elif hasattr(model, 'count_params'):
            return model.count_params()
        
        return 0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all efficiency metrics."""
        return self.metrics.copy()
    
    def save_metrics(self, output_path: Path):
        """Save efficiency metrics to file."""
        import json
        
        metrics_data = {
            'efficiency_metrics': self.metrics,
            'timestamp': time.time()
        }
        
        with open(output_path / 'efficiency_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)

class EfficiencyComparison:
    """
    Compare efficiency metrics across different models.
    """
    
    def __init__(self, model_metrics: Dict[str, Dict[str, float]]):
        """
        Initialize efficiency comparison.
        
        Args:
            model_metrics: Dictionary with model names as keys and efficiency metrics as values
        """
        self.model_metrics = model_metrics
    
    def rank_by_efficiency(self, metric: str = 'inference_time') -> Dict[str, int]:
        """Rank models by a specific efficiency metric."""
        sorted_models = sorted(
            self.model_metrics.items(),
            key=lambda x: x[1].get(metric, float('inf'))
        )
        
        rankings = {}
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            rankings[model_name] = rank
        
        return rankings
    
    def efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive efficiency report."""
        report = {
            'fastest_inference': self.rank_by_efficiency('inference_time'),
            'smallest_model': self.rank_by_efficiency('model_size_mb'),
            'most_memory_efficient': self.rank_by_efficiency('memory_peak'),
            'fastest_training': self.rank_by_efficiency('training_time'),
            'parameter_efficiency': self.rank_by_efficiency('parameters_count')
        }
        
        # Calculate efficiency scores (lower is better)
        efficiency_scores = {}
        for model_name, metrics in self.model_metrics.items():
            # Normalize metrics and calculate composite score
            normalized_metrics = {
                'inference_time': metrics.get('inference_time', 0) / max(m.get('inference_time', 1) for m in self.model_metrics.values()),
                'model_size': metrics.get('model_size_mb', 0) / max(m.get('model_size_mb', 1) for m in self.model_metrics.values()),
                'memory_peak': metrics.get('memory_peak', 0) / max(m.get('memory_peak', 1) for m in self.model_metrics.values())
            }
            
            # Composite efficiency score (weighted average)
            efficiency_scores[model_name] = (
                0.4 * normalized_metrics['inference_time'] +
                0.3 * normalized_metrics['model_size'] +
                0.3 * normalized_metrics['memory_peak']
            )
        
        report['efficiency_scores'] = efficiency_scores
        report['most_efficient_overall'] = min(efficiency_scores, key=efficiency_scores.get)
        
        return report
