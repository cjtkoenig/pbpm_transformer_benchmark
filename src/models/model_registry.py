"""
Model registry for PBPM transformer models with adapter support.
Supports both TensorFlow and PyTorch models with standardized data interfaces.
"""

import tensorflow as tf
from tensorflow import keras
import pytorch_lightning as lightning
from typing import Dict, Any, Union, Callable
import torch
import torch.nn as nn

from .process_transformer import (
    get_next_activity_model, get_next_time_model, get_remaining_time_model
)


class ModelAdapter:
    """Base class for model adapters that convert standardized data to model-specific format."""
    
    def __init__(self, model_name: str, task: str):
        self.model_name = model_name
        self.task = task
    
    def adapt_input(self, data: Dict[str, Any]) -> Any:
        """Convert standardized input data to model-specific format."""
        raise NotImplementedError
    
    def adapt_output(self, model_output: Any) -> Any:
        """Convert model-specific output to standardized format."""
        raise NotImplementedError
    
    def get_model_config(self, **kwargs) -> Dict[str, Any]:
        """Get model-specific configuration."""
        raise NotImplementedError


class ProcessTransformerAdapter(ModelAdapter):
    """Adapter for ProcessTransformer models."""
    
    def __init__(self, task: str):
        super().__init__("process_transformer", task)
    
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt input for ProcessTransformer models."""
        # ProcessTransformer expects tokenized sequences directly
        return {
            'input_sequences': data['token_x'],
            'time_features': data.get('time_x', None),
            'max_case_length': data.get('max_case_length', 50)
        }
    
    def adapt_output(self, model_output: Any) -> Any:
        """Adapt output for ProcessTransformer models."""
        # ProcessTransformer outputs logits directly
        return model_output
    
    def get_model_config(self, **kwargs) -> Dict[str, Any]:
        """Get ProcessTransformer configuration."""
        return {
            'max_case_length': kwargs.get('max_case_length', 50),
            'vocab_size': kwargs.get('vocab_size', 1000),
            'embed_dim': kwargs.get('embed_dim', 36),
            'num_heads': kwargs.get('num_heads', 4),
            'ff_dim': kwargs.get('ff_dim', 64)
        }


class BERTAdapter(ModelAdapter):
    """Adapter for BERT-based process models (example for future models)."""
    
    def __init__(self, task: str):
        super().__init__("bert_process", task)
    
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt input for BERT models."""
        # BERT might expect different tokenization or additional features
        return {
            'input_ids': data['token_x'],
            'attention_mask': data.get('attention_mask', None),
            'token_type_ids': data.get('token_type_ids', None),
            'time_features': data.get('time_x', None)
        }
    
    def adapt_output(self, model_output: Any) -> Any:
        """Adapt output for BERT models."""
        # BERT might output hidden states that need to be processed
        return model_output
    
    def get_model_config(self, **kwargs) -> Dict[str, Any]:
        """Get BERT configuration."""
        return {
            'max_case_length': kwargs.get('max_case_length', 50),
            'vocab_size': kwargs.get('vocab_size', 1000),
            'hidden_size': kwargs.get('hidden_size', 768),
            'num_attention_heads': kwargs.get('num_attention_heads', 12),
            'intermediate_size': kwargs.get('intermediate_size', 3072)
        }


class ModelRegistry:
    """This registry manages transformer models for PBPM benchmarking, supporting both TensorFlow and PyTorch."""
    
    def __init__(self):
        self.factories = {
            "process_transformer": {
                "framework": "tensorflow",
                "factory": self._create_process_transformer,
                "adapter": ProcessTransformerAdapter
            },
            "bert_process": {
                "framework": "pytorch", 
                "factory": self._create_bert_model,
                "adapter": BERTAdapter
            }
        }
    
    def register_model(self, name: str, framework: str, factory: Callable, adapter: ModelAdapter):
        """Register a new model with its factory and adapter."""
        self.factories[name] = {
            "framework": framework,
            "factory": factory,
            "adapter": adapter
        }
    
    def create_model(self, name: str, task: str, **kwargs) -> Union[lightning.LightningModule, keras.Model]:
        """Create a model with the specified name and task."""
        if name not in self.factories:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self.factories.keys())}")
        
        factory_info = self.factories[name]
        factory = factory_info["factory"]
        
        return factory(task=task, **kwargs)
    
    def get_adapter(self, name: str, task: str) -> ModelAdapter:
        """Get the adapter for a specific model and task."""
        if name not in self.factories:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self.factories.keys())}")
        
        adapter_class = self.factories[name]["adapter"]
        return adapter_class(task)
    
    def _create_process_transformer(self, task: str, **kwargs) -> keras.Model:
        """Create Process Transformer model for PBPM tasks."""
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        
        try:
            if task == "next_activity":
                return get_next_activity_model(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "next_time":
                return get_next_time_model(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=1,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "remaining_time":
                return get_remaining_time_model(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=1,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            else:
                raise NotImplementedError(f"Task '{task}' not implemented for Process Transformer")
        except TypeError as e:
            if "training" in str(e):
                # This is expected - TensorFlow models need training context
                print(f"Warning: TensorFlow model creation requires training context. Error: {e}")
                return None
            else:
                raise e
    
    def _create_bert_model(self, task: str, **kwargs) -> lightning.LightningModule:
        """Create BERT-based model for PBPM tasks (placeholder for future implementation)."""
        # This is a placeholder for future BERT model implementation
        raise NotImplementedError("BERT model not yet implemented")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            name: {
                "framework": info["framework"],
                "supported_tasks": self._get_supported_tasks(name)
            }
            for name, info in self.factories.items()
        }
    
    def _get_supported_tasks(self, model_name: str) -> list:
        """Get list of supported tasks for a model."""
        if model_name == "process_transformer":
            return ["next_activity", "next_time", "remaining_time"]
        elif model_name == "bert_process":
            return ["next_activity", "next_time", "remaining_time"]
        else:
            return []


# Global registry instance
model_registry = ModelRegistry()


def create_model(name: str, task: str, **kwargs) -> Union[lightning.LightningModule, keras.Model]:
    """Convenience function to create a model."""
    return model_registry.create_model(name, task, **kwargs)


def get_adapter(name: str, task: str) -> ModelAdapter:
    """Convenience function to get a model adapter."""
    return model_registry.get_adapter(name, task)
