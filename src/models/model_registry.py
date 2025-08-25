from typing import Dict, Any, Optional, Callable, Union
import torch
import torch.nn as nn
import pytorch_lightning as lightning
from tensorflow import keras
import tensorflow as tf

from .process_transformer import (
    get_next_activity_model as tf_next_activity,
    get_next_time_model as tf_next_time,
    get_remaining_time_model as tf_remaining_time
)
from .mtlformer import (
    get_next_activity_model as mtl_next_activity,
    get_next_time_model as mtl_next_time,
    get_remaining_time_model as mtl_remaining_time,
    get_predict_model as mtl_predict_model
)


class ModelRegistry:
    """This registry manages transformer models for PBPM benchmarking, supporting both TensorFlow and PyTorch."""
    def __init__(self):
        self._models = {}
        self._model_factories = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register transformer models for benchmarking."""
        
        # Process Transformer Model (TensorFlow)
        self.register_model(
            name="process_transformer",
            framework="tensorflow", 
            factory=self._create_process_transformer,
            description="Process Transformer for PBPM tasks (TensorFlow)"
        )
        
        # MTLFormer Model (TensorFlow) - Single Task Models
        self.register_model(
            name="mtlformer",
            framework="tensorflow", 
            factory=self._create_mtlformer,
            description="Multi-Task Learning Transformer for PBPM tasks (TensorFlow) - Single Task Models"
        )
        
        # MTLFormer Multi-Task Model (TensorFlow) - True Multi-Task Learning
        self.register_model(
            name="mtlformer_multi",
            framework="tensorflow", 
            factory=self._create_mtlformer_multi,
            description="True Multi-Task Learning Transformer with shared parameters (TensorFlow)"
        )
        
        # Placeholder for future models
        # self.register_model(
        #     name="model_2",
        #     framework="tensorflow", 
        #     factory=self._create_model_2,
        #     description="Second TensorFlow model"
        # )
        # 
        # self.register_model(
        #     name="model_3",
        #     framework="pytorch", 
        #     factory=self._create_model_3,
        #     description="First PyTorch model"
        # )
        # 
        # self.register_model(
        #     name="model_4",
        #     framework="pytorch", 
        #     factory=self._create_model_4,
        #     description="Second PyTorch model"
        # )
    
    def register_model(self, name: str, framework: str, factory: Callable, 
                      description: str = ""):
        """Register a new model factory."""
        self._model_factories[name] = {
            "framework": framework,
            "factory": factory,
            "description": description
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their metadata."""
        return self._model_factories.copy()
    
    def create_model(self, name: str, task: str, **kwargs) -> Union[lightning.LightningModule, keras.Model]:
        """Create a model instance by name and task."""
        if name not in self._model_factories:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._model_factories.keys())}")
        
        factory_info = self._model_factories[name]
        factory = factory_info["factory"]
        
        return factory(task=task, **kwargs)
    
    def _create_process_transformer(self, task: str, **kwargs) -> keras.Model:
        """Create Process Transformer model for PBPM tasks."""
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        
        try:
            if task == "next_activity":
                return tf_next_activity(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "next_time":
                return tf_next_time(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=1,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "remaining_time":
                return tf_remaining_time(
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
    
    def _create_mtlformer(self, task: str, **kwargs) -> keras.Model:
        """Create MTLFormer model for PBPM tasks."""
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        
        try:
            if task == "next_activity":
                return mtl_next_activity(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "next_time":
                return mtl_next_time(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=1,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            elif task == "remaining_time":
                return mtl_remaining_time(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=1,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
            else:
                raise NotImplementedError(f"Task '{task}' not implemented for MTLFormer")
        except TypeError as e:
            if "training" in str(e):
                # This is expected - TensorFlow models need training context
                print(f"Warning: TensorFlow model creation requires training context. Error: {e}")
                return None
            else:
                raise e
    
    def _create_mtlformer_multi(self, task: str, **kwargs) -> keras.Model:
        """Create true multi-task learning MTLFormer model."""
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        
        try:
            # The multi-task model handles all tasks simultaneously
            # Task parameter is ignored for this model type
            return mtl_predict_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size,
                output_dim=vocab_size,  # For next activity classification
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim
            )
        except TypeError as e:
            if "training" in str(e):
                # This is expected - TensorFlow models need training context
                print(f"Warning: TensorFlow model creation requires training context. Error: {e}")
                return None
            else:
                raise e
    
    # Additional transformer models will be added here as they are implemented
    # Example PyTorch model factory:
    # def _create_model_3(self, task: str, **kwargs) -> lightning.LightningModule:
    #     """Create PyTorch Lightning model."""
    #     # Implementation for PyTorch model
    #     pass


# Global registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return model_registry


def register_custom_model(name: str, framework: str, factory: Callable, description: str = ""):
    """Register a custom model in the global registry."""
    model_registry.register_model(name, framework, factory, description)


def create_model(name: str, task: str, **kwargs) -> Union[lightning.LightningModule, keras.Model]:
    """Create a model using the global registry."""
    return model_registry.create_model(name, task, **kwargs)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models in the registry."""
    return model_registry.get_available_models()

# Additional transformer variants can be added here as needed
# Example:
# @register_custom_model("transformer_attention")
# def create_transformer_attention_model(task: str, vocab_size: int, **kwargs) -> Any:
#     """Transformer with enhanced attention mechanisms."""
#     # Implementation would go here
#     pass
