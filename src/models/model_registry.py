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
from .mtlformer import get_predict_model as get_mtlformer_model
from .mtlformer import TokenAndPositionEmbedding, TransformerBlock


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


# PLACEHOLDER FOR FUTURE MODELS
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
            "mtlformer": {
                "framework": "tensorflow",
                "factory": self._create_mtlformer,
                "adapter": ProcessTransformerAdapter  # will override via get_adapter for custom handling if needed
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
                # vocab_size includes special tokens like [PAD], [UNK], [EOC]
                # but we only want to predict actual activities
                output_dim = kwargs.get("output_dim", vocab_size)
                return get_next_activity_model(
                    max_case_length=max_case_length,
                    vocab_size=vocab_size,
                    output_dim=output_dim,
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
    
    def _create_mtlformer(self, task: str, **kwargs) -> keras.Model:
        """Create MTLFormer model.
        If task is one of single-task names, wrap the multi-output model to expose only the relevant head.
        If task == 'multitask', return the full multi-output model.
        """
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        # Determine output_dim for next_activity
        output_dim = kwargs.get("output_dim", vocab_size)
        base_model = get_mtlformer_model(
            max_case_length=max_case_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        if task in ("next_activity", "next_time", "remaining_time"):
            # Build minimal single-task graphs to avoid unused-head variables.
            # Shared config
            embed_dim_local = embed_dim
            num_heads_local = num_heads
            ff_dim_local = ff_dim

            if task == "next_activity":
                # Inputs: tokens only (tasks expect single input)
                tokens = keras.Input(shape=(max_case_length,), name="tokens")
                # Internally duplicate tokens and use zeros time (as before)
                tokens2 = tokens
                zeros_time = keras.layers.Lambda(
                    lambda t: tf.zeros((tf.shape(t)[0], 3), dtype=tf.float32),
                    name="zeros_time_from_tokens"
                )(tokens)

                # Branch 1 (inputs1)
                x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim_local)(tokens)
                x = TransformerBlock(embed_dim_local, num_heads_local, ff_dim_local)(x)
                x = keras.layers.GlobalAveragePooling1D()(x)
                # Branch 2 (inputs2)
                x1 = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim_local)(tokens2)
                x1 = TransformerBlock(embed_dim_local, num_heads_local, ff_dim_local)(x1)
                x1 = keras.layers.GlobalAveragePooling1D()(x1)
                # Time features branch (shared feature layers only)
                x_t1 = keras.layers.Dense(32, activation="relu")(zeros_time)
                x_2 = keras.layers.Concatenate()([x1, x_t1])
                x_2 = keras.layers.Dropout(0.1)(x_2)
                x_2 = keras.layers.Dense(64, activation="relu", name="next_time", kernel_regularizer='l2')(x_2)
                x_t2 = keras.layers.Dense(32, activation="relu")(zeros_time)
                x_3 = keras.layers.Concatenate()([x1, x_t2])
                x_3 = keras.layers.Dropout(0.1)(x_3)
                x_3 = keras.layers.Dense(64, activation="relu", name="remain_time", kernel_regularizer='l2')(x_3)
                # Next activity features
                x_1 = keras.layers.Dropout(0.1)(x)
                x_1 = keras.layers.Dense(32, activation="relu", name="next_act", kernel_regularizer='l2')(x_1)
                # Shared concat
                shared = keras.layers.Concatenate()([x_1, x_2, x_3])
                # Head out1 only
                out1 = keras.layers.Dropout(0.1)(shared)
                out1 = keras.layers.Dense(128, activation="relu")(out1)
                out1 = keras.layers.Dropout(0.1)(out1)
                out1 = keras.layers.Dense(32, activation="relu")(out1)
                out1 = keras.layers.Dropout(0.1)(out1)
                outputs = keras.layers.Dense(output_dim, activation="linear", name='out1')(out1)
                return keras.Model(inputs=tokens, outputs=outputs, name="mtlformer_next_activity")

            # For time tasks: inputs are [tokens, time]
            tokens = keras.Input(shape=(max_case_length,), name="tokens")
            time_in = keras.Input(shape=(3,), name="time_inputs")
            tokens2 = tokens

            # Branch 1
            x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim_local)(tokens)
            x = TransformerBlock(embed_dim_local, num_heads_local, ff_dim_local)(x)
            x = keras.layers.GlobalAveragePooling1D()(x)
            # Branch 2
            x1 = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim_local)(tokens2)
            x1 = TransformerBlock(embed_dim_local, num_heads_local, ff_dim_local)(x1)
            x1 = keras.layers.GlobalAveragePooling1D()(x1)
            # Time feature branches
            x_t1 = keras.layers.Dense(32, activation="relu")(time_in)
            x_2 = keras.layers.Concatenate()([x1, x_t1])
            x_2 = keras.layers.Dropout(0.1)(x_2)
            x_2 = keras.layers.Dense(64, activation="relu", name="next_time", kernel_regularizer='l2')(x_2)
            x_t2 = keras.layers.Dense(32, activation="relu")(time_in)
            x_3 = keras.layers.Concatenate()([x1, x_t2])
            x_3 = keras.layers.Dropout(0.1)(x_3)
            x_3 = keras.layers.Dense(64, activation="relu", name="remain_time", kernel_regularizer='l2')(x_3)
            # Next activity features
            x_1 = keras.layers.Dropout(0.1)(x)
            x_1 = keras.layers.Dense(32, activation="relu", name="next_act", kernel_regularizer='l2')(x_1)
            shared = keras.layers.Concatenate()([x_1, x_2, x_3])

            if task == "next_time":
                head = keras.layers.Dropout(0.1)(shared)
                head = keras.layers.Dense(128, activation="relu")(head)
                head = keras.layers.Dropout(0.1)(head)
                head = keras.layers.Dense(32, activation="relu")(head)
                head = keras.layers.Dropout(0.1)(head)
                outputs = keras.layers.Dense(1, activation="linear", name="out2")(head)
                return keras.Model(inputs=[tokens, time_in], outputs=outputs, name="mtlformer_next_time")
            else:  # remaining_time
                head = keras.layers.Dropout(0.1)(shared)
                head = keras.layers.Dense(128, activation="relu")(head)
                head = keras.layers.Dropout(0.1)(head)
                head = keras.layers.Dense(32, activation="relu")(head)
                head = keras.layers.Dropout(0.1)(head)
                outputs = keras.layers.Dense(1, activation="linear", name="out3")(head)
                return keras.Model(inputs=[tokens, time_in], outputs=outputs, name="mtlformer_remaining_time")

        elif task == "multitask":
            return base_model
        else:
            raise NotImplementedError(f"Task '{task}' not implemented for MTLFormer")
    
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
        elif model_name == "mtlformer":
            return ["next_activity", "next_time", "remaining_time", "multitask"]
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
