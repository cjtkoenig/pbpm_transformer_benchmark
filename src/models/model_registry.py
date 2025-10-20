"""
Model registry for PBPM transformer models with adapter support.
Supports both TensorFlow and PyTorch models with standardized data interfaces.
"""

from tensorflow import keras
import pytorch_lightning as lightning
from typing import Dict, Any, Union, Callable

from .process_transformer import (
    get_next_activity_model, get_next_time_model, get_remaining_time_model
)
from .mtlformer import get_predict_model as get_mtlformer_model


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


class SharedLSTMAdapter(ModelAdapter):
    """Adapter for Shared LSTM in extended attribute mode.
    
    Responsibilities:
    - Enforce extended attribute inputs (activities + resources + delta_t) and map them to tensors.
    - Delegate tokenization, padding and dtype casting to CanonicalLogsDataLoader to keep content canonical.
    - Return ([ac_x, rl_x, t_x], y) suitable for Keras .fit()/.evaluate()/.predict().
    
    Expected input (data dict):
    - df: pandas DataFrame from a canonical split CSV (must include resource_prefix and delta_t_prefix)
    - loader: CanonicalLogsDataLoader for the same dataset (provides vocabularies)
    - max_case_length: int (sequence length T)
    - shuffle: bool
    """
    def __init__(self, task: str):
        super().__init__("shared_lstm", task)
    
    def adapt_input(self, data: Dict[str, Any]):
        df = data.get('df')
        loader = data.get('loader')
        max_case_length = int(data.get('max_case_length', 50))
        if df is None or loader is None:
            raise ValueError("SharedLSTMAdapter requires 'df' and 'loader' in data.")
        return loader.prepare_next_activity_extended_data(df, max_case_length, shuffle=data.get('shuffle', True))
    
    def adapt_output(self, model_output: Any) -> Any:
        # The model already outputs per-class probabilities (softmax), passthrough is fine.
        return model_output
    
    def get_model_config(self, **kwargs) -> Dict[str, Any]:
        return {
            'max_case_length': kwargs.get('max_case_length', 50),
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
        """Create Process Transformer model for PBPM tasks.
        Model expects canonical minimal inputs as per adapter.
        """
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
    
    
    def _create_mtlformer(self, task: str, **kwargs) -> keras.Model:
        """Create MTLFormer model.
        In this benchmark setup, MTLFormer is only allowed for task='multitask'.
        Attribute mode removed; model expects canonical minimal inputs as per adapter.
        """
        if task != "multitask":
            raise NotImplementedError("MTLFormer in this benchmark only supports task='multitask'. Run the MultiTaskLearningTask and compare per-task reports.")
        max_case_length = kwargs.get("max_case_length", 50)
        vocab_size = kwargs.get("vocab_size", 1000)
        embed_dim = kwargs.get("embed_dim", 36)
        num_heads = kwargs.get("num_heads", 4)
        ff_dim = kwargs.get("ff_dim", 64)
        output_dim = kwargs.get("output_dim", vocab_size)
        base_model = get_mtlformer_model(
            max_case_length=max_case_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        return base_model
    
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
        """Get a list of supported tasks for a model."""
        if model_name == "process_transformer":
            return ["next_activity", "next_time", "remaining_time"]
        elif model_name == "mtlformer":
            return ["multitask"]
        elif model_name == "specialised_lstm":
            return ["next_activity"]
        elif model_name == "shared_lstm":
            return ["next_activity"]
        elif model_name == "activity_only_lstm":
            return ["next_activity"]
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


def activity_only_lstm_factory(task: str, **kwargs):
    """Factory for activity_only_lstm (supports only next_activity; activities-only inputs)."""
    if task != "next_activity":
        raise NotImplementedError("activity_only_lstm supports only task='next_activity'")
    from .activity_only_lstm import get_next_activity_model as get_wick
    return get_wick(
        max_case_length=kwargs.get("max_case_length", 50),
        vocab_size=kwargs.get("vocab_size", 1000),
        output_dim=kwargs.get("output_dim", kwargs.get("vocab_size", 1000)),
        embed_dim=kwargs.get("embed_dim", 64),
        lstm_size_alpha=kwargs.get("lstm_size_alpha", 50),
        lstm_size_beta=kwargs.get("lstm_size_beta", 50),
        dropout_input=kwargs.get("dropout_input", 0.15),
        dropout_context=kwargs.get("dropout_context", 0.15),
        l2reg=kwargs.get("l2reg", 1e-4),
    )


def shared_lstm_factory(task: str, **kwargs):
    """Factory for shared_lstm (next_activity, expects extended inputs from adapter).
    Requires dataset metadata to obtain resource vocab size if not provided.
    """
    if task != "next_activity":
        raise NotImplementedError("shared_lstm supports only task='next_activity'")
    dataset_name = kwargs.get("dataset_name", None)
    rl_vocab_size = kwargs.get("rl_vocab_size", None)
    if rl_vocab_size is None:
        if dataset_name is None:
            raise ValueError("shared_lstm_factory requires 'dataset_name' to read resource vocab size from metadata, or pass rl_vocab_size explicitly")
        # Lazy import to avoid circular deps
        try:
            from ..data.loader import CanonicalLogsDataLoader
        except Exception as e:
            from src.data.loader import CanonicalLogsDataLoader  # fallback absolute import
        loader = CanonicalLogsDataLoader(dataset_name, str(kwargs.get("processed_dir", "data/processed")), allow_runtime_resource_vocab=True)
        rl_vocab_size = int(getattr(loader, 'rl_vocab_size', 0) or 0)
        if rl_vocab_size <= 0:
            raise ValueError(f"Resource vocabulary not found for dataset '{dataset_name}'. Ensure preprocessing captured resource attributes.")
    # Build model
    from .shared_lstm import get_next_activity_model as get_shared
    return get_shared(
        max_case_length=kwargs.get("max_case_length", 50),
        ac_vocab_size=kwargs.get("vocab_size", 1000),
        rl_vocab_size=rl_vocab_size,
        output_dim=kwargs.get("output_dim", kwargs.get("vocab_size", 1000)),
        embed_dim_ac=kwargs.get("embed_dim", 64),
        embed_dim_rl=kwargs.get("embed_dim", 64),
        lstm_size_alpha=kwargs.get("lstm_size_alpha", 50),
        lstm_size_beta=kwargs.get("lstm_size_beta", 50),
        dropout_input=kwargs.get("dropout_input", 0.15),
        dropout_context=kwargs.get("dropout_context", 0.15),
        l2reg=kwargs.get("l2reg", 1e-4),
    )


model_registry.register_model(
    name="activity_only_lstm",
    framework="tensorflow",
    factory=activity_only_lstm_factory,
    adapter=ProcessTransformerAdapter,
)

model_registry.register_model(
    name="shared_lstm",
    framework="tensorflow",
    factory=shared_lstm_factory,
    adapter=SharedLSTMAdapter,
)



class SpecialisedLSTMAdapter(ModelAdapter):
    """Adapter for Specialised LSTM in extended attribute mode.

    Identical contract to SharedLSTMAdapter; we keep a separate class for clearer logging and future tweaks.
    Expects CanonicalLogsDataLoader.prepare_next_activity_extended_data to provide ([ac_x, rl_x, t_x], y).
    """
    def __init__(self, task: str):
        super().__init__("specialised_lstm", task)

    def adapt_input(self, data: Dict[str, Any]):
        df = data.get('df')
        loader = data.get('loader')
        max_case_length = int(data.get('max_case_length', 50))
        if df is None or loader is None:
            raise ValueError("SpecialisedLSTMAdapter requires 'df' and 'loader' in data.")
        return loader.prepare_next_activity_extended_data(df, max_case_length, shuffle=data.get('shuffle', True))

    def adapt_output(self, model_output: Any) -> Any:
        return model_output

    def get_model_config(self, **kwargs) -> Dict[str, Any]:
        return {
            'max_case_length': kwargs.get('max_case_length', 50),
        }


def specialised_lstm_factory(task: str, **kwargs):
    """Factory for specialised_lstm (next_activity, expects extended inputs from adapter).
    Requires dataset metadata to obtain resource vocab size if not provided.
    """
    if task != "next_activity":
        raise NotImplementedError("specialised_lstm supports only task='next_activity'")
    dataset_name = kwargs.get("dataset_name", None)
    rl_vocab_size = kwargs.get("rl_vocab_size", None)
    if rl_vocab_size is None:
        if dataset_name is None:
            raise ValueError("specialised_lstm_factory requires 'dataset_name' to read resource vocab size from metadata, or pass rl_vocab_size explicitly")
        # Lazy import to avoid circular deps
        try:
            from ..data.loader import CanonicalLogsDataLoader
        except Exception:
            from src.data.loader import CanonicalLogsDataLoader  # fallback absolute import
        loader = CanonicalLogsDataLoader(dataset_name, str(kwargs.get("processed_dir", "data/processed")), allow_runtime_resource_vocab=True)
        rl_vocab_size = int(getattr(loader, 'rl_vocab_size', 0) or 0)
        if rl_vocab_size <= 0:
            raise ValueError(f"Resource vocabulary not found for dataset '{dataset_name}'. Ensure preprocessing captured resource attributes.")
    # Build model
    from .specialised_lstm import get_next_activity_model as get_spec
    return get_spec(
        max_case_length=kwargs.get("max_case_length", 50),
        ac_vocab_size=kwargs.get("vocab_size", 1000),
        rl_vocab_size=rl_vocab_size,
        output_dim=kwargs.get("output_dim", kwargs.get("vocab_size", 1000)),
        embed_dim_ac=kwargs.get("embed_dim", 64),
        embed_dim_rl=kwargs.get("embed_dim", 64),
        lstm_size_alpha=kwargs.get("lstm_size_alpha", 50),
        lstm_size_beta=kwargs.get("lstm_size_beta", 50),
        dropout_input=kwargs.get("dropout_input", 0.15),
        dropout_context=kwargs.get("dropout_context", 0.15),
        l2reg=kwargs.get("l2reg", 1e-4),
    )


model_registry.register_model(
    name="specialised_lstm",
    framework="tensorflow",
    factory=specialised_lstm_factory,
    adapter=SpecialisedLSTMAdapter,
)
