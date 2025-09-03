import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import pytorch_lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn import metrics

from ..data.encoders import Vocabulary
from ..data.preprocessor import SimplePreprocessor
from ..data.loader import CanonicalLogsDataLoader
from ..metrics import accuracy_score, f1_score, classification_report_metrics
from ..utils.logging import create_logger
from ..utils.cross_validation import run_canonical_cross_validation


class NextActivityTask:
    """
    Task implementation for next activity prediction in process mining.
    
    This task predicts the next activity in a process trace given a prefix
    of activities that have already occurred.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = None
        self.model = None
        self.results = {}
        self.logger = None  # Will be initialized in run method
        self.current_dataset = None
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and prepare data for next activity prediction using canonical pipeline."""
        # First ensure data is processed using canonical format
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            processing_info = preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # Now use the canonical loader
        data_loader = CanonicalLogsDataLoader(dataset_name, str(processed_directory))
        
        # Load all data for the task (we'll split by folds later)
        # For now, we'll load fold 0 to get the structure
        train_df, val_df = data_loader.load_fold_data("next_activity", 0)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Get metadata
        metadata = {
            "x_word_dict": data_loader.x_word_dict,
            "y_word_dict": data_loader.y_word_dict,
            "vocab_size": data_loader.vocab_size,
            "num_activities": data_loader.num_activities
        }
        
        return combined_df, metadata
    
    def build_vocabulary(self, all_activity_tokens: List[str]) -> Vocabulary:
        """Build vocabulary from all activity tokens across datasets."""
        all_activity_tokens.append(self.config["data"]["end_of_case_token"])
        return Vocabulary(all_activity_tokens)
    
    def merge_vocabularies(self, vocabularies: List[Vocabulary]) -> Vocabulary:
        """Merge multiple vocabularies into a single vocabulary."""
        all_tokens = set()
        for vocab in vocabularies:
            # Add all tokens from each vocabulary
            all_tokens.update(vocab.index_to_token)
        
        # Create a new vocabulary with all unique tokens
        return Vocabulary(list(all_tokens))
    
    def create_model(self, vocab_size: int, max_case_length: int = None, output_dim: int = None):
        """Create the next activity prediction model using transformer models."""
        from ..models.model_registry import create_model
        if max_case_length is None:
            max_case_length = self.config["model"].get("max_case_length", 50)
        # Resolve per-model hyperparameters with fallback to global model.*
        _m = self.config.get("model", {})
        _name = _m.get("name", "process_transformer")
        _pm = (_m.get("per_model", {}) or {}).get(_name, {})
        embed_dim = _pm.get("embed_dim", _m.get("embed_dim", 36))
        num_heads = _pm.get("num_heads", _m.get("num_heads", 4))
        ff_dim = _pm.get("ff_dim", _m.get("ff_dim", 64))
        # Build kwargs and include guards info for LSTM variants
        kwargs = dict(
            name=_name,
            task="next_activity",
            vocab_size=vocab_size,
            max_case_length=max_case_length,
            output_dim=output_dim,  # pass correct output dimension
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            attribute_mode=self.config.get("data", {}).get("attribute_mode", "minimal"),
        )
        ds = self.config.get("data", {}).get("datasets")
        if isinstance(ds, list) and len(ds) == 1:
            kwargs["dataset_name"] = ds[0]
        return create_model(**kwargs)
    
    def create_trainer(self, model, checkpoint_dir: Path = None):
        """Create training setup for both TensorFlow and PyTorch models."""
        if isinstance(model, keras.Model):
            # TensorFlow model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["train"]["learning_rate"]),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            # Attach TF early stopping params via config (used in fit callbacks)
            self._tf_early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.config.get("train", {}).get("early_stopping_monitor", "val_loss"),
                patience=self.config.get("train", {}).get("early_stopping_patience", None) or 0,
                min_delta=self.config.get("train", {}).get("early_stopping_min_delta", 0.0),
                mode=self.config.get("train", {}).get("early_stopping_mode", "min"),
                restore_best_weights=self.config.get("train", {}).get("restore_best_weights", True)
            ) if self.config.get("train", {}).get("early_stopping_patience", None) is not None else None
            return model
        else:
            # PyTorch Lightning model
            callbacks = []
            
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                monitor = self.config["train"].get("early_stopping_monitor", 'val_loss')
                mode = self.config["train"].get("early_stopping_mode", 'min')
                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename='model-{epoch:02d}-{'+monitor+':.2f}',
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    save_last=True
                )
                callbacks.append(checkpoint_callback)
            
            early_stopping = EarlyStopping(
                monitor=self.config["train"].get("early_stopping_monitor", 'val_loss'),
                patience=self.config["train"].get("early_stopping_patience", 5),
                mode=self.config["train"].get("early_stopping_mode", 'min'),
                min_delta=self.config["train"].get("early_stopping_min_delta", 0.0)
            )
            callbacks.append(early_stopping)
            
            trainer = lightning.Trainer(
                max_epochs=self.config["train"]["max_epochs"],
                accelerator=self.config["train"]["accelerator"],
                devices=self.config["train"]["devices"],
                callbacks=callbacks,
                enable_progress_bar=True,
                log_every_n_steps=10
            )
            
            return trainer
    
    def train_and_evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_idx: int) -> Dict[str, Any]:
        """
        Train and evaluate model on a specific fold.
        
        Args:
            train_df: Training data for this fold
            val_df: Validation data for this fold
            fold_idx: Fold index
            
        Returns:
            Dictionary with training and evaluation results
        """
        print(f"Training fold {fold_idx + 1}...")
        
        # Create checkpoint directory
        checkpoint_dir = Path("outputs/checkpoints") / self.current_dataset / f"fold_{fold_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metrics output dir
        model_name = self.config.get("model", {}).get("name", "unknown_model")
        metrics_dir = Path("outputs") / self.current_dataset / "next_activity" / model_name / f"fold_{fold_idx}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vocabulary from metadata
        from ..data.loader import CanonicalLogsDataLoader
        loader = CanonicalLogsDataLoader(self.current_dataset, str(Path(self.config["data"]["path_processed"])))
        x_word_dict = loader.x_word_dict
        y_word_dict = loader.y_word_dict
        
        # Create vocabulary
        self.vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create model
        vocab_size = len(self.vocabulary.index_to_token)
        
        # Get max_case_length from metadata and allow trimming via config
        observed_max = loader.get_max_case_length(train_df)
        cfg_data_max = self.config.get("data", {}).get("max_prefix_length")
        cfg_model_max = self.config.get("model", {}).get("max_case_length")
        # pick the minimum positive integer among observed and configured limits
        limits = [v for v in [observed_max, cfg_data_max, cfg_model_max] if v is not None]
        max_case_length = min(limits) if limits else observed_max
        
        output_dim = len(y_word_dict)
        
        model = self.create_model(vocab_size, max_case_length, output_dim)
        
        # Create trainer
        trainer = self.create_trainer(model, checkpoint_dir)
        
        # Train model
        import time, json as _json
        start_time = time.time()
        if isinstance(trainer, keras.Model):
            # TensorFlow training
            # Prepare data
            x_train, y_train = self._prepare_tensorflow_data(train_df, max_case_length)
            x_val, y_val = self._prepare_tensorflow_data(val_df, max_case_length)
            
            # Train
            callbacks = []
            if getattr(self, "_tf_early_stopping", None) is not None:
                callbacks.append(self._tf_early_stopping)
            history = trainer.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=self.config["train"]["max_epochs"],
                batch_size=self.config["train"]["batch_size"],
                callbacks=callbacks if callbacks else None,
                verbose=1
            )
            train_time = time.time() - start_time
            
            # Evaluate
            eval_start = time.time()
            val_loss, val_accuracy = trainer.evaluate(x_val, y_val, verbose=0)
            y_pred = trainer.predict(x_val, verbose=0)
            infer_time = time.time() - eval_start
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred_classes)
            f1 = f1_score(y_val, y_pred_classes, average='weighted')
            param_count = int(trainer.count_params())
            
            # Early-stopping transparency (TF/Keras)
            es_cfg = self.config.get("train", {})
            monitor = es_cfg.get("early_stopping_monitor", "val_loss")
            mode = es_cfg.get("early_stopping_mode", "min")
            min_delta = float(es_cfg.get("early_stopping_min_delta", 0.0))
            patience = es_cfg.get("early_stopping_patience", None)

            # History-based computations
            history_dict = getattr(history, 'history', {}) or {}
            # pick monitored series if available; fallback to common keys
            series_key = monitor if monitor in history_dict else (
                'val_loss' if 'val_loss' in history_dict else (
                'val_accuracy' if 'val_accuracy' in history_dict else (next(iter(history_dict.keys())) if history_dict else None)))
            series = history_dict.get(series_key, []) if series_key else []
            epochs_run = int(len(series)) if series else int(len(history_dict.get('loss', [])))

            # Determine best epoch index (0-based) from series
            best_epoch_idx = None
            if series:
                if mode == 'min':
                    best_epoch_idx = int(np.argmin(series))
                else:
                    best_epoch_idx = int(np.argmax(series))
            # Keras EarlyStopping has attribute stopped_epoch (>0 when stopped)
            early_stopped = False
            if getattr(self, "_tf_early_stopping", None) is not None:
                try:
                    stopped_epoch = int(getattr(self._tf_early_stopping, 'stopped_epoch', 0) or 0)
                except Exception:
                    stopped_epoch = 0
                early_stopped = stopped_epoch > 0 or (epochs_run < int(self.config['train']['max_epochs']))

            metrics = {
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'train_time_sec': float(train_time),
                'infer_time_sec': float(infer_time),
                'param_count': param_count,
                # transparency fields
                'best_epoch': (None if best_epoch_idx is None else int(best_epoch_idx + 1)),
                'epochs_run': int(epochs_run),
                'early_stopped': bool(early_stopped),
                'early_stopping': {
                    'monitor': str(monitor),
                    'mode': str(mode),
                    'min_delta': float(min_delta),
                    'patience': (None if patience is None else int(patience))
                }
            }
            
        else:
            # PyTorch Lightning training
            from ..training.datamodule import CanonicalNextActivityDataModule
            
            # Create data module
            datamodule = CanonicalNextActivityDataModule(
                dataset_name=self.current_dataset,
                task="next_activity",
                fold_idx=fold_idx,
                processed_dir=str(Path(self.config["data"]["path_processed"])),
                batch_size=self.config["train"]["batch_size"]
            )
            
            # Train
            trainer.fit(model, datamodule)
            train_time = time.time() - start_time
            
            # Evaluate
            eval_start = time.time()
            results = trainer.validate(model, datamodule, ckpt_path='best')
            infer_time = time.time() - eval_start
            
            # Extract metrics
            res0 = results[0] if isinstance(results, list) and results else {}
            val_loss = float(res0.get('val_loss', 0.0))
            val_acc = float(res0.get('val_accuracy', 0.0))
            f1v = float(res0.get('val_f1', 0.0))
            # parameter count if available
            try:
                param_count = int(sum(p.numel() for p in model.parameters()))
            except Exception:
                param_count = 0
            # Early-stopping transparency (PyTorch Lightning)
            es_cfg = self.config.get('train', {})
            monitor = es_cfg.get('early_stopping_monitor', 'val_loss')
            mode = es_cfg.get('early_stopping_mode', 'min')
            min_delta = float(es_cfg.get('early_stopping_min_delta', 0.0))
            patience = es_cfg.get('early_stopping_patience', None)

            # Compute epochs_run
            try:
                epochs_run = int(getattr(trainer, 'current_epoch', 0)) + 1
            except Exception:
                epochs_run = None

            # Determine early_stopped
            early_stopped = None
            try:
                early_stopped = bool(getattr(trainer, 'should_stop', False)) or (
                    epochs_run is not None and epochs_run < int(self.config['train']['max_epochs'])
                )
            except Exception:
                early_stopped = epochs_run is not None and epochs_run < int(self.config['train']['max_epochs'])

            # Try to infer best_epoch from checkpoint callback best_model_path
            best_epoch = None
            try:
                for cb in getattr(trainer, 'callbacks', []) or []:
                    if hasattr(cb, 'best_model_path') and cb.best_model_path:
                        # filename pattern: model-{epoch:02d}-{val_loss:.2f}
                        import re
                        m = re.search(r"model-(\d+)-", cb.best_model_path)
                        if m:
                            best_epoch = int(m.group(1))
                            break
            except Exception:
                best_epoch = None

            metrics = {
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'accuracy': val_acc,  # For compatibility
                'f1_score': f1v,
                'train_time_sec': float(train_time),
                'infer_time_sec': float(infer_time),
                'param_count': param_count,
                # transparency fields
                'best_epoch': best_epoch,
                'epochs_run': epochs_run,
                'early_stopped': bool(early_stopped) if early_stopped is not None else None,
                'early_stopping': {
                    'monitor': str(monitor),
                    'mode': str(mode),
                    'min_delta': float(min_delta),
                    'patience': (None if patience is None else int(patience))
                }
            }
        
        # Persist metrics.json
        try:
            with open(metrics_dir / "metrics.json", 'w') as f:
                _json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write metrics.json: {e}")
        
        return {
            'fold_idx': fold_idx,
            'metrics': metrics,
            'checkpoint_dir': str(checkpoint_dir)
        }
    
    def _prepare_tensorflow_data(self, df: pd.DataFrame, max_case_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for TensorFlow training."""
        # This is a simplified version - in practice, you'd use the canonical loader
        x = df["prefix"].values
        y = df["next_act"].values
        
        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.vocabulary.token_to_index[s] for s in _x.split()])
        
        # Pad sequences to fixed length
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')
        
        # Tokenize labels using the same vocabulary as training
        from ..data.loader import CanonicalLogsDataLoader
        loader = CanonicalLogsDataLoader(self.current_dataset, str(Path(self.config["data"]["path_processed"])))
        y_word_dict = loader.y_word_dict
        token_y = np.array([y_word_dict[act] for act in y])
        
        return token_x, token_y
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """
        Run next activity prediction task on multiple datasets.
        
        Args:
            datasets: List of dataset names to process
            raw_directory: Directory containing raw data
            outputs_dir: Directory to save outputs
            
        Returns:
            Dictionary with results for all datasets
        """
        # Initialize logger
        self.logger = create_logger("next_activity_task", outputs_dir)
        
        # Set up processed directory
        processed_directory = Path(self.config["data"]["path_processed"])
        
        # Check if force preprocessing is requested
        force_reprocess = self.config.get("force_preprocess", False)
        
        results = {}
        
        for dataset_name in datasets:
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            self.current_dataset = dataset_name
            
            try:
                # Load and prepare data
                combined_df, metadata = self.prepare_data(
                    dataset_name, raw_directory, processed_directory, force_reprocess
                )
                
                print(f"Total samples: {len(combined_df)}")
                print(f"Vocabulary size: {metadata['vocab_size']}")
                print(f"Number of activities: {metadata['num_activities']}")
                
                # Create vocabulary
                self.vocabulary = Vocabulary(list(metadata["x_word_dict"].keys()))
                
                # Run canonical cross-validation
                cv_results = run_canonical_cross_validation(
                    task_class=self.__class__,
                    config=self.config,
                    df=combined_df,
                    dataset_name=dataset_name,
                    processed_dir=processed_directory,
                    task_name="next_activity"
                )
                
                # Store results
                results[dataset_name] = cv_results
                
                # Print summary
                print(f"\n=== Cross-Validation Results for {dataset_name} ===")
                for metric, value in cv_results['cv_summary'].items():
                    if metric.endswith('_mean'):
                        print(f"{metric}: {value:.4f}")
                
                # Save detailed results into model-scoped directory to avoid overwrites across models
                model_name = self.config.get("model", {}).get("name", "unknown_model")
                model_dir = outputs_dir / dataset_name / "next_activity" / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                results_file = model_dir / "cv_results.json"
                with open(results_file, 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if self.logger:
                    print(f"Error processing {dataset_name}: {e}")
                continue
        
        return results
