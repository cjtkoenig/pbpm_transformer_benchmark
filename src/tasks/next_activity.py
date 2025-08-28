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
        return create_model(
            name="process_transformer",
            task="next_activity",
            vocab_size=vocab_size,
            max_case_length=max_case_length,
            output_dim=output_dim,  # pass correct output dimension
            embed_dim=self.config["model"].get("embed_dim", 36),
            num_heads=self.config["model"].get("num_heads", 4),
            ff_dim=self.config["model"].get("ff_dim", 64)
        )
    
    def create_trainer(self, model, checkpoint_dir: Path = None):
        """Create training setup for both TensorFlow and PyTorch models."""
        if isinstance(model, keras.Model):
            # TensorFlow model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["train"]["learning_rate"]),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        else:
            # PyTorch Lightning model
            callbacks = []
            
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename='model-{epoch:02d}-{val_loss:.2f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3,
                    save_last=True
                )
                callbacks.append(checkpoint_callback)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config["train"].get("patience", 5),
                mode='min'
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
        
        # Create vocabulary from metadata
        from ..data.loader import CanonicalLogsDataLoader
        loader = CanonicalLogsDataLoader(self.current_dataset, str(Path(self.config["data"]["path_processed"])))
        x_word_dict = loader.x_word_dict
        y_word_dict = loader.y_word_dict
        
        # Create vocabulary
        self.vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create model
        vocab_size = len(self.vocabulary.index_to_token)
        
        # Get max_case_length from metadata to ensure consistency
        max_case_length = loader.get_max_case_length(train_df)
        
        output_dim = len(y_word_dict)
        
        model = self.create_model(vocab_size, max_case_length, output_dim)
        
        # Create trainer
        trainer = self.create_trainer(model, checkpoint_dir)
        
        # Train model
        if isinstance(trainer, keras.Model):
            # TensorFlow training
            # Prepare data
            x_train, y_train = self._prepare_tensorflow_data(train_df, max_case_length)
            x_val, y_val = self._prepare_tensorflow_data(val_df, max_case_length)
            
            # Train
            history = trainer.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=self.config["train"]["max_epochs"],
                batch_size=self.config["train"]["batch_size"],
                verbose=1
            )
            
            # Evaluate
            val_loss, val_accuracy = trainer.evaluate(x_val, y_val, verbose=0)
            y_pred = trainer.predict(x_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred_classes)
            f1 = f1_score(y_val, y_pred_classes, average='weighted')
            
            metrics = {
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'accuracy': accuracy,
                'f1_score': f1
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
            
            # Evaluate
            results = trainer.validate(model, datamodule)
            
            # Extract metrics
            metrics = {
                'val_loss': results[0]['val_loss'],
                'val_accuracy': results[0]['val_accuracy'],
                'accuracy': results[0]['val_accuracy'],  # For compatibility
                'f1_score': results[0].get('val_f1', 0.0)
            }
        
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
                
                # Save detailed results
                results_file = outputs_dir / f"{dataset_name}_next_activity_cv_results.json"
                with open(results_file, 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if self.logger:
                    print(f"Error processing {dataset_name}: {e}")
                continue
        
        return results
