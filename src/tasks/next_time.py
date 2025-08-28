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
from ..data.loader import LogsDataLoader
from ..metrics import mean_absolute_error, mean_squared_error, r2_score
from ..utils.logging import create_logger
from ..utils.cross_validation import run_cross_validation


class NextTimeTask:
    """
    Task implementation for next time prediction in process mining.
    
    This task predicts the time until the next event in a process trace 
    given a prefix of activities that have already occurred.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = None
        self.model = None
        self.results = {}
        self.logger = None  # Will be initialized in run method
        self.current_dataset = None
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load and prepare data for next time prediction using ProcessTransformer pipeline."""
        # First ensure data is processed using ProcessTransformer format
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # Now use the ProcessTransformer loader
        data_loader = LogsDataLoader(dataset_name, str(processed_directory))
        
        # Load data for next time task
        train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("next_time")
        
        # Combine train and test for cross-validation
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create vocabulary from the metadata
        vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create labels series
        labels_series = combined_df["next_time"]
        
        return combined_df, labels_series, vocabulary
    
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
    
    def create_model(self, vocab_size: int, max_case_length: int = None):
        """Create the next time prediction model using transformer models."""
        from ..models.model_registry import create_model
        if max_case_length is None:
            max_case_length = self.config["model"].get("max_case_length", 50)
        return create_model(
            name="process_transformer",
            task="next_time",
            vocab_size=vocab_size,
            max_case_length=max_case_length,
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
                loss='log_cosh',  # Use LogCosh loss as in original ProcessTransformer
                metrics=['mae']  # Use MAE as metric
            )
            
            # Set up callbacks
            callbacks = []
            
            # Model checkpointing
            if checkpoint_dir:
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_dir / "model-{epoch:02d}-{val_loss:.2f}.h5"),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False
                )
                callbacks.append(checkpoint_callback)
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stopping)
            
            return callbacks
        else:
            # PyTorch Lightning model
            callbacks = []
            
            # Model checkpointing
            if checkpoint_dir:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="model-{epoch:02d}-{val_loss:.2f}",
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3,
                    save_last=True
                )
                callbacks.append(checkpoint_callback)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stopping)
            
            # Create trainer
            trainer = lightning.Trainer(
                max_epochs=self.config["train"]["max_epochs"],
                accelerator=self.config["train"]["accelerator"],
                devices=self.config["train"]["devices"],
                callbacks=callbacks,
                enable_progress_bar=True,
                log_every_n_steps=10
            )
            
            return trainer
    
    def evaluate_model(self, model, test_data) -> Dict[str, float]:
        """Evaluate the model and compute metrics."""
        if isinstance(model, keras.Model):
            # TensorFlow model evaluation
            y_pred = model.predict(test_data[0])
            y_true = test_data[1]
        else:
            # PyTorch Lightning model evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(test_data[0])
                if isinstance(outputs, tuple):
                    y_pred = outputs[0].cpu().numpy()
                else:
                    y_pred = outputs.cpu().numpy()
                y_true = test_data[1].cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train_and_evaluate_fold(self, train_prefixes: pd.DataFrame, train_labels: pd.Series,
                               val_prefixes: pd.DataFrame, val_labels: pd.Series, 
                               fold_idx: int) -> Dict[str, Any]:
        """
        Train and evaluate a model on a single fold.
        
        Args:
            train_prefixes: Training data prefixes
            train_labels: Training labels
            val_prefixes: Validation data prefixes
            val_labels: Validation labels
            fold_idx: Current fold index
            
        Returns:
            Dictionary with fold results
        """
        print(f"Training fold {fold_idx + 1}...")
        
        # Prepare data using ProcessTransformer format
        data_loader = LogsDataLoader(self.current_dataset, str(Path(self.config["data"]["path_processed"])))
        
        # Get vocabulary and model parameters
        _, _, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("next_time")
        
        # Prepare training data
        train_token_x, train_time_x, train_token_y, time_scaler, y_scaler = data_loader.prepare_data_next_time(
            train_prefixes, x_word_dict, max_case_length
        )
        
        # Prepare validation data using the same scalers
        val_token_x, val_time_x, val_token_y, _, _ = data_loader.prepare_data_next_time(
            val_prefixes, x_word_dict, max_case_length, time_scaler=time_scaler, y_scaler=y_scaler
        )
        
        # Create model
        model = self.create_model(vocab_size, max_case_length=max_case_length)
        
        # Create checkpoint directory for this fold
        checkpoint_dir = Path(self.config.get("outputs_dir", "outputs")) / "checkpoints" / self.current_dataset / f"fold_{fold_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, keras.Model):
            # TensorFlow model training
            callbacks = self.create_trainer(model, checkpoint_dir)
            
            # Train the model
            history = model.fit(
                [train_token_x, train_time_x], train_token_y,
                epochs=self.config["train"]["max_epochs"],
                batch_size=self.config["train"]["batch_size"],
                validation_data=([val_token_x, val_time_x], val_token_y),
                shuffle=True,
                verbose=1,
                callbacks=callbacks
            )
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, ([val_token_x, val_time_x], val_token_y))
            
        else:
            # PyTorch Lightning model training
            trainer = self.create_trainer(model, checkpoint_dir)
            
            # Create data module for this fold
            from ..training.datamodule import NextTimeDataModule
            datamodule = NextTimeDataModule(
                train_df=train_prefixes,
                test_df=val_prefixes,  # Use validation data as test for this fold
                x_word_dict=x_word_dict,
                y_word_dict=y_word_dict,
                max_case_length=max_case_length,
                batch_size=self.config["train"]["batch_size"]
            )
            
            # Train the model
            trainer.fit(model, datamodule)
            
            # Evaluate on validation set
            test_results = trainer.test(model, datamodule)
            val_metrics = {
                'mae': test_results[0]['test_mae'],
                'rmse': test_results[0].get('test_rmse', 0.0),
                'r2': test_results[0].get('test_r2', 0.0)
            }
        
        return {
            'fold_idx': fold_idx,
            'metrics': val_metrics,
            'train_samples': len(train_prefixes),
            'val_samples': len(val_prefixes)
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """Run the next time prediction task on multiple datasets using 5-fold cross-validation."""
        processed_directory = Path(self.config["data"]["path_processed"])
        processed_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = create_logger(self.config, outputs_dir)
        
        # Store outputs_dir in config for use in train_and_evaluate_fold
        self.config["outputs_dir"] = str(outputs_dir)

        # Train and evaluate per dataset
        results = {}
        
        for dataset_name in datasets:
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            self.current_dataset = dataset_name
            
            try:
                # Load and prepare data
                prefixes_df, labels_series, vocabulary = self.prepare_data(
                    dataset_name, raw_directory, processed_directory, 
                    force_reprocess=self.config.get("force_preprocess", False)
                )
                
                print(f"Total samples: {len(prefixes_df)}")
                print(f"Vocabulary size: {len(vocabulary.index_to_token)}")
                
                # Run 5-fold cross-validation
                cv_results = run_cross_validation(
                    task_class=self.__class__,
                    config=self.config,
                    prefixes_df=prefixes_df,
                    labels_series=labels_series,
                    vocabulary=vocabulary,
                    cv_config=self.config["cv"],
                    dataset_name=dataset_name
                )
                
                # Store results
                results[dataset_name] = cv_results
                
                # Print summary
                print(f"\n=== Cross-Validation Results for {dataset_name} ===")
                for metric, value in cv_results['cv_summary'].items():
                    if metric.endswith('_mean'):
                        print(f"{metric}: {value:.4f}")
                
                # Save detailed results
                results_file = outputs_dir / f"{dataset_name}_next_time_cv_results.json"
                with open(results_file, 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if self.logger:
                    self.logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        return results
