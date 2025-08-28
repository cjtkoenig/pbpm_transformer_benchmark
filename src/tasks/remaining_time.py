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


class RemainingTimeTask:
    """
    Task implementation for remaining time prediction in process mining.
    
    This task predicts the remaining time until process completion 
    given a prefix of activities that have already occurred.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = None
        self.model = None
        self.results = {}
        self.logger = None  # Will be initialized in run method
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load and prepare data for remaining time prediction using ProcessTransformer pipeline."""
        # First ensure data is processed using ProcessTransformer format
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # Now use the ProcessTransformer loader
        data_loader = LogsDataLoader(dataset_name, str(processed_directory))
        
        # Load data for remaining time task
        train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("remaining_time")
        
        # Combine train and test for evaluation
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create vocabulary from the metadata
        vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create labels series
        labels_series = combined_df["remaining_time_days"]
        
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
        """Create the remaining time prediction model using transformer models."""
        from ..models.model_registry import create_model
        if max_case_length is None:
            max_case_length = self.config["model"].get("max_case_length", 50)
        return create_model(
            name="process_transformer",
            task="remaining_time",
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
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """Run the remaining time prediction task on multiple datasets."""
        processed_directory = Path(self.config["data"]["path_processed"])
        processed_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = create_logger(self.config, outputs_dir)
        
        # Train and evaluate per dataset
        results = {}
        
        for dataset_name in datasets:
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            
            try:
                # Load and prepare data using ProcessTransformer format
                data_loader = LogsDataLoader(dataset_name, str(processed_directory))
                
                # Check if data is processed
                if not data_loader.is_processed():
                    print(f"Data not processed for {dataset_name}, preprocessing...")
                    preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
                    preprocessor.preprocess_dataset(dataset_name, force_reprocess=False)
                
                # Load data for remaining time task
                train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("remaining_time")
                
                print(f"Vocabulary size: {vocab_size}")
                print(f"Total classes: {total_classes}")
                print(f"Max case length: {max_case_length}")
                print(f"Train samples: {len(train_df)}")
                print(f"Test samples: {len(test_df)}")
                
                # Prepare training data
                train_token_x, train_time_x, train_token_y, time_scaler, y_scaler = data_loader.prepare_data_remaining_time(
                    train_df, x_word_dict, max_case_length
                )
                
                # Create and train model
                model = self.create_model(vocab_size, max_case_length=max_case_length)
                
                if isinstance(model, keras.Model):
                    # TensorFlow model training
                    print("Training TensorFlow model...")
                    
                    # Create callbacks
                    callbacks = self.create_trainer(model)
                    
                    # Train the model with dual inputs
                    history = model.fit(
                        [train_token_x, train_time_x], train_token_y,
                        epochs=self.config["train"]["max_epochs"],
                        batch_size=self.config["train"]["batch_size"],
                        validation_split=0.2,
                        shuffle=True,
                        verbose=2,
                        callbacks=callbacks
                    )
                    
                    # Evaluate over all prefixes (k) and save the results
                    k, maes, mses, rmses = [], [], [], []
                    
                    for i in range(max_case_length):
                        test_data_subset = test_df[test_df["k"] == i]
                        if len(test_data_subset) > 0:
                            test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
                                test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False
                            )
                            
                            y_pred = model.predict([test_token_x, test_time_x])
                            
                            # Inverse transform predictions and true values
                            _test_y = y_scaler.inverse_transform(test_y)
                            _y_pred = y_scaler.inverse_transform(y_pred)
                            
                            k.append(i)
                            maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                            mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                            rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))
                    
                    # Add average metrics
                    k.append(max_case_length)
                    maes.append(np.mean(maes))
                    mses.append(np.mean(mses))
                    rmses.append(np.mean(rmses))
                    
                    print(f'Average MAE across all prefixes: {np.mean(maes):.4f}')
                    print(f'Average MSE across all prefixes: {np.mean(mses):.4f}')
                    print(f'Average RMSE across all prefixes: {np.mean(rmses):.4f}')
                    
                    # Save results
                    results_df = pd.DataFrame({
                        "k": k, "mean_absolute_error": maes,
                        "mean_squared_error": mses,
                        "root_mean_squared_error": rmses
                    })
                    
                    result_path = outputs_dir / f"{dataset_name}_remaining_time_results.csv"
                    results_df.to_csv(result_path, index=False)
                    
                    results[dataset_name] = {
                        'avg_mae': np.mean(maes),
                        'avg_mse': np.mean(mses),
                        'avg_rmse': np.mean(rmses),
                        'prefix_results': results_df.to_dict('records')
                    }
                    
                else:
                    # PyTorch Lightning model training
                    print("Training PyTorch Lightning model...")
                    trainer = self.create_trainer(model)
                    
                    # Create data module for training
                    from ..training.datamodule import RemainingTimeDataModule
                    datamodule = RemainingTimeDataModule(
                        train_df=train_df,
                        test_df=test_df,
                        x_word_dict=x_word_dict,
                        y_word_dict=y_word_dict,
                        max_case_length=max_case_length,
                        batch_size=self.config["train"]["batch_size"]
                    )
                    
                    # Train the model
                    trainer.fit(model, datamodule)
                    
                    # Evaluate
                    test_results = trainer.test(model, datamodule)
                    
                    results[dataset_name] = {
                        'test_mae': test_results[0]['test_mae'],
                        'test_loss': test_results[0]['test_loss']
                    }
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if self.logger:
                    print(f"Error processing {dataset_name}: {e}")  # Use print instead of logger.error
                continue
        
        return results
