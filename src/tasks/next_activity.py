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
from ..metrics import accuracy_score, f1_score, classification_report_metrics
from ..utils.logging import create_logger


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
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load and prepare data for next activity prediction using ProcessTransformer pipeline."""
        # First ensure data is processed using ProcessTransformer format
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # Now use the ProcessTransformer loader
        data_loader = LogsDataLoader(dataset_name, str(processed_directory))
        
        # Load data for next activity task
        train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("next_activity")
        
        # Combine train and test for evaluation
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create vocabulary from the metadata
        vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Create labels series
        labels_series = combined_df["next_act"]
        
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
        """Create the next activity prediction model using transformer models."""
        from ..models.model_registry import create_model
        if max_case_length is None:
            max_case_length = self.config["model"].get("max_case_length", 50)
        return create_model(
            name="process_transformer",
            task="next_activity",
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
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
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
            y_pred = np.argmax(model.predict(test_data[0]), axis=1)
            y_true = test_data[1]
        else:
            # PyTorch Lightning model evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(test_data[0])
                if isinstance(outputs, tuple):
                    y_pred = torch.argmax(outputs[0], dim=1).cpu().numpy()
                else:
                    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true = test_data[1].cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """Run the next activity prediction task on multiple datasets."""
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
                
                # Load data for next activity task
                train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes = data_loader.load_data("next_activity")
                
                print(f"Vocabulary size: {vocab_size}")
                print(f"Total classes: {total_classes}")
                print(f"Max case length: {max_case_length}")
                print(f"Train samples: {len(train_df)}")
                print(f"Test samples: {len(test_df)}")
                
                # Prepare training data
                train_token_x, train_token_y = data_loader.prepare_data_next_activity(
                    train_df, x_word_dict, y_word_dict, max_case_length
                )
                
                # Create and train model
                model = self.create_model(vocab_size, max_case_length=max_case_length)
                
                if isinstance(model, keras.Model):
                    # TensorFlow model training
                    print("Training TensorFlow model...")
                    
                    # Create callbacks
                    callbacks = self.create_trainer(model)
                    
                    # Train the model
                    history = model.fit(
                        train_token_x, train_token_y,
                        epochs=self.config["train"]["max_epochs"],
                        batch_size=self.config["train"]["batch_size"],
                        validation_split=0.2,
                        shuffle=True,
                        verbose=2,
                        callbacks=callbacks
                    )
                    
                    # Evaluate over all prefixes (k) and save the results
                    k, accuracies, fscores, precisions, recalls = [], [], [], [], []
                    
                    for i in range(max_case_length):
                        test_data_subset = test_df[test_df["k"] == i]
                        if len(test_data_subset) > 0:
                            test_token_x, test_token_y = data_loader.prepare_data_next_activity(
                                test_data_subset, x_word_dict, y_word_dict, max_case_length
                            )
                            y_pred = np.argmax(model.predict(test_token_x), axis=1)
                            accuracy = metrics.accuracy_score(test_token_y, y_pred)
                            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                                test_token_y, y_pred, average="weighted"
                            )
                            k.append(i)
                            accuracies.append(accuracy)
                            fscores.append(fscore)
                            precisions.append(precision)
                            recalls.append(recall)
                    
                    # Add average metrics
                    k.append(max_case_length)
                    accuracies.append(np.mean(accuracies))
                    fscores.append(np.mean(fscores))
                    precisions.append(np.mean(precisions))
                    recalls.append(np.mean(recalls))
                    
                    print(f'Average accuracy across all prefixes: {np.mean(accuracies):.4f}')
                    print(f'Average f-score across all prefixes: {np.mean(fscores):.4f}')
                    print(f'Average precision across all prefixes: {np.mean(precisions):.4f}')
                    print(f'Average recall across all prefixes: {np.mean(recalls):.4f}')
                    
                    # Save results
                    results_df = pd.DataFrame({
                        "k": k, "accuracy": accuracies, "fscore": fscores,
                        "precision": precisions, "recall": recalls
                    })
                    
                    result_path = outputs_dir / f"{dataset_name}_next_activity_results.csv"
                    results_df.to_csv(result_path, index=False)
                    
                    results[dataset_name] = {
                        'avg_accuracy': np.mean(accuracies),
                        'avg_fscore': np.mean(fscores),
                        'avg_precision': np.mean(precisions),
                        'avg_recall': np.mean(recalls),
                        'prefix_results': results_df.to_dict('records')
                    }
                    
                else:
                    # PyTorch Lightning model training
                    print("Training PyTorch Lightning model...")
                    trainer = self.create_trainer(model)
                    
                    # Create data module for training
                    from ..training.datamodule import NextActivityDataModule
                    datamodule = NextActivityDataModule(
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
                        'test_accuracy': test_results[0]['test_accuracy'],
                        'test_loss': test_results[0]['test_loss']
                    }
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if self.logger:
                    print(f"Error processing {dataset_name}: {e}")  # Use print instead of logger.error
                continue
        
        return results
