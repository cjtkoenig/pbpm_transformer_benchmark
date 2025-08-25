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

from ..data.encoders import Vocabulary
from ..data.preprocessor import SimplePreprocessor
from ..metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from ..utils.logging import create_logger
from ..training.datamodule import MultiTaskDataModule


class MultiTaskTask:
    """
    Multi-task learning implementation for MTLFormer.
    
    This task handles next_activity, next_time, and remaining_time prediction
    simultaneously using a single model with shared parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = None
        self.model = None
        self.results = {}
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], Vocabulary]:
        """Load and prepare data for all three tasks."""
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        
        # Prepare data for each task
        activity_prefixes_df, activity_labels, activity_vocab = preprocessor.preprocess_dataset(dataset_name, force_reprocess, task="next_activity")
        time_prefixes_df, time_labels, time_vocab = preprocessor.preprocess_dataset(dataset_name, force_reprocess, task="next_time")
        remaining_prefixes_df, remaining_labels, remaining_vocab = preprocessor.preprocess_dataset(dataset_name, force_reprocess, task="remaining_time")
        
        # Merge vocabularies (they should be the same, but just in case)
        all_tokens = set()
        for vocab in [activity_vocab, time_vocab, remaining_vocab]:
            all_tokens.update(vocab.index_to_token)
        vocabulary = Vocabulary(list(all_tokens))
        
        return {
            'activity': activity_prefixes_df,
            'time': time_prefixes_df,
            'remaining': remaining_prefixes_df
        }, {
            'activity': activity_labels,
            'time': time_labels,
            'remaining': remaining_labels
        }, vocabulary
    
    def build_vocabulary(self, all_activity_tokens: List[str]) -> Vocabulary:
        """Build vocabulary from all activity tokens across datasets."""
        all_activity_tokens.append(self.config["data"]["end_of_case_token"])
        return Vocabulary(all_activity_tokens)
    
    def merge_vocabularies(self, vocabularies: List[Vocabulary]) -> Vocabulary:
        """Merge multiple vocabularies into a single vocabulary."""
        all_tokens = set()
        for vocab in vocabularies:
            all_tokens.update(vocab.index_to_token)
        return Vocabulary(list(all_tokens))
    
    def create_model(self, vocab_size: int):
        """Create the multi-task MTLFormer model."""
        from ..models.model_registry import create_model
        return create_model(
            name="mtlformer_multi",
            task="multi_task",
            vocab_size=vocab_size,
            max_case_length=self.config["model"].get("max_case_length", 50),
            embed_dim=self.config["model"].get("embed_dim", 36),
            num_heads=self.config["model"].get("num_heads", 4),
            ff_dim=self.config["model"].get("ff_dim", 64)
        )
    
    def create_trainer(self, model, checkpoint_dir: Path = None):
        """Create training setup for multi-task TensorFlow model."""
        if isinstance(model, keras.Model):
            # Multi-task loss with weights as in the original main.py
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["train"]["learning_rate"]),
                loss={
                    'out1': 'sparse_categorical_crossentropy',  # next_activity
                    'out2': 'logcosh',  # next_time
                    'out3': 'logcosh'   # remaining_time
                },
                loss_weights={
                    'out1': 0.6,  # next_activity weight
                    'out2': 2.0,  # next_time weight
                    'out3': 0.3   # remaining_time weight
                },
                metrics={
                    'out1': ['accuracy'],
                    'out2': ['mae'],
                    'out3': ['mae']
                }
            )
            
            # Set up callbacks
            callbacks = []
            
            # Model checkpointing
            if checkpoint_dir:
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_dir / "multitask-model-{epoch:02d}-{val_loss:.2f}.h5"),
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
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    def evaluate_model(self, model, test_data) -> Dict[str, float]:
        """Evaluate the multi-task model and compute metrics for each task."""
        if isinstance(model, keras.Model):
            # TensorFlow model
            predictions = model.predict(test_data[0], verbose=0)
            
            # Extract predictions for each task
            next_activity_pred = predictions[0]  # out1
            next_time_pred = predictions[1]      # out2
            remaining_time_pred = predictions[2] # out3
            
            # Extract targets for each task
            next_activity_true = test_data[1][0]  # activity targets
            next_time_true = test_data[1][1]      # time targets
            remaining_time_true = test_data[1][2]  # remaining time targets
            
            # Calculate metrics for each task
            # Next Activity (Classification)
            predicted_classes = np.argmax(next_activity_pred, axis=-1)
            activity_accuracy = accuracy_score(next_activity_true, predicted_classes, 
                                             num_classes=len(self.vocabulary.index_to_token), ignore_index=0)
            activity_f1 = f1_score(next_activity_true, predicted_classes, 
                                 num_classes=len(self.vocabulary.index_to_token), ignore_index=0)
            
            # Next Time (Regression)
            next_time_mae = mean_absolute_error(next_time_true, next_time_pred)
            next_time_mse = mean_squared_error(next_time_true, next_time_pred)
            next_time_rmse = np.sqrt(next_time_mse)
            next_time_r2 = r2_score(next_time_true, next_time_pred)
            
            # Remaining Time (Regression)
            remaining_time_mae = mean_absolute_error(remaining_time_true, remaining_time_pred)
            remaining_time_mse = mean_squared_error(remaining_time_true, remaining_time_pred)
            remaining_time_rmse = np.sqrt(remaining_time_mse)
            remaining_time_r2 = r2_score(remaining_time_true, remaining_time_pred)
            
            return {
                # Next Activity metrics
                'next_activity_accuracy': activity_accuracy,
                'next_activity_f1': activity_f1,
                
                # Next Time metrics
                'next_time_mae': next_time_mae,
                'next_time_mse': next_time_mse,
                'next_time_rmse': next_time_rmse,
                'next_time_r2': next_time_r2,
                
                # Remaining Time metrics
                'remaining_time_mae': remaining_time_mae,
                'remaining_time_mse': remaining_time_mse,
                'remaining_time_rmse': remaining_time_rmse,
                'remaining_time_r2': remaining_time_r2
            }
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """Run multi-task learning evaluation across all datasets."""
        logger = create_logger(outputs_dir / "multitask_evaluation.log")
        logger.info("Starting multi-task learning evaluation")
        
        all_results = {}
        
        for dataset_name in datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Prepare data for all tasks
            prefixes_dict, labels_dict, vocabulary = self.prepare_data(
                dataset_name, raw_directory, outputs_dir / "processed"
            )
            
            self.vocabulary = vocabulary
            
            # Create model
            model = self.create_model(len(vocabulary.index_to_token))
            self.model = model
            
            # Set up cross-validation
            from ..utils.cross_validation import create_cv_splits
            
            # Create CV splits for activity data (use as reference)
            cv_splits = create_cv_splits(
                prefixes_dict['activity'], 
                labels_dict['activity'],
                n_folds=self.config["cv"]["n_folds"],
                stratify=self.config["cv"]["stratify"],
                split_by_cases=self.config["cv"]["split_by_cases"]
            )
            
            dataset_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                logger.info(f"Processing fold {fold_idx + 1}/{len(cv_splits)}")
                
                # Split data for all tasks using the same indices
                train_activity_df = prefixes_dict['activity'].iloc[train_idx]
                val_activity_df = prefixes_dict['activity'].iloc[val_idx]
                train_activity_labels = labels_dict['activity'].iloc[train_idx]
                val_activity_labels = labels_dict['activity'].iloc[val_idx]
                
                train_time_df = prefixes_dict['time'].iloc[train_idx]
                val_time_df = prefixes_dict['time'].iloc[val_idx]
                train_time_labels = labels_dict['time'].iloc[train_idx]
                val_time_labels = labels_dict['time'].iloc[val_idx]
                
                train_remaining_df = prefixes_dict['remaining'].iloc[train_idx]
                val_remaining_df = prefixes_dict['remaining'].iloc[val_idx]
                train_remaining_labels = labels_dict['remaining'].iloc[train_idx]
                val_remaining_labels = labels_dict['remaining'].iloc[val_idx]
                
                # Create multi-task data module
                datamodule = MultiTaskDataModule(
                    train_activity_df, train_activity_labels,
                    val_activity_df, val_activity_labels,
                    train_time_df, train_time_labels,
                    val_time_df, val_time_labels,
                    train_remaining_df, train_remaining_labels,
                    val_remaining_df, val_remaining_labels,
                    vocabulary,
                    batch_size=self.config["train"]["batch_size"]
                )
                
                # Train model
                callbacks = self.create_trainer(model, outputs_dir / "checkpoints" / dataset_name / f"fold_{fold_idx}")
                
                # Prepare training data
                train_data = [
                    [datamodule.train_dataset[i][0] for i in range(len(datamodule.train_dataset))],
                    [datamodule.train_dataset[i][1] for i in range(len(datamodule.train_dataset))],
                    [datamodule.train_dataset[i][2] for i in range(len(datamodule.train_dataset))]
                ]
                
                val_data = [
                    [datamodule.val_dataset[i][0] for i in range(len(datamodule.val_dataset))],
                    [datamodule.val_dataset[i][1] for i in range(len(datamodule.val_dataset))],
                    [datamodule.val_dataset[i][2] for i in range(len(datamodule.val_dataset))]
                ]
                
                # Train
                model.fit(
                    train_data,
                    [datamodule.train_dataset[i][3] for i in range(len(datamodule.train_dataset))],
                    validation_data=(val_data, [datamodule.val_dataset[i][3] for i in range(len(datamodule.val_dataset))]),
                    epochs=self.config["train"]["max_epochs"],
                    batch_size=self.config["train"]["batch_size"],
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate
                fold_metrics = self.evaluate_model(model, (val_data, [
                    [datamodule.val_dataset[i][3] for i in range(len(datamodule.val_dataset))],
                    [datamodule.val_dataset[i][4] for i in range(len(datamodule.val_dataset))],
                    [datamodule.val_dataset[i][5] for i in range(len(datamodule.val_dataset))]
                ]))
                
                fold_metrics['fold'] = fold_idx
                dataset_results.append(fold_metrics)
                
                logger.info(f"Fold {fold_idx + 1} results: {fold_metrics}")
            
            # Aggregate results across folds
            aggregated_results = self._aggregate_results(dataset_results)
            all_results[dataset_name] = aggregated_results
            
            # Save results
            results_file = outputs_dir / f"multitask_results_{dataset_name}.json"
            with open(results_file, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            
            logger.info(f"Completed dataset {dataset_name}. Results saved to {results_file}")
        
        # Save overall results
        overall_results_file = outputs_dir / "multitask_overall_results.json"
        with open(overall_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Multi-task evaluation completed. Overall results saved to {overall_results_file}")
        
        return all_results
    
    def _aggregate_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results across cross-validation folds."""
        aggregated = {}
        
        # Get all metric names from the first fold
        metric_names = [k for k in fold_results[0].keys() if k != 'fold']
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)
        
        # Add fold details
        aggregated['n_folds'] = len(fold_results)
        aggregated['fold_results'] = fold_results
        
        return aggregated
