import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models.base import BaseNextActivityModel
from src.training.datamodule import PrefixDataModule
from src.data.encoders import Vocabulary
from src.data.prefixer import generate_prefixes
from src.data.loader import load_event_log
from src.metrics import accuracy_score, f1_score, classification_report_metrics
from src.utils.cross_validation import run_cross_validation
from src.utils.logging import create_logger


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
        
    def prepare_data(self, dataset_name: str, raw_directory: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for next activity prediction."""
        csv_candidate = raw_directory / f"{dataset_name}.csv"
        xes_candidate = raw_directory / f"{dataset_name}.xes"

        if csv_candidate.exists():
            event_log = load_event_log(str(csv_candidate))
        elif xes_candidate.exists():
            event_log = load_event_log(str(xes_candidate))
        else:
            raise FileNotFoundError(
                f"Neither {csv_candidate.name} nor {xes_candidate.name} exist in {raw_directory}"
            )

        prefixes_dataframe, label_series = generate_prefixes(
            event_log_dataframe=event_log,
            end_of_case_token=self.config["data"]["end_of_case_token"],
            max_prefix_length=self.config["data"]["max_prefix_length"],
            attribute_mode=self.config["data"]["attribute_mode"]
        )
        return prefixes_dataframe, label_series
    
    def build_vocabulary(self, all_activity_tokens: List[str]) -> Vocabulary:
        """Build vocabulary from all activity tokens across datasets."""
        all_activity_tokens.append(self.config["data"]["end_of_case_token"])
        return Vocabulary(all_activity_tokens)
    
    def create_model(self, vocab_size: int) -> BaseNextActivityModel:
        """Create the next activity prediction model."""
        return BaseNextActivityModel(
            vocab_size=vocab_size,
            hidden_size=self.config["model"]["hidden_size"],
            num_layers=self.config["model"]["num_layers"],
            num_heads=self.config["model"]["num_heads"],
            dropout_probability=self.config["model"]["dropout_probability"],
            learning_rate=self.config["train"]["learning_rate"]
        )
    
    def create_trainer(self, checkpoint_dir: Path = None) -> Trainer:
        """Create PyTorch Lightning trainer with callbacks."""
        trainer_accelerator = self.config["train"]["accelerator"]
        if trainer_accelerator == "auto":
            trainer_accelerator = (
                "gpu" if torch.cuda.is_available() 
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() 
                      else "cpu")
            )
        
        # Set up callbacks
        callbacks = []
        
        # Model checkpointing
        if checkpoint_dir:
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='model-{epoch:02d}-{val_loss:.2f}',
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
        
        return Trainer(
            accelerator=trainer_accelerator,
            devices=self.config["train"]["devices"],
            max_epochs=self.config["train"]["max_epochs"],
            enable_progress_bar=True,
            log_every_n_steps=50,
            callbacks=callbacks
        )
    
    def evaluate_model(self, model: BaseNextActivityModel, test_dataloader) -> Dict[str, float]:
        """Evaluate the model and compute metrics."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_sequences, target_indices = batch
                if next(model.parameters()).is_cuda:
                    input_sequences = input_sequences.cuda()
                    target_indices = target_indices.cuda()
                
                logits = model(input_sequences)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target_indices.cpu().numpy())
        
        # Calculate metrics using our comprehensive metrics module
        vocab_size = len(self.vocabulary.index_to_token)
        acc = accuracy_score(all_targets, all_predictions, num_classes=vocab_size, ignore_index=0)
        f1 = f1_score(all_targets, all_predictions, num_classes=vocab_size, ignore_index=0)
        
        # Generate classification report
        target_names = [self.vocabulary.index_to_token[i] for i in range(len(self.vocabulary.index_to_token)) 
                       if i != 0]  # Exclude pad token
        classification_rep = classification_report_metrics(
            all_targets, all_predictions, 
            target_names=target_names
        )
        
        return {
            "accuracy": acc,
            "f1_score": f1,
            "classification_report": classification_rep
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """
        Run the complete next activity prediction task.
        
        Args:
            datasets: List of dataset names to process
            raw_directory: Path to raw data directory
            outputs_dir: Path to save outputs
            
        Returns:
            Dictionary containing results for all datasets
        """
        print(f"Running Next Activity Prediction Task for datasets: {datasets}")
        
        # Set up logging
        logger = create_logger(self.config, outputs_dir)

        # Aggregate vocab across all configured datasets
        all_activity_tokens = []
        dataset_cache = {}
        
        for dataset_name in datasets:
            print(f"Loading dataset: {dataset_name}")
            prefixes_dataframe, label_series = self.prepare_data(dataset_name, raw_directory)
            dataset_cache[dataset_name] = (prefixes_dataframe, label_series)
            all_activity_tokens.extend(prefixes_dataframe["prefix_activities"].explode().tolist())
        
        # Build vocabulary
        self.vocabulary = self.build_vocabulary(all_activity_tokens)
        print(f"Vocabulary size: {len(self.vocabulary.index_to_token)}")
        
        # Train and evaluate per dataset
        results = {}
        
        for dataset_name, (prefixes_dataframe, label_series) in dataset_cache.items():
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            
            # Split data
            split_index = int(0.8 * len(prefixes_dataframe))
            train_prefixes = prefixes_dataframe.iloc[:split_index].reset_index(drop=True)
            train_labels = label_series.iloc[:split_index].reset_index(drop=True)
            val_prefixes = prefixes_dataframe.iloc[split_index:].reset_index(drop=True)
            val_labels = label_series.iloc[split_index:].reset_index(drop=True)
            
            # Create data module
            data_module = PrefixDataModule(
                train_prefix_df=train_prefixes,
                train_labels=train_labels,
                val_prefix_df=val_prefixes,
                val_labels=val_labels,
                vocabulary=self.vocabulary,
                batch_size=self.config["train"]["batch_size"]
            )
            
            # Create trainer with checkpointing
            checkpoint_dir = outputs_dir / "checkpoints" / dataset_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer = self.create_trainer(checkpoint_dir=checkpoint_dir)
            
            # Create and train model
            model = self.create_model(len(self.vocabulary.index_to_token))
            trainer.fit(model, data_module)
            
            # Evaluate model
            evaluation_results = self.evaluate_model(model, data_module.val_dataloader())
            
            # Store results
            results[dataset_name] = {
                "vocabulary_size": len(self.vocabulary.index_to_token),
                "train_samples": len(train_prefixes),
                "val_samples": len(val_prefixes),
                "metrics": evaluation_results
            }
            
            # Log results
            logger.log_results(results[dataset_name], dataset_name)
            
            print(f"Dataset {dataset_name} - Accuracy: {evaluation_results['accuracy']:.4f}, "
                  f"F1 Score: {evaluation_results['f1_score']:.4f}")
        
        # Save results
        results_file = outputs_dir / "next_activity_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.finish()
        print(f"\nResults saved to: {results_file}")
        return results
    
    def train_and_evaluate_fold(self, train_prefixes: pd.DataFrame, train_labels: pd.Series,
                               val_prefixes: pd.DataFrame, val_labels: pd.Series, 
                               fold_idx: int) -> Dict[str, Any]:
        """
        Train and evaluate a single fold for cross-validation.
        
        Args:
            train_prefixes: Training prefix data
            train_labels: Training labels
            val_prefixes: Validation prefix data
            val_labels: Validation labels
            fold_idx: Fold index
            
        Returns:
            Dictionary with fold results
        """
        # Create data module
        data_module = PrefixDataModule(
            train_prefix_df=train_prefixes,
            train_labels=train_labels,
            val_prefix_df=val_prefixes,
            val_labels=val_labels,
            vocabulary=self.vocabulary,
            batch_size=self.config["train"]["batch_size"]
        )
        
        # Create model
        model = self.create_model(len(self.vocabulary.index_to_token))
        
        # Create trainer with checkpointing
        checkpoint_dir = Path(f"outputs/checkpoints/fold_{fold_idx}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer = self.create_trainer(checkpoint_dir=checkpoint_dir)
        
        # Train model
        trainer.fit(model, data_module)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data_module.val_dataloader())
        
        return {
            "fold_idx": fold_idx,
            "train_samples": len(train_prefixes),
            "val_samples": len(val_prefixes),
            "metrics": evaluation_results,
            "checkpoint_path": str(checkpoint_dir) if checkpoint_dir.exists() else None
        }
    
    def run_with_cv(self, datasets: List[str], raw_directory: Path, 
                   outputs_dir: Path) -> Dict[str, Any]:
        """
        Run next activity prediction with cross-validation.
        
        Args:
            datasets: List of dataset names
            raw_directory: Path to raw data directory
            outputs_dir: Path to save outputs
            
        Returns:
            Dictionary with CV results
        """
        print(f"Running Next Activity Prediction with Cross-Validation for datasets: {datasets}")
        
        # Set up logging
        logger = create_logger(self.config, outputs_dir)
        
        # Aggregate vocab across all configured datasets
        all_activity_tokens = []
        dataset_cache = {}
        
        for dataset_name in datasets:
            print(f"Loading dataset: {dataset_name}")
            prefixes_dataframe, label_series = self.prepare_data(dataset_name, raw_directory)
            dataset_cache[dataset_name] = (prefixes_dataframe, label_series)
            all_activity_tokens.extend(prefixes_dataframe["prefix_activities"].explode().tolist())
        
        # Build vocabulary
        self.vocabulary = self.build_vocabulary(all_activity_tokens)
        print(f"Vocabulary size: {len(self.vocabulary.index_to_token)}")
        
        # Run cross-validation for each dataset
        cv_results = {}
        for dataset_name, (prefixes_dataframe, label_series) in dataset_cache.items():
            print(f"\n=== Running CV for Dataset: {dataset_name} ===")
            
            # Run cross-validation
            cv_result = run_cross_validation(
                task_class=NextActivityTask,
                config=self.config,
                prefixes_df=prefixes_dataframe,
                labels_series=label_series,
                vocabulary=self.vocabulary,
                cv_config=self.config.get("cv", {})
            )
            
            cv_results[dataset_name] = cv_result
            
            # Log results
            logger.log_results(cv_result, dataset_name)
            
            print(f"CV Results for {dataset_name}:")
            for metric, value in cv_result['cv_summary'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
        
        # Save CV results
        cv_results_file = outputs_dir / "cv_results.json"
        with open(cv_results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        logger.finish()
        print(f"\nCV Results saved to: {cv_results_file}")
        return cv_results


def run_next_activity_task(config: Dict[str, Any], datasets: List[str], 
                          raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to run the next activity prediction task.
    
    Args:
        config: Configuration dictionary
        datasets: List of dataset names
        raw_directory: Path to raw data directory
        outputs_dir: Path to save outputs
        
    Returns:
        Results dictionary
    """
    task = NextActivityTask(config)
    
    # Check if cross-validation is enabled
    cv_config = config.get("cv", {})
    if cv_config.get("n_folds", 0) > 1:
        return task.run_with_cv(datasets, raw_directory, outputs_dir)
    else:
        return task.run(datasets, raw_directory, outputs_dir)
