import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ..training.datamodule import SuffixPrefixDataModule
from ..data.encoders import Vocabulary
from ..data.preprocessor import SimplePreprocessor
from ..data.loader import load_event_log
from ..metrics import normalized_damerau_levenshtein
from ..utils.cross_validation import run_cross_validation
from ..utils.logging import create_logger


class SuffixTask:
    """
    Task implementation for suffix prediction in process mining.
    
    This task predicts the remaining sequence of activities (suffix) in a process trace 
    given a prefix of activities that have already occurred.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = None
        self.model = None
        self.results = {}
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.Series, Vocabulary]:
        """Load and prepare data for suffix prediction with simple preprocessing."""
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        prefixes_df, labels_series, vocabulary = preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # For suffix prediction, we need to create suffix targets
        # We need to load the original event log for suffix extraction
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
        
        suffix_targets = self._create_suffix_targets(prefixes_df, event_log)
        
        return prefixes_df, suffix_targets, vocabulary
    
    def _create_suffix_targets(self, prefixes_df: pd.DataFrame, event_log: pd.DataFrame) -> pd.Series:
        """Create suffix targets for training."""
        max_suffix_length = 10
        suffix_targets = []
        
        for _, row in prefixes_df.iterrows():
            case_id = row["case_id"]
            prefix_length = row["prefix_length"]
            
            # Get the case data
            case_data = event_log[event_log["case:concept:name"] == case_id].sort_values("time:timestamp")
            
            if prefix_length < len(case_data):
                # Extract actual suffix from the case
                suffix_activities = case_data.iloc[prefix_length:]["concept:name"].tolist()
                # Truncate to max_suffix_length
                suffix_activities = suffix_activities[:max_suffix_length]
                # Pad with zeros if shorter than max_suffix_length
                while len(suffix_activities) < max_suffix_length:
                    suffix_activities.append(0)
            else:
                # End of case - no suffix
                suffix_activities = [0] * max_suffix_length
            
            suffix_targets.append(suffix_activities)
        
        return pd.Series(suffix_targets, index=prefixes_df.index)
    
    def build_vocabulary(self, all_activity_tokens: List[str]) -> Vocabulary:
        """Build vocabulary from all activity tokens across datasets."""
        all_activity_tokens.append(self.config["data"]["end_of_case_token"])
        return Vocabulary(all_activity_tokens)
    
    def create_model(self, vocab_size: int):
        """Create the suffix prediction model using transformer models."""
        from ..models.model_registry import create_model
        return create_model(
            name="process_transformer",
            task="suffix",
            vocab_size=vocab_size,
            max_case_length=self.config["model"].get("max_case_length", 50),
            embed_dim=self.config["model"].get("embed_dim", 36),
            num_heads=self.config["model"].get("num_heads", 4),
            ff_dim=self.config["model"].get("ff_dim", 64)
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
                filename='suffix-model-{epoch:02d}-{val_loss:.2f}',
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
    
    def evaluate_model(self, model, test_dataloader) -> Dict[str, float]:
        """Evaluate the model and compute metrics."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_sequences, target_suffixes = batch
                if next(model.parameters()).is_cuda:
                    input_sequences = input_sequences.cuda()
                    target_suffixes = target_suffixes.cuda()
                
                logits = model(input_sequences)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target_suffixes.cpu().numpy())
        
        # Calculate normalized Damerau-Levenshtein distance
        distances = []
        for pred, target in zip(all_predictions, all_targets):
            # Convert indices to tokens (handle both string and integer indices)
            pred_tokens = []
            for i in pred:
                if i != 0 and i < len(self.vocabulary.index_to_token):
                    pred_tokens.append(self.vocabulary.index_to_token[i])
            
            target_tokens = []
            for i in target:
                if i != 0 and i < len(self.vocabulary.index_to_token):
                    target_tokens.append(self.vocabulary.index_to_token[i])
            
            distance = normalized_damerau_levenshtein(pred_tokens, target_tokens)
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        return {
            "normalized_damerau_levenshtein": avg_distance,
            "num_samples": len(distances)
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """
        Run the complete suffix prediction task.
        
        Args:
            datasets: List of dataset names to process
            raw_directory: Path to raw data directory
            outputs_dir: Path to save outputs
            
        Returns:
            Dictionary containing results for all datasets
        """
        print(f"Running Suffix Prediction Task for datasets: {datasets}")
        
        # Set up logging
        logger = create_logger(self.config, outputs_dir)

        # Aggregate vocab across all configured datasets
        all_activity_tokens = []
        dataset_cache = {}
        
        # Set up processed directory
        processed_directory = Path(self.config["data"]["path_processed"])
        
        # Check if force preprocessing is requested
        force_reprocess = self.config.get("force_preprocess", False)
        
        for dataset_name in datasets:
            print(f"Loading dataset: {dataset_name}")
            prefixes_dataframe, suffix_targets, vocabulary = self.prepare_data(
                dataset_name, raw_directory, processed_directory, force_reprocess
            )
            dataset_cache[dataset_name] = (prefixes_dataframe, suffix_targets)
            all_activity_tokens.extend(prefixes_dataframe["prefix_activities"].explode().tolist())
        
        # Build vocabulary
        self.vocabulary = self.build_vocabulary(all_activity_tokens)
        print(f"Vocabulary size: {len(self.vocabulary.index_to_token)}")
        
        # Train and evaluate per dataset
        results = {}
        
        for dataset_name, (prefixes_dataframe, suffix_targets) in dataset_cache.items():
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            
            # Split data
            split_index = int(0.8 * len(prefixes_dataframe))
            train_prefixes = prefixes_dataframe.iloc[:split_index].reset_index(drop=True)
            train_suffixes = suffix_targets.iloc[:split_index].reset_index(drop=True)
            val_prefixes = prefixes_dataframe.iloc[split_index:].reset_index(drop=True)
            val_suffixes = suffix_targets.iloc[split_index:].reset_index(drop=True)
            
            # Create data module
            data_module = SuffixPrefixDataModule(
                train_prefix_df=train_prefixes,
                train_labels=train_suffixes,
                val_prefix_df=val_prefixes,
                val_labels=val_suffixes,
                vocabulary=self.vocabulary,
                batch_size=self.config["train"]["batch_size"]
            )
            
            # Create model
            model = self.create_model(len(self.vocabulary.index_to_token))
            
            # Create trainer
            checkpoint_dir = outputs_dir / "checkpoints" / dataset_name / "suffix"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer = self.create_trainer(checkpoint_dir)
            
            # Train model
            trainer.fit(model, data_module)
            
            # Evaluate model
            test_results = trainer.test(model, data_module.test_dataloader())
            
            # Additional evaluation with our custom metrics
            custom_metrics = self.evaluate_model(model, data_module.test_dataloader())
            
            results[dataset_name] = {
                "test_loss": test_results[0]["test_loss"],
                **custom_metrics
            }
            
            print(f"Results for {dataset_name}: {results[dataset_name]}")
        
        # Save results
        results_file = outputs_dir / "suffix_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Suffix prediction results saved to {results_file}")
        return results
