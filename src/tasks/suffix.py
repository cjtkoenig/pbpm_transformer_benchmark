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
        self.current_dataset = None
        
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
            prefix_length = row["k"]  # CRITICAL FIX: Use 'k' column from canonical preprocessing
            
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
    
    def create_trainer(self, checkpoint_dir: Path):
        """Create PyTorch Lightning trainer for suffix prediction."""
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
        trainer = Trainer(
            max_epochs=self.config["train"]["max_epochs"],
            accelerator=self.config["train"]["accelerator"],
            devices=self.config["train"]["devices"],
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        
        return trainer
    
    def evaluate_model(self, model, test_dataloader) -> Dict[str, float]:
        """Evaluate the model using normalized Damerau-Levenshtein distance."""
        model.eval()
        distances = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                
                # Convert outputs to predicted sequences
                predicted_sequences = torch.argmax(outputs, dim=-1)
                
                # Convert targets to sequences (remove padding)
                for pred_seq, target_seq in zip(predicted_sequences, targets):
                    # Remove padding (zeros) from sequences
                    pred_seq = pred_seq[pred_seq != 0].cpu().numpy()
                    target_seq = target_seq[target_seq != 0].cpu().numpy()
                    
                    # Calculate normalized Damerau-Levenshtein distance
                    distance = normalized_damerau_levenshtein(pred_seq, target_seq)
                    distances.append(distance)
        
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        return {
            "normalized_damerau_levenshtein": avg_distance,
            "num_samples": len(distances)
        }
    
    def train_and_evaluate_fold(self, train_prefixes: pd.DataFrame, train_labels: pd.Series,
                               val_prefixes: pd.DataFrame, val_labels: pd.Series, 
                               fold_idx: int) -> Dict[str, Any]:
        """
        Train and evaluate a model on a single fold.
        
        Args:
            train_prefixes: Training data prefixes
            train_labels: Training labels (suffix targets)
            val_prefixes: Validation data prefixes
            val_labels: Validation labels (suffix targets)
            fold_idx: Current fold index
            
        Returns:
            Dictionary with fold results
        """
        print(f"Training fold {fold_idx + 1}...")
        
        # Create data module for this fold
        data_module = SuffixPrefixDataModule(
            train_prefix_df=train_prefixes,
            train_labels=train_labels,
            val_prefix_df=val_prefixes,
            val_labels=val_labels,
            vocabulary=self.vocabulary,
            batch_size=self.config["train"]["batch_size"]
        )
        
        # Create model
        model = self.create_model(len(self.vocabulary.index_to_token))
        
        # Create checkpoint directory for this fold
        checkpoint_dir = Path(self.config.get("outputs_dir", "outputs")) / "checkpoints" / self.current_dataset / f"fold_{fold_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trainer
        trainer = self.create_trainer(checkpoint_dir)
        
        # Train model
        trainer.fit(model, data_module)
        
        # Evaluate model
        test_results = trainer.test(model, data_module.test_dataloader())
        
        # Additional evaluation with custom metrics
        custom_metrics = self.evaluate_model(model, data_module.test_dataloader())
        
        return {
            'fold_idx': fold_idx,
            'metrics': {
                'test_loss': test_results[0]["test_loss"],
                **custom_metrics
            },
            'train_samples': len(train_prefixes),
            'val_samples': len(val_prefixes)
        }
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """
        Run the complete suffix prediction task using 5-fold cross-validation.
        
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
        
        # Store outputs_dir in config for use in train_and_evaluate_fold
        self.config["outputs_dir"] = str(outputs_dir)

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
            # CRITICAL FIX: Use 'prefix' column from canonical preprocessing
            all_activity_tokens.extend(prefixes_dataframe["prefix"].str.split().explode().tolist())
        
        # Build vocabulary
        self.vocabulary = self.build_vocabulary(all_activity_tokens)
        print(f"Vocabulary size: {len(self.vocabulary.index_to_token)}")
        
        # Train and evaluate per dataset
        results = {}
        
        for dataset_name, (prefixes_dataframe, suffix_targets) in dataset_cache.items():
            print(f"\n=== Processing Dataset: {dataset_name} ===")
            self.current_dataset = dataset_name
            
            try:
                print(f"Total samples: {len(prefixes_dataframe)}")
                print(f"Vocabulary size: {len(self.vocabulary.index_to_token)}")
                
                # Run 5-fold cross-validation
                cv_results = run_cross_validation(
                    task_class=self.__class__,
                    config=self.config,
                    prefixes_df=prefixes_dataframe,
                    labels_series=suffix_targets,
                    vocabulary=self.vocabulary,
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
                results_file = outputs_dir / f"{dataset_name}_suffix_cv_results.json"
                with open(results_file, 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                if logger:
                    logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        return results
