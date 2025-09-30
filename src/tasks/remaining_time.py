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
from ..metrics import mean_absolute_error, mean_squared_error, r2_score
from ..utils.logging import create_logger
from ..utils.cross_validation import run_canonical_cross_validation
# External PGTNet integration
try:
    from ..models.pgtnet_external_runner import PGTNetExternalRunner
    from ..data.pgtnet_preprocessor import PGTNetPreprocessor
except Exception:
    # Allow the file to be importable even if optional modules are missing
    PGTNetExternalRunner = None
    PGTNetPreprocessor = None


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
        self.current_dataset = None

    # ------------------------- PGTNet external path -------------------------
    def _run_pgtnet(self, datasets: List[str], outputs_dir: Path) -> Dict[str, Any]:
        if PGTNetExternalRunner is None or PGTNetPreprocessor is None:
            raise RuntimeError("PGTNet integration not available: missing optional modules or files.")
        results: Dict[str, Any] = {}
        model_cfg = self.config.get("model", {}) or {}
        # Merge base model cfg with optional nested model.pgtnet overrides for flags
        base_model = dict(model_cfg)
        nested_over = dict((model_cfg.get("pgtnet", {}) or {}))
        def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(a)
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        merged_cfg = _deep_merge(base_model, nested_over)
        execute = bool(merged_cfg.get("execute", False))
        auto_collect = bool(merged_cfg.get("auto_collect_predictions", False))
        n_folds = int(self.config.get("cv", {}).get("n_folds", 5))
        seeds = model_cfg.get("seeds") or [self.config.get("seed", 42)]
        seed = int(seeds[0])
        for dataset_name in datasets:
            self.current_dataset = dataset_name
            print(f"\n=== PGTNet Integration: {dataset_name} ===")
            # Prepare dirs and mapping
            pre = PGTNetPreprocessor(dataset_name, self.config)
            pre.ensure_conversion_workspace()
            mapping_path = pre.ensure_mapping()
            # Orchestrate folds
            fold_results = []
            manifest_paths: List[Dict[str, str]] = []
            dataset_failed = False
            for fold_idx in range(n_folds):
                _ = pre.write_fold_masks(fold_idx)
                runner = PGTNetExternalRunner(dataset_name, self.config)
                manifest = runner.run_full_fold(fold_idx=fold_idx, seed=seed, execute=execute)
                # Compute or stub metrics per fold
                fold_metrics = self._compute_pgtnet_fold_metrics(dataset_name, fold_idx, mapping_path, outputs_dir) if auto_collect else None
                # Persist also under canonical outputs path
                model_name = "pgtnet"
                model_dir = outputs_dir / dataset_name / "remaining_time" / model_name / f"fold_{fold_idx}"
                model_dir.mkdir(parents=True, exist_ok=True)
                if fold_metrics is not None:
                    with open(model_dir / "metrics.json", "w") as f:
                        json.dump(fold_metrics, f, indent=2)
                # Always dump the fold plan/manifest
                fold_manifest_path = model_dir / "run_manifest.json"
                with open(fold_manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                manifest_paths.append({
                    "fold": f"{fold_idx}",
                    "path": str(fold_manifest_path),
                })
                # failure detection for the fold
                calls = list(manifest.get("calls", []) or [])
                fold_failed = bool(manifest.get("overall_success") is False or any(int(c.get("returncode", 0) or 0) != 0 for c in calls)) if execute else False
                if fold_failed:
                    dataset_failed = True
                fold_results.append({
                    "fold_idx": fold_idx,
                    "metrics": fold_metrics or {"mae": None, "num_test_prefixes": 0},
                    "manifest": manifest,
                    "failed": fold_failed,
                })
                # Console summary for faster smoke-test feedback
                try:
                    if execute:
                        if calls:
                            def _fmt_call(c):
                                try:
                                    rc = int(c.get('returncode', 0) or 0)
                                except Exception:
                                    rc = -1
                                secs = int(c.get('seconds', 0) or 0)
                                name = c.get('name')
                                extra = f" stderr={c.get('stderr')}" if rc != 0 and c.get('stderr') else ""
                                return f"{name}:rc={rc} ({secs}s){extra}"
                            status = ", ".join([_fmt_call(c) for c in calls])
                            prefix = "FAIL" if fold_failed else "OK  "
                            print(f"  Fold {fold_idx} [{prefix}] external calls -> {status}")
                        else:
                            print(f"  Fold {fold_idx}: executed but no call logs captured (see outputs/pgtnet/{dataset_name}/fold_{fold_idx}/)")
                    else:
                        print(f"  Fold {fold_idx}: plan-only (no external execution). Manifest saved.")
                except Exception:
                    pass
            # Aggregate simple CV stats if metrics present
            have_vals = [fr for fr in fold_results if fr["metrics"] and isinstance(fr["metrics"].get("mae"), (int, float))]
            if have_vals:
                maes = [fr["metrics"]["mae"] for fr in have_vals]
                cv_summary = {
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "mae_min": float(np.min(maes)),
                    "mae_max": float(np.max(maes)),
                }
            else:
                cv_summary = {}
            # Write a dataset-level plan file to make smoke tests self-explanatory
            dataset_plan_dir = outputs_dir / dataset_name / "remaining_time" / "pgtnet"
            dataset_plan_dir.mkdir(parents=True, exist_ok=True)
            plan_payload = {
                "dataset": dataset_name,
                "task": "remaining_time",
                "model": "pgtnet",
                "execute": bool(execute),
                "auto_collect_predictions": bool(auto_collect),
                "n_folds": int(n_folds),
                "seed": int(seed),
                "runner_outputs": str(PGTNetExternalRunner(dataset_name, self.config).outputs_root),
                "manifests": manifest_paths,
                "had_failures": bool(dataset_failed) if execute else False,
                "notes": (
                    "Plan-only mode: no training/inference was executed. Set model.pgtnet.execute=true "
                    "to actually run PGTNet/GraphGPS and produce predictions/metrics."
                ) if not execute else (
                    "If predictions_test.csv were produced by the external repos, they will be auto-collected "
                    "into per-fold directories when auto_collect_predictions=true."
                ),
            }
            try:
                with open(dataset_plan_dir / "plan.json", "w") as f:
                    json.dump(plan_payload, f, indent=2)
            except Exception as e:
                print(f"Warning: failed to write dataset-level plan.json: {e}")
            results[dataset_name] = {
                "fold_results": fold_results,
                "cv_summary": cv_summary,
                "dataset_name": dataset_name,
                "had_failures": bool(dataset_failed) if execute else False,
            }
        return results

    def _compute_pgtnet_fold_metrics(self, dataset_name: str, fold_idx: int, mapping_path: Path, outputs_dir: Path) -> Dict[str, Any]:
        """Try to read predictions_test.csv and compute MAE. Returns a dict or None if unavailable."""
        # Prefer runner outputs under outputs/pgtnet/<dataset>/fold_k/
        runner_pred = Path("outputs/pgtnet") / dataset_name / f"fold_{fold_idx}" / "predictions_test.csv"
        # Fallback to model-scoped dir if user placed it there
        model_pred = outputs_dir / dataset_name / "remaining_time" / "pgtnet" / f"fold_{fold_idx}" / "predictions_test.csv"
        pred_path = runner_pred if runner_pred.exists() else (model_pred if model_pred.exists() else None)
        if pred_path is None or not pred_path.exists():
            return None
        try:
            preds = pd.read_csv(pred_path)
            mapping = pd.read_csv(mapping_path)
        except Exception as e:
            print(f"Warning: failed reading predictions or mapping for fold {fold_idx}: {e}")
            return None
        # Identify join keys
        join_keys = None
        for cand in [("case_id", "prefix_len"), ("case_id", "k")]:
            if all(c in preds.columns for c in cand):
                join_keys = cand
                break
        if join_keys is None:
            # If predictions include indices that match mapping.graph_idx, try that
            if "graph_idx" in preds.columns:
                merged = preds.merge(mapping[["graph_idx", "case_id", "prefix_len"]], on="graph_idx", how="left")
            else:
                print(f"Warning: predictions for fold {fold_idx} lack join keys; expected (case_id,prefix_len) or graph_idx")
                return None
        else:
            left_cols = list(join_keys)
            # Normalize types for robust join
            preds[left_cols[0]] = preds[left_cols[0]].astype(str)
            mapping["case_id"] = mapping["case_id"].astype(str)
            mapping.rename(columns={"prefix_len": left_cols[1]}, inplace=True)
            merged = preds.merge(mapping, on=list(join_keys), how="inner")
        # Heuristics to locate y_true / y_pred
        y_true_col = None
        for c in ["remaining_time_days", "y_true", "target", "label", "ground_truth", "remaining_time"]:
            if c in merged.columns:
                y_true_col = c
                break
        y_pred_col = None
        for c in ["y_pred", "prediction", "pred", "pred_remaining_time", "pred_remaining_time_days"]:
            if c in merged.columns:
                y_pred_col = c
                break
        if y_true_col is None or y_pred_col is None:
            print(f"Warning: could not find y_true/y_pred columns in predictions for fold {fold_idx}")
            return None
        try:
            y_true = merged[y_true_col].astype(float).values
            y_pred = merged[y_pred_col].astype(float).values
            mae = float(np.mean(np.abs(y_true - y_pred)))
            return {"mae": mae, "num_test_prefixes": int(len(y_true))}
        except Exception as e:
            print(f"Warning: failed to compute MAE for fold {fold_idx}: {e}")
            return None
        
    def prepare_data(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and prepare data for remaining time prediction using canonical pipeline."""
        # First ensure data is processed using canonical format
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            processing_info = preprocessor.preprocess_dataset(dataset_name, force_reprocess)
        
        # Now use the canonical loader
        data_loader = CanonicalLogsDataLoader(dataset_name, str(processed_directory))
        
        # Load all data for the task (we'll split by folds later)
        # For now, we'll load fold 0 to get the structure
        train_df, val_df = data_loader.load_fold_data("remaining_time", 0)
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
    
    def create_model(self, vocab_size: int, max_case_length: int = None, output_dim: int = 1):
        """Create the remaining time prediction model using transformer models."""
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
        return create_model(
            name=_name,
            task="remaining_time",
            vocab_size=vocab_size,
            max_case_length=max_case_length,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
    
    def create_trainer(self, model, checkpoint_dir: Path = None):
        """Create training setup for both TensorFlow and PyTorch models."""
        if isinstance(model, keras.Model):
            # TensorFlow model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config["train"]["learning_rate"]),
                loss='mse',
                metrics=['mae']
            )
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
        metrics_dir = Path("outputs") / self.current_dataset / "remaining_time" / model_name / f"fold_{fold_idx}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vocabulary from metadata (recreate per fold due to fresh task instance)
        from ..data.loader import CanonicalLogsDataLoader
        loader = CanonicalLogsDataLoader(self.current_dataset, str(Path(self.config["data"]["path_processed"])))
        x_word_dict = loader.x_word_dict
        # Initialize vocabulary
        self.vocabulary = Vocabulary(list(x_word_dict.keys()))
        
        # Determine model dims
        vocab_size = len(self.vocabulary.index_to_token)
        observed_max = loader.get_max_case_length(train_df)
        cfg_data_max = self.config.get("data", {}).get("max_prefix_length")
        cfg_model_max = self.config.get("model", {}).get("max_case_length")
        limits = [v for v in [observed_max, cfg_data_max, cfg_model_max] if v is not None]
        max_case_length = min(limits) if limits else observed_max
        
        model = self.create_model(vocab_size, max_case_length, output_dim=1)
        
        # Create trainer
        trainer = self.create_trainer(model, checkpoint_dir)
        
        # Train model
        import time, json as _json
        start_time = time.time()
        if isinstance(trainer, keras.Model):
            # TensorFlow training
            # Prepare data
            x_train, time_x_train, y_train = self._prepare_tensorflow_data(train_df, max_case_length)
            x_val, time_x_val, y_val = self._prepare_tensorflow_data(val_df, max_case_length)
            
            # Train
            callbacks = []
            if getattr(self, "_tf_early_stopping", None) is not None:
                callbacks.append(self._tf_early_stopping)
            history = trainer.fit(
                [x_train, time_x_train], y_train,
                validation_data=([x_val, time_x_val], y_val),
                epochs=self.config["train"]["max_epochs"],
                batch_size=self.config["train"]["batch_size"],
                callbacks=callbacks if callbacks else None,
                verbose=1
            )
            train_time = time.time() - start_time
            
            # Evaluate
            eval_start = time.time()
            val_loss, val_mae = trainer.evaluate([x_val, time_x_val], y_val, verbose=0)
            y_pred = trainer.predict([x_val, time_x_val], verbose=0)
            infer_time = time.time() - eval_start
            
            # Calculate metrics (ensure matching 1D shapes)
            y_true = np.asarray(y_val).reshape(-1)
            y_hat = np.asarray(y_pred).reshape(-1)
            mae = float(mean_absolute_error(y_true, y_hat))
            mse = float(mean_squared_error(y_true, y_hat))
            r2 = float(r2_score(y_true, y_hat))
            param_count = int(trainer.count_params())
            
            # Early-stopping transparency (TF/Keras)
            es_cfg = self.config.get("train", {})
            monitor = es_cfg.get("early_stopping_monitor", "val_loss")
            mode = es_cfg.get("early_stopping_mode", "min")
            min_delta = float(es_cfg.get("early_stopping_min_delta", 0.0))
            patience = es_cfg.get("early_stopping_patience", None)

            history_dict = getattr(history, 'history', {}) or {}
            series_key = monitor if monitor in history_dict else (
                'val_loss' if 'val_loss' in history_dict else (
                'val_mae' if 'val_mae' in history_dict else (next(iter(history_dict.keys())) if history_dict else None)))
            series = history_dict.get(series_key, []) if series_key else []
            epochs_run = int(len(series)) if series else int(len(history_dict.get('loss', [])))

            best_epoch_idx = None
            if series:
                if mode == 'min':
                    best_epoch_idx = int(np.argmin(series))
                else:
                    best_epoch_idx = int(np.argmax(series))
            early_stopped = False
            if getattr(self, "_tf_early_stopping", None) is not None:
                try:
                    stopped_epoch = int(getattr(self._tf_early_stopping, 'stopped_epoch', 0) or 0)
                except Exception:
                    stopped_epoch = 0
                early_stopped = stopped_epoch > 0 or (epochs_run < int(self.config['train']['max_epochs']))

            metrics = {
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'train_time_sec': float(train_time),
                'infer_time_sec': float(infer_time),
                'param_count': param_count,
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
            from ..training.datamodule import CanonicalRemainingTimeDataModule
            
            # Create data module
            datamodule = CanonicalRemainingTimeDataModule(
                dataset_name=self.current_dataset,
                task="remaining_time",
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
            val_mae = float(res0.get('val_mae', 0.0))
            mse = float(res0.get('val_mse', 0.0))
            r2 = float(res0.get('val_r2', 0.0))
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

            try:
                epochs_run = int(getattr(trainer, 'current_epoch', 0)) + 1
            except Exception:
                epochs_run = None

            early_stopped = None
            try:
                early_stopped = bool(getattr(trainer, 'should_stop', False)) or (
                    epochs_run is not None and epochs_run < int(self.config['train']['max_epochs'])
                )
            except Exception:
                early_stopped = epochs_run is not None and epochs_run < int(self.config['train']['max_epochs'])

            best_epoch = None
            try:
                for cb in getattr(trainer, 'callbacks', []) or []:
                    if hasattr(cb, 'best_model_path') and cb.best_model_path:
                        import re
                        m = re.search(r"model-(\d+)-", cb.best_model_path)
                        if m:
                            best_epoch = int(m.group(1))
                            break
            except Exception:
                best_epoch = None

            metrics = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'mae': val_mae,  # For compatibility
                'mse': mse,
                'r2': r2,
                'train_time_sec': float(train_time),
                'infer_time_sec': float(infer_time),
                'param_count': param_count,
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
    
    def _prepare_tensorflow_data(self, df: pd.DataFrame, max_case_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for TensorFlow training with fixed padding to max_case_length."""
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["remaining_time_days"].values.astype(np.float32)
        
        # Tokenize sequences
        token_x = []
        for _x in x:
            token_x.append([self.vocabulary.token_to_index[s] for s in _x.split()])
        
        # Pad sequences to the model's fixed max_case_length
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')
        
        return token_x, time_x, y
    
    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        """
        Run remaining time prediction task on multiple datasets.
        
        Args:
            datasets: List of dataset names to process
            raw_directory: Directory containing raw data
            outputs_dir: Directory to save outputs
            
        Returns:
            Dictionary with results for all datasets
        """
        # Initialize logger
        self.logger = create_logger("remaining_time_task", outputs_dir)

        # Special-case integration: PGTNet external runner
        model_cfg = self.config.get("model", {}) or {}
        if (model_cfg.get("name") == "pgtnet"):
            return self._run_pgtnet(datasets, outputs_dir)
        
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
                    task_name="remaining_time"
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
                model_dir = outputs_dir / dataset_name / "remaining_time" / model_name
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
