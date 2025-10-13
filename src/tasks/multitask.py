import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..data.loader import CanonicalLogsDataLoader
from ..data.preprocessor import SimplePreprocessor
from ..data.encoders import Vocabulary
from ..metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from ..utils.cross_validation import CanonicalCrossValidation, aggregate_cv_results


class MultiTaskLearningTask:
    """
    Joint training over next_activity, next_time, remaining_time using a single multi-output model.
    - Does NOT change canonical preprocessing, labels, or splits.
    - For each fold, we form the training/validation sets by intersecting the case sets
      across the three single-task canonical splits to prevent leakage and maintain comparability.
    - Metrics are reported per task using the same definitions as single-task runs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary: Vocabulary | None = None
        self.current_dataset: str | None = None

    def _ensure_processed(self, dataset_name: str, raw_directory: Path, processed_directory: Path, force_reprocess: bool = False):
        preprocessor = SimplePreprocessor(raw_directory, processed_directory, self.config)
        if not preprocessor.is_processed(dataset_name) or force_reprocess:
            preprocessor.preprocess_dataset(dataset_name, force_reprocess)

    def _load_fold_dfs(self, loader: CanonicalLogsDataLoader, fold_idx: int) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        tasks = ["next_activity", "next_time", "remaining_time"]
        data = {}
        for t in tasks:
            train_df, val_df = loader.load_fold_data(t, fold_idx)
            data[t] = (train_df, val_df)
        return data

    def _intersect_by_case(self, dfs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]):
        # Compute common case sets for train and val
        train_sets = [set(df[0]['case_id'].unique()) for df in dfs.values()]
        val_sets = [set(df[1]['case_id'].unique()) for df in dfs.values()]
        common_train_cases = set.intersection(*train_sets) if train_sets else set()
        common_val_cases = set.intersection(*val_sets) if val_sets else set()

        filtered = {}
        for t, (train_df, val_df) in dfs.items():
            ftr = train_df[train_df['case_id'].isin(common_train_cases)].copy()
            fvl = val_df[val_df['case_id'].isin(common_val_cases)].copy()
            # Canonical stable ordering to align across tasks
            if 'prefix' in ftr.columns:
                ftr = ftr.sort_values(['case_id', 'prefix']).reset_index(drop=True)
                fvl = fvl.sort_values(['case_id', 'prefix']).reset_index(drop=True)
            else:
                ftr = ftr.sort_values(['case_id']).reset_index(drop=True)
                fvl = fvl.sort_values(['case_id']).reset_index(drop=True)
            filtered[t] = (ftr, fvl)
        return filtered

    def _prepare_arrays(self, loader: CanonicalLogsDataLoader, ta_df: pd.DataFrame, tt_df: pd.DataFrame, tr_df: pd.DataFrame, max_case_length: int):
        # Align rows from the three task-specific DataFrames by case_id and prefix
        base = ta_df[["case_id", "prefix", "next_act"]].copy()
        tt_cols = ["case_id", "prefix", "recent_time", "latest_time", "time_passed", "next_time"]
        tr_cols = ["case_id", "prefix", "remaining_time_days"]
        tt_sel = tt_df[tt_cols].copy()
        tr_sel = tr_df[tr_cols].copy()
        merged = base.merge(tt_sel, on=["case_id", "prefix"], how="inner").merge(tr_sel, on=["case_id", "prefix"], how="inner")
        # Stable ordering
        merged = merged.sort_values(["case_id", "prefix"]).reset_index(drop=True)

        # Next Activity tokens and labels
        xa = merged["prefix"].values
        ya = merged["next_act"].values
        token_x = []
        for _x in xa:
            token_x.append([loader.x_word_dict[s] for s in _x.split()])
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length, padding='post', truncating='post')
        y_word_dict = loader.y_word_dict
        token_ya = np.array([y_word_dict[_y] for _y in ya], dtype=np.int32)

        # Time features and targets (raw, as in single-task runners)
        time_feats = merged[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        yn = merged["next_time"].values.astype(np.float32)
        yr = merged["remaining_time_days"].values.astype(np.float32)
        yn = yn.reshape(-1, 1)
        yr = yr.reshape(-1, 1)

        # MTLFormer expects two token inputs and one time input
        inputs1 = np.asarray(token_x, dtype=np.int32)
        inputs2 = np.asarray(token_x, dtype=np.int32)
        time_inputs = np.asarray(time_feats, dtype=np.float32)
        return inputs1, inputs2, time_inputs, token_ya, yn, yr

    def _create_model(self, vocab_size: int, max_case_length: int):
        from ..models.model_registry import create_model
        # Resolve per-model hyperparameters with fallback to global model.*
        _m = self.config.get("model", {})
        _name = _m.get("name", "mtlformer")
        _pm = (_m.get("per_model", {}) or {}).get(_name, {})
        embed_dim = _pm.get("embed_dim", _m.get("embed_dim", 36))
        num_heads = _pm.get("num_heads", _m.get("num_heads", 4))
        ff_dim = _pm.get("ff_dim", _m.get("ff_dim", 64))
        return create_model(
            name=_name,
            task="multitask",
            vocab_size=vocab_size,
            max_case_length=max_case_length,
            output_dim=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )

    def _compile_model(self, model: keras.Model):
        loss_weights = self.config.get("model", {}).get("loss_weights", {"out1": 1.0, "out2": 1.0, "out3": 1.0})
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["train"]["learning_rate"]),
            loss={
                'out1': keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                'out2': 'mse',
                'out3': 'mse'
            },
            loss_weights=loss_weights,
            metrics={
                'out1': ['accuracy'],
                'out2': ['mae'],
                'out3': ['mae']
            }
        )

    def train_and_evaluate_fold(self, loader: CanonicalLogsDataLoader, fold_idx: int) -> Dict[str, Any]:
        # Load fold data from each single-task split and intersect cases
        dfs = self._load_fold_dfs(loader, fold_idx)
        dfs = self._intersect_by_case(dfs)
        ta_tr, ta_val = dfs['next_activity']
        tt_tr, tt_val = dfs['next_time']
        tr_tr, tr_val = dfs['remaining_time']
        # After intersection and sorting, prefixes should align. We'll use next_activity frames as base.
        train_df = ta_tr
        val_df = ta_val

        # Vocabulary and model dims
        vocab_size = loader.vocab_size
        observed_max = loader.get_max_case_length(train_df)
        cfg_data_max = self.config.get("data", {}).get("max_prefix_length")
        cfg_model_max = self.config.get("model", {}).get("max_case_length")
        limits = [v for v in [observed_max, cfg_data_max, cfg_model_max] if v is not None]
        max_case_length = min(limits) if limits else observed_max

        model = self._create_model(vocab_size, max_case_length)
        self._compile_model(model)

        # Prepare arrays
        X1_tr, X2_tr, T_tr, Ya_tr, Yn_tr, Yr_tr = self._prepare_arrays(loader, ta_tr, tt_tr, tr_tr, max_case_length)
        X1_va, X2_va, T_va, Ya_va, Yn_va, Yr_va = self._prepare_arrays(loader, ta_val, tt_val, tr_val, max_case_length)

        # Train
        callbacks = []
        if self.config.get("train", {}).get("early_stopping_patience", None) is not None:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=self.config.get("train", {}).get("early_stopping_monitor", "val_loss"),
                patience=self.config.get("train", {}).get("early_stopping_patience", 0),
                min_delta=self.config.get("train", {}).get("early_stopping_min_delta", 0.0),
                mode=self.config.get("train", {}).get("early_stopping_mode", "min"),
                restore_best_weights=self.config.get("train", {}).get("restore_best_weights", True)
            ))

        import time
        train_start = time.time()
        history = model.fit(
            [X1_tr, X2_tr, T_tr], [Ya_tr, Yn_tr, Yr_tr],
            validation_data=([X1_va, X2_va, T_va], [Ya_va, Yn_va, Yr_va]),
            epochs=self.config["train"]["max_epochs"],
            batch_size=self.config["train"]["batch_size"],
            callbacks=callbacks if callbacks else None,
            verbose=1
        )
        train_time = float(time.time() - train_start)

        # Evaluate with consistent metrics
        infer_start = time.time()
        preds = model.predict([X1_va, X2_va, T_va], verbose=0)
        infer_time = float(time.time() - infer_start)
        pred_a, pred_nt, pred_rt = preds

        # Activity accuracy (argmax over logits)
        y_true_a = Ya_va.reshape(-1)
        y_hat_a = np.argmax(pred_a, axis=-1).reshape(-1)
        acc = float(accuracy_score(y_true_a, y_hat_a))

        # Time metrics on raw scale (consistent with single-task tasks)
        y_true_nt = Yn_va.reshape(-1)
        y_hat_nt = pred_nt.reshape(-1)
        y_true_rt = Yr_va.reshape(-1)
        y_hat_rt = pred_rt.reshape(-1)
        nt_mae = float(mean_absolute_error(y_true_nt, y_hat_nt))
        nt_mse = float(mean_squared_error(y_true_nt, y_hat_nt))
        nt_r2 = float(r2_score(y_true_nt, y_hat_nt))
        rt_mae = float(mean_absolute_error(y_true_rt, y_hat_rt))
        rt_mse = float(mean_squared_error(y_true_rt, y_hat_rt))
        rt_r2 = float(r2_score(y_true_rt, y_hat_rt))

        # Determine epochs run from history
        try:
            h = getattr(history, 'history', {}) or {}
            epochs_run = int(len(h.get('loss', []))) if h else None
        except Exception:
            epochs_run = None

        # Compose per-task metrics
        metrics = {
            'next_activity_accuracy': acc,
            'next_time_mae': nt_mae,
            'next_time_mse': nt_mse,
            'next_time_r2': nt_r2,
            'remaining_time_mae': rt_mae,
            'remaining_time_mse': rt_mse,
            'remaining_time_r2': rt_r2,
            'param_count': int(model.count_params()),
            'train_time_sec': float(train_time),
            'infer_time_sec': float(infer_time),
            'epochs_run': epochs_run,
        }
        return metrics

    def run(self, datasets: List[str], raw_directory: Path, outputs_dir: Path) -> Dict[str, Any]:
        results = {}
        processed_directory = Path(self.config["data"]["path_processed"])
        force_reprocess = self.config.get("force_preprocess", False)

        for dataset_name in datasets:
            print(f"\n=== Processing Dataset (multitask): {dataset_name} ===")
            self.current_dataset = dataset_name
            self._ensure_processed(dataset_name, raw_directory, processed_directory, force_reprocess)
            loader = CanonicalLogsDataLoader(dataset_name, str(processed_directory))

            # CV folds are derived from per-task canonical splits; we iterate over fold indices based on one task
            splits_info_na = loader.get_splits_info("next_activity")
            splits_info_nt = loader.get_splits_info("next_time")
            splits_info_rt = loader.get_splits_info("remaining_time")
            n_folds = (splits_info_na.get('n_folds')
                       or self.config['cv'].get('n_folds', 5) or 5)

            fold_metrics = []
            for fold_idx in range(n_folds):
                fold_m = self.train_and_evaluate_fold(loader, fold_idx)
                fold_metrics.append(fold_m)
                print(f"Fold {fold_idx+1} metrics: {fold_m}")

            # Aggregate: simple mean/std like aggregate_cv_results
            def agg(values):
                arr = np.array(values, dtype=float)
                return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), [float(v) for v in arr]

            aggregated = {}
            for key in fold_metrics[0].keys():
                if isinstance(fold_metrics[0][key], (int, float)):
                    vals = [fm[key] for fm in fold_metrics]
                    m, s, mn, mx, lst = agg(vals)
                    aggregated[f"{key}_mean"] = m
                    aggregated[f"{key}_std"] = s
                    aggregated[f"{key}_min"] = mn
                    aggregated[f"{key}_max"] = mx
                    aggregated[f"{key}_values"] = lst

            # Persist multitask aggregate
            model_name = self.config.get("model", {}).get("name", "mtlformer")
            out_dir = outputs_dir / dataset_name / "multitask" / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "cv_metrics.json").write_text(json.dumps({
                'fold_metrics': fold_metrics,
                'cv_summary': aggregated,
                'dataset_name': dataset_name,
            }, indent=2))

            
            # Additionally, emit per-task reports to match single-task layout for comparability
            # Build per-task fold results and write fold-level metrics.json
            # Next Activity
            na_fold_results = []
            for i, fm in enumerate(fold_metrics):
                fold_dir = outputs_dir / dataset_name / "next_activity" / model_name / f"fold_{i}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                na_metrics = {
                    'accuracy': float(fm['next_activity_accuracy']),
                    'train_time_sec': float(fm.get('train_time_sec')) if fm.get('train_time_sec') is not None else None,
                    'infer_time_sec': float(fm.get('infer_time_sec')) if fm.get('infer_time_sec') is not None else None,
                    'param_count': int(fm.get('param_count')) if fm.get('param_count') is not None else None,
                    'epochs_run': int(fm.get('epochs_run')) if fm.get('epochs_run') is not None else None,
                }
                (fold_dir / "metrics.json").write_text(json.dumps(na_metrics, indent=2))
                na_fold_results.append({'fold_idx': i, 'metrics': na_metrics})
            na_cv = aggregate_cv_results(na_fold_results)
            na_dir = outputs_dir / dataset_name / "next_activity" / model_name
            na_dir.mkdir(parents=True, exist_ok=True)
            (na_dir / "cv_results.json").write_text(json.dumps({
                'fold_results': na_fold_results,
                'cv_summary': na_cv,
                'dataset_name': dataset_name,
            }, indent=2))

            # Next Time
            nt_fold_results = []
            for i, fm in enumerate(fold_metrics):
                fold_dir = outputs_dir / dataset_name / "next_time" / model_name / f"fold_{i}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                nt_metrics = {
                    'mae': float(fm['next_time_mae']),
                    'mse': float(fm['next_time_mse']),
                    'r2': float(fm['next_time_r2']),
                    'train_time_sec': float(fm.get('train_time_sec')) if fm.get('train_time_sec') is not None else None,
                    'infer_time_sec': float(fm.get('infer_time_sec')) if fm.get('infer_time_sec') is not None else None,
                    'param_count': int(fm.get('param_count')) if fm.get('param_count') is not None else None,
                    'epochs_run': int(fm.get('epochs_run')) if fm.get('epochs_run') is not None else None,
                }
                (fold_dir / "metrics.json").write_text(json.dumps(nt_metrics, indent=2))
                nt_fold_results.append({'fold_idx': i, 'metrics': nt_metrics})
            nt_cv = aggregate_cv_results(nt_fold_results)
            nt_dir = outputs_dir / dataset_name / "next_time" / model_name
            nt_dir.mkdir(parents=True, exist_ok=True)
            (nt_dir / "cv_results.json").write_text(json.dumps({
                'fold_results': nt_fold_results,
                'cv_summary': nt_cv,
                'dataset_name': dataset_name,
            }, indent=2))

            # Remaining Time
            rt_fold_results = []
            for i, fm in enumerate(fold_metrics):
                fold_dir = outputs_dir / dataset_name / "remaining_time" / model_name / f"fold_{i}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                rt_metrics = {
                    'mae': float(fm['remaining_time_mae']),
                    'mse': float(fm['remaining_time_mse']),
                    'r2': float(fm['remaining_time_r2']),
                    'train_time_sec': float(fm.get('train_time_sec')) if fm.get('train_time_sec') is not None else None,
                    'infer_time_sec': float(fm.get('infer_time_sec')) if fm.get('infer_time_sec') is not None else None,
                    'param_count': int(fm.get('param_count')) if fm.get('param_count') is not None else None,
                    'epochs_run': int(fm.get('epochs_run')) if fm.get('epochs_run') is not None else None,
                }
                (fold_dir / "metrics.json").write_text(json.dumps(rt_metrics, indent=2))
                rt_fold_results.append({'fold_idx': i, 'metrics': rt_metrics})
            rt_cv = aggregate_cv_results(rt_fold_results)
            rt_dir = outputs_dir / dataset_name / "remaining_time" / model_name
            rt_dir.mkdir(parents=True, exist_ok=True)
            (rt_dir / "cv_results.json").write_text(json.dumps({
                'fold_results': rt_fold_results,
                'cv_summary': rt_cv,
                'dataset_name': dataset_name,
            }, indent=2))

            results[dataset_name] = {
                'fold_metrics': fold_metrics,
                'cv_summary': aggregated,
            }
        return results
