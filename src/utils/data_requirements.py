from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from omegaconf import DictConfig, OmegaConf

from ..data.preprocessor import SimplePreprocessor


def _has_extended_columns(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ["resource_prefix", "delta_t_prefix"])


def _load_any_fold(processed_dir: Path, dataset: str, task: str) -> pd.DataFrame | None:
    fold0 = processed_dir / dataset / "splits" / task / "fold_0"
    train = fold0 / "train.csv"
    val = fold0 / "val.csv"
    try:
        if train.exists() and val.exists():
            # Prefer train.csv to check schema
            return pd.read_csv(train)
    except Exception:
        return None
    return None


def validate_datasets_or_raise(config: DictConfig, project_root: Path) -> None:
    """Validate that the correct processed datasets exist for the selected model/task.

    Rules implemented:
    - Models that use only activities/time (process_transformer, mtlformer, activity_only_lstm):
      Accept canonical processed datasets regardless of presence of extended columns.
    - Extended next-activity models (shared_lstm, specialised_lstm):
      Require extended columns in canonical next_activity splits. If missing, raise
      with instructions to force preprocessing.
    - PGTNet (remaining_time only):
      Requires canonical remaining_time splits to exist; its own graph conversion
      cache lives under data/processed_pgtnet and is orchestrated by the runner.
    """
    model = str(getattr(config.model, "name", "process_transformer"))
    task = str(getattr(config, "task", "next_activity"))

    processed_dir = project_root / getattr(config.data, "path_processed", "data/processed")
    datasets: List[str] = list(getattr(config.data, "datasets", []) or [])

    pre = SimplePreprocessor(project_root / getattr(config.data, "path_raw", "data/raw"), processed_dir, OmegaConf.to_container(config, resolve=True))

    # First, ensure canonical base artifacts exist where expected
    for ds in datasets:
        if not pre.is_processed(ds):
            # Let the task layer trigger preprocess as designed, but provide a clear error upfront
            raise FileNotFoundError(
                f"Canonical processed data not found for dataset '{ds}' under {processed_dir}. "
                f"Run: uv run python -m src.cli preprocess_action=force data.datasets=\"[\"{ds}\"]\""
            )

    # Extended models require extended columns
    if model in {"shared_lstm", "specialised_lstm"}:
        if task != "next_activity":
            # These models are only used for next_activity; no further checks needed here.
            return
        for ds in datasets:
            df = _load_any_fold(processed_dir, ds, task="next_activity")
            if df is None or not _has_extended_columns(df):
                raise RuntimeError(
                    "Extended attribute columns not found in canonical next_activity splits for dataset '"
                    + ds + "'. Please rebuild canonical processed data to include extended attributes.\n"
                    "Try: uv run python -m src.cli preprocess_action=force data.datasets=\"[\"" + ds + "\"]\""
                )

    # Minimal-mode models need nothing else; extended columns are optional and must be ignored by adapters
    return
