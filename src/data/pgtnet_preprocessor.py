"""
PGTNetPreprocessor (external runner)

Responsibilities:
- Create a per-dataset workspace under data/processed_pgtnet/<dataset>.
- Ensure one-time conversion cache directory exists (actual conversion is handled by the external runner).
- Produce a mapping.csv that links canonical identities to PGTNet-side example identities:
  columns: graph_idx, case_id, prefix_len
  We derive this from the canonical remaining_time processed CSVs (across folds),
  ensuring unique rows by (case_id, k) and assigning a stable graph_idx.
- Write per-fold masks from canonical 5-fold case-based splits into
  data/processed_pgtnet/<dataset>/fold_<k>/masks.json with train/val/test case ID lists.

Notes:
- This preprocessor intentionally does not modify canonical artifacts.
- The actual graph conversion is expected to be performed by the PGTNet/GraphGPS pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import pandas as pd

from ..utils.cross_validation import CanonicalCrossValidation


class PGTNetPreprocessor:
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.config = config or {}
        self.processed_canonical = Path(self.config.get("data", {}).get("path_processed", "data/processed"))
        self.processed_pgtnet_root = Path("data/processed_pgtnet") / dataset_name
        self.converted_dir = self.processed_pgtnet_root / "converted"
        self.mapping_file = self.processed_pgtnet_root / "mapping.csv"

    # -------------- Public API --------------
    def ensure_conversion_workspace(self) -> None:
        """Ensure directory structure exists for PGTNet conversion cache.
        The conversion itself is handled by the external runner.
        """
        self.converted_dir.mkdir(parents=True, exist_ok=True)

    def ensure_mapping(self) -> Path:
        """Create mapping.csv if missing. The mapping links (case_id, k) to a stable graph_idx.
        Returns path to mapping.csv
        """
        self.processed_pgtnet_root.mkdir(parents=True, exist_ok=True)
        if self.mapping_file.exists():
            return self.mapping_file

        # Aggregate unique (case_id, k) pairs from canonical remaining_time splits across folds
        splits_dir = self._get_task_splits_dir("remaining_time")
        # If splits do not exist yet, we cannot build mapping; instruct the caller to preprocess canonically first
        if not splits_dir.exists():
            raise FileNotFoundError(
                f"Canonical splits not found at {splits_dir}. Run canonical preprocessing or the remaining_time task once to create splits."
            )

        pairs: Set[Tuple[str, int]] = set()
        # Find all folds
        for fold_dir in sorted(splits_dir.glob("fold_*")):
            train_csv = fold_dir / "train.csv"
            val_csv = fold_dir / "val.csv"
            for csv_path in [train_csv, val_csv]:
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if not {"case_id", "k"}.issubset(df.columns):
                    raise ValueError(f"Expected columns 'case_id' and 'k' in {csv_path}")
                for tup in df[["case_id", "k"]].itertuples(index=False, name=None):
                    # Normalize case_id to string for consistency
                    pairs.add((str(tup[0]), int(tup[1])))

        # Assign stable indices by sorting
        sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
        rows = [
            {"graph_idx": i, "case_id": cid, "prefix_len": k}
            for i, (cid, k) in enumerate(sorted_pairs)
        ]
        map_df = pd.DataFrame(rows, columns=["graph_idx", "case_id", "prefix_len"])  # stable column order
        map_df.to_csv(self.mapping_file, index=False)
        return self.mapping_file

    def write_fold_masks(self, fold_idx: int) -> Path:
        """Write masks.json with train/val/test case IDs for the given fold based on canonical splits.
        Returns masks.json path.
        """
        # Load canonical split metadata
        cv = CanonicalCrossValidation(
            n_folds=int(self.config.get("cv", {}).get("n_folds", 5)),
            random_state=int(self.config.get("seed", 42)),
        )
        # Ensure splits exist; use metadata to get case IDs
        splits_info = cv.get_splits_info(self.dataset_name, self.processed_canonical, task_name="remaining_time")
        if not splits_info:
            # Try to create splits by reading any one fold data via create_case_based_splits
            # We need a DF; attempt to assemble from any existing canonical fold CSVs.
            # If unavailable, raise a helpful error.
            candidate_dir = self._get_task_splits_dir("remaining_time")
            if not candidate_dir.exists():
                raise FileNotFoundError(
                    f"Canonical remaining_time splits not found for {self.dataset_name}. Run canonical preprocessing first."
                )
            # If metadata is missing but CSVs are present, we can reconstruct using case lists within the files
            # Fall back to reading the specific fold's CSVs
        # If metadata exists, proceed
        folds = splits_info.get("folds", {})
        key = f"fold_{fold_idx}"
        if key not in folds:
            raise ValueError(f"Fold {fold_idx} not present in canonical splits metadata for {self.dataset_name}.")
        fold_info = folds[key]

        fold_dir = self.processed_pgtnet_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        masks_path = fold_dir / "masks.json"
        train_case_ids: List[Any] = fold_info.get("train_case_ids", [])
        val_case_ids: List[Any] = fold_info.get("val_case_ids", [])

        masks = {
            "train_case_ids": [str(x) for x in train_case_ids],
            "val_case_ids": [str(x) for x in val_case_ids],
            # We use canonical val cases as test set for external evaluation
            "test_case_ids": [str(x) for x in val_case_ids],
        }
        with open(masks_path, "w") as f:
            json.dump(masks, f, indent=2)
        return masks_path

    # -------------- Helpers --------------
    def _get_task_splits_dir(self, task_name: str) -> Path:
        base = self.processed_canonical / self.dataset_name / "splits" / task_name
        return base
