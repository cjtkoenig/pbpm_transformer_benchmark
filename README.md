# PBPM Transformer Benchmark

A standardized benchmark for Predictive Business Process Monitoring (PBPM) with Transformer-based models. It follows the methodology of Rama‑Maneiro, Vidal & Lama (2021) with canonical preprocessing, strict case-based 5‑fold cross‑validation, and task‑specific metrics.

Table of Contents
- Overview
- Quick Start
- Installation and Environment
- Datasets and Preprocessing
- Smoke Tests
- Running Tasks
- Configuration (Hydra)
- Cross-Validation Protocol
- Models and Adapters
- Metrics
- Analysis and Reporting
- Hierarchical Bayes (enable PyMC/ArviZ)
- Make Targets
- Troubleshooting and Known Pitfalls
- Project Layout
- License

## Overview
- Canonical tasks: next activity, next time, remaining time.
  - multitask for MTLFormer only.
- Centralized preprocessing: prefixes, activity encoding, numeric normalization, End‑of‑Case token (<eoc>), cached under data/processed.
- Reproducible 5‑fold cross‑validation with persisted, case-based splits and fixed seeds.
- Adapters reshape data only; they must not change content, labels, vocabularies, or splits.
- Minimal environment snapshot is written to outputs/env.json for reproducibility.

## Quick Start

Prerequisites
- Python 3.11
- uv package manager

Fast path (recommended)
```bash
# One-command onboarding: create venv and install minimal deps
make setup
```
Notes
- Minimal install includes pandas 2.3.x, numpy 1.26.x, scikit-learn >=1.4,<1.6, scipy 1.16.x — sufficient for CLI and preprocessing utilities.
- PyTorch (for PyTorch-based utilities) runs on Apple Silicon with MPS when train.accelerator=mps.
- TensorFlow: pinned to 2.20.0 in requirements.txt (Python 3.11 compatible).


## Installation and Environment

Full installation
```bash
make install
# Or using uv directly
uv venv
uv pip install -r requirements.txt
```

Minimal manual install (no Make)
```bash
uv venv
uv run python -m ensurepip --upgrade || true
uv pip install pandas==2.3.1 numpy==1.26.4 "scikit-learn>=1.4,<1.6" scipy==1.16.1
```

Environment notes
- Python: 3.11
- Torch: 2.8.0 (MPS supported on Apple Silicon)
- TensorFlow: prefer >= 2.20.0 on Python 3.11; avoid 2.15.1 on macOS with Python 3.11.

### Note on PGTNet (removed from this repo)
PGTNet has been moved out of this repository due to its heavy third-party dependencies and specialized preprocessing pipeline. It now runs in its own dedicated repository/environment.

What changed
- All in-repo PGTNet code and Make targets have been removed.
- This repository focuses on the canonical in-repo models (ProcessTransformer, MTLFormer, and LSTMs).
- You can still include PGTNet results in analyses via the External Results Ingestion described below.

External PGTNet runs
- Run PGTNet in its own repository/environment following its instructions.
- Export metrics per dataset/task and drop them into outputs/external_results/ here to include them in aggregated tables with provenance labels such as: "PGTNet (external, splits=canonical)".

See: External Results Ingestion section below.

## Datasets and Preprocessing
- Place raw CSV logs in data/raw with at least columns case:concept:name, concept:name, and timestamps where required.
- Preprocessing runs on first use and caches under data/processed/.
- Two attribute modes: minimal (activities only) and extended (if model supports attributes).

Manage preprocessing
```bash
# Using Make (preferred)
make preprocess_info
make preprocess_force DATASETS="[\"Helpdesk\"]"
make preprocess_clear                  # clears all processed datasets
make preprocess_clear DATASET=Helpdesk # clears a single dataset

# Or using uv directly
uv run python -m src.cli preprocess_action=info
uv run python -m src.cli preprocess_action=force data.datasets="[Helpdesk]"
uv run python -m src.cli preprocess_action=clear
```

Note: You can also force preprocessing during a task run via Hydra override:
`uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]" force_preprocess=true`

Dataset-specific attribute handling (extended mode)
- Tourism: If column `event_log_activity_type` is present, it is treated as a categorical resource-like attribute by stringifying and normalizing tokens.
- Traffic Fines: Prefer `vehicleClass` (4 categories). It is forward-filled per case to all events and used as the resource-like attribute. If `vehicleClass` is absent, we fall back to `org:resource`.
  - Missing values are mapped to [UNK].
  - If you processed this dataset before this change, re-run preprocessing with:
    `uv run python -m src.cli preprocess_action=force data.datasets="[Traffic_Fines]"`

Missing values policy (all datasets)
- Core identifiers: Rows with missing Case ID, Activity, or unparseable timestamps are dropped during standardization with a warning message. This avoids injecting the literal string "nan" into vocabularies or labels.
- Activities: normalized to lowercase with spaces replaced by dashes. Missing activities are not allowed; affected events are removed before building prefixes and labels.
- Resources (extended mode): Missing categorical values are mapped to [UNK]. We normalize after filling missing values to ensure NaNs are not turned into the string "nan". For Traffic Fines, `vehicleClass` is forward-filled per case before normalization.
- Time features: Timestamps that cannot be parsed are dropped; delta-time sequences fill missing entries implicitly by construction (they are derived from valid timestamps only).

## Smoke Tests
These quick commands verify that your environment, data pipeline, and CLI wiring are set up correctly. They are lightweight and should run within seconds to a couple of minutes depending on your machine.

Preferred Make-based smoke tests
```bash
# Fast minimal smoke (next_activity on Helpdesk, uses defaults)
make smoke_test_minimal

# ProcessTransformer across its three tasks on Helpdesk (2 epochs each)
make smoke_test_process_transformer

# MTLFormer multitask on Helpdesk (2 epochs)
make smoke_test_mtlformer

# Include external results (PGTNet or others): see External Results Ingestion section below
```

Alternative: direct uv CLI
```bash
# Minimal end-to-end run via Hydra CLI; adjust accelerator as needed
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]" train.accelerator=cpu
# On Apple Silicon with Metal (optional):
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]" train.accelerator=mps
```
Notes: The first run will trigger preprocessing and write outputs/ and logs. TensorFlow >= 2.20.0 is recommended on Python 3.11 if training proceeds.

Preprocessing pipeline smoke
```bash
# Using Make (preferred)
make preprocess_info
make preprocess_force DATASETS="[\"Helpdesk\"]"

# Or using uv directly
uv run python -m src.cli preprocess_action=info
uv run python -m src.cli preprocess_action=force data.datasets="[Helpdesk]"
```
You should see summaries under data/processed and no errors. Use clear if you need a fresh state: `make preprocess_clear` or `uv run python -m src.cli preprocess_action=clear`.

## Full Benchmark Run (Step-by-Step)
This walkthrough shows the typical flow from setup to analysis. Adjust ACCELERATOR and dataset lists as needed.

1) Environment setup
- Minimal (fast; enough for preprocessing and CV tests):
  - make setup
- Full install (TF + all extras from requirements.txt):
  - make install
- Optional: Create a dedicated GraphGPS/PGTNet env for PGTNet:
  - make graphgps_env

2) Prepare data
- Place CSVs in data/raw (e.g., data/raw/Helpdesk.csv)
- or unlock them with git lfs:
  - git lfs install
  - git lfs pull (to fetch any LFS-tracked XES files)
- Inspect what’s cached:
  - make preprocess_info
- Force preprocessing for selected datasets (safe and fast for first time):
  - make preprocess_force DATASETS="[\"Helpdesk\"]"

3) Run baseline models
- ProcessTransformer (single tasks):
  - uv run python -m src.cli model.name=process_transformer task=next_activity data.datasets="[Helpdesk]"
  - uv run python -m src.cli model.name=process_transformer task=next_time data.datasets="[Helpdesk]"
  - uv run python -m src.cli model.name=process_transformer task=remaining_time data.datasets="[Helpdesk]"
- MTLFormer (multitask):
  - uv run python -m src.cli model.name=mtlformer task=multitask data.datasets="[Helpdesk]"

4) Include external results (optional)
- If you have results from external repos (e.g., PGTNet), place a JSON/CSV file under outputs/external_results/ using one of the accepted formats described in the External Results Ingestion section. These will be merged with in-repo results during analysis and labeled with provenance, e.g., "PGTNet (external, splits=canonical)".

5) Aggregate and analyze
- Quick stats across folds:
  - make analyze
- Full stats per task:
  - make analyze_full TASK=remaining_time

Tips
- For a quick overall dry run, see smoke tests: make smoke_test_process_transformer and make smoke_test_mtlformer.

## Running Tasks
```bash
# Next activity (ProcessTransformer)
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]" model.name=process_transformer

# Next time (ProcessTransformer)
uv run python -m src.cli task=next_time data.datasets="[Helpdesk]" model.name=process_transformer

# Remaining time (ProcessTransformer)
uv run python -m src.cli task=remaining_time data.datasets="[Helpdesk]" model.name=process_transformer

# Multitask (MTLFormer) — trains a single multi-output model and reports per-task metrics comparable to single-task runs
uv run python -m src.cli task=multitask data.datasets="[Helpdesk]" model.name=mtlformer
```

Safety constraints (enforced by CLI)
- ProcessTransformer supports only single tasks: next_activity, next_time, remaining_time. Using task=multitask will raise an error.
- MTLFormer supports only multitask. Using it on single tasks will raise an error.

Reporting for MTLFormer
- Although MTLFormer runs in multitask mode only, the runner writes per-task outputs so they are directly comparable with single-task models:
  - outputs/<dataset>/next_activity/mtlformer/cv_results.json
  - outputs/<dataset>/next_time/mtlformer/cv_results.json
  - outputs/<dataset>/remaining_time/mtlformer/cv_results.json
- This preserves the benchmark methodology and analysis compatibility.

## Configuration (Hydra)
Entry point: src/cli.py with configs under configs/. Canonical config is configs/benchmark.yaml. Override via Hydra dot-notation.

Examples
```bash
# Override model params
uv run python -m src.cli task=next_activity model.hidden_size=512 model.num_layers=6

# Override training params
uv run python -m src.cli task=next_activity train.max_epochs=20 train.batch_size=64

# Override data params
uv run python -m src.cli task=next_activity data.max_prefix_length=20 data.attribute_mode=extended

# Choose datasets
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk,Tourism]"
```

## Cross-Validation Protocol
- Strict case-based splits; no case appears in both train and validation of a fold.
- 5 folds persisted per dataset/task with a stable seed.
- Aggregation utilities compute mean/std/min/max across folds.

## Models and Adapters

### External models (PGTNet)
PGTNet is not part of this repository anymore. To compare against it, run PGTNet externally and ingest its results using the External Results Ingestion mechanism described above. This preserves comparability (shared canonical splits) without introducing heavy third-party code into this repo.

Adapters bridge canonical data and model I/O. They may reshape and cast, but must not change content.

Allowed
- Reshaping [B,T,D], generating attention masks from pad_id.
- One-hot ↔ index conversion; dtype changes; padding/clipping to max_prefix_len.

Not allowed
- Changing splits, re-encoding vocab/labels, altering normalization/time units, filtering/augmenting data.

Implemented models
Experiment Run 1: attribute_mode=minimal (activities only)
- process_transformer (TensorFlow)
- mtlformer (TensorFlow)
- specialised_lstm (TensorFlow; next_activity only)
- shared_lstm (TensorFlow; next_activity only)

## Metrics
- Next Activity: Accuracy (primary), optionally F1.
- Time predictions: MAE (primary). RMSE/R² optional.

## Analysis and Reporting
A minimal analysis utility exists at src/analysis/summary.py that aggregates per-fold metrics from outputs/ and produces a naïve ranking.

### External Results Ingestion (PGTNet and others)
You can include results produced outside this repository (e.g., PGTNet) in your aggregated reports. Place JSON/CSV files under outputs/external_results/ and they will be merged into analysis with clear provenance labels like "PGTNet (external, splits=canonical)".

Accepted formats
1) JSON list of records
- Example:
```
[
  {"dataset":"Helpdesk","task":"remaining_time","model":"PGTNet","metric":"mae","score":1.234,"splits":"canonical"},
  {"dataset":"Helpdesk","task":"remaining_time","model":"PGTNet","metric":"mae","fold":0,"value":1.20,"splits":"canonical"},
  {"dataset":"Helpdesk","task":"remaining_time","model":"PGTNet","metric":"mae","fold":1,"value":1.27,"splits":"canonical"}
]
```
2) Nested JSON dict
- Example:
```
{
  "model": "PGTNet",
  "splits": "canonical",
  "results": {
    "Helpdesk": {
      "remaining_time": {"metric": "mae", "score": 1.234, "folds": [1.20, 1.27, 1.25, 1.23, 1.21]}
    }
  }
}
```
3) CSV with columns: dataset, task, model, metric, score (optional), folds (JSON array) or (fold,value) rows, splits (optional)
- Example rows:
```
dataset,task,model,metric,score,splits
Helpdesk,remaining_time,PGTNet,mae,1.234,canonical
```

Notes
- Metrics: use accuracy for next_activity; MAE for time tasks. The ingester will invert MAE to the higher-is-better convention internally for ranking summaries; fold-level MAE values are kept as-is for confidence intervals.
- Labels: the ingester labels entries as "<Model> (external, splits=<label>)"; ensure your model field is "PGTNet" to appear under extended track in reports.
- Location: files must be placed under outputs/external_results/.

Quick analysis
```bash
make analyze
# writes outputs/analysis/summary.json
```

Full statistical analysis (if enabled in your setup)
```bash
make analyze_full TASK=next_activity
# writes outputs/analysis/full_report_next_activity.json

make analyze_all
```


## Make Targets
```bash
make setup             # one-command onboarding (minimal deps)
make install           # full dependency install
make install_minimal   # minimal dependency install

# Smoke tests (quick checks; run on Helpdesk, 2-3 epochs)
make smoke_test_minimal               # process_transformer next_activity on Helpdesk (3 epochs)
make smoke_test_process_transformer   # process_transformer on 3 tasks (2 epochs each)
make smoke_test_mtlformer             # mtlformer multitask (2 epochs)
make smoke_test_specialised_lstm      # specialised_lstm next_activity (2 epochs)
make smoke_test_shared_lstm           # shared_lstm next_activity (2 epochs)

make analyze           # lightweight analysis summary
make analyze_full TASK=next_activity  # full analysis for a task
make analyze_all       # full analysis for all tasks
make stats DATASET=Helpdesk  # dataset statistics
make stats_all
make sysinfo           # system info snapshot
make clean             # clear caches only
make clean_outputs     # remove outputs/ and lightning_logs/
make clean_processed   # clear data/processed cache
make clean_all         # all cleanup targets
```

## Troubleshooting and Known Pitfalls
- TensorFlow 2.15.1 is often incompatible with Python 3.11/macOS; prefer TF >= 2.20.0 as in pyproject.toml/requirements.txt.
- If .venv lacks pip, bootstrap with: .venv/bin/python -m ensurepip --upgrade
- For consistent reporting across machines, note accelerator and versions; MPS behavior can differ.
- Data availability: ensure CSVs exist under data/raw; use preprocess_action=force to (re)build processed artifacts.

## Project Layout
```
pbpm_transformer_benchmark/
├── configs/                 # Hydra configuration files
│   └── benchmark.yaml       # Main configuration
├── data/
│   ├── raw/                 # Raw event logs (.csv)
│   └── processed/           # Preprocessed cache (auto-generated)
├── src/
│   ├── cli.py               # Entry point (Hydra)
│   ├── data/                # Preprocessing pipeline
│   ├── models/              # Models and adapters
│   ├── tasks/               # Task runners
│   ├── metrics/             # Metrics
│   └── utils/               # Utilities
├── outputs/                 # Experiment outputs
├── lightning_logs/          # Training logs (ignored by VCS)
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Project metadata
```

## License
See LICENSE.txt for licensing information.



## Full Benchmark: Minimal Mode (Activities-Only)
Run the full minimal-mode experiment across all canonical datasets and models.

Command
```bash
make run_benchmark_minimal_mode
```
What it does
- Forces preprocessing for datasets: ["BPI_Challenge_2012", "Helpdesk", "Traffic_Fines", "Sepsis", "Tourism"].
- Trains/evaluates using config-defined epochs and per-model learning rates with data.attribute_mode=minimal:
  - ProcessTransformer on: next_activity, next_time, remaining_time
  - MTLFormer on: multitask
  - Specialised LSTM on: next_activity
  - Shared LSTM on: next_activity

Notes
- Early stopping is enabled and configurable via train.early_stopping_* in configs/benchmark.yaml. Defaults: patience=3, min_delta=0.001, monitor=val_loss, mode=min, restore_best_weights=true. For Next-Activity you may set train.early_stopping_monitor=val_accuracy and train.early_stopping_mode=max. This keeps runs time-bounded while still allowing warmup/regularization.
- Outputs are written under outputs/<dataset>/<task>/<model>/ with fold-level metrics.json and cv_results.json; a run-level environment snapshot is also updated at outputs/env.json.






