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
- Make Targets
- Troubleshooting and Known Pitfalls
- Project Layout
- License

## Overview
- Canonical tasks: next activity, next time, remaining time.
  - multitask for MTLFormer only.
- Centralized preprocessing: prefixes, activity encoding, numeric normalization, End‑of‑Case token (<eoc>), cached under data/processed.
  - Different Graph-based preprocessing for PGTNet only.
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
- TensorFlow: prefer >= 2.20.0 on Python 3.11 (requirements.txt’s 2.15.1 pin may be incompatible in some environments).


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

Config reference (excerpt)
```yaml
# Task selection
task: next_activity  # next_activity | next_time | remaining_time | (multitask for mtlformer only)

# Data configuration
data:
  path_raw: data/raw
  path_processed: data/processed
  end_of_case_token: "<eoc>"
  max_prefix_length: null
  attribute_mode: minimal  # minimal | extended
  datasets: ["BPI_Challenge_2012", "Helpdesk", "Road_Traffic_Fine_Management_Process", "Sepsis Cases - Event Log", "Tourism"]

# Model configuration
model:
  name: process_transformer # process_transformer | mtlformer | specialised_lstm | shared_lstm
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  dropout_probability: 0.1

# Training configuration
train:
  batch_size: 128
  max_epochs: 10
  learning_rate: 3e-4
  accelerator: auto  # auto | cpu | gpu | mps
  devices: 1

# Cross-validation
cv:
  n_folds: 5
  stratify: null
  split_by_cases: true
```

## Cross-Validation Protocol
- Strict case-based splits; no case appears in both train and validation of a fold.
- 5 folds persisted per dataset/task with a stable seed.
- Aggregation utilities compute mean/std/min/max across folds.

## Models and Adapters
Adapters bridge canonical data and model I/O. They may reshape and cast, but must not change content.

Allowed
- Reshaping [B,T,D], generating attention masks from pad_id.
- One-hot ↔ index conversion; dtype changes; padding/clipping to max_prefix_len.

Not allowed
- Changing splits, re-encoding vocab/labels, altering normalization/time units, filtering/augmenting data.

Implemented models
- process_transformer (TensorFlow)
- mtlformer (TensorFlow)
- specialised_lstm (TensorFlow; next_activity only)
- shared_lstm (TensorFlow; next_activity only)

## Metrics
- Next Activity: Accuracy (primary), optionally F1.
- Time predictions: MAE (primary). RMSE/R² optional.

## Analysis and Reporting
A minimal analysis utility exists at src/analysis/summary.py that aggregates per-fold metrics from outputs/ and produces a naïve ranking.

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

