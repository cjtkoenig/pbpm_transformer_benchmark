# PBPM Transformer Benchmark

A comprehensive benchmark for Predictive Business Process Monitoring (PBPM) using Transformer models, following the methodology of Rama-Maneiro et al. (2021).

## Overview

This project implements a standardized benchmark for evaluating Transformer models in PBPM across four prediction tasks:
- **Next Activity Prediction**: Predict the next activity in a process trace
- **Suffix Prediction**: Predict the remaining sequence of activities
- **Next Event Time Prediction**: Predict the time until the next event
- **Remaining Time Prediction**: Predict the time until case completion

## Features

- **Standardized Methodology**: Follows Rama-Maneiro et al. (2021) framework
- **Multiple Datasets**: Support for real-world event logs including Tourism dataset
- **Efficient Preprocessing**: Simple file-based preprocessing with automatic reuse
- **Cross-Validation**: 5-fold cross-validation with case-based splitting
- **Multiple Models**: PyTorch Lightning and TensorFlow/Keras implementations
- **Comprehensive Metrics**: Task-specific evaluation metrics

## Quick Start

### Prerequisites

- Python 3.13+
- uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pbpm_transformer_benchmark

# Install dependencies
uv sync
```

### Basic Usage

```bash
# Run next activity prediction
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]"

# Run suffix prediction
uv run python -m src.cli task=suffix data.datasets="[Helpdesk]"

# Run next time prediction
uv run python -m src.cli task=next_time data.datasets="[Helpdesk]"

# Run remaining time prediction
uv run python -m src.cli task=remaining_time data.datasets="[Helpdesk]"

# Run multi-task learning (all three tasks simultaneously)
uv run python -m src.cli task=multi_task model.name=mtlformer_multi data.datasets="[Helpdesk]"
```

## Data Preprocessing

The system automatically handles data preprocessing with intelligent caching:

### Automatic Preprocessing
- Datasets are automatically preprocessed on first use
- Preprocessed data is saved in `data/processed/` for reuse
- Subsequent runs use cached data for efficiency

### Preprocessing Management

```bash
# View processed datasets
uv run python -m src.cli preprocess_action=info

# Force reprocessing of datasets
uv run python -m src.cli preprocess_action=force data.datasets="[Helpdesk]"

# Force reprocessing during task execution
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk]" force_preprocess=true

# Clear processed data
uv run python -m src.cli preprocess_action=clear
```

## Available Models

The benchmark supports multiple transformer models:

### Process Transformer
- **Framework**: TensorFlow/Keras
- **Description**: Standard transformer model for PBPM tasks
- **Usage**: `model.name=process_transformer`

### MTLFormer
- **Framework**: TensorFlow/Keras  
- **Description**: Multi-Task Learning Transformer for PBPM tasks (Single Task Models)
- **Usage**: `model.name=mtlformer`
- **Note**: Implements the same architecture as Process Transformer, taken from the original repository

### MTLFormer Multi-Task
- **Framework**: TensorFlow/Keras  
- **Description**: True Multi-Task Learning Transformer with shared parameters across all tasks
- **Usage**: `model.name=mtlformer_multi`
- **Task**: `task=multi_task`
- **Note**: Creates a single model with three outputs (next_activity, next_time, remaining_time) that shares parameters

## Configuration

The benchmark is configured via `configs/benchmark.yaml`:

```yaml
# Task selection
task: next_activity  # next_activity | suffix | next_time | remaining_time | multi_task

# Data configuration
data:
  path_raw: data/raw
  path_processed: data/processed
  end_of_case_token: "<eoc>"
  max_prefix_length: null
  attribute_mode: minimal  # minimal | extended
  datasets: ["Helpdesk", "BPI_Challenge_2012", "Tourism"]

# Model configuration
model:
  name: process_transformer  # process_transformer | mtlformer | mtlformer_multi
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

## Project Layout

```
pbpm_transformer_benchmark/
├── configs/                 # Configuration files
│   └── benchmark.yaml      # Main configuration
├── data/                   # Data directories
│   ├── raw/               # Raw event logs (.csv, .xes)
│   └── processed/         # Preprocessed data (auto-generated)
├── src/                   # Source code
│   ├── cli.py            # Command-line interface
│   ├── data/             # Data processing modules
│   ├── models/           # Model implementations
│   ├── tasks/            # Task implementations
│   ├── training/         # Training utilities
│   ├── metrics/          # Evaluation metrics
│   └── utils/            # Utility functions
├── outputs/              # Experiment outputs
├── lightning_logs/       # PyTorch Lightning logs
└── requirements.txt      # Python dependencies
```

## Make Targets

```bash
# Install dependencies
make install

# Run tests
make test

# Clean outputs
make clean

# Format code
make format

# Lint code
make lint
```

## Hydra Config Overrides

You can override any configuration parameter from the command line:

```bash
# Override model parameters
uv run python -m src.cli task=next_activity model.hidden_size=512 model.num_layers=6

# Override training parameters
uv run python -m src.cli task=next_activity train.max_epochs=20 train.batch_size=64

# Override data parameters
uv run python -m src.cli task=next_activity data.max_prefix_length=20 data.attribute_mode=extended

# Use specific datasets
uv run python -m src.cli task=next_activity data.datasets="[Helpdesk,Tourism]"
```

## Supported Datasets

The benchmark supports the following datasets:
- **BPI_Challenge_2012**: Business process intelligence challenge 2012
- **BPI_Challenge_2019**: Business process intelligence challenge 2019
- **Helpdesk**: IT service desk process
- **Road_Traffic_Fine_Management_Process**: Traffic fine management
- **Sepsis Cases - Event Log**: Healthcare process
- **Tourism**: Custom tourism industry dataset

## Evaluation Metrics

Each task uses specific evaluation metrics:

- **Next Activity**: Accuracy, F1-Score
- **Suffix**: Normalized Damerau-Levenshtein Distance
- **Time Predictions**: MAE, RMSE, R²

## Licenses

This project uses the following models and their respective licenses:

- **ProcessTransformer**: Apache License 2.0
  - Source: [ProcessTransformer Repository](https://github.com/processmining/process-transformer)
  - Used for: Transformer-based process prediction models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
