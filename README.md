# PBPM Transformer Benchmark

Minimal benchmark scaffold for transformer-based predictive process monitoring (PPM).  
Configuration via Hydra (`configs/benchmark.yaml`), training entrypoint at `src/cli.py`.

## Requirements
- Python 3.13.5
- [uv](https://github.com/astral-sh/uv) installed (`pipx install uv` or `pip install uv`)
- `make` (macOS/Linux; on Windows use Git Bash or run the commands inside the targets manually with `uv run ...`)

## Quick start

```bash
# 1) Create venv and install dependencies
make install

# 2) (Optional) print a short system report (Python/OS/Torch/CUDA/MPS)
make sysinfo

# 3) Add your event logs to:
#    data/raw/<dataset>.xes  or  data/raw/<dataset>.csv
#    
#    For XES files, you can automatically convert them to CSV:
#    make convert_xes        # Convert to data/processed/
#    make convert_xes_local  # Convert in same directory

# 4) Tell the project which dataset(s) to use by editing configs/benchmark.yaml:
#    data:
#      datasets: ["<dataset_name_without_extension>"]
```

## Project layout
```
pbpm_transformer_benchmark/
├── configs/               # Hydra config (benchmark.yaml)
├── data/
│   ├── raw/               # your raw event logs (.xes/.csv)
│   └── processed/         # preprocessed data
├── outputs/               # results, checkpoints
├── src/
│   ├── cli.py             # Hydra CLI entrypoint
│   ├── data/              # loaders, preprocessing
│   ├── models/            # transformer variants
│   ├── metrics/           # accuracy, edit distance, MAE
│   └── utils/
```

## Make targets

```bash
make install        # create venv + install requirements.txt
make convert_xes    # convert XES files to CSV
make run            # run with configs/benchmark.yaml defaults
make run_fast       # quick test (1 epoch, small model)
make sysinfo        # print Python/OS/Torch/CUDA/MPS info
make lint           # run flake8 + pylint (if dev deps installed)
make clean          # remove __pycache__ and .pyc files
make clean_outputs  # clear outputs/*
```

> Dev tools (optional): add them with `make install_dev` and then use `make lint`.

## Config overrides (Hydra)
You can override any config value from the command line:

```bash
# change epochs and hidden size temporarily
uv run python -m src.cli train.max_epochs=50 model.hidden_size=512

# run for a single dataset without editing files (expects data/raw/helpdesk.xes or .csv)
uv run python -m src.cli data.datasets="[helpdesk]"
```



## Notes
- `.xes` columns follow XES naming (`case:concept:name`, `concept:name`, `time:timestamp`).
- If you prefer friendlier names, add a rename step in your loader.
- Windows: if `make` isn't available, open Git Bash or run the commands inside the targets manually with `uv run ...`.
