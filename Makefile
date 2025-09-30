# ===== Environment Setup =====

create_venv:
	uv venv

# Install full dependencies from requirements.txt (may include TensorFlow pin)
install: create_venv
	uv pip install -r requirements.txt

# Install minimal dependencies sufficient to run test_cv.py quickly (no TF required)
install_minimal: create_venv
	uv run python -m ensurepip --upgrade || true
	uv pip install pandas==2.3.1 numpy==1.26.4 "scikit-learn>=1.4,<1.6" scipy==1.16.1

# ===== Linting used for code conformance =====

flake8:
	uv run python -m flake8 .

pylint:
	uv run python -m pylint src

lint: flake8 pylint

# ===== Convenience Help =====
.PHONY: help
help:
	@echo "PBPM Transformer Benchmark - Make targets"; \
	echo "  setup           Auto-detect OS (macOS/Linux) and run setup_*"; \
	echo "  setup_macos     Create venv, install minimal deps, run test"; \
	echo "  setup_linux     Create venv, install minimal deps, run test"; \
	echo "  setup_windows   Create venv, install minimal deps, run test (Git Bash/PowerShell)"; \
	echo "  install         Install full deps from requirements.txt"; \
	echo "  install_minimal Install minimal deps to run tests fast"; \
	echo "  test            Run CV sanity test (test_cv.py)"; \
	echo "  preprocess_*    Inspect/force/clear processed data cache"; \
	echo "  graphgps_env    Provision helper venv for PGTNet/GraphGPS (CPU Torch 2.8 + PyG + deps)"; \
	echo "  smoke_test_pgtnet_run [DATASET=Helpdesk] [XES=/abs/path/file.xes]  Execute PGTNet pipeline for 1 fold (uses helper venv if present)"; \
	echo "  smoke_test_pgtnet_bootstrap  Provision helper venv then run smoke_test_pgtnet_run"; \
	echo "  smoke_test      Run chosen MODEL across all datasets and tasks for 1 epoch (mtlformer adds multitask) (ACCELERATOR=cpu|mps|gpu)"; \
	echo "  run_both_lstms  Run specialised_lstm and shared_lstm on all datasets (extended) for 1 epoch each"; \
	echo "  run_benchmark_minimal_mode  Force preprocess and run all models/tasks on minimal attributes (uses config-defined epochs and per-model LRs)"; \
	echo "  clean_*         Clean caches, outputs, processed data";

# Ensure standard folders exist
dirs:
	uv run python -c "import pathlib; [pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/processed','outputs']]"

# ===== Dataset Statistics =====
# Note: All datasets are already converted to clean CSV format in data/processed/

# Show statistics for a specific dataset
stats: dirs
	@echo "Usage: make stats DATASET=your_dataset_name"
	@echo "Example: make stats DATASET=sample_synth"
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: Please specify DATASET=your_dataset_name"; \
		exit 1; \
	fi
	@if [ -f "data/raw/$(DATASET).csv" ]; then \
		uv run python scripts/dataset_stats.py data/raw/$(DATASET).csv; \
	else \
		echo "Error: Dataset $(DATASET) not found in data/raw/"; \
		echo "Available datasets:"; \
		ls -1 data/raw/*.csv 2>/dev/null | sed 's/.*\///' | sed 's/\.[^.]*$$//' | sort -u || echo "No datasets found"; \
		exit 1; \
	fi

# Show statistics for all available datasets
stats_all: dirs
	@echo "Calculating statistics for all available datasets..."
	@for file in data/raw/*.csv; do \
		if [ -f "$$file" ]; then \
			echo ""; \
			uv run python scripts/dataset_stats.py "$$file"; \
		fi; \
	done 2>/dev/null || echo "No datasets found in data/raw/"

# System info snapshot
sysinfo:
	uv run python -m src.utils.system_report

# ===== Cleanup Commands =====

# Clean Python cache files
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.pyd" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Clean output files
clean_outputs:
	@echo "Cleaning output files..."
	rm -rf outputs/* 2>/dev/null || true
	rm -rf lightning_logs/* 2>/dev/null || true

# Clean log files
clean_logs:
	@echo "Cleaning log files..."
	rm -rf *.log 2>/dev/null || true
	rm -rf logs/* 2>/dev/null || true

# Clean processed data
clean_processed:
	@echo "Cleaning processed data (removing all cached artifacts under data/processed)..."
	# Preserve the directory but remove all contents (files and subdirectories)
	rm -rf data/processed/* 2>/dev/null || true
	rm -rf data/processed/.* 2>/dev/null || true

clean_all: clean clean_outputs clean_logs clean_processed
	@echo "Cleaned all generated files and caches"

# ===== Preprocessing Only =====
# Run preprocessing management actions without training any model.
# Usage examples:
#   make preprocess_info
#   make preprocess_force DATASETS="[\"Helpdesk\"]"
#   make preprocess_clear                 # clears all processed artifacts
#   make preprocess_clear DATASET=Helpdesk # clears just one dataset
.PHONY: preprocess_info preprocess_force preprocess_clear

# Onboarding: OS-specific setup targets
# These targets aim to get a new user to a passing test_cv.py quickly.
# They install minimal deps first (fast), then run a sanity test.
setup_macos: install_minimal dirs
	@echo "Detected macOS. Python 3.11 recommended. MPS available via train.accelerator=mps."
	@echo "Note: TensorFlow 2.15.x is often incompatible on Python 3.11/macOS. Prefer TF >= 2.20 if you need TF models."
	make test
	@echo "macOS setup complete. Try: make smoke MODEL=process_transformer DATASET=Helpdesk ACCELERATOR=mps"

setup_linux: install_minimal dirs
	@echo "Detected Linux. Python 3.11 recommended."
	make test
	@echo "Linux setup complete."

# Windows notes: Make often runs under Git Bash; uv handles venv well. We use uv run for portability.
setup_windows: create_venv
	@echo "Windows setup starting (PowerShell or Git Bash)."
	uv run python -m ensurepip --upgrade || true
	uv pip install pandas==2.3.1 numpy==1.26.4 "scikit-learn>=1.4,<1.6" scipy==1.16.1
	@echo "Running CV sanity test..."
	uv run python test_cv.py
	@echo "Windows setup complete. If you need TensorFlow, prefer >=2.20 on Python 3.11."

# Auto-detect setup for Unix shells; prints hint for Windows.
setup:
	@if [ "$(OS)" = "Windows_NT" ]; then \
		echo "On Windows, run: make setup_windows"; \
	else \
		UNAME_S=$$(uname -s); \
		if echo $$UNAME_S | grep -qi "darwin"; then \
			make setup_macos; \
		else \
			make setup_linux; \
		fi; \
	fi

preprocess_info: dirs
	@echo "Showing processed data cache info..."
	uv run python -m src.cli preprocess_action=info

preprocess_force: dirs
	@echo "Forcing preprocessing for specified datasets (no training)..."
	@if [ -z "$(DATASETS)" ]; then \
		echo "Error: Please provide DATASETS in Hydra list syntax, e.g."; \
		echo "       make preprocess_force DATASETS=\"[\"Helpdesk\"]\""; \
		exit 1; \
	fi; \
	uv run python -m src.cli preprocess_action=force data.datasets="$(DATASETS)"

preprocess_clear: dirs
	@if [ -z "$(DATASET)" ]; then \
		echo "Clearing ALL processed datasets under data/processed (cache only)..."; \
		uv run python -m src.cli preprocess_action=clear; \
	else \
		echo "Clearing processed artifacts for dataset: $(DATASET)"; \
		uv run python -m src.cli preprocess_action=clear dataset_name=$(DATASET); \
	fi

# ===== Analysis =====
.PHONY: analyze analyze_full analyze_all analyze_thesis
analyze: dirs
	@echo "Running lightweight analysis summary (outputs/analysis/summary.json)..."
	uv run python -m src.cli analysis.action=run_stats

# Usage: make analyze_full TASK=next_activity
analyze_full: dirs
	@if [ -z "$(TASK)" ]; then \
		echo "Error: Please specify TASK=next_activity|next_time|remaining_time"; \
		exit 1; \
	fi; \
	echo "Running full statistical analysis for task=$(TASK)..."; \
	uv run python -m src.utils.statistical_analysis --task $(TASK)

# Run full analysis for all tasks
analyze_all: dirs
	@for t in next_activity next_time remaining_time; do \
		echo "Running full statistical analysis for task=$$t..."; \
		uv run python -m src.utils.statistical_analysis --task $$t || exit $$?; \
	done

# Thesis-aligned report across tasks (Minimal vs Extended, uplifts, MTLFormer efficiency)
# Usage: make analyze_thesis [TASK=all|next_activity|next_time|remaining_time]
 analyze_thesis: dirs
	@echo "Generating thesis-aligned report (outputs/analysis/thesis_report*.json)..."; \
	if [ -z "$(TASK)" ] || [ "$(TASK)" = "all" ]; then \
		uv run python -m src.utils.statistical_analysis --thesis --task all; \
	else \
		uv run python -m src.utils.statistical_analysis --thesis --task $(TASK); \
	fi

# ===== Smoke Tests =====

# Split smoke tests into fixed, non-variable targets.
# 1) process_transformer on its three tasks (next_activity, next_time, remaining_time)
# 2) mtlformer on multitask
# 3) specialised_lstm (next_activity only)
# 4) shared_lstm (next_activity only)
# Each runs exactly 2 epochs on a small dataset for speed.

.PHONY: smoke_test_minimal
smoke_test_minimal: dirs
	@echo "=== Smoke: process_transformer on Helpdesk (next_activity, 15 epochs) ==="; \
	uv run python -m src.cli task=next_activity data.datasets="[\"Helpdesk\"]" train.max_epochs=15;

.PHONY: smoke_test_process_transformer
smoke_test_process_transformer: dirs
	@echo "=== Smoke: process_transformer on Helpdesk (3 tasks, 2 epochs) ==="; \
	uv run python -m src.cli model.name=process_transformer task=next_activity data.datasets="[\"Helpdesk\"]" train.max_epochs=2; \
	uv run python -m src.cli model.name=process_transformer task=next_time data.datasets="[\"Helpdesk\"]" train.max_epochs=2; \
	uv run python -m src.cli model.name=process_transformer task=remaining_time data.datasets="[\"Helpdesk\"]" train.max_epochs=2;

.PHONY: smoke_test_mtlformer
smoke_test_mtlformer: dirs
	@echo "=== Smoke: mtlformer multitask on Helpdesk (15 epochs) ==="; \
	uv run python -m src.cli model.name=mtlformer task=multitask data.datasets="[\"Helpdesk\"]" train.max_epochs=15;

.PHONY: smoke_test_activity_only_lstm
smoke_test_activity_only_lstm: dirs
	@echo "=== Smoke: activity_only_lstm next_activity on Helpdesk (15 epochs) ==="; \
	uv run python -m src.cli model.name=activity_only_lstm task=next_activity data.datasets="[\"Helpdesk\"]" train.max_epochs=15;

.PHONY: smoke_test_specialised_lstm
smoke_test_specialised_lstm: dirs
	@echo "=== Smoke: specialised_lstm next_activity on Helpdesk (2 epochs, extended attrs) ==="; \
	uv run python -m src.cli model.name=specialised_lstm task=next_activity data.datasets="[\"Helpdesk\"]" data.attribute_mode=extended train.max_epochs=2;

.PHONY: smoke_test_shared_lstm
smoke_test_shared_lstm: dirs
	@echo "=== Smoke: shared_lstm next_activity on Helpdesk (2 epochs, extended attrs) ==="; \
	uv run python -m src.cli model.name=shared_lstm task=next_activity data.datasets="[\"Helpdesk\"]" data.attribute_mode=extended train.max_epochs=2;

# PGTNet smoke test (plan-only by default)
# Usage:
#   make smoke_test_pgtnet [DATASET=Helpdesk] [ACCELERATOR=cpu|mps|gpu]
# Notes:
# - This target does not actually train PGTNet; it generates per-fold plans/manifests
#   to run the external PGTNet/GraphGPS pipeline. It also writes mapping/masks under
#   data/processed_pgtnet/<dataset> based on canonical splits.
# - To execute training/inference externally later, re-run the CLI with +model.pgtnet.execute=true
#   and configure model.pgtnet.python, pgtnet_repo, graphgps_repo.
.PHONY: smoke_test_pgtnet
smoke_test_pgtnet: dirs
	@DATASET_NAME=$${DATASET:-Helpdesk}; \
	echo "=== Smoke: PGTNet (plan-only) on $$DATASET_NAME (remaining_time, extended) ==="; \
	uv run python -m src.cli preprocess_action=force data.datasets="[\"$$DATASET_NAME\"]" || exit $$?; \
	uv run python -m src.cli task=remaining_time data.datasets="[\"$$DATASET_NAME\"]" data.attribute_mode=extended model.name=pgtnet +model.pgtnet.execute=false +model.pgtnet.auto_collect_predictions=false || exit $$?; \
	echo "Plan written under outputs/pgtnet/$$DATASET_NAME/ and outputs/$$DATASET_NAME/remaining_time/pgtnet/"

# PGTNet smoke test (execute actual external pipeline for 1 fold if repos are available)
# Usage:
#   make smoke_test_pgtnet_run [DATASET=Helpdesk]
# Prerequisites:
#   - third_party/PGTNet and third_party/GraphGPS must point to working checkouts
#   - model.pgtnet.python should be a Python interpreter with their deps
# Behavior:
#   - Runs fold 0 only (cv.n_folds=1)
#   - Uses extended attribute mode and writes per-fold manifests/logs under outputs/pgtnet/<dataset>/
.PHONY: smoke_test_pgtnet_run
smoke_test_pgtnet_run: dirs
	@DATASET_NAME=$${DATASET:-Helpdesk}; \
	PY_PATH=$$( [ -x third_party/graphgps_venv/bin/python ] && echo "third_party/graphgps_venv/bin/python" || echo "" ); \
	[ -n "$$PY_PATH" ] && echo "Using helper interpreter: $$PY_PATH" || echo "No helper interpreter found. You can create it via: make graphgps_env"; \
	if [ -n "$$PY_PATH" ]; then $(MAKE) check_graphgps_env || { echo "GraphGPS helper env is missing modules. Run: make graphgps_env"; exit 1; }; fi; \
	if [ -n "$(XES)" ]; then XES_ARG="+model.pgtnet.converter.input_xes=$(XES)"; else XES_ARG=""; fi; \
	echo "=== Smoke: PGTNet (execute) on $$DATASET_NAME (remaining_time, extended, 1 fold) ==="; \
	if [ -n "$$PY_PATH" ]; then EXTRA="+model.pgtnet.python=$$PY_PATH"; else EXTRA=""; fi; \
	uv run python -m src.cli task=remaining_time data.datasets="[\"$$DATASET_NAME\"]" data.attribute_mode=extended model.name=pgtnet cv.n_folds=1 +model.pgtnet.execute=true +model.pgtnet.auto_collect_predictions=false $$EXTRA $$XES_ARG || exit $$?; \
	echo "Tip: Place the XES file under third_party/PGTNet/raw_dataset with the exact name from conversion_configs/<cfg>.yaml, or pass XES=/abs/path/file.xes"; \
	echo "Check outputs/pgtnet/$$DATASET_NAME/fold_0/ for logs and outputs."

# Helpers to list available items
.PHONY: list_datasets list_tasks list_models
list_datasets:
	@echo "Available datasets in data/raw:" && \
	ls -1 data/raw/*.csv 2>/dev/null | sed 's/.*\///' | sed 's/\.[^.]*$$//' | sort -u || echo "(none)"

list_tasks:
	@echo "Supported tasks:" && \
	echo "  - next_activity" && \
	echo "  - next_time" && \
	echo "  - remaining_time" && \
	echo "  - multitask (for mtlformer only)"

# Keep this list in sync with src/models/model_registry.py registrations
list_models:
	@echo "Registered models:" && \
		echo "  - process_transformer" && \
		echo "  - mtlformer" && \
		echo "  - activity_only_lstm" && \
		echo "  - specialised_lstm" && \
		echo "  - shared_lstm" && \
		echo "  - pgtnet (external)"

# Basic test target (README mentions make test). Runs the CV sanity test.
.PHONY: test
test:
	@echo "Running cross-validation sanity test (test_cv.py)..."
	uv run python test_cv.py


# ===== Full Benchmark: Minimal Mode (activities-only) =====
.PHONY: run_benchmark_minimal_mode
run_benchmark_minimal_mode: dirs
	@echo "=== PBPM Benchmark: Minimal Mode (activities only) ==="; \
	echo "Datasets: BPI_Challenge_2012, Helpdesk, Road_Traffic_Fine_Management_Process, Sepsis Cases - Event Log, Tourism"; \
	echo "Step 1/2: Force preprocessing for all datasets..."; \
	UV="uv run python -m src.cli"; \
	DATASETS="[\"BPI_Challenge_2012\", \"Helpdesk\", \"Road_Traffic_Fine_Management_Process\", \"Sepsis Cases - Event Log\", \"Tourism\"]"; \
	$$UV preprocess_action=force data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "Step 2/2: Train/Evaluate all models and tasks (using config-defined epochs and per-model learning rates, attribute_mode=minimal)"; \
	echo "--- ProcessTransformer: next_activity ---"; \
	$$UV model.name=process_transformer task=next_activity data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "--- ProcessTransformer: next_time ---"; \
	$$UV model.name=process_transformer task=next_time data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "--- ProcessTransformer: remaining_time ---"; \
	$$UV model.name=process_transformer task=remaining_time data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "--- MTLFormer: multitask ---"; \
	$$UV model.name=mtlformer task=multitask data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "--- Activity-Only LSTM: next_activity ---"; \
	$$UV model.name=activity_only_lstm task=next_activity data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \
	echo "=== Benchmark completed. Outputs written under outputs/ ==="; \
	echo "Note: outputs/env.json is overwritten by each run; refer to per-model outputs under outputs/<dataset>/<task>/<model>/";

# ===== Extended Attributes: LSTMs on all datasets for 1 epoch =====
.PHONY: run_both_lstms
run_both_lstms: dirs
	@echo "=== Extended attributes: specialised_lstm and shared_lstm over all datasets (1 epoch each) ==="; \
	UV="uv run python -m src.cli"; \
	DATASETS="[\"BPI_Challenge_2012\", \"Helpdesk\", \"Road_Traffic_Fine_Management_Process\", \"Sepsis Cases - Event Log\", \"Tourism\"]"; \
	echo "--- specialised_lstm (next_activity, extended) ---"; \
	$$UV model.name=specialised_lstm task=next_activity data.datasets="$$DATASETS" data.attribute_mode=extended train.max_epochs=1 || exit $$?; \
	echo "--- shared_lstm (next_activity, extended) ---"; \
	$$UV model.name=shared_lstm task=next_activity data.datasets="$$DATASETS" data.attribute_mode=extended train.max_epochs=1 || exit $$?;

.PHONY: thesis_report
thesis_report: dirs
	@echo "Generating thesis-aligned report (within-track rankings and uplift)..."; \
	if [ -z "$(TASK)" ]; then \
		uv run python -m src.cli analysis.action=thesis_report; \
	else \
		echo "Task specified: $(TASK)"; \
		uv run python -m src.cli analysis.action=thesis_report analysis.task=$(TASK); \
	fi


# Create a dedicated Python environment for GraphGPS/PGTNet with Torch + PyG (CPU)
.PHONY: graphgps_env
graphgps_env:
	@echo "Creating GraphGPS Python venv under third_party/graphgps_venv ..."; \
	uv venv third_party/graphgps_venv; \
	third_party/graphgps_venv/bin/python -m ensurepip --upgrade || true; \
	third_party/graphgps_venv/bin/python -m pip install --upgrade pip; \
	# Install CPU-only PyTorch 2.8.0 for maximum wheel availability across machines
	third_party/graphgps_venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0; \
	# Install PyG packages against torch 2.8.0 CPU wheels from the official wheel index
	third_party/graphgps_venv/bin/python -m pip install -f https://data.pyg.org/whl/torch-2.8.0+cpu.html torch_geometric torch_scatter; \
	# GraphGym expects pytorch_lightning
	third_party/graphgps_venv/bin/python -m pip install pytorch_lightning==2.5.3; \
	# Converter/runtime utilities
	third_party/graphgps_venv/bin/python -m pip install pyyaml; \
	third_party/graphgps_venv/bin/python -m pip install pm4py; \
	third_party/graphgps_venv/bin/python -m pip install "scikit-learn>=1.4,<1.6"; \
	third_party/graphgps_venv/bin/python -m pip install yacs; \
	third_party/graphgps_venv/bin/python -m pip install pandas==2.3.1; \
	# Additional GraphGPS/PGTNet runtime dependencies
	third_party/graphgps_venv/bin/python -m pip install einops; \
	third_party/graphgps_venv/bin/python -m pip install local-attention; \
	third_party/graphgps_venv/bin/python -m pip install axial-positional-embedding; \
	third_party/graphgps_venv/bin/python -m pip install performer-pytorch; \
	third_party/graphgps_venv/bin/python -m pip install ogb
	third_party/graphgps_venv/bin/python -m pip install 'setuptools<70'
	# Guard against third-party 'importlib' shadowing the stdlib 'importlib' module
	third_party/graphgps_venv/bin/python -m pip uninstall -y importlib >/dev/null 2>&1 || true; \
	echo "GraphGPS Python ready at third_party/graphgps_venv/bin/python";

# Bootstrap env then run the PGTNet execute smoke test
.PHONY: smoke_test_pgtnet_bootstrap
smoke_test_pgtnet_bootstrap: graphgps_env
	@$(MAKE) smoke_test_pgtnet_run

.PHONY: check_graphgps_env
check_graphgps_env:
	@PY=third_party/graphgps_venv/bin/python; \
	if [ ! -x "$$PY" ]; then echo "GraphGPS helper interpreter not found at third_party/graphgps_venv/bin/python. Run: make graphgps_env"; exit 1; fi; \
	MODS="yaml pm4py sklearn pandas torch torch_geometric pytorch_lightning yacs torch_scatter"; \
	MISSING=""; \
	for M in $$MODS; do \
		$$PY -c "import $$M" >/dev/null 2>&1 || MISSING="$$MISSING $$M"; \
	done; \
	if [ -z "$$MISSING" ]; then echo "OK"; else echo "Missing:$$MISSING"; exit 1; fi


# ===== CSV ↔ XES helpers =====
.PHONY: csv_to_xes csv_to_xes_all xes_to_csv

# Convert a single CSV under data/raw to XES.
# Usage:
#   make csv_to_xes DATASET=Helpdesk
#   make csv_to_xes CSV=/abs/path/in.csv OUT=/abs/path/out.xes
csv_to_xes:
	@# Prefer the helper interpreter (has pm4py) if available
	if [ -x third_party/graphgps_venv/bin/python ]; then PY=third_party/graphgps_venv/bin/python; UV=""; else PY="uv run python"; fi; \
	if [ -n "$(DATASET)" ]; then \
	  echo "Converting dataset '$(DATASET)' (data/raw/$(DATASET).csv → data/raw/$(DATASET).xes)"; \
	  $$PY scripts/csv_to_xes.py --dataset "$(DATASET)" $(if $(OUT),--out "$(OUT)",) $(if $(OVERWRITE),--overwrite,) || exit $$?; \
	elif [ -n "$(CSV)" ]; then \
	  echo "Converting CSV '$(CSV)' → $(if $(OUT),$(OUT),auto .xes)"; \
	  $$PY scripts/csv_to_xes.py --csv "$(CSV)" $(if $(OUT),--out "$(OUT)",) $(if $(OVERWRITE),--overwrite,) || exit $$?; \
	else \
	  echo "Specify DATASET=<name> (uses data/raw/<name>.csv) or CSV=/path/to/file.csv"; exit 2; \
	fi

# Convert XES in data/raw to CSV.
# Usage:
#   make xes_to_csv DATASET=BPI_Challenge_2012 [OVERWRITE=1]
#   make xes_to_csv XES=/abs/path/in.xes OUT=/abs/path/out.csv
xes_to_csv:
	@# Prefer the helper interpreter (has pm4py) if available
	if [ -x third_party/graphgps_venv/bin/python ]; then PY=third_party/graphgps_venv/bin/python; else PY="uv run python"; fi; \
	if [ -n "$(DATASET)" ]; then \
	  echo "Converting dataset '$(DATASET)' (data/raw/$(DATASET).xes → data/raw/$(DATASET).csv)"; \
	  $$PY scripts/xes_to_csv.py --dataset "$(DATASET)" $(if $(OUT),--out "$(OUT)",) $(if $(OVERWRITE),--overwrite,) || exit $$?; \
	elif [ -n "$(XES)" ]; then \
	  echo "Converting XES '$(XES)' → $(if $(OUT),$(OUT),auto .csv)"; \
	  $$PY scripts/xes_to_csv.py --xes "$(XES)" $(if $(OUT),--out "$(OUT)",) $(if $(OVERWRITE),--overwrite,) || exit $$?; \
	else \
	  echo "Specify DATASET=<name> (uses data/raw/<name>.xes) or XES=/path/to/file.xes"; exit 2; \
	fi

# Convert all CSVs in data/raw to XES next to them (skips existing unless OVERWRITE=1)
csv_to_xes_all:
	@set -e; \
	if [ -x third_party/graphgps_venv/bin/python ]; then PY=third_party/graphgps_venv/bin/python; else PY="uv run python"; fi; \
	shopt -s nullglob; \
	for f in data/raw/*.csv; do \
	  base=$${f%.csv}; out="$$base.xes"; \
	  if [ -f "$$out" ] && [ "$(OVERWRITE)" != "1" ]; then \
	    echo "Skip existing $$out (use OVERWRITE=1 to overwrite)"; \
	  else \
	    echo "Converting $$f → $$out"; \
	    $$PY scripts/csv_to_xes.py --csv "$$f" --out "$$out" $(if $(OVERWRITE),--overwrite,) || exit $$?; \
	  fi; \
	done; \
	echo "Done."
