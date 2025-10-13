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
	echo "  preprocess_all  Force preprocessing for all datasets found under data/raw (unified for minimal/extended)"; \
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
.PHONY: preprocess_info preprocess_force preprocess_clear preprocess_all preprocess_extended preprocess_all_extended

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

# Force preprocessing for all datasets found under data/raw
# Usage: make preprocess_all
preprocess_all: dirs
	@FILES=$$(ls -1 data/raw/*.csv 2>/dev/null || true); \
	if [ -z "$$FILES" ]; then \
		echo "No datasets found under data/raw"; \
		exit 0; \
	fi; \
	NAMES=$$(for f in $$FILES; do b=$$(basename "$$f"); n=$${b%.*}; printf '"%s",' "$$n"; done); \
	DATASETS="[$${NAMES%,}]"; \
	echo "Forcing preprocessing for datasets: $$DATASETS (unified minimal/extended)"; \
	uv run python -m src.cli preprocess_action=force data.datasets="$$DATASETS"

# Force preprocessing for specific datasets in extended attribute mode
# Usage: make preprocess_extended DATASETS='["Helpdesk", "BPI_Challenge_2012"]'
preprocess_extended: dirs
	@if [ -z "$(DATASETS)" ]; then \
		echo "Error: Please provide DATASETS in Hydra list syntax, e.g."; \
		echo "       make preprocess_extended DATASETS=\"[\"Helpdesk\"]\""; \
		exit 1; \
	fi; \
 echo "Forcing preprocessing for datasets: $(DATASETS) (unified minimal/extended)"; \
	uv run python -m src.cli preprocess_action=force data.datasets="$(DATASETS)"

# Force preprocessing for all datasets in extended attribute mode
# Usage: make preprocess_all_extended
preprocess_all_extended: dirs
	@$(MAKE) preprocess_all MODE=extended

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
		echo "  - shared_lstm"

# Basic test target (README mentions make test). Runs the CV sanity test.
.PHONY: test
test:
	@echo "Running cross-validation sanity test (test_cv.py)..."
	uv run python test_cv.py

# ===== Smoke Tests =====
.PHONY: smoke_test_minimal
smoke_test_minimal: dirs
	@echo "=== Smoke: process_transformer on a single dataset (next_activity) ==="; \
	uv run python -m src.cli task=next_activity data.datasets="[\"Traffic_Fines\"]" train.max_epochs=1;

.PHONY: smoke_test_process_transformer
smoke_test_process_transformer: dirs
	@echo "=== Smoke: process_transformer on Helpdesk (3 tasks, 2 epochs) ==="; \
	uv run python -m src.cli model.name=process_transformer task=next_activity data.datasets="[\"Helpdesk\"]" train.max_epochs=2; \
	uv run python -m src.cli model.name=process_transformer task=next_time data.datasets="[\"Helpdesk\"]" train.max_epochs=2; \
	uv run python -m src.cli model.name=process_transformer task=remaining_time data.datasets="[\"Helpdesk\"]" train.max_epochs=2;

.PHONY: smoke_test_mtlformer
smoke_test_mtlformer: dirs
	@echo "=== Smoke: mtlformer multitask on Helpdesk (15 epochs) ==="; \
	uv run python -m src.cli model.name=mtlformer task=multitask data.datasets="[\"Helpdesk\"]" train.max_epochs=3;

.PHONY: smoke_test_activity_only_lstm
smoke_test_activity_only_lstm: dirs
	@echo "=== Smoke: activity_only_lstm next_activity on Helpdesk (15 epochs) ==="; \
	uv run python -m src.cli model.name=activity_only_lstm task=next_activity data.datasets="[\"Helpdesk\"]" train.max_epochs=3;

.PHONY: smoke_test_specialised_lstm
smoke_test_specialised_lstm: dirs
	@echo "=== Smoke: specialised_lstm next_activity on Helpdesk (2 epochs, extended attrs) ==="; \
	uv run python -m src.cli model.name=specialised_lstm task=next_activity data.datasets="[\"Sepsis\"]" data.attribute_mode=extended train.max_epochs=1;

.PHONY: smoke_test_shared_lstm
smoke_test_shared_lstm: dirs
	@echo "=== Smoke: shared_lstm next_activity on Helpdesk (2 epochs, extended attrs) ==="; \
	uv run python -m src.cli model.name=shared_lstm task=next_activity data.datasets="[\"Tourism\"]" data.attribute_mode=extended train.max_epochs=1;

run_mtl_former_again: dirs
	@echo "=== PBPM Benchmark: MTLFormer with efficiency metrics ==="; \
	echo "Datasets: BPI_Challenge_2012, Helpdesk, Traffic_Fines, Sepsis, Tourism"; \
	UV="uv run python -m src.cli"; \
	DATASETS="[\"BPI_Challenge_2012\", \"Helpdesk\", \"Traffic_Fines\", \"Sepsis\", \"Tourism\"]"; \
	echo "--- MTLFormer: multitask ---"; \
	$$UV model.name=mtlformer task=multitask data.datasets="$$DATASETS" data.attribute_mode=minimal || exit $$?; \

# ===== Minimal Benchmark Run =====
.PHONY: run_benchmark_minimal_mode
run_benchmark_minimal_mode: dirs
	@echo "=== PBPM Benchmark: Minimal Mode (activities only) ==="; \
	echo "Datasets: BPI_Challenge_2012, Helpdesk, Traffic_Fines, Sepsis, Tourism"; \
	echo "Step 1/2: Force preprocessing for all datasets..."; \
	UV="uv run python -m src.cli"; \
	DATASETS="[\"BPI_Challenge_2012\", \"Helpdesk\", \"Traffic_Fines\", \"Sepsis\", \"Tourism\"]"; \
	$$UV preprocess_action=force data.datasets="$$DATASETS" data.attribute_mode=extended || exit $$?; \
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

# ===== Extended Benchmark Run =====
.PHONY: run_benchmark_extended_mode
run_benchmark_extended_mode: dirs
	@echo "=== PBPM Benchmark: Extended Mode (LSTMs) ==="; \
	UV="uv run python -m src.cli"; \
	DATASETS="[\"BPI_Challenge_2012\", \"Helpdesk\", \"Traffic_Fines\", \"Sepsis\", \"Tourism\"]"; \
	echo "Force preprocessing for all datasets (extended ready)"; \
	$$UV preprocess_action=force data.datasets="$$DATASETS" || exit $$?; \
	echo "--- specialised_lstm (next_activity, extended) ---"; \
	$$UV model.name=specialised_lstm task=next_activity data.datasets="$$DATASETS" data.attribute_mode=extended || exit $$?; \
	echo "--- shared_lstm (next_activity, extended) ---"; \
	$$UV model.name=shared_lstm task=next_activity data.datasets="$$DATASETS" data.attribute_mode=extended || exit $$?;

.PHONY: thesis_report
thesis_report: dirs
	@echo "Generating thesis-aligned report (within-track rankings and uplift)..."; \
	if [ -z "$(TASK)" ]; then \
		uv run python -m src.cli analysis.action=thesis_report; \
	else \
		echo "Task specified: $(TASK)"; \
		uv run python -m src.cli analysis.action=thesis_report analysis.task=$(TASK); \
	fi
