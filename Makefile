# ===== Environment Setup =====

create_venv:
	uv venv

install: create_venv
	uv pip install -r requirements.txt

# Optional: dev tools (flake8, pylint, etc.)
install_dev: install
	uv pip install -r dev-requirements.txt

# ===== Linting used for code conformance =====

flake8:
	uv run python -m flake8 .

pylint:
	uv run python -m pylint src

lint: flake8 pylint

# Ensure standard folders exist
dirs:
	uv run python -c "import pathlib; [pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/processed','outputs']]"

# Generate sample data
sample_data: dirs
	uv run python scripts/get_sample_data.py

# Note: All datasets are already converted to clean CSV format in data/processed/
# ===== Dataset Statistics =====

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

# ===== Benchmark Commands =====

# Run full benchmark with cross-validation (5 folds)
run: dirs
	uv run python -m src.cli train.accelerator=cpu

# Run with custom dataset
run_dataset: dirs
	@echo "Usage: make run_dataset DATASET=your_dataset_name"
	@echo "Example: make run_dataset DATASET=helpdesk"
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: Please specify DATASET=your_dataset_name"; \
		exit 1; \
	fi
	uv run python -m src.cli data.datasets="[$(DATASET)]" train.accelerator=cpu

# Run with custom training settings
run_custom: dirs
	@echo "Usage: make run_custom EPOCHS=10 BATCH_SIZE=64"
	@if [ -z "$(EPOCHS)" ] || [ -z "$(BATCH_SIZE)" ]; then \
		echo "Error: Please specify EPOCHS and BATCH_SIZE"; \
		echo "Example: make run_custom EPOCHS=10 BATCH_SIZE=64"; \
		exit 1; \
	fi
	uv run python -m src.cli train.max_epochs=$(EPOCHS) train.batch_size=$(BATCH_SIZE) train.accelerator=cpu

# System info snapshot
sysinfo:
	uv run python -m src.utils.system_report

clean_all: clean clean_outputs clean_logs
	@echo "Cleaned all generated files and caches"
