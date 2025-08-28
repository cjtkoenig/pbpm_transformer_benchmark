# PBPM Transformer Benchmark

A comprehensive benchmark for Predictive Business Process Monitoring (PBPM) using Transformer models, following the methodology of Rama-Maneiro et al. (2021).

## Benchmark-Setup

Die Methodik des Benchmarks orientiert sich am Rahmenwerk von Rama-Maneiro, Vidal & Lama (2021) und stellt eine vergleichbare sowie reproduzierbare Evaluationsgrundlage im Predictive Business Process Monitoring (PBPM) sicher.

### Benchmark-Aufgaben
- **Next Activity Prediction** – Vorhersage der nächsten Aktivität  
- **Suffix Prediction** – Vorhersage des verbleibenden Aktivitätsverlaufs  
- **Next Time Prediction** – Vorhersage der Zeit bis zum nächsten Event  
- **Remaining Time Prediction** – Schätzung der Restlaufzeit eines Prozesses  

### Datensätze
- Nutzung der von Rama-Maneiro et al. eingesetzten Logs (verschiedene Domänen & Komplexität)  
- Ergänzend: eigener Datensatz aus der Tourismusbranche zur Überprüfung der Generalisierbarkeit  

### Preprocessing
- Generierung von Präfixen unterschiedlicher Länge  
- Kodierung der Aktivitäten  
- Normalisierung numerischer Attribute  
- Hinzufügen eines *End-of-Case*-Tokens  
- Zwei Varianten:  
  - **Minimalistisch**: nur Aktivitäten  
  - **Erweitert**: zusätzliche Attribute (falls Modellarchitektur unterstützt)  

### Evaluation
- **5-fold Cross-Validation**  
- Innerhalb der Folds: 80/20-Split in Trainings- und Validierungsdaten  
- Einheitliche Hardware: MacBook Pro (Apple M3 Pro, 36 GB RAM)  
- Software: Python 3.13.5, PyTorch 2.8.0 (MPS)  

### Metriken
- Next Activity → **Accuracy**  
- Suffix Prediction → **normalisierte Damerau-Levenshtein-Distanz**  
- Zeitprognosen → **Mean Absolute Error (MAE)**  
- Ergänzend: Trainingszeit, Inferenzzeit, Modellgröße  

### Statistische Auswertung
- Rangordnung der Modelle mittels **Plackett-Luce-Modell**  
- Paarweise Vergleiche der Top-Modelle mit **hierarchischem Bayes-Test**  
- Robustheitsprüfung durch **Friedman- / Wilcoxon-Tests**  

### Modellauswahl
- Berücksichtigt werden nur Modelle mit verfügbarer/zugänglicher Implementierung  
- Fokus: **Transformer-Modelle** (seit 2020)  
- Aktuell nur ProcessTransformer implementiert 

## Model Adapters

Um unterschiedliche Modellarchitekturen in den Benchmark einzubinden, wird ein **einheitliches, kanonisches Datenformat** genutzt.  
Alle Datensätze werden **einmalig** vorverarbeitet (Präfix-Generierung, Kodierung, Normalisierung, Padding, Splits).  
Diese Artefakte bestehen u. a. aus:

- `X_{train,val,test}.npy` – Eingabesequenzen  
- `y_{train,val,test}.npy` – Zielwerte (abhängig von der Benchmark-Aufgabe)  
- `meta.json` – Metadaten (Vokabulare, `pad_id`, `eoc_id`, Normalisierungswerte, Feature-Liste, max. Präfixlänge)  
- `splits.json` – fixe Case-IDs für Train/Val/Test  

### Adapter-Konzept
Da verschiedene Modelle Eingaben in unterschiedlicher Form erwarten, kommen **Adapter** zum Einsatz.  
Adapter sind *leichte Umwandlungsschichten*, die nur die **Datenform** anpassen, nicht deren Inhalt.

- **Erlaubt**:  
  - Indizes ↔ One-Hot Konvertierung  
  - Aufsplitten von `[B,T,D]` in mehrere Eingaben (z. B. Aktivitäten, Ressourcen, Zeitfeature)  
  - Erzeugen von Attention-Masks aus `pad_id`  
  - Änderung von Datentypen (`int64` → `int32`)  
  - Padding/Clipping auf `max_prefix_len`  

- **Nicht erlaubt**:  
  - Neue Splits erstellen  
  - Labels verschieben oder neu berechnen  
  - Re-Kodierung von Aktivitäten/Vokabular  
  - Abweichende Normalisierung oder Zeit-Einheiten  
  - Daten augmentieren oder filtern  

### Konformität mit dem Benchmark
Die Benchmark-Invarianten gelten für **alle Modelle**:
- Einheitliche **Tasks & Targets** (Next Activity, Next Time, Remaining Time, Suffix)  
- Gleiche **Train/Val/Test-Splits** nach Case-ID  
- Einheitliche **Preprocessing-Pipeline** (einmalig ausgeführt)  
- Einheitliche **Metriken** pro Task  
- Dokumentierte **Hardware-/Software-Umgebung**  

Adapter sind somit nur „Stecker“, die gewährleisten, dass jedes Modell exakt dieselben Daten konsumiert – lediglich in der Form, die es benötigt.

### Validierung
- Hashes der kanonischen Artefakte werden veröffentlicht.  
- Unit Tests prüfen u. a.:  
  - Round-Trip Konsistenz (`argmax(one_hot(y)) == y_index`)  
  - Masken entsprechen dem `pad_id`  
  - Sample-Anzahl pro Split ist identisch über alle Adapter  

Dieses Vorgehen stellt sicher, dass die Ergebnisse der Modelle **vergleichbar und reproduzierbar** bleiben.


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
