import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.seeding import set_all_seeds
from src.data.preprocessor import SimplePreprocessor


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(config: DictConfig):
    # Resolve learning rate with explicit per-model defaults
    model_name = str(getattr(config.model, 'name', 'process_transformer'))
    lr_scalar = getattr(config.train, 'learning_rate', None)
    lr_map = getattr(config.train, 'learning_rates', None)
    if lr_scalar is not None:
        resolved_lr = float(lr_scalar)
        lr_source = 'train.learning_rate override'
    elif lr_map is not None and model_name in lr_map:
        resolved_lr = float(lr_map[model_name])
        lr_source = f'train.learning_rates[{model_name}]'
    else:
        raise ValueError(
            f"No learning rate defined for model '{model_name}'. "
            "Provide train.learning_rate or add an entry under train.learning_rates."
        )
    config.train.learning_rate = resolved_lr

    # Fixed outputs directory under project root (no config)
    project_root = Path(hydra.utils.get_original_cwd())
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Print clean configuration header
    print("\n" + "=" * 80)
    print(" PBPM TRANSFORMER BENCHMARK")
    print("=" * 80)
    
    print(f"Task:           {config.task}")
    print(f"Model:          {config.model.name}")
    print(f"Datasets:       {', '.join(config.data.datasets)}")
    print(f"Cross-validation: {config.cv.n_folds} folds")
    print(f"Batch size:     {config.train.batch_size}")
    print(f"Max epochs:     {config.train.max_epochs}")
    print(f"Learning rate:  {config.train.learning_rate} ({lr_source})")
    print(f"Seed:           {config.seed}")
    print(f"Outputs dir:    {outputs_dir}")
    print()

    set_all_seeds(config.seed)

    # Convert config to dict for task usage
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Handle analysis management commands
    if hasattr(config, 'analysis') and getattr(config.analysis, 'action', None) == 'run_stats':
        from src.analysis.summary import run_stats
        report = run_stats(str(outputs_dir))
        print(f"\nAnalysis summary saved to {outputs_dir}/analysis/summary.json")
        print(json.dumps(report, indent=2))
        return
    if hasattr(config, 'analysis') and getattr(config.analysis, 'action', None) == 'run_full_stats':
        # Full statistical analysis: Plackett–Luce, Hierarchical Bayes, Friedman/Wilcoxon
        from src.utils.statistical_analysis import _load_summary_or_collect, BenchmarkStatisticalAnalysis
        # Determine task to analyze; default to config.task
        task_to_analyze = str(getattr(config, 'task', 'next_activity'))
        data = _load_summary_or_collect(outputs_dir)
        datasets = sorted([d for d, tmap in data.items() if task_to_analyze in tmap])
        models = sorted({m for d in datasets for m in data[d][task_to_analyze].keys()})
        if not datasets or not models:
            print(f"No data found for task {task_to_analyze} under {outputs_dir}")
            return
        results = {d: {task_to_analyze: data[d][task_to_analyze]} for d in datasets}
        BA = BenchmarkStatisticalAnalysis(results, models, datasets)
        report = BA.generate_comprehensive_report(task_to_analyze, include_plackett_luce=True, include_hierarchical_bayes=True)
        out_dir = outputs_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"full_report_{task_to_analyze}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nFull statistical report saved to {out_path}")
        return
    if hasattr(config, 'analysis') and getattr(config.analysis, 'action', None) == 'thesis_report':
        # Thesis-aligned report: within-track comparisons, uplift for extended
        from src.utils.statistical_analysis import generate_thesis_report
        task_to_analyze = str(getattr(getattr(config, 'analysis', {}), 'task', 'all'))
        rep = generate_thesis_report(outputs_dir, task=task_to_analyze)
        out_dir = outputs_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("thesis_report.json" if task_to_analyze == 'all' else f"thesis_report_{task_to_analyze}.json")
        out_path.write_text(json.dumps(rep, indent=2))
        print(f"\nThesis-aligned report saved to {out_path}")
        return

    # Handle preprocessing management commands
    if hasattr(config, 'preprocess_action') and config.preprocess_action:
        handle_preprocess_action(config, project_root)
        return
    
    # Safety barrier: enforce model-task compatibility
    model_name = str(getattr(config.model, 'name', 'process_transformer'))
    task_name = str(config.task)
    single_tasks = {"next_activity", "next_time", "remaining_time"}
    if model_name == "mtlformer" and task_name in single_tasks:
        raise ValueError(
            "Invalid configuration: mtlformer only supports task=multitask. "
            "Please run: uv run python -m src.cli task=multitask model.name=mtlformer"
        )
    if model_name == "process_transformer" and task_name == "multitask":
        raise ValueError(
            "Invalid configuration: process_transformer does not support task=multitask. "
            "Please choose one of: next_activity, next_time, remaining_time"
        )
    # Enforce minimal attribute mode for ProcessTransformer and MTLFormer
    if model_name in {"process_transformer", "mtlformer"}:
        attr_mode = str(getattr(config.data, 'attribute_mode', 'minimal'))
        if attr_mode != 'minimal':
            raise ValueError(
                "Invalid configuration: process_transformer and mtlformer only support data.attribute_mode=minimal in this benchmark."
            )
    if model_name == "activity_only_lstm":
        if task_name != "next_activity":
            raise ValueError(
                "Invalid configuration: activity_only_lstm supports only task=next_activity."
            )
        attr_mode = str(getattr(config.data, 'attribute_mode', 'minimal'))
        if attr_mode != 'minimal':
            raise ValueError(
                "Invalid configuration: activity_only_lstm is an activities-only model. "
                "Please set data.attribute_mode=minimal."
            )
    if model_name in {"shared_lstm", "specialised_lstm"}:
        if task_name != "next_activity":
            raise ValueError("Invalid configuration: shared_lstm/specialised_lstm support only task=next_activity.")
        attr_mode = str(getattr(config.data, 'attribute_mode', 'minimal'))
        if attr_mode != 'extended':
            raise ValueError(
                "Invalid configuration: shared_lstm and specialised_lstm are reserved for extended attribute mode. "
                "Please set data.attribute_mode=extended. For activities-only, use model.name=activity_only_lstm."
            )
    
    # Route to appropriate task based on config
    if config.task == "next_activity":
        from src.tasks.next_activity import NextActivityTask
        task = NextActivityTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\n✓ Next Activity Task completed successfully")
        print(f"  Results saved to outputs/")
    elif config.task == "next_time":
        from src.tasks.next_time import NextTimeTask
        task = NextTimeTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\n✓ Next Time Task completed successfully")
        print(f"  Results saved to outputs/")
    elif config.task == "remaining_time":
        from src.tasks.remaining_time import RemainingTimeTask
        task = RemainingTimeTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\n✓ Remaining Time Task completed successfully")
        print(f"  Results saved to outputs/")
    elif config.task == "multitask":
        from src.tasks.multitask import MultiTaskLearningTask
        task = MultiTaskLearningTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\n✓ Multi-Task Learning completed successfully")
        print(f"  Results saved to outputs/")

    else:
        raise ValueError(f"Task '{config.task}' not implemented yet")

    # environment capture
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except Exception:
        tf_version = None
    import sys, platform
    # Determine track
    _model = str(config.model.name)
    if _model in {"process_transformer", "mtlformer", "activity_only_lstm"}:
        track = "minimal"
    elif _model in {"shared_lstm", "specialised_lstm", "pgtnet"}:
        track = "extended"
    else:
        track = None
    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "tensorflow": tf_version,
        "accelerator": config.train.accelerator,
        "mps": bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
        "cuda": torch.cuda.is_available(),
        "devices": int(getattr(config.train, 'devices', 1)),
        "seed": int(config.seed),
        "task": config.task,
        "model": config.model.name,
        "data_attribute_mode": getattr(config.data, 'attribute_mode', None),
        "track": track,
    }
    (outputs_dir / "env.json").write_text(json.dumps(env_info, indent=2))
    print(f"\nEnvironment info saved to outputs/env.json")


def handle_preprocess_action(config: DictConfig, project_root: Path):
    """Handle preprocessing management actions."""
    processed_dir = project_root / config.data.path_processed
    raw_dir = project_root / config.data.path_raw
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    preprocessor = SimplePreprocessor(raw_dir, processed_dir, config_dict)
    
    if config.preprocess_action == "info":
        processed_info = preprocessor.get_processed_info()
        print("\n" + "-" * 80)
        print(" PROCESSED DATA INFORMATION")
        print("-" * 80)
        print(f"Processed directory: {processed_info['processed_dir']}")
        print(f"Total processed files: {processed_info['total_files']}")
        if processed_info['processed_datasets']:
            print("Processed datasets:")
            for dataset_name in processed_info['processed_datasets']:
                print(f"  - {dataset_name}")
        else:
            print("No processed datasets found.")
    
    elif config.preprocess_action == "clear":
        dataset_name = getattr(config, 'dataset_name', None)
        preprocessor.clear_processed_data(dataset_name)
    
    elif config.preprocess_action == "force":
        # Force preprocessing of specified datasets
        datasets = getattr(config, 'datasets', config.data.datasets)
        print(f"\nForce preprocessing datasets: {datasets}")
        
        for dataset_name in datasets:
            try:
                processing_info = preprocessor.preprocess_dataset(
                    dataset_name, force_reprocess=True
                )
                print(f"✓ Successfully preprocessed {dataset_name}")
                # Report canonical summary consistent with new preprocessing pipeline
                vocab = processing_info.get("vocabularies", {})
                x_word_dict = vocab.get("x_word_dict", {})
                y_word_dict = vocab.get("y_word_dict", {})
                total_cases = processing_info.get("total_cases")
                total_events = processing_info.get("total_events")
                tasks = processing_info.get("tasks", [])
                if total_cases is not None and total_events is not None:
                    print(f"  - Cases: {total_cases}, Events: {total_events}")
                print(f"  - Vocabulary size: {len(x_word_dict) if isinstance(x_word_dict, dict) else 'n/a'} (input), {len(y_word_dict) if isinstance(y_word_dict, dict) else 'n/a'} (labels)")
                if tasks:
                    print(f"  - Tasks prepared: {', '.join(tasks)}")
            except Exception as e:
                print(f"✗ Error preprocessing {dataset_name}: {e}")
    
    else:
        print(f"Unknown preprocess action: {config.preprocess_action}")
        print("Available actions: info, clear, force")


if __name__ == "__main__":
    main()
