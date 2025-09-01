import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.seeding import set_all_seeds
from src.data.preprocessor import SimplePreprocessor


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(config: DictConfig):
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
    print(f"Learning rate:  {config.train.learning_rate}")
    print(f"Seed:           {config.seed}")
    print()

    project_root = Path(hydra.utils.get_original_cwd())
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(config.seed)

    # Convert config to dict for task usage
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Handle analysis management commands
    if hasattr(config, 'analysis') and getattr(config.analysis, 'action', None) == 'run_stats':
        from src.analysis.stat_tests import run_stats
        report = run_stats(str(project_root / "outputs"))
        print("\nAnalysis summary saved to outputs/analysis/summary.json")
        print(json.dumps(report, indent=2))
        return

    # Handle preprocessing management commands
    if hasattr(config, 'preprocess_action') and config.preprocess_action:
        handle_preprocess_action(config, project_root)
        return
    
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
    elif config.task == "suffix":
        from src.tasks.suffix import SuffixTask
        task = SuffixTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\n✓ Suffix Task completed successfully")
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

    # Minimal environment capture (extended)
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except Exception:
        tf_version = None
    import sys, platform
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
