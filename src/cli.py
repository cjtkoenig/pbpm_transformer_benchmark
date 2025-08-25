import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.seeding import set_all_seeds
from src.tasks.next_activity import NextActivityTask
from src.tasks.suffix import SuffixTask
from src.tasks.next_time import NextTimeTask
from src.tasks.remaining_time import RemainingTimeTask
from src.tasks.multi_task import MultiTaskTask
from src.data.preprocessor import SimplePreprocessor


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(config: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(config))

    project_root = Path(hydra.utils.get_original_cwd())
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(config.seed)

    # Convert config to dict for task usage
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Handle preprocessing management commands
    if hasattr(config, 'preprocess_action') and config.preprocess_action:
        handle_preprocess_action(config, project_root)
        return
    
    # Route to appropriate task based on config
    if config.task == "next_activity":
        task = NextActivityTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nNext Activity Task completed. Results: {results}")
    elif config.task == "suffix":
        task = SuffixTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nSuffix Task completed. Results: {results}")
    elif config.task == "next_time":
        task = NextTimeTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nNext Time Task completed. Results: {results}")
    elif config.task == "remaining_time":
        task = RemainingTimeTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nRemaining Time Task completed. Results: {results}")
    elif config.task == "multi_task":
        task = MultiTaskTask(config_dict)
        results = task.run(
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nMulti-Task Task completed. Results: {results}")
    else:
        raise ValueError(f"Task '{config.task}' not implemented yet")

    # Minimal environment capture
    env_info = {
        "torch": torch.__version__,
        "accelerator": config.train.accelerator,
        "seed": int(config.seed),
        "task": config.task
    }
    (outputs_dir / "env.json").write_text(json.dumps(env_info, indent=2))
    print("\nSaved environment info to outputs/env.json")


def handle_preprocess_action(config: DictConfig, project_root: Path):
    """Handle preprocessing management actions."""
    processed_dir = project_root / config.data.path_processed
    raw_dir = project_root / config.data.path_raw
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    preprocessor = SimplePreprocessor(raw_dir, processed_dir, config_dict)
    
    if config.preprocess_action == "info":
        processed_info = preprocessor.get_processed_info()
        print("\n=== Processed Data Information ===")
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
        print(f"Force preprocessing datasets: {datasets}")
        
        for dataset_name in datasets:
            try:
                prefixes_df, labels_series, vocabulary = preprocessor.preprocess_dataset(
                    dataset_name, force_reprocess=True
                )
                print(f"Successfully preprocessed {dataset_name}")
                print(f"  - Prefixes: {len(prefixes_df)}")
                print(f"  - Vocabulary size: {len(vocabulary.index_to_token)}")
            except Exception as e:
                print(f"Error preprocessing {dataset_name}: {e}")
    
    else:
        print(f"Unknown preprocess action: {config.preprocess_action}")
        print("Available actions: info, clear, force")


if __name__ == "__main__":
    main()
