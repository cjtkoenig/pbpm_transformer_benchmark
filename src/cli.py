import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.seeding import set_all_seeds
from src.tasks.next_activity import run_next_activity_task



@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(config: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(config))

    project_root = Path(hydra.utils.get_original_cwd())
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(config.seed)

    # Convert config to dict for task usage
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Route to appropriate task based on config
    if config.task == "next_activity":
        results = run_next_activity_task(
            config=config_dict,
            datasets=config.data.datasets,
            raw_directory=project_root / config.data.path_raw,
            outputs_dir=outputs_dir
        )
        print(f"\nNext Activity Task completed. Results: {results}")
    # TODO: other tasks
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


if __name__ == "__main__":
    main()
