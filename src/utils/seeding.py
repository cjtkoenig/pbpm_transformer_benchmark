import random
import numpy as numpy_lib
import torch

def set_all_seeds(seed_value: int) -> None:
    """Set Python, NumPy and Torch seeds for reproducibility."""
    random.seed(seed_value)
    numpy_lib.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
