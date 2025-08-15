"""Print minimal system info for reproducibility."""
import platform
import sys
import torch

def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("Torch version:", torch.__version__)
    mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    print("MPS available:", bool(mps_available))

if __name__ == "__main__":
    main()
