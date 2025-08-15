import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class LocalLogger:
    """Simple logger for experiment tracking."""
    
    def __init__(self, outputs_dir: Path):
        """Initialize  logger"""
        self.outputs_dir = outputs_dir
        self.logs_dir = outputs_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"experiment_{timestamp}.log"
        
        print(f"Logging to: {self.log_file}")
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to local file"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "metrics": metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # print to console for immediate feedback
        if step is not None:
            print(f"Step {step}: {metrics}")
        else:
            print(f"Metrics: {metrics}")
    
    def log_results(self, results: Dict[str, Any], dataset_name: str):
        """Save results to local file."""
        results_file = self.outputs_dir / f"results_{dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
    
    def finish(self):
        """Finish logging."""
        print(f"Experiment log saved to: {self.log_file}")


def create_logger(config: Dict[str, Any], outputs_dir: Path) -> LocalLogger:
    return LocalLogger(outputs_dir)
