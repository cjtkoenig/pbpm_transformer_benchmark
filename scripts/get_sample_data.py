
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def generate_synthetic_event_log(n_cases=100, max_events_per_case=10):
    """
    Generate a synthetic event log for testing.
    Args:
        n_cases: Number of process cases to generate
        max_events_per_case: Maximum number of events per case
    Returns:
        pandas.DataFrame: Event log with columns case:concept:name, concept:name, time:timestamp
    """
    
    activities = ["Start", "Review", "Approve", "Reject", "Complete", "Archive"]
    patterns = [
        ["Start", "Review", "Approve", "Complete"],
        ["Start", "Review", "Reject", "Complete"],
        ["Start", "Review", "Approve", "Archive"],
        ["Start", "Review", "Reject", "Archive"],
    ]
    
    event_records = []
    base_time = datetime(2024, 1, 1)
    
    for case_id in range(1, n_cases + 1):
        pattern = random.choice(patterns)
        if random.random() < 0.3:  # 30% chance to skip an activity
            pattern = [act for act in pattern if random.random() > 0.2]
        case_start_time = base_time + timedelta(days=random.randint(0, 30))
        
        for event_idx, activity in enumerate(pattern):
            event_time = case_start_time + timedelta(
                hours=event_idx * random.randint(1, 4),
                minutes=random.randint(0, 59)
            )
            
            event_records.append({
                "case:concept:name": f"Case_{case_id:04d}",
                "concept:name": activity,
                "time:timestamp": event_time
            })
    
    return pd.DataFrame(event_records)

def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic event log...")
    event_log = generate_synthetic_event_log(n_cases=200, max_events_per_case=8)
    csv_path = data_dir / "sample_synth.csv"
    event_log.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()
