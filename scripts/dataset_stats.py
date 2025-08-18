
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime

def calculate_dataset_stats(file_path: str):
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.data.loader import load_event_log
    
    print(f"Loading dataset: {file_path}")
    
    try:
        df = load_event_log(file_path)

        # Calculate basic statistics
        total_events = len(df)
        total_cases = df["case:concept:name"].nunique()
        total_activities = df["concept:name"].nunique()
        
        # Calculate case length statistics
        case_lengths = df.groupby("case:concept:name").size()
        avg_case_length = case_lengths.mean()
        min_case_length = case_lengths.min()
        max_case_length = case_lengths.max()

        stats = {
            "file_path": file_path,
            "total_events": total_events,
            "total_cases": total_cases,
            "total_activities": total_activities,
            "avg_case_length": avg_case_length,
            "min_case_length": min_case_length,
            "max_case_length": max_case_length,
            "case_length_distribution": case_lengths.value_counts().sort_index().to_dict()
        }
        
        return stats
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def print_stats(stats):
    """
    Print statistics in a simple format for table use.
    
    Args:
        stats: Dictionary containing dataset statistics
    """
    if not stats:
        return
    
    dataset_name = Path(stats['file_path']).stem
    
    print(f"{dataset_name}:")
    print(f"Fälle (Traces): {stats['total_cases']:,}")
    print(f"Anzahl an Aktivitäten: {stats['total_activities']:,}")
    print(f"Anzahl an Events: {stats['total_events']:,}")
    print(f"Ø Trace-Länge: {stats['avg_case_length']:.2f} events")
    print()

def main():
    """Main function to handle command line arguments and process datasets."""
    parser = argparse.ArgumentParser(description="Calculate statistics for event log datasets")
    parser.add_argument("file_path", help="Path to the event log CSV file")
    parser.add_argument("--output", "-o", help="Output file for statistics (JSON format)")
    
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist")
        sys.exit(1)
    
    stats = calculate_dataset_stats(args.file_path)
    
    if stats:
        print_stats(stats)
        
        if args.output:
            import json
            # Convert datetime objects and numpy types to JSON serializable format
            stats_json = {}
            for k, v in stats.items():
                if isinstance(v, (pd.Timedelta, datetime)):
                    stats_json[k] = str(v)
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    stats_json[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    stats_json[k] = float(v)
                else:
                    stats_json[k] = v
            
            with open(args.output, 'w') as f:
                json.dump(stats_json, f, indent=2)
            print(f"\nStatistics saved to: {args.output}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
