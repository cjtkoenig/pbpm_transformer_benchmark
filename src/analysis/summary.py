"""
Lightweight benchmark summary: aggregate per-fold metrics and produce simple rankings.

- Input: outputs/<dataset>/<task>/<model>/<fold>/metrics.json produced by tasks.
- Output: outputs/analysis/summary.json with per-dataset, per-task ranked model lists.
  For multitask, emits two rankings: multitask.classification (accuracy) and multitask.time (mae).

When to use this:
- Quick, dependency-light overview right after running experiments.
- No statistical significance testing; suitable for sanity checks and reports that
  only need mean metrics and a naive ranking.

For full statistical methodology (Plackett–Luce, hierarchical Bayes, Friedman/Wilcoxon),
use src.utils.statistical_analysis.BenchmarkStatisticalAnalysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Minimal analysis scaffold: aggregates per-task metrics across folds and ranks models.
# Define per-task metric specs. For tasks with multiple metrics (e.g., multitask),
# we emit separate rankings per metric alias.
# Each spec is a tuple: (alias, metric_key_in_metrics_json, direction)
# direction: 'max' or 'min'
TASK_METRICS = {
    "next_activity": [("accuracy", "accuracy", "max")],
    "next_time": [("mae", "mae", "min")],
    "remaining_time": [("mae", "mae", "min")],
    # Multitask emits two leaderboards: classification (accuracy) and time (mae)
    "multitask": [
        ("classification", "accuracy", "max"),
        ("time", "mae", "min"),
    ],
}


def _collect_metrics(outputs_dir: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Walk outputs directory and collect metrics per dataset/task/model.
    Returns nested dict: dataset -> task -> model -> list of metric dicts
    """
    results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for dataset_dir in outputs_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in ("checkpoints",):
            continue
        dataset = dataset_dir.name
        results.setdefault(dataset, {})
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            results[dataset].setdefault(task, {})
            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                fold_metrics = []
                for fold_dir in model_dir.iterdir():
                    if not fold_dir.is_dir():
                        continue
                    mfile = fold_dir / "metrics.json"
                    if mfile.exists():
                        try:
                            fold_metrics.append(json.loads(mfile.read_text()))
                        except Exception:
                            pass
                if fold_metrics:
                    results[dataset][task].setdefault(model, []).extend(fold_metrics)
    return results


def _aggregate_model_scores(metrics_list: List[Dict[str, Any]], metric_key: str, direction: str) -> float:
    if not metric_key or not metrics_list:
        return float("nan")
    vals = [m.get(metric_key) for m in metrics_list if isinstance(m.get(metric_key), (int, float))]
    if not vals:
        return float("nan")
    mean_val = float(np.mean(vals))
    # Convert to a unified score where higher is better by inverting when minimizing
    if direction == "min":
        return -mean_val
    return mean_val


def run_stats(outputs_dir: str) -> Dict[str, Any]:
    """
    Aggregate metrics and produce a simple ranking per dataset/task.

    Notes
    -----
    - This function does not perform statistical significance testing.
      It does not run Plackett–Luce, hierarchical Bayes, Friedman, or Wilcoxon.
    """
    base = Path(outputs_dir)
    collected = _collect_metrics(base)
    ranking_report: Dict[str, Any] = {}

    for dataset, task_map in collected.items():
        ranking_report.setdefault(dataset, {})
        for task, model_map in task_map.items():
            metric_specs = TASK_METRICS.get(task, [])
            if not metric_specs:
                # Fallback: try to infer from common keys
                metric_specs = [("score", "accuracy", "max")]
            # If there's a single metric spec, keep backward-compatible shape
            if len(metric_specs) == 1:
                alias, metric_key, direction = metric_specs[0]
                scores = []
                for model, mlist in model_map.items():
                    score = _aggregate_model_scores(mlist, metric_key, direction)
                    if not np.isnan(score):
                        scores.append((model, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                ranking_report[dataset][task] = {
                    "metric": metric_key,
                    "direction": direction,
                    "ranking": [{"model": m, "score": s} for m, s in scores],
                }
            else:
                # Multiple metrics: nest under task
                ranking_report[dataset].setdefault(task, {})
                for alias, metric_key, direction in metric_specs:
                    scores = []
                    for model, mlist in model_map.items():
                        score = _aggregate_model_scores(mlist, metric_key, direction)
                        if not np.isnan(score):
                            scores.append((model, score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    ranking_report[dataset][task][alias] = {
                        "metric": metric_key,
                        "direction": direction,
                        "ranking": [{"model": m, "score": s} for m, s in scores],
                    }
    # Save summary
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(ranking_report, indent=2))
    return ranking_report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Summarize benchmark outputs into simple rankings.")
    ap.add_argument("--outputs", default="outputs", help="Path to outputs directory")
    args = ap.parse_args()
    rep = run_stats(args.outputs)
    print(json.dumps(rep, indent=2))
