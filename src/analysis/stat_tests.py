import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Minimal analysis scaffold: aggregates per-task metrics across folds and ranks models.
PREFER_MAX = {
    "next_activity": "accuracy",
    "suffix": "accuracy",  # placeholder until sequence metric is implemented
    "next_time": "mae",    # lower is better; we will invert for ranking
    "remaining_time": "mae" # lower is better; we will invert for ranking
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


def _aggregate_model_scores(metrics_list: List[Dict[str, Any]], task: str) -> float:
    key = PREFER_MAX.get(task, None)
    if not key or not metrics_list:
        return float("nan")
    vals = [m.get(key) for m in metrics_list if isinstance(m.get(key), (int, float))]
    if not vals:
        return float("nan")
    mean_val = float(np.mean(vals))
    # For MAE-like metrics we minimize; convert to score by negative
    if task in ("next_time", "remaining_time") and key in ("mae",):
        return -mean_val
    return mean_val


def run_stats(outputs_dir: str) -> Dict[str, Any]:
    base = Path(outputs_dir)
    collected = _collect_metrics(base)
    ranking_report: Dict[str, Any] = {}

    for dataset, task_map in collected.items():
        ranking_report.setdefault(dataset, {})
        for task, model_map in task_map.items():
            scores = []
            for model, mlist in model_map.items():
                score = _aggregate_model_scores(mlist, task)
                if not np.isnan(score):
                    scores.append((model, score))
            # sort descending by score (since we inverted MAE)
            scores.sort(key=lambda x: x[1], reverse=True)
            ranking_report[dataset][task] = {
                "ranking": [{"model": m, "score": s} for m, s in scores]
            }
    # Save summary
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(ranking_report, indent=2))
    return ranking_report
