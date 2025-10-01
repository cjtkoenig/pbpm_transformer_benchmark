"""
Statistical analysis utilities for benchmark evaluation.
Implements Plackett-Luce ranking models, hierarchical Bayes testing, and non-parametric tests
following Rama-Maneiro et al. (2021).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import warnings
import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

import pymc as pm
import arviz as az
from scipy.optimize import minimize


class RankingDirection(Enum):
    """Direction for ranking (higher is better vs lower is better)."""
    HIGHER_BETTER = "higher"
    LOWER_BETTER = "lower"


@dataclass
class RankingData:
    """Container for ranking data across datasets."""
    rankings: List[List[int]]  # List of rankings per dataset
    model_names: List[str]
    dataset_names: List[str]
    direction: RankingDirection


class PlackettLuceModel:
    """
    Implementation of the Plackett-Luce ranking model.
    
    The Plackett-Luce model assumes that the probability of observing a ranking
    is proportional to the product of choice probabilities at each position.
    """
    
    def __init__(self, ranking_data: RankingData):
        """
        Initialize Plackett-Luce model.
        
        Args:
            ranking_data: RankingData object containing rankings and metadata
        """
        self.ranking_data = ranking_data
        self.n_models = len(ranking_data.model_names)
        self.n_datasets = len(ranking_data.dataset_names)
        
    def _log_likelihood(self, strengths: np.ndarray) -> float:
        """
        Calculate log-likelihood of the Plackett-Luce model.
        
        Args:
            strengths: Array of model strengths (log-scale)
            
        Returns:
            Log-likelihood value
        """
        log_likelihood = 0.0
        
        for ranking in self.ranking_data.rankings:
            if len(ranking) == 0:
                continue
            
            # ranking[i] gives the rank of model i (1-based)
            # We need to find which model has rank 1, 2, 3, etc.
            ranked_models = [0] * len(ranking)  # ranked_models[rank-1] = model_index
            for model_idx, rank in enumerate(ranking):
                ranked_models[rank - 1] = model_idx
            
            # Calculate log-likelihood for this ranking
            for i, model_idx in enumerate(ranked_models):
                # Remaining models at position i
                remaining_models = ranked_models[i:]
                remaining_strengths = strengths[remaining_models]
                
                # Log probability of choosing model_idx from remaining models
                log_prob = strengths[model_idx] - np.log(np.sum(np.exp(remaining_strengths)))
                log_likelihood += log_prob
        
        return log_likelihood
    
    def fit(self, method: str = 'BFGS', max_iter: int = 1000) -> Dict[str, Any]:
        """
        Fit Plackett-Luce model using maximum likelihood estimation.
        
        Args:
            method: Optimization method ('BFGS', 'L-BFGS-B', etc.)
            max_iter: Maximum iterations for optimization
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        # Initialize strengths at zero (equal probability)
        initial_strengths = np.zeros(self.n_models)
        
        # Constrain one strength to zero for identifiability
        bounds = [(-10, 10)] * self.n_models
        bounds[0] = (0, 0)  # Fix first model strength to 0
        
        # Optimize log-likelihood
        result = minimize(
            fun=lambda x: -self._log_likelihood(x),  # Minimize negative log-likelihood
            x0=initial_strengths,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        if not result.success:
            warnings.warn(f"Plackett-Luce optimization failed: {result.message}")
        
        # Convert strengths to probabilities
        fitted_strengths = result.x
        probabilities = np.exp(fitted_strengths) / np.sum(np.exp(fitted_strengths))
        
        return {
            'strengths': fitted_strengths,
            'probabilities': probabilities,
            'log_likelihood': -result.fun,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit
        }
    
    def predict_ranking_probability(self, ranking: List[int], strengths: np.ndarray) -> float:
        """
        Predict probability of observing a specific ranking.
        
        Args:
            ranking: List of model ranks (1-based)
            strengths: Model strengths from fitted model
            
        Returns:
            Probability of the ranking
        """
        if len(ranking) == 0:
            return 1.0
        
        log_prob = 0.0
        
        # ranking[i] gives the rank of model i (1-based)
        # We need to find which model has rank 1, 2, 3, etc.
        ranked_models = [0] * len(ranking)  # ranked_models[rank-1] = model_index
        for model_idx, rank in enumerate(ranking):
            ranked_models[rank - 1] = model_idx
        
        for i, model_idx in enumerate(ranked_models):
            remaining_models = ranked_models[i:]
            remaining_strengths = strengths[remaining_models]
            
            log_prob += strengths[model_idx] - np.log(np.sum(np.exp(remaining_strengths)))
        
        return np.exp(log_prob)


class HierarchicalBayesModel:
    """
    Hierarchical Bayesian model for benchmark analysis.
    
    Implements a hierarchical model where:
    - Dataset-level effects are modeled as random effects
    - Model performance is modeled as fixed effects
    - Uncertainty is properly quantified
    """
    
    def __init__(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                 model_names: List[str], dataset_names: List[str]):
        """
        Initialize hierarchical Bayes model.
        
        Args:
            results: Dictionary with results structure {dataset: {task: {model: metric_value}}}
            model_names: List of model names
            dataset_names: List of dataset names
        """
        self.results = results
        self.model_names = model_names
        self.dataset_names = dataset_names
        self.n_models = len(model_names)
        self.n_datasets = len(dataset_names)
        

    
    def fit_hierarchical_model(self, task: str, 
                              n_samples: int = 2000, 
                              n_tune: int = 1000) -> Dict[str, Any]:
        """
        Fit hierarchical Bayesian model for a specific task.
        
        Args:
            task: Task name
            n_samples: Number of posterior samples
            n_tune: Number of tuning samples
            
        Returns:
            Dictionary with model results and diagnostics
        """
        # Prepare data for PyMC
        data = self._prepare_data_for_pymc(task)
        
        if data is None:
            return {'error': 'No valid data for hierarchical model'}
        
        with pm.Model() as model:
            # Prior for global model effects
            model_effects = pm.Normal('model_effects', mu=0, sigma=1, shape=self.n_models)
            
            # Prior for dataset random effects
            dataset_effects = pm.Normal('dataset_effects', mu=0, sigma=0.5, shape=self.n_datasets)
            
            # Prior for observation noise
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Expected performance
            mu = (model_effects[data['model_idx']] + 
                  dataset_effects[data['dataset_idx']])
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data['values'])
            
            # Sample from posterior
            trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True)
        
        # Extract results
        model_effects_samples = trace.posterior['model_effects'].values
        dataset_effects_samples = trace.posterior['dataset_effects'].values
        
        # Calculate posterior summaries
        model_effects_summary = az.summary(trace, var_names=['model_effects'])
        dataset_effects_summary = az.summary(trace, var_names=['dataset_effects'])
        
        # Calculate pairwise differences
        pairwise_differences = self._calculate_pairwise_differences(model_effects_samples)
        
        return {
            'trace': trace,
            'model_effects_summary': model_effects_summary,
            'dataset_effects_summary': dataset_effects_summary,
            'pairwise_differences': pairwise_differences,
            'model': model,
            'data': data
        }
    
    def _prepare_data_for_pymc(self, task: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Prepare data for PyMC hierarchical model.
        
        Args:
            task: Task name
            
        Returns:
            Dictionary with prepared data or None if insufficient data
        """
        values = []
        model_idx = []
        dataset_idx = []
        
        for i, dataset in enumerate(self.dataset_names):
            if dataset not in self.results or task not in self.results[dataset]:
                continue
                
            for j, model in enumerate(self.model_names):
                if model in self.results[dataset][task]:
                    values.append(self.results[dataset][task][model])
                    model_idx.append(j)
                    dataset_idx.append(i)
        
        if len(values) < 3:
            return None
        
        return {
            'values': np.array(values),
            'model_idx': np.array(model_idx),
            'dataset_idx': np.array(dataset_idx)
        }
    
    def _calculate_pairwise_differences(self, model_effects_samples: np.ndarray) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Calculate pairwise differences between models with uncertainty.
        
        Args:
            model_effects_samples: Posterior samples of model effects
            
        Returns:
            Dictionary with pairwise difference statistics
        """
        pairwise_results = {}
        
        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names[i+1:], i+1):
                # Calculate differences
                differences = model_effects_samples[:, :, i] - model_effects_samples[:, :, j]
                
                # Calculate statistics
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                quantiles = np.percentile(differences, [2.5, 25, 50, 75, 97.5])
                
                # Calculate probability that model1 > model2
                prob_better = np.mean(differences > 0)
                
                pairwise_results[(model1, model2)] = {
                    'mean_difference': mean_diff,
                    'std_difference': std_diff,
                    'quantiles': quantiles,
                    'prob_model1_better': prob_better,
                    'credible_interval_95': [quantiles[0], quantiles[4]],
                    'credible_interval_50': [quantiles[1], quantiles[3]]
                }
        
        return pairwise_results


class BenchmarkStatisticalAnalysis:
    """
    Comprehensive statistical analysis for benchmark results following Rama-Maneiro et al. (2021).
    
    Implements:
    - Plackett-Luce ranking models
    - Hierarchical Bayesian models
    - Non-parametric tests (Friedman, Wilcoxon)
    - Average ranks (Borda)
    """
    
    def __init__(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                 model_names: List[str], dataset_names: List[str]):
        """
        Initialize statistical analysis.
        
        Args:
            results: Dictionary with results structure {dataset: {task: {model: metric_value}}}
            model_names: List of model names
            dataset_names: List of dataset names
        """
        self.results = results
        self.model_names = model_names
        self.dataset_names = dataset_names
        
        # Define task-specific ranking directions
        self.task_directions = {
            'next_activity': RankingDirection.HIGHER_BETTER,      # Accuracy - higher is better
            'next_time': RankingDirection.LOWER_BETTER,           # MAE - lower is better
            'remaining_time': RankingDirection.LOWER_BETTER       # MAE - lower is better
        }
    
    def _get_rankings_for_task(self, task: str) -> RankingData:
        """
        Extract rankings for a specific task across all datasets.
        
        Args:
            task: Task name
            
        Returns:
            RankingData object with rankings and metadata
        """
        rankings = []
        valid_datasets = []
        
        for dataset in self.dataset_names:
            if (dataset in self.results and 
                task in self.results[dataset]):
                
                # Get models that have results for this dataset
                available_models = [model for model in self.model_names 
                                  if model in self.results[dataset][task]]
                
                if len(available_models) == len(self.model_names):
                    # Create ranking
                    dataset_results = self.results[dataset][task]
                    reverse = (self.task_directions.get(task, RankingDirection.HIGHER_BETTER) ==
                              RankingDirection.HIGHER_BETTER)

                    # Sort models by performance
                    sorted_models = sorted(available_models,
                                         key=lambda m: dataset_results[m],
                                         reverse=reverse)
                    
                    # Convert to ranks (1-based)
                    model_to_rank = {model: rank + 1 for rank, model in enumerate(sorted_models)}
                    ranking = [model_to_rank[model] for model in self.model_names]
                    
                    rankings.append(ranking)
                    valid_datasets.append(dataset)
        
        return RankingData(
            rankings=rankings,
            model_names=self.model_names,
            dataset_names=valid_datasets,
            direction=self.task_directions.get(task, RankingDirection.HIGHER_BETTER)
        )
    
    def fit_plackett_luce_model(self, task: str) -> Dict[str, Any]:
        """
        Fit Plackett-Luce ranking model for a specific task.
        
        Args:
            task: Task name
            
        Returns:
            Dictionary with fitted model results
        """
        ranking_data = self._get_rankings_for_task(task)
        
        if len(ranking_data.rankings) < 2:
            warnings.warn(f"Insufficient rankings ({len(ranking_data.rankings)}) for Plackett-Luce model")
            return {'error': 'Insufficient data'}
        
        pl_model = PlackettLuceModel(ranking_data)
        results = pl_model.fit()
        
        # Add model information
        results['model_names'] = self.model_names
        results['n_datasets'] = len(ranking_data.rankings)
        results['ranking_data'] = ranking_data
        
        return results
    
    def fit_hierarchical_bayes_model(self, task: str, 
                                   n_samples: int = 2000, 
                                   n_tune: int = 1000) -> Dict[str, Any]:
        """
        Fit hierarchical Bayesian model for a specific task.
        
        Args:
            task: Task name
            n_samples: Number of posterior samples
            n_tune: Number of tuning samples
            
        Returns:
            Dictionary with fitted model results
        """
        hb_model = HierarchicalBayesModel(self.results, self.model_names, self.dataset_names)
        return hb_model.fit_hierarchical_model(task, n_samples, n_tune)
    
    def average_ranks(self, task: str) -> Dict[str, float]:
        """
        Calculate average ranks (Borda) across datasets for a specific task.
        
        Args:
            task: Task name
            
        Returns:
            Dictionary with model average ranks
        """
        ranking_data = self._get_rankings_for_task(task)
        
        if not ranking_data.rankings:
            warnings.warn(f"No valid rankings found for task {task}")
            return {model: np.inf for model in self.model_names}
        
        # Calculate average ranks
        model_ranks = {model: [] for model in self.model_names}
        
        for ranking in ranking_data.rankings:
            for model_idx, rank in enumerate(ranking):
                model_ranks[self.model_names[model_idx]].append(rank)
        
        # Calculate average ranks
        avg_ranks = {}
        for model in self.model_names:
            if model_ranks[model]:
                avg_ranks[model] = np.mean(model_ranks[model])
            else:
                avg_ranks[model] = np.inf
        
        return avg_ranks
    
    def friedman_test(self, task: str) -> Tuple[float, float]:
        """
        Perform Friedman test for multiple model comparison.
        
        Args:
            task: Task name
            
        Returns:
            Tuple of (statistic, p_value)
        """
        ranking_data = self._get_rankings_for_task(task)
        
        if len(ranking_data.rankings) < 2:
            warnings.warn(f"Insufficient datasets ({len(ranking_data.rankings)}) for Friedman test")
            return np.nan, np.nan
        
        # Convert rankings to performance matrix for Friedman test
        data_matrix = np.array(ranking_data.rankings)
        
        # Perform Friedman test
        try:
            statistic, p_value = friedmanchisquare(*data_matrix.T)
            return statistic, p_value
        except Exception as e:
            warnings.warn(f"Friedman test failed: {e}")
            return np.nan, np.nan
    
    def pairwise_wilcoxon_tests(self, task: str, 
                               alpha: float = 0.05) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Perform pairwise Wilcoxon signed-rank tests with multiple testing correction.
        
        Args:
            task: Task name
            alpha: Significance level
            
        Returns:
            Dictionary with pairwise test results
        """
        ranking_data = self._get_rankings_for_task(task)
        
        if len(ranking_data.rankings) < 3:
            warnings.warn(f"Insufficient datasets ({len(ranking_data.rankings)}) for Wilcoxon tests")
            return {}
        
        pairwise_results = {}
        all_p_values = []
        test_pairs = []
        
        # Perform all pairwise tests
        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names[i+1:], i+1):
                # Collect paired samples (ranks)
                model1_ranks = []
                model2_ranks = []
                
                for ranking in ranking_data.rankings:
                    model1_ranks.append(ranking[i])
                    model2_ranks.append(ranking[j])
                
                # Perform Wilcoxon test
                try:
                    statistic, p_value = wilcoxon(
                        model1_ranks, 
                        model2_ranks,
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                    
                    all_p_values.append(p_value)
                    test_pairs.append((model1, model2))
                    
                    pairwise_results[(model1, model2)] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'mean_difference': np.mean(model1_ranks) - np.mean(model2_ranks),
                        'model1_mean_rank': np.mean(model1_ranks),
                        'model2_mean_rank': np.mean(model2_ranks)
                    }
                    
                except Exception as e:
                    warnings.warn(f"Wilcoxon test failed for {model1} vs {model2}: {e}")
        
        # Apply multiple testing correction (Holm-Bonferroni)
        if all_p_values:
            _, corrected_p_values, _, _ = multipletests(
                all_p_values, 
                alpha=alpha, 
                method='holm'
            )
            
            # Update results with corrected p-values
            for i, (model1, model2) in enumerate(test_pairs):
                if (model1, model2) in pairwise_results:
                    pairwise_results[(model1, model2)]['p_value_corrected'] = corrected_p_values[i]
                    pairwise_results[(model1, model2)]['significant'] = corrected_p_values[i] < alpha
        
        return pairwise_results
    
    def summary_statistics(self, task: str) -> Dict[str, Any]:
        """
        Calculate summary statistics for model performance across datasets.
        
        Args:
            task: Task name
            
        Returns:
            Dictionary with summary statistics
        """
        valid_datasets = []
        for dataset in self.dataset_names:
            if (dataset in self.results and 
                task in self.results[dataset]):
                available_models = [model for model in self.model_names 
                                  if model in self.results[dataset][task]]
                if len(available_models) == len(self.model_names):
                    valid_datasets.append(dataset)
        
        model_stats = {}
        for model in self.model_names:
            scores = []
            for dataset in valid_datasets:
                scores.append(self.results[dataset][task][model])
            
            if scores:
                model_stats[model] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'n_datasets': len(scores),
                    'cv': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf
                }
            else:
                model_stats[model] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'n_datasets': 0,
                    'cv': np.nan
                }
        
        return model_stats
    
    def generate_comprehensive_report(self, task: str, 
                                   include_plackett_luce: bool = True,
                                   include_hierarchical_bayes: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report for a task.
        
        Args:
            task: Task name
            include_plackett_luce: Whether to include Plackett-Luce analysis
            include_hierarchical_bayes: Whether to include hierarchical Bayes analysis
            
        Returns:
            Dictionary with complete statistical analysis
        """
        report = {
            'task': task,
            'task_direction': self.task_directions.get(task, RankingDirection.HIGHER_BETTER).value,
            'average_ranks': self.average_ranks(task),
            'friedman_test': {
                'statistic': None,
                'p_value': None
            },
            'pairwise_tests': {},
            'summary_statistics': {},
            'plackett_luce': {},
            'hierarchical_bayes': {},
            'summary': {}
        }
        
        # Perform Friedman test
        try:
            stat, p_val = self.friedman_test(task)
            report['friedman_test']['statistic'] = stat
            report['friedman_test']['p_value'] = p_val
        except Exception as e:
            warnings.warn(f"Friedman test failed: {e}")
        
        # Perform pairwise tests
        try:
            report['pairwise_tests'] = self.pairwise_wilcoxon_tests(task)
        except Exception as e:
            warnings.warn(f"Pairwise tests failed: {e}")
        
        # Calculate summary statistics
        try:
            report['summary_statistics'] = self.summary_statistics(task)
        except Exception as e:
            warnings.warn(f"Summary statistics failed: {e}")
        
        # Fit Plackett-Luce model
        if include_plackett_luce:
            try:
                report['plackett_luce'] = self.fit_plackett_luce_model(task)
            except Exception as e:
                warnings.warn(f"Plackett-Luce model failed: {e}")
                report['plackett_luce'] = {'error': str(e)}
        
        # Fit hierarchical Bayes model
        if include_hierarchical_bayes:
            try:
                report['hierarchical_bayes'] = self.fit_hierarchical_bayes_model(task)
            except Exception as e:
                warnings.warn(f"Hierarchical Bayes model failed: {e}")
                report['hierarchical_bayes'] = {'error': str(e)}
        
        # Generate summary
        ranks = report['average_ranks']
        valid_ranks = {k: v for k, v in ranks.items() if v != np.inf}
        
        if valid_ranks:
            best_model = min(valid_ranks, key=valid_ranks.get)
            report['summary'] = {
                'best_model': best_model,
                'best_rank': ranks[best_model],
                'total_models': len(self.model_names),
                'valid_datasets': len(self._get_rankings_for_task(task).rankings),
                'total_datasets': len(self.dataset_names)
            }
        else:
            report['summary'] = {
                'best_model': None,
                'best_rank': None,
                'total_models': len(self.model_names),
                'valid_datasets': 0,
                'total_datasets': len(self.dataset_names)
            }
        
        return report


def _load_summary_or_collect(outputs_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build {dataset: {task: {model: score}}} by reading outputs/analysis/summary.json
    if present; otherwise, collect from raw fold metrics mimicking analysis.summary.
    For time tasks we assume higher-is-better scores already inverted in summary.json.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    summary_path = outputs_dir / "analysis" / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            for dataset, task_map in summary.items():
                results.setdefault(dataset, {})
                for task, data in task_map.items():
                    results[dataset].setdefault(task, {})
                    for rec in data.get("ranking", []):
                        m = rec.get("model")
                        s = rec.get("score")
                        if isinstance(s, (int, float)):
                            results[dataset][task][m] = float(s)
            # Merge external results if any
            ext_summary, _, _ = _ingest_external_results(outputs_dir)
            for d, tmap in ext_summary.items():
                for t, mmap in tmap.items():
                    results.setdefault(d, {}).setdefault(t, {}).update(mmap)
            return results
        except Exception:
            pass
    # Fallback: collect and average like analysis.summary, but we need to import lazily
    from pathlib import Path as _P
    import glob, json as _json
    import numpy as _np
    prefer_max = {
        "next_activity": "accuracy",
        "next_time": "mae",
        "remaining_time": "mae",
    }
    for dataset_dir in outputs_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in ("checkpoints", "analysis"):
            continue
        dataset = dataset_dir.name
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                key = prefer_max.get(task)
                vals = []
                for fold_dir in model_dir.iterdir():
                    if not fold_dir.is_dir():
                        continue
                    mfile = fold_dir / "metrics.json"
                    if mfile.exists():
                        try:
                            md = _json.loads(mfile.read_text())
                            v = md.get(key)
                            if isinstance(v, (int, float)):
                                vals.append(float(v))
                        except Exception:
                            pass
                if vals:
                    score = float(_np.mean(vals))
                    if task in ("next_time", "remaining_time") and key == "mae":
                        score = -score
                    results.setdefault(dataset, {}).setdefault(task, {})[model] = score
    # Merge external results if any
    ext_summary, _, _ = _ingest_external_results(outputs_dir)
    for d, tmap in ext_summary.items():
        for t, mmap in tmap.items():
            results.setdefault(d, {}).setdefault(t, {}).update(mmap)
    return results


def _collect_fold_scores(outputs_dir: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Collect per-fold scores for each dataset/task/model.
    Returns {dataset: {task: {model: [fold_scores...]}}}
    Uses accuracy for next_activity; MAE for time tasks (kept as-is, lower is better).
    """
    data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    prefer_key = {
        "next_activity": "accuracy",
        "next_time": "mae",
        "remaining_time": "mae",
    }
    if not outputs_dir.exists():
        return data
    for dataset_dir in outputs_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in ("checkpoints", "analysis"):
            continue
        dataset = dataset_dir.name
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            key = prefer_key.get(task)
            if key is None:
                continue
            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                scores: List[float] = []
                for fold_dir in model_dir.iterdir():
                    if not fold_dir.is_dir():
                        continue
                    mfile = fold_dir / "metrics.json"
                    if mfile.exists():
                        try:
                            md = json.loads(mfile.read_text())
                            v = md.get(key)
                            if isinstance(v, (int, float)):
                                scores.append(float(v))
                        except Exception:
                            pass
                if scores:
                    data.setdefault(dataset, {}).setdefault(task, {})[model] = scores
    return data


def _model_track(model_name: str) -> Optional[str]:
    """Infer track name from model. Returns 'minimal', 'extended', or None.
    External PGTNet labels (e.g., 'PGTNet (external, splits=...)') are treated as 'extended'.
    """
    name = (model_name or "").strip()
    lname = name.lower()
    minimal = {"process_transformer", "mtlformer", "activity_only_lstm"}
    extended = {"shared_lstm", "specialised_lstm"}
    if name in minimal:
        return "minimal"
    if name in extended:
        return "extended"
    # Heuristic: any model label mentioning 'pgtnet' belongs to extended track
    if "pgtnet" in lname or lname.startswith("pgt"):
        return "extended"
    return None


def _compute_ci(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    import numpy as _np
    if not values:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    arr = _np.array(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "n": int(arr.size)}
    # t-based CI across folds
    sem = float(arr.std(ddof=1) / _np.sqrt(arr.size))
    from scipy.stats import t
    h = float(sem * t.ppf((1 + confidence) / 2.0, df=arr.size - 1))
    return {"mean": mean, "lower": mean - h, "upper": mean + h, "n": int(arr.size)}


def generate_thesis_report(outputs_dir: Path, task: str = "all") -> Dict[str, Any]:
    """Generate thesis-aligned reports per the user’s schema.
    Returns a dict with keys: minimal, extended, efficiency, stratified, metadata.
    """
    outputs_dir = Path(outputs_dir)
    fold_scores = _collect_fold_scores(outputs_dir)
    summary = _load_summary_or_collect(outputs_dir)
    # Ingest external additions (scores, folds, efficiency)
    ext_summary, ext_folds, ext_eff = _ingest_external_results(outputs_dir)
    # Merge external folds into fold_scores for CI computations
    for d, tmap in ext_folds.items():
        for tsk, mmap in tmap.items():
            for model, folds in mmap.items():
                fold_scores.setdefault(d, {}).setdefault(tsk, {})[model] = folds
    tasks = ["next_activity", "next_time", "remaining_time"] if task == "all" else [task]

    def filter_models_for_task(t: str, track: str) -> List[str]:
        all_models = set()
        for d in summary:
            if t in summary[d]:
                all_models |= set(summary[d][t].keys())
        # Track filter only; include mtlformer per-task results produced by multitask training
        return [m for m in sorted(all_models) if _model_track(m) == track]

    report: Dict[str, Any] = {"metadata": {"tasks": tasks}, "per_task": {}, "notes": {}}

    for t in tasks:
        per_task = {"minimal": {}, "extended": {}, "rankings": {}, "significance": {}}
        # Minimal track models per constraints
        if t == "next_activity":
            min_models = [m for m in ["process_transformer", "mtlformer", "activity_only_lstm"] if m in filter_models_for_task(t, "minimal")]
            ext_models = [m for m in ["shared_lstm", "specialised_lstm"] if m in filter_models_for_task(t, "extended")]
            metric_name = "accuracy"; higher_better = True
        elif t == "next_time":
            min_models = [m for m in ["process_transformer", "mtlformer"] if m in filter_models_for_task(t, "minimal")]
            ext_models = []
            metric_name = "mae"; higher_better = False
        else:  # remaining_time
            min_models = [m for m in ["process_transformer", "mtlformer"] if m in filter_models_for_task(t, "minimal")]
            # Include any extended-track models available for remaining_time (e.g., external PGTNet labels)
            ext_models = filter_models_for_task(t, "extended")
            metric_name = "mae"; higher_better = False

        # Build per-dataset tables with CI
        datasets = sorted(d for d in summary if t in summary[d])
        minimal_table = {}
        extended_table = {}
        for d in datasets:
            if t not in fold_scores.get(d, {}):
                continue
            # Minimal
            row_min = {}
            for m in min_models:
                vals = fold_scores[d][t].get(m, [])
                ci = _compute_ci(vals)
                row_min[m] = ci
            if row_min:
                # Ranking within dataset for minimal
                if higher_better:
                    ranking = sorted(((m, row_min[m]["mean"]) for m in row_min), key=lambda x: x[1], reverse=True)
                else:
                    ranking = sorted(((m, row_min[m]["mean"]) for m in row_min), key=lambda x: x[1])
                rank_map = {m: i + 1 for i, (m, _) in enumerate(ranking)}
                for m in row_min:
                    row_min[m]["rank"] = rank_map[m]
                minimal_table[d] = row_min
            # Extended and uplift
            row_ext = {}
            for m in ext_models:
                vals = fold_scores[d][t].get(m, [])
                ci = _compute_ci(vals)
                # compute minimal best mean for uplift
                if minimal_table.get(d):
                    min_best_mean = None
                    if higher_better:
                        min_best_mean = max(minimal_table[d][mm]["mean"] for mm in minimal_table[d])
                        uplift = (ci["mean"] - min_best_mean) / (min_best_mean if min_best_mean else 1.0)
                    else:
                        min_best_mean = min(minimal_table[d][mm]["mean"] for mm in minimal_table[d])
                        uplift = (min_best_mean - ci["mean"]) / (min_best_mean if min_best_mean else 1.0)
                    ci["uplift_vs_minimal_best"] = uplift
                row_ext[m] = ci
            if row_ext:
                extended_table[d] = row_ext
        per_task["minimal"]["per_dataset"] = minimal_table
        per_task["extended"]["per_dataset"] = extended_table

        # Rankings & significance within-track
        # Use BenchmarkStatisticalAnalysis for means across datasets
        def build_stats(models: List[str]):
            if not models:
                return {}
            task_results = {d: {t: {m: summary[d][t][m] for m in models if m in summary[d][t]}} for d in datasets if t in summary[d]}
            BA = BenchmarkStatisticalAnalysis(task_results, models, [d for d in datasets if t in summary[d]])
            comp = BA.generate_comprehensive_report(t, include_plackett_luce=True, include_hierarchical_bayes=False)
            # Limit pairwise tests to top models (min avg rank)
            avg = comp.get("average_ranks", {})
            if avg:
                min_rank = min(avg.values())
                top_models = [m for m, r in avg.items() if r == min_rank]
                if len(top_models) >= 2:
                    BA_top = BenchmarkStatisticalAnalysis(task_results, top_models, [d for d in datasets if t in summary[d]])
                    comp_top = {
                        "pairwise_top_only": BA_top.pairwise_wilcoxon_tests(t)
                    }
                else:
                    comp_top = {"pairwise_top_only": {}}
            else:
                comp_top = {"pairwise_top_only": {}}
            comp.update(comp_top)
            return comp

        per_task["rankings"]["minimal"] = build_stats(min_models)
        # Extended-Track: descriptive only (no inferential tests)
        if ext_models:
            per_task["rankings"]["extended"] = {"note": "Extended-Track is descriptive only. No Plackett–Luce, Bayes, or pairwise significance tests run."}
        else:
            per_task["rankings"]["extended"] = {"note": "No Extended-Track models for this task."}

        # Efficiency: attempt to read optional efficiency.json at model level per dataset
        efficiency = {}
        for d in datasets:
            eff_d = {}
            base = outputs_dir / d / t
            if not base.exists():
                continue
            for m in sorted(min_models + ext_models):
                mdir = base / m
                if not mdir.exists():
                    continue
                efile = mdir / "efficiency.json"
                if efile.exists():
                    try:
                        eff = json.loads(efile.read_text())
                        eff_d[m] = eff
                    except Exception:
                        pass
            # Merge external efficiency for this dataset/task
            if d in ext_eff and t in ext_eff[d]:
                for m_ext, eff_vals in ext_eff[d][t].items():
                    # do not overwrite existing file-based entries; only fill missing or update
                    cur = eff_d.setdefault(m_ext, {})
                    for k, v in eff_vals.items():
                        if k not in cur:
                            cur[k] = v
                        else:
                            # keep existing; optionally ensure numeric
                            try:
                                cur[k] = float(cur[k])
                            except Exception:
                                pass
            if eff_d:
                efficiency[d] = eff_d
        per_task["efficiency"] = efficiency

        # Stratified: attempt to read optional stratified_metrics.json per model
        stratified = {}
        for d in datasets:
            s_d = {}
            base = outputs_dir / d / t
            if not base.exists():
                continue
            for m in sorted(min_models + ext_models):
                mdir = base / m
                if not mdir.exists():
                    continue
                sfile = mdir / "stratified_metrics.json"
                if sfile.exists():
                    try:
                        s = json.loads(sfile.read_text())
                        s_d[m] = s
                    except Exception:
                        pass
            if s_d:
                stratified[d] = s_d
        per_task["stratified"] = stratified

        # Explicit limitation note for NET (next_time): no extended models
        if t == "next_time":
            per_task.setdefault("extended", {})["limitation"] = "No extended model available for next_time (NET); reported explicitly as a limitation."

        report["per_task"][t] = per_task

    # Global MTLFormer efficiency comparison: one multitask vs sum of three single-task runs
    mtl_eff: Dict[str, Any] = {}
    for d in sorted(summary.keys()):
        multi_path = outputs_dir / d / "multitask" / "mtlformer" / "efficiency.json"
        multi = None
        if multi_path.exists():
            try:
                multi = json.loads(multi_path.read_text())
            except Exception:
                multi = None
        # Sum single-task runs across NA/NET/RT using process_transformer primarily
        single_sum = {"train_time_seconds": 0.0, "infer_time_seconds": 0.0, "params": None, "count": 0}
        single_tasks = ["next_activity", "next_time", "remaining_time"]
        for tsk in single_tasks:
            base = outputs_dir / d / tsk
            if not base.exists():
                continue
            # prefer process_transformer; fallback to any minimal
            candidates = ["process_transformer", "activity_only_lstm"]
            found = None
            for cand in candidates:
                efile = base / cand / "efficiency.json"
                if efile.exists():
                    found = efile
                    break
            if not found:
                # last resort: try any model in dir
                for mdir in base.iterdir():
                    if mdir.is_dir():
                        efile = mdir / "efficiency.json"
                        if efile.exists():
                            found = efile
                            break
            if found:
                try:
                    eff = json.loads(found.read_text())
                    single_sum["train_time_seconds"] += float(eff.get("train_time_seconds", 0.0))
                    single_sum["infer_time_seconds"] += float(eff.get("infer_time_seconds", 0.0))
                    # params not additive; store max across single runs for reference
                    p = eff.get("params")
                    if isinstance(p, (int, float)):
                        single_sum["params"] = max(int(p), int(single_sum["params"])) if single_sum["params"] is not None else int(p)
                    single_sum["count"] += 1
                except Exception:
                    pass
        if multi or single_sum["count"] > 0:
            mtl_eff[d] = {
                "multitask_mtlformer": multi or {},
                "single_task_sum": single_sum,
                "comparison": {
                    "train_time_ratio_multi_vs_sum": (float(multi.get("train_time_seconds", 0.0)) / single_sum["train_time_seconds"]) if (multi and single_sum["train_time_seconds"]) else None,
                    "infer_time_ratio_multi_vs_sum": (float(multi.get("infer_time_seconds", 0.0)) / single_sum["infer_time_seconds"]) if (multi and single_sum["infer_time_seconds"]) else None
                }
            }
    report["mtlformer_efficiency"] = mtl_eff

    # General notes
    report["notes"]["inference"] = "Inferential statistics (Plackett–Luce, Bayesian tests, non-parametric tests) are restricted to the Minimal-Track. Extended-Track results are descriptive with uplifts vs best minimal model."

    return report


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Statistical analysis CLI: per-task full report or thesis-aligned report")
    ap.add_argument("--task", choices=["next_activity", "next_time", "remaining_time", "all"], default="all", help="Task to analyze (for full report) or 'all'")
    ap.add_argument("--outputs", default="outputs", help="Path to outputs directory")
    ap.add_argument("--no-pl", dest="pl", action="store_false", help="Disable Plackett–Luce fit (full report mode)")
    ap.add_argument("--no-hb", dest="hb", action="store_false", help="Disable Hierarchical Bayes fit (full report mode)")
    ap.add_argument("--thesis", action="store_true", help="Generate thesis-aligned report across tasks (Minimal vs Extended, uplifts, MTLFormer efficiency)")
    ap.set_defaults(pl=True, hb=True)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)

    out_dir = outputs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.thesis:
        thesis = generate_thesis_report(outputs_dir, task=args.task)
        out_path = out_dir / ("thesis_report.json" if args.task == "all" else f"thesis_report_{args.task}.json")
        out_path.write_text(json.dumps(thesis, indent=2))
        print(f"Saved thesis-aligned report to {out_path}")
    else:
        data = _load_summary_or_collect(outputs_dir)
        # Collect model and dataset names where this task appears
        datasets = sorted([d for d, tmap in data.items() if args.task in tmap])
        models = sorted({m for d in datasets for m in data[d][args.task].keys()})
        if not datasets or not models:
            print(f"No data found for task {args.task} under {outputs_dir}")
            raise SystemExit(1)

        # Build results dict limited to the selected task
        results = {d: {args.task: data[d][args.task]} for d in datasets}
        BA = BenchmarkStatisticalAnalysis(results, models, datasets)
        report = BA.generate_comprehensive_report(args.task, include_plackett_luce=args.pl, include_hierarchical_bayes=args.hb)
        out_path = out_dir / f"full_report_{args.task}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved full statistical report to {out_path}")



def _ingest_external_results(outputs_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, List[float]]]], Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]: 
    """
    Load external benchmark results placed under outputs/external_results/ and return
    two structures:
      - (summary_add): {dataset: {task: {model_label: score}}} where score follows
        _load_summary_or_collect conventions (accuracy higher-is-better; time tasks inverted sign)
      - (fold_add): {dataset: {task: {model_label: [fold_scores...]}}} where fold scores
        are raw (accuracy or MAE as-is) for inclusion in confidence intervals.

    Accepted format (per file):
      - JSON list of records, each record with keys:
        {"dataset": str, "task": str, "model": str, "metric": "accuracy|mae",
         "score": float (optional if folds given),
         "fold": int (optional; if present with 'value', ingested as a single-fold score),
         "value": float (used with 'fold'),
         "folds": [float, ...] (alternative to single fold records),
         "splits": str (optional provenance label, e.g., 'canonical' or 'author'),
         "train_time_seconds"|"training_time": float (optional),
         "infer_time_seconds"|"inference_time": float (optional),
         "params"|"model_size": float (optional)
        }

    Any other file format or JSON structure is ignored.
    """
    from pathlib import Path as _P
    import json as _json

    summary_add: Dict[str, Dict[str, Dict[str, float]]] = {}
    fold_add: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    efficiency_add: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    ext_dir = _P(outputs_dir) / "external_results"
    if not ext_dir.exists():
        return summary_add, fold_add, efficiency_add

    def _label(model: str, splits: Optional[str]) -> str:
        s = f"{model} (external" + (f", splits={splits}" if splits else ")")
        if not s.endswith(")"):
            s = s + ")"
        return s

    def _add(dataset: str, task: str, model_label: str, metric: str, score: Optional[float], folds: Optional[List[float]]):
        # Normalize
        dataset = str(dataset)
        task = str(task)
        metric = str(metric).lower() if metric else None
        if folds:
            try:
                fvals = [float(x) for x in folds]
            except Exception:
                fvals = []
        else:
            fvals = []
        if score is not None:
            try:
                sc = float(score)
            except Exception:
                sc = None
        else:
            sc = None
        # Merge into fold_add
        if fvals:
            fold_add.setdefault(dataset, {}).setdefault(task, {})[model_label] = fvals
        # Derive summary score
        if sc is None and fvals:
            sc = float(np.mean(fvals))
        if sc is None:
            return
        # Invert time tasks to match summary convention (higher is better)
        if task in ("next_time", "remaining_time") and metric == "mae":
            sc = -float(sc)
        summary_add.setdefault(dataset, {}).setdefault(task, {})[model_label] = float(sc)

    # Iterate only JSON files and only accept top-level lists of records
    for f in ext_dir.glob("*.json"):
        try:
            obj = _json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(obj, list):
            # Not the accepted format
            continue
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            ds = rec.get("dataset")
            t = rec.get("task")
            model = rec.get("model")
            metric = rec.get("metric")
            splits = rec.get("splits")
            folds = rec.get("folds")
            # Support single fold record via {fold, value}
            if folds is None and rec.get("fold") is not None and rec.get("value") is not None:
                folds = [rec.get("value")]
            score = rec.get("score")
            if ds and t and model and metric:
                label = _label(model, splits)
                _add(ds, t, label, metric, score, folds)
                # Optionally ingest efficiency metrics per record
                eff_keys = {
                    "train_time_seconds": rec.get("train_time_seconds", rec.get("training_time")),
                    "infer_time_seconds": rec.get("infer_time_seconds", rec.get("inference_time")),
                    "params": rec.get("params", rec.get("model_size")),
                }
                # If any efficiency field is present, record it
                if any(v is not None for v in eff_keys.values()):
                    eff_clean: Dict[str, float] = {}
                    for k, v in eff_keys.items():
                        try:
                            if v is not None:
                                eff_clean[k] = float(v)
                        except Exception:
                            continue
                    if eff_clean:
                        efficiency_add.setdefault(str(ds), {}).setdefault(str(t), {}).setdefault(label, {}).update(eff_clean)

    return summary_add, fold_add, efficiency_add
