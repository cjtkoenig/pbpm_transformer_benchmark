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
            'suffix': RankingDirection.LOWER_BETTER,              # Distance - lower is better
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
            for model2 in self.model_names[i+1:]:
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


# Example usage
if __name__ == "__main__":
    # Example results structure: {dataset: {task: {model: metric_value}}}
    example_results = {
        "Helpdesk": {
            "next_activity": {
                "process_transformer": 0.85,
                "mtlformer": 0.83,
                "mtlformer_multi": 0.87
            },
            "next_time": {
                "process_transformer": 2.1,
                "mtlformer": 2.3,
                "mtlformer_multi": 1.9
            }
        },
        "BPI_Challenge_2012": {
            "next_activity": {
                "process_transformer": 0.78,
                "mtlformer": 0.76,
                "mtlformer_multi": 0.79
            },
            "next_time": {
                "process_transformer": 3.2,
                "mtlformer": 3.4,
                "mtlformer_multi": 3.0
            }
        }
    }
    
    model_names = ["process_transformer", "mtlformer", "mtlformer_multi"]
    dataset_names = ["Helpdesk", "BPI_Challenge_2012"]
    
    # Initialize analysis
    analysis = BenchmarkStatisticalAnalysis(example_results, model_names, dataset_names)
    
    # Generate comprehensive reports for each task
    for task in ["next_activity", "next_time"]:
        print(f"\n=== Comprehensive Statistical Report for {task} ===")
        report = analysis.generate_comprehensive_report(task)
        
        print(f"Task direction: {report['task_direction']}")
        print(f"Average ranks: {report['average_ranks']}")
        print(f"Friedman test: {report['friedman_test']}")
        print(f"Best model: {report['summary']['best_model']}")
        print(f"Valid datasets: {report['summary']['valid_datasets']}")
        
        # Print Plackett-Luce results if available
        if 'plackett_luce' in report and 'error' not in report['plackett_luce']:
            pl_results = report['plackett_luce']
            print(f"Plackett-Luce probabilities: {dict(zip(model_names, pl_results['probabilities']))}")
            print(f"Plackett-Luce log-likelihood: {pl_results['log_likelihood']}")
        
        # Print hierarchical Bayes results if available
        if 'hierarchical_bayes' in report and 'error' not in report['hierarchical_bayes']:
            hb_results = report['hierarchical_bayes']
            print(f"Hierarchical Bayes model fitted successfully")
            if 'pairwise_differences' in hb_results:
                print(f"Number of pairwise comparisons: {len(hb_results['pairwise_differences'])}")
