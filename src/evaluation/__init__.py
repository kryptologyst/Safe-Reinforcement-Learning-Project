"""Evaluation metrics and utilities for safe reinforcement learning."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import compute_cvar


class SafeRLEvaluator:
    """Evaluator for safe reinforcement learning algorithms."""
    
    def __init__(self, cost_limit: float = 0.01, alpha: float = 0.05):
        """Initialize evaluator.
        
        Args:
            cost_limit: Cost limit for safety constraints
            alpha: Risk level for CVaR computation
        """
        self.cost_limit = cost_limit
        self.alpha = alpha
        self.results = []
    
    def evaluate_episode(
        self,
        episode_rewards: List[float],
        episode_costs: List[float],
        episode_safety_violations: List[bool],
        episode_length: int,
        algorithm_name: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Evaluate a single episode.
        
        Args:
            episode_rewards: List of rewards in episode
            episode_costs: List of costs in episode
            episode_safety_violations: List of safety violations
            episode_length: Length of episode
            algorithm_name: Name of algorithm
            seed: Random seed
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_reward = sum(episode_rewards)
        total_cost = sum(episode_costs)
        safety_violations = sum(episode_safety_violations)
        safety_violation_rate = safety_violations / episode_length if episode_length > 0 else 0
        
        # Safety metrics
        constraint_satisfaction = 1.0 if total_cost <= self.cost_limit else 0.0
        safety_score = 1.0 - min(1.0, safety_violation_rate)
        
        # Risk metrics
        cvar_reward = compute_cvar(np.array(episode_rewards), self.alpha)
        cvar_cost = compute_cvar(np.array(episode_costs), self.alpha)
        
        # Episode metrics
        episode_metrics = {
            "algorithm": algorithm_name,
            "seed": seed,
            "episode_length": episode_length,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "safety_violations": safety_violations,
            "safety_violation_rate": safety_violation_rate,
            "constraint_satisfaction": constraint_satisfaction,
            "safety_score": safety_score,
            "cvar_reward": cvar_reward,
            "cvar_cost": cvar_cost,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_cost": np.mean(episode_costs),
            "std_cost": np.std(episode_costs),
        }
        
        self.results.append(episode_metrics)
        return episode_metrics
    
    def evaluate_algorithm(
        self,
        algorithm_name: str,
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate algorithm across multiple episodes and seeds.
        
        Args:
            algorithm_name: Name of algorithm
            num_episodes: Number of episodes per seed
            seeds: List of seeds to evaluate
            
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 101112]
        
        # Filter results for this algorithm
        algorithm_results = [r for r in self.results if r["algorithm"] == algorithm_name]
        
        if not algorithm_results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(algorithm_results)
        
        # Aggregate metrics
        metrics = {
            "algorithm": algorithm_name,
            "num_episodes": len(algorithm_results),
            "num_seeds": len(df["seed"].unique()),
        }
        
        # Performance metrics
        metrics.update({
            "mean_reward": df["total_reward"].mean(),
            "std_reward": df["total_reward"].std(),
            "mean_reward_ci": self._compute_ci(df["total_reward"]),
            "mean_episode_length": df["episode_length"].mean(),
            "std_episode_length": df["episode_length"].std(),
        })
        
        # Safety metrics
        metrics.update({
            "mean_cost": df["total_cost"].mean(),
            "std_cost": df["total_cost"].std(),
            "mean_cost_ci": self._compute_ci(df["total_cost"]),
            "constraint_satisfaction_rate": df["constraint_satisfaction"].mean(),
            "mean_safety_violations": df["safety_violations"].mean(),
            "std_safety_violations": df["safety_violations"].std(),
            "mean_safety_violation_rate": df["safety_violation_rate"].mean(),
            "std_safety_violation_rate": df["safety_violation_rate"].std(),
            "mean_safety_score": df["safety_score"].mean(),
            "std_safety_score": df["safety_score"].std(),
        })
        
        # Risk metrics
        metrics.update({
            "mean_cvar_reward": df["cvar_reward"].mean(),
            "std_cvar_reward": df["cvar_reward"].std(),
            "mean_cvar_cost": df["cvar_cost"].mean(),
            "std_cvar_cost": df["cvar_cost"].std(),
        })
        
        # Statistical significance tests
        if len(df["seed"].unique()) > 1:
            metrics["reward_seed_pvalue"] = self._test_seed_significance(df, "total_reward")
            metrics["cost_seed_pvalue"] = self._test_seed_significance(df, "total_cost")
        
        return metrics
    
    def compare_algorithms(self, algorithm_names: List[str]) -> Dict[str, Any]:
        """Compare multiple algorithms.
        
        Args:
            algorithm_names: List of algorithm names to compare
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {}
        
        # Individual algorithm metrics
        for name in algorithm_names:
            comparison[name] = self.evaluate_algorithm(name)
        
        # Pairwise comparisons
        if len(algorithm_names) >= 2:
            comparison["pairwise_tests"] = {}
            
            for i, alg1 in enumerate(algorithm_names):
                for alg2 in algorithm_names[i+1:]:
                    test_results = self._compare_two_algorithms(alg1, alg2)
                    comparison["pairwise_tests"][f"{alg1}_vs_{alg2}"] = test_results
        
        return comparison
    
    def _compute_ci(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval.
        
        Args:
            data: Data series
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = data.mean()
        std = data.std()
        se = std / np.sqrt(n)
        
        # t-distribution for small samples
        if n < 30:
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            t_val = stats.norm.ppf((1 + confidence) / 2)
        
        margin = t_val * se
        return (mean - margin, mean + margin)
    
    def _test_seed_significance(self, df: pd.DataFrame, metric: str) -> float:
        """Test significance across seeds.
        
        Args:
            df: DataFrame with results
            metric: Metric to test
            
        Returns:
            p-value
        """
        seed_groups = [group[metric].values for _, group in df.groupby("seed")]
        
        if len(seed_groups) < 2:
            return 1.0
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*seed_groups)
        return p_value
    
    def _compare_two_algorithms(self, alg1: str, alg2: str) -> Dict[str, Any]:
        """Compare two algorithms.
        
        Args:
            alg1: First algorithm name
            alg2: Second algorithm name
            
        Returns:
            Dictionary of comparison results
        """
        df1 = pd.DataFrame([r for r in self.results if r["algorithm"] == alg1])
        df2 = pd.DataFrame([r for r in self.results if r["algorithm"] == alg2])
        
        if df1.empty or df2.empty:
            return {}
        
        comparison = {}
        
        # Reward comparison
        reward_test = stats.ttest_ind(df1["total_reward"], df2["total_reward"])
        comparison["reward_ttest"] = {
            "statistic": reward_test.statistic,
            "pvalue": reward_test.pvalue,
            "mean_diff": df1["total_reward"].mean() - df2["total_reward"].mean(),
        }
        
        # Cost comparison
        cost_test = stats.ttest_ind(df1["total_cost"], df2["total_cost"])
        comparison["cost_ttest"] = {
            "statistic": cost_test.statistic,
            "pvalue": cost_test.pvalue,
            "mean_diff": df1["total_cost"].mean() - df2["total_cost"].mean(),
        }
        
        # Safety comparison
        safety_test = stats.ttest_ind(df1["safety_score"], df2["safety_score"])
        comparison["safety_ttest"] = {
            "statistic": safety_test.statistic,
            "pvalue": safety_test.pvalue,
            "mean_diff": df1["safety_score"].mean() - df2["safety_score"].mean(),
        }
        
        return comparison
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot evaluation results.
        
        Args:
            save_path: Path to save plot
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Safe RL Evaluation Results", fontsize=16)
        
        # Reward distribution
        sns.boxplot(data=df, x="algorithm", y="total_reward", ax=axes[0, 0])
        axes[0, 0].set_title("Total Reward Distribution")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cost distribution
        sns.boxplot(data=df, x="algorithm", y="total_cost", ax=axes[0, 1])
        axes[0, 1].set_title("Total Cost Distribution")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Safety score distribution
        sns.boxplot(data=df, x="algorithm", y="safety_score", ax=axes[0, 2])
        axes[0, 2].set_title("Safety Score Distribution")
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Safety violation rate
        sns.boxplot(data=df, x="algorithm", y="safety_violation_rate", ax=axes[1, 0])
        axes[1, 0].set_title("Safety Violation Rate")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # CVaR reward
        sns.boxplot(data=df, x="algorithm", y="cvar_reward", ax=axes[1, 1])
        axes[1, 1].set_title("CVaR Reward")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Episode length
        sns.boxplot(data=df, x="algorithm", y="episode_length", ax=axes[1, 2])
        axes[1, 2].set_title("Episode Length")
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save results to file.
        
        Args:
            filepath: Path to save results
        """
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
    
    def load_results(self, filepath: str) -> None:
        """Load results from file.
        
        Args:
            filepath: Path to load results from
        """
        df = pd.read_csv(filepath)
        self.results = df.to_dict('records')
