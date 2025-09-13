"""
Statistical testing and confidence interval utilities for compression analysis.

This module provides statistical tools for evaluating the significance of
compression performance differences and computing confidence intervals.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from dataclasses import dataclass
import pandas as pd
import logging
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None


class StatisticalTester:
    """Performs statistical significance tests on model performance."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical tester.

        Args:
            confidence_level: Confidence level for tests (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def paired_t_test(self,
                     original_scores: np.ndarray,
                     compressed_scores: np.ndarray) -> StatisticalTestResult:
        """
        Perform paired t-test between original and compressed model scores.

        Args:
            original_scores: Scores from original model
            compressed_scores: Scores from compressed model

        Returns:
            Statistical test result
        """
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(original_scores, compressed_scores)

        # Calculate effect size (Cohen's d)
        diff = original_scores - compressed_scores
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        # Calculate confidence interval for mean difference
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci = stats.t.interval(self.confidence_level, len(diff) - 1,
                             loc=mean_diff, scale=se_diff)

        # Interpret results
        if p_value < self.alpha:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def wilcoxon_signed_rank_test(self,
                                 original_scores: np.ndarray,
                                 compressed_scores: np.ndarray) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            original_scores: Scores from original model
            compressed_scores: Scores from compressed model

        Returns:
            Statistical test result
        """
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(original_scores, compressed_scores)

        # Calculate effect size (rank-biserial correlation)
        n = len(original_scores)
        effect_size = 1 - (2 * statistic) / (n * (n + 1))

        # Interpret results
        if p_value < self.alpha:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def mcnemar_test(self,
                    original_predictions: np.ndarray,
                    compressed_predictions: np.ndarray,
                    ground_truth: np.ndarray) -> StatisticalTestResult:
        """
        Perform McNemar's test for paired binary classification results.

        Args:
            original_predictions: Binary predictions from original model
            compressed_predictions: Binary predictions from compressed model
            ground_truth: True labels

        Returns:
            Statistical test result
        """
        # Create contingency table
        original_correct = (original_predictions == ground_truth)
        compressed_correct = (compressed_predictions == ground_truth)

        # Count disagreements
        n01 = np.sum(original_correct & ~compressed_correct)  # Original correct, compressed wrong
        n10 = np.sum(~original_correct & compressed_correct)  # Original wrong, compressed correct

        # Perform McNemar's test
        if n01 + n10 > 0:
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        else:
            statistic = 0
            p_value = 1.0

        # Interpret results
        if p_value < self.alpha:
            if n01 > n10:
                interpretation = "Original model significantly better"
            else:
                interpretation = "Compressed model significantly better"
        else:
            interpretation = "No significant difference in accuracy"

        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )

    def bootstrap_confidence_interval(self,
                                     scores: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     statistic: callable = np.mean) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval for a statistic.

        Args:
            scores: Array of scores
            n_bootstrap: Number of bootstrap samples
            statistic: Function to compute statistic (default: mean)

        Returns:
            Point estimate and confidence interval
        """
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_sample = resample(scores, replace=True, n_samples=len(scores))
            boot_stat = statistic(boot_sample)
            bootstrap_stats.append(boot_stat)

        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        # Point estimate
        point_estimate = statistic(scores)

        return point_estimate, (lower, upper)

    def permutation_test(self,
                        group1: np.ndarray,
                        group2: np.ndarray,
                        n_permutations: int = 10000) -> StatisticalTestResult:
        """
        Perform permutation test for difference in means.

        Args:
            group1: First group scores
            group2: Second group scores
            n_permutations: Number of permutations

        Returns:
            Statistical test result
        """
        # Observed difference
        observed_diff = np.mean(group1) - np.mean(group2)

        # Combine groups
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        # Perform permutations
        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            permuted_diffs.append(perm_diff)

        # Calculate p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

        # Interpret results
        if p_value < self.alpha:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Permutation test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )


class MultipleComparisonCorrector:
    """Handles multiple comparison corrections for statistical tests."""

    @staticmethod
    def bonferroni_correction(p_values: List[float],
                             alpha: float = 0.05) -> Tuple[List[bool], float]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Adjusted significance decisions and adjusted alpha
        """
        n_tests = len(p_values)
        adjusted_alpha = alpha / n_tests
        significant = [p < adjusted_alpha for p in p_values]
        return significant, adjusted_alpha

    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float],
                                     alpha: float = 0.05) -> List[bool]:
        """
        Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Adjusted significance decisions
        """
        n = len(p_values)
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

        significant = [False] * n
        for i in range(n - 1, -1, -1):
            idx, p = sorted_p[i]
            threshold = alpha * (i + 1) / n
            if p <= threshold:
                for j in range(i + 1):
                    significant[sorted_p[j][0]] = True
                break

        return significant


class PerformanceVarianceAnalyzer:
    """Analyzes variance in model performance across different conditions."""

    def __init__(self):
        """Initialize variance analyzer."""
        pass

    def analyze_variance(self,
                        results_df: pd.DataFrame,
                        factors: List[str],
                        response: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform ANOVA to analyze variance across factors.

        Args:
            results_df: DataFrame with experimental results
            factors: List of factor columns
            response: Response variable column

        Returns:
            ANOVA results
        """
        from scipy.stats import f_oneway

        anova_results = {}

        for factor in factors:
            if factor not in results_df.columns:
                continue

            # Group by factor levels
            groups = []
            levels = results_df[factor].unique()

            for level in levels:
                group_data = results_df[results_df[factor] == level][response].values
                if len(group_data) > 0:
                    groups.append(group_data)

            if len(groups) > 1:
                # Perform one-way ANOVA
                f_stat, p_value = f_oneway(*groups)

                anova_results[factor] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_groups': len(groups),
                    'group_means': [np.mean(g) for g in groups],
                    'group_stds': [np.std(g) for g in groups]
                }

        return anova_results

    def calculate_effect_sizes(self,
                             results_df: pd.DataFrame,
                             group_col: str,
                             value_col: str) -> Dict[str, float]:
        """
        Calculate effect sizes (eta-squared, omega-squared) for group differences.

        Args:
            results_df: DataFrame with results
            group_col: Column defining groups
            value_col: Column with values

        Returns:
            Effect size metrics
        """
        # Calculate group statistics
        groups = results_df.groupby(group_col)[value_col]
        group_means = groups.mean()
        group_sizes = groups.size()
        grand_mean = results_df[value_col].mean()

        # Sum of squares
        ss_between = sum(n * (mean - grand_mean) ** 2
                        for mean, n in zip(group_means, group_sizes))
        ss_total = sum((results_df[value_col] - grand_mean) ** 2)
        ss_within = ss_total - ss_between

        # Degrees of freedom
        df_between = len(group_means) - 1
        df_within = len(results_df) - len(group_means)

        # Effect sizes
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        omega_squared = ((ss_between - df_between * ms_within) /
                        (ss_total + ms_within)) if (ss_total + ms_within) > 0 else 0

        return {
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'interpretation': self._interpret_effect_size(eta_squared)
        }

    @staticmethod
    def _interpret_effect_size(eta_squared: float) -> str:
        """Interpret effect size magnitude."""
        if eta_squared < 0.01:
            return "Negligible effect"
        elif eta_squared < 0.06:
            return "Small effect"
        elif eta_squared < 0.14:
            return "Medium effect"
        else:
            return "Large effect"


def run_comprehensive_statistical_analysis(original_results: np.ndarray,
                                          compressed_results: np.ndarray,
                                          test_types: List[str] = None) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis comparing models.

    Args:
        original_results: Results from original model
        compressed_results: Results from compressed model
        test_types: List of test types to run

    Returns:
        Dictionary with all test results
    """
    if test_types is None:
        test_types = ['t_test', 'wilcoxon', 'bootstrap']

    tester = StatisticalTester()
    results = {}

    if 't_test' in test_types:
        results['t_test'] = tester.paired_t_test(original_results, compressed_results)

    if 'wilcoxon' in test_types:
        results['wilcoxon'] = tester.wilcoxon_signed_rank_test(original_results, compressed_results)

    if 'bootstrap' in test_types:
        orig_mean, orig_ci = tester.bootstrap_confidence_interval(original_results)
        comp_mean, comp_ci = tester.bootstrap_confidence_interval(compressed_results)
        results['bootstrap'] = {
            'original': {'mean': orig_mean, 'ci': orig_ci},
            'compressed': {'mean': comp_mean, 'ci': comp_ci}
        }

    if 'permutation' in test_types:
        results['permutation'] = tester.permutation_test(original_results, compressed_results)

    return results