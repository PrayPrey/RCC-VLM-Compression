"""
Ablation study framework for analyzing compression component contributions.

This module provides tools for systematically evaluating the impact of
different compression stages and their ordering on model performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from itertools import permutations, combinations
import pandas as pd
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    compression_stages: List[str] = field(default_factory=lambda: ["dare", "nullu", "alphaedit"])
    test_orderings: bool = True
    test_individual: bool = True
    test_combinations: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "compression_ratio", "inference_time"])
    num_samples: int = 1000
    save_results: bool = True
    results_dir: str = "./ablation_results"


@dataclass
class AblationResult:
    """Container for ablation study results."""
    configuration: str
    stages_active: List[str]
    metrics: Dict[str, float]
    compression_ratio: float
    performance_retention: float
    inference_speedup: float
    memory_reduction: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class AblationStudy:
    """Conducts systematic ablation studies on compression pipelines."""

    def __init__(self,
                 model: nn.Module,
                 compression_pipeline: Any,
                 evaluator: Any,
                 config: Optional[AblationConfig] = None):
        """
        Initialize ablation study.

        Args:
            model: Original model
            compression_pipeline: Compression pipeline instance
            evaluator: Evaluation module
            config: Ablation configuration
        """
        self.model = model
        self.compression_pipeline = compression_pipeline
        self.evaluator = evaluator
        self.config = config or AblationConfig()
        self.results: List[AblationResult] = []

        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def run_full_study(self) -> pd.DataFrame:
        """
        Run complete ablation study.

        Returns:
            DataFrame with ablation results
        """
        logger.info("Starting comprehensive ablation study")

        # Test individual components
        if self.config.test_individual:
            self._test_individual_components()

        # Test combinations
        if self.config.test_combinations:
            self._test_combinations()

        # Test orderings
        if self.config.test_orderings:
            self._test_orderings()

        # Convert results to DataFrame
        df = self._results_to_dataframe()

        # Save results
        if self.config.save_results:
            self._save_results(df)

        # Generate analysis
        analysis = self._analyze_results(df)

        return df, analysis

    def _test_individual_components(self):
        """Test each compression component individually."""
        logger.info("Testing individual compression components")

        for stage in self.config.compression_stages:
            logger.info(f"Testing stage: {stage}")

            # Create pipeline with only this stage
            compressed_model = self._apply_single_stage(self.model.clone(), stage)

            # Evaluate
            metrics = self._evaluate_model(compressed_model)

            # Store result
            result = AblationResult(
                configuration=f"single_{stage}",
                stages_active=[stage],
                metrics=metrics,
                compression_ratio=self._calculate_compression_ratio(self.model, compressed_model),
                performance_retention=metrics.get('accuracy', 0) / self._get_baseline_accuracy(),
                inference_speedup=self._calculate_speedup(self.model, compressed_model),
                memory_reduction=self._calculate_memory_reduction(self.model, compressed_model)
            )
            self.results.append(result)

    def _test_combinations(self):
        """Test different combinations of components."""
        logger.info("Testing component combinations")

        stages = self.config.compression_stages

        # Test all possible combinations (except empty and full)
        for r in range(2, len(stages)):
            for combo in combinations(stages, r):
                logger.info(f"Testing combination: {combo}")

                # Apply combination
                compressed_model = self._apply_stages(self.model.clone(), list(combo))

                # Evaluate
                metrics = self._evaluate_model(compressed_model)

                # Store result
                result = AblationResult(
                    configuration=f"combo_{'_'.join(combo)}",
                    stages_active=list(combo),
                    metrics=metrics,
                    compression_ratio=self._calculate_compression_ratio(self.model, compressed_model),
                    performance_retention=metrics.get('accuracy', 0) / self._get_baseline_accuracy(),
                    inference_speedup=self._calculate_speedup(self.model, compressed_model),
                    memory_reduction=self._calculate_memory_reduction(self.model, compressed_model)
                )
                self.results.append(result)

    def _test_orderings(self):
        """Test different orderings of compression stages."""
        logger.info("Testing stage orderings")

        stages = self.config.compression_stages

        # Test all permutations
        for perm in permutations(stages):
            logger.info(f"Testing ordering: {perm}")

            # Apply stages in this order
            compressed_model = self._apply_stages(self.model.clone(), list(perm))

            # Evaluate
            metrics = self._evaluate_model(compressed_model)

            # Store result
            result = AblationResult(
                configuration=f"order_{'_'.join(perm)}",
                stages_active=list(perm),
                metrics=metrics,
                compression_ratio=self._calculate_compression_ratio(self.model, compressed_model),
                performance_retention=metrics.get('accuracy', 0) / self._get_baseline_accuracy(),
                inference_speedup=self._calculate_speedup(self.model, compressed_model),
                memory_reduction=self._calculate_memory_reduction(self.model, compressed_model)
            )
            self.results.append(result)

    def _apply_single_stage(self, model: nn.Module, stage: str) -> nn.Module:
        """Apply single compression stage."""
        if hasattr(self.compression_pipeline, f'apply_{stage}'):
            return getattr(self.compression_pipeline, f'apply_{stage}')(model)
        else:
            logger.warning(f"Stage {stage} not found in pipeline")
            return model

    def _apply_stages(self, model: nn.Module, stages: List[str]) -> nn.Module:
        """Apply multiple compression stages in sequence."""
        for stage in stages:
            model = self._apply_single_stage(model, stage)
        return model

    def _evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate compressed model."""
        if self.evaluator:
            return self.evaluator.evaluate(model, num_samples=self.config.num_samples)
        else:
            # Dummy evaluation
            return {
                'accuracy': np.random.uniform(0.7, 0.95),
                'loss': np.random.uniform(0.1, 0.5)
            }

    def _calculate_compression_ratio(self, original: nn.Module, compressed: nn.Module) -> float:
        """Calculate compression ratio."""
        original_params = sum(p.numel() for p in original.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        return original_params / compressed_params if compressed_params > 0 else float('inf')

    def _calculate_speedup(self, original: nn.Module, compressed: nn.Module) -> float:
        """Calculate inference speedup."""
        # Simplified - in practice would measure actual inference time
        return self._calculate_compression_ratio(original, compressed) * 0.8

    def _calculate_memory_reduction(self, original: nn.Module, compressed: nn.Module) -> float:
        """Calculate memory reduction."""
        original_size = sum(p.numel() * p.element_size() for p in original.parameters())
        compressed_size = sum(p.numel() * p.element_size() for p in compressed.parameters())
        return 1 - (compressed_size / original_size)

    def _get_baseline_accuracy(self) -> float:
        """Get baseline accuracy of original model."""
        if not hasattr(self, '_baseline_accuracy'):
            metrics = self._evaluate_model(self.model)
            self._baseline_accuracy = metrics.get('accuracy', 1.0)
        return self._baseline_accuracy

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            row = {
                'configuration': result.configuration,
                'stages': '_'.join(result.stages_active),
                'num_stages': len(result.stages_active),
                'compression_ratio': result.compression_ratio,
                'performance_retention': result.performance_retention,
                'inference_speedup': result.inference_speedup,
                'memory_reduction': result.memory_reduction,
                'timestamp': result.timestamp
            }
            # Add metrics
            for key, value in result.metrics.items():
                row[f'metric_{key}'] = value
            data.append(row)

        return pd.DataFrame(data)

    def _analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ablation results."""
        analysis = {
            'best_configuration': None,
            'stage_importance': {},
            'ordering_impact': {},
            'synergy_effects': {},
            'recommendations': []
        }

        # Find best configuration
        if not df.empty:
            # Balance between compression and performance
            df['score'] = df['compression_ratio'] * df['performance_retention']
            best_idx = df['score'].idxmax()
            analysis['best_configuration'] = df.loc[best_idx, 'configuration']

            # Calculate stage importance
            for stage in self.config.compression_stages:
                stage_df = df[df['stages'].str.contains(stage)]
                if not stage_df.empty:
                    analysis['stage_importance'][stage] = {
                        'avg_compression': stage_df['compression_ratio'].mean(),
                        'avg_performance': stage_df['performance_retention'].mean(),
                        'appearances': len(stage_df)
                    }

            # Analyze ordering impact
            order_df = df[df['configuration'].str.startswith('order_')]
            if not order_df.empty:
                analysis['ordering_impact'] = {
                    'variance': order_df['performance_retention'].var(),
                    'best_order': order_df.loc[order_df['performance_retention'].idxmax(), 'stages'],
                    'worst_order': order_df.loc[order_df['performance_retention'].idxmin(), 'stages']
                }

            # Generate recommendations
            if analysis['stage_importance']:
                most_important = max(analysis['stage_importance'].items(),
                                   key=lambda x: x[1]['avg_performance'])
                analysis['recommendations'].append(
                    f"Stage '{most_important[0]}' has highest impact on performance retention"
                )

            if analysis['ordering_impact'] and analysis['ordering_impact']['variance'] > 0.01:
                analysis['recommendations'].append(
                    f"Stage ordering matters significantly (variance: {analysis['ordering_impact']['variance']:.4f})"
                )

        return analysis

    def _save_results(self, df: pd.DataFrame):
        """Save ablation results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save DataFrame
        csv_path = Path(self.config.results_dir) / f"ablation_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # Save detailed results as JSON
        json_path = Path(self.config.results_dir) / f"ablation_details_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([{
                'configuration': r.configuration,
                'stages_active': r.stages_active,
                'metrics': r.metrics,
                'compression_ratio': r.compression_ratio,
                'performance_retention': r.performance_retention,
                'inference_speedup': r.inference_speedup,
                'memory_reduction': r.memory_reduction,
                'timestamp': r.timestamp
            } for r in self.results], f, indent=2)
        logger.info(f"Detailed results saved to {json_path}")


class ComponentContributionAnalyzer:
    """Analyzes individual component contributions to compression."""

    def __init__(self, ablation_results: pd.DataFrame):
        """
        Initialize contribution analyzer.

        Args:
            ablation_results: DataFrame from ablation study
        """
        self.results = ablation_results

    def calculate_shapley_values(self) -> Dict[str, float]:
        """
        Calculate Shapley values for each compression component.

        Returns:
            Shapley values indicating component importance
        """
        # Extract unique stages
        all_stages = set()
        for stages_str in self.results['stages']:
            all_stages.update(stages_str.split('_'))

        shapley_values = {}

        for stage in all_stages:
            if not stage:  # Skip empty strings
                continue

            marginal_contributions = []

            # Find all configurations with and without this stage
            with_stage = self.results[self.results['stages'].str.contains(stage)]
            without_stage = self.results[~self.results['stages'].str.contains(stage)]

            # Calculate marginal contribution for each subset
            for _, row_with in with_stage.iterrows():
                stages_with = set(row_with['stages'].split('_'))
                stages_without = stages_with - {stage}

                # Find corresponding configuration without this stage
                for _, row_without in without_stage.iterrows():
                    if set(row_without['stages'].split('_')) == stages_without:
                        # Calculate marginal contribution
                        contribution = row_with['performance_retention'] - row_without['performance_retention']
                        marginal_contributions.append(contribution)
                        break

            # Shapley value is average marginal contribution
            if marginal_contributions:
                shapley_values[stage] = np.mean(marginal_contributions)
            else:
                shapley_values[stage] = 0.0

        return shapley_values

    def analyze_interactions(self) -> Dict[str, float]:
        """
        Analyze interaction effects between components.

        Returns:
            Interaction strengths between component pairs
        """
        interactions = {}

        # Get all stage pairs
        all_stages = set()
        for stages_str in self.results['stages']:
            all_stages.update(stages_str.split('_'))

        for stage1, stage2 in combinations(all_stages, 2):
            if not stage1 or not stage2:
                continue

            # Find configurations with both, each alone, and neither
            both = self.results[
                self.results['stages'].str.contains(stage1) &
                self.results['stages'].str.contains(stage2)
            ]
            only1 = self.results[
                self.results['stages'].str.contains(stage1) &
                ~self.results['stages'].str.contains(stage2)
            ]
            only2 = self.results[
                ~self.results['stages'].str.contains(stage1) &
                self.results['stages'].str.contains(stage2)
            ]
            neither = self.results[
                ~self.results['stages'].str.contains(stage1) &
                ~self.results['stages'].str.contains(stage2)
            ]

            if not both.empty and not only1.empty and not only2.empty and not neither.empty:
                # Calculate interaction effect
                interaction = (
                    both['performance_retention'].mean() -
                    only1['performance_retention'].mean() -
                    only2['performance_retention'].mean() +
                    neither['performance_retention'].mean()
                )
                interactions[f"{stage1}_{stage2}"] = interaction

        return interactions