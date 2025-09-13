"""
Base interfaces for evaluation metrics.

This module provides abstract base classes and common utilities
for all evaluation metrics in the RCC system.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Base configuration for metrics."""
    name: str
    compute_on_cpu: bool = False
    reduction: str = "mean"  # mean, sum, none
    threshold: Optional[float] = None
    k_values: List[int] = None  # For top-k metrics
    average: str = "macro"  # For classification: macro, micro, weighted


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    """

    def __init__(self, config: MetricConfig):
        """
        Initialize metric.

        Args:
            config: Metric configuration
        """
        self.config = config
        self.reset()

    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Update metric with new predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional metric-specific arguments
        """
        pass

    @abstractmethod
    def compute(self) -> Union[float, Dict[str, float]]:
        """
        Compute the metric value.

        Returns:
            Metric value or dictionary of values
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor,
                 **kwargs) -> Union[float, Dict[str, float]]:
        """
        Compute metric for given predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments

        Returns:
            Metric value
        """
        self.reset()
        self.update(predictions, targets, **kwargs)
        return self.compute()


class AccumulativeMetric(BaseMetric):
    """
    Base class for metrics that accumulate values over batches.
    """

    def __init__(self, config: MetricConfig):
        """Initialize accumulative metric."""
        super().__init__(config)
        self.accumulated_values = []
        self.accumulated_weights = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               weights: Optional[torch.Tensor] = None, **kwargs):
        """
        Update metric with new batch.

        Args:
            predictions: Predictions
            targets: Targets
            weights: Sample weights (optional)
            **kwargs: Additional arguments
        """
        value = self._compute_batch(predictions, targets, **kwargs)

        if self.config.compute_on_cpu:
            value = value.cpu()

        self.accumulated_values.append(value)

        if weights is not None:
            self.accumulated_weights.append(weights)
        else:
            batch_size = predictions.size(0)
            self.accumulated_weights.append(torch.ones(batch_size))

    @abstractmethod
    def _compute_batch(self, predictions: torch.Tensor,
                      targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute metric for a single batch.

        Args:
            predictions: Batch predictions
            targets: Batch targets
            **kwargs: Additional arguments

        Returns:
            Batch metric value
        """
        pass

    def compute(self) -> float:
        """
        Compute final metric value.

        Returns:
            Aggregated metric value
        """
        if not self.accumulated_values:
            return 0.0

        values = torch.cat(self.accumulated_values)
        weights = torch.cat(self.accumulated_weights)

        if self.config.reduction == "mean":
            return (values * weights).sum().item() / weights.sum().item()
        elif self.config.reduction == "sum":
            return values.sum().item()
        else:
            return values

    def reset(self):
        """Reset accumulator."""
        self.accumulated_values = []
        self.accumulated_weights = []


class MetricCollection:
    """
    Collection of metrics to compute together.
    """

    def __init__(self, metrics: Dict[str, BaseMetric]):
        """
        Initialize metric collection.

        Args:
            metrics: Dictionary of metrics
        """
        self.metrics = metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Update all metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        """
        for metric in self.metrics.values():
            metric.update(predictions, targets, **kwargs)

    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                for k, v in value.items():
                    results[f"{name}_{k}"] = v
            else:
                results[name] = value
        return results

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def add_metric(self, name: str, metric: BaseMetric):
        """
        Add a new metric to the collection.

        Args:
            name: Metric name
            metric: Metric instance
        """
        self.metrics[name] = metric

    def remove_metric(self, name: str):
        """
        Remove a metric from the collection.

        Args:
            name: Metric name to remove
        """
        if name in self.metrics:
            del self.metrics[name]


class ConfusionMatrix:
    """
    Confusion matrix for classification metrics.
    """

    def __init__(self, num_classes: int):
        """
        Initialize confusion matrix.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix.

        Args:
            predictions: Predicted class indices
            targets: True class indices
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        for t, p in zip(targets, predictions):
            self.matrix[t.long(), p.long()] += 1

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute various metrics from confusion matrix.

        Returns:
            Dictionary of metrics
        """
        # Per-class metrics
        tp = torch.diag(self.matrix)
        fp = self.matrix.sum(dim=0) - tp
        fn = self.matrix.sum(dim=1) - tp
        tn = self.matrix.sum() - (tp + fp + fn)

        # Precision
        precision = tp / (tp + fp + 1e-8)

        # Recall
        recall = tp / (tp + fn + 1e-8)

        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Accuracy
        accuracy = (tp + tn) / self.matrix.sum()

        return {
            'confusion_matrix': self.matrix.cpu().numpy(),
            'precision_per_class': precision.cpu().numpy(),
            'recall_per_class': recall.cpu().numpy(),
            'f1_per_class': f1.cpu().numpy(),
            'accuracy_per_class': accuracy.cpu().numpy(),
            'precision_macro': precision.mean().item(),
            'recall_macro': recall.mean().item(),
            'f1_macro': f1.mean().item(),
            'accuracy': tp.sum().item() / self.matrix.sum().item()
        }

    def reset(self):
        """Reset confusion matrix."""
        self.matrix = torch.zeros(self.num_classes, self.num_classes)

    def plot(self, class_names: Optional[List[str]] = None):
        """
        Plot confusion matrix.

        Args:
            class_names: Names of classes
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = self.matrix / self.matrix.sum(dim=1, keepdim=True)
        cm_normalized = cm_normalized.cpu().numpy()

        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names or range(self.num_classes),
            yticklabels=class_names or range(self.num_classes),
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized Confusion Matrix')

        return fig


def create_metric(metric_type: str, **kwargs) -> BaseMetric:
    """
    Factory function to create metrics.

    Args:
        metric_type: Type of metric to create
        **kwargs: Metric-specific arguments

    Returns:
        Metric instance
    """
    from .classification import (
        AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric
    )
    from .retrieval import RetrievalMetrics
    from .captioning import CaptioningMetrics

    metric_map = {
        'accuracy': AccuracyMetric,
        'precision': PrecisionMetric,
        'recall': RecallMetric,
        'f1': F1Metric,
        'retrieval': RetrievalMetrics,
        'captioning': CaptioningMetrics
    }

    if metric_type not in metric_map:
        raise ValueError(f"Unknown metric type: {metric_type}")

    config = MetricConfig(name=metric_type, **kwargs)
    return metric_map[metric_type](config)