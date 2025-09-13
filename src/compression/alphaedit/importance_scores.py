"""
Importance score computation for AlphaEdit adaptive weighting.

This module implements various importance metrics for weight adaptation,
including gradient-based, magnitude-based, and Taylor expansion methods.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ImportanceMetric(Enum):
    """Importance metric types."""
    GRADIENT = "gradient"
    MAGNITUDE = "magnitude"
    TAYLOR = "taylor"
    FISHER = "fisher"
    ACTIVATION = "activation"
    HYBRID = "hybrid"


@dataclass
class ImportanceConfig:
    """Configuration for importance computation."""
    metric: ImportanceMetric = ImportanceMetric.GRADIENT
    normalize: bool = True
    use_running_average: bool = True
    averaging_momentum: float = 0.9
    taylor_order: int = 1
    fisher_samples: int = 100
    activation_percentile: float = 95.0
    hybrid_weights: Dict[str, float] = None


class ImportanceScorer:
    """Computes importance scores for adaptive weight scaling."""

    def __init__(self, config: ImportanceConfig):
        """
        Initialize importance scorer.

        Args:
            config: Importance computation configuration
        """
        self.config = config
        self.importance_scores = {}
        self.running_scores = {}
        self.gradient_accumulator = {}
        self.activation_accumulator = {}
        self.fisher_accumulator = {}

        # Default hybrid weights
        if config.hybrid_weights is None and config.metric == ImportanceMetric.HYBRID:
            self.config.hybrid_weights = {
                'gradient': 0.3,
                'magnitude': 0.2,
                'taylor': 0.3,
                'activation': 0.2
            }

    def compute_importance(self,
                          layer_name: str,
                          weight: Tensor,
                          gradient: Optional[Tensor] = None,
                          activation: Optional[Tensor] = None,
                          output_gradient: Optional[Tensor] = None) -> Tensor:
        """
        Compute importance scores for layer weights.

        Args:
            layer_name: Name of the layer
            weight: Weight tensor
            gradient: Weight gradient
            activation: Layer activation
            output_gradient: Output gradient for Taylor expansion

        Returns:
            Importance scores tensor
        """
        if self.config.metric == ImportanceMetric.GRADIENT:
            scores = self._gradient_importance(weight, gradient)
        elif self.config.metric == ImportanceMetric.MAGNITUDE:
            scores = self._magnitude_importance(weight)
        elif self.config.metric == ImportanceMetric.TAYLOR:
            scores = self._taylor_importance(weight, gradient, output_gradient)
        elif self.config.metric == ImportanceMetric.FISHER:
            scores = self._fisher_importance(layer_name, weight, gradient)
        elif self.config.metric == ImportanceMetric.ACTIVATION:
            scores = self._activation_importance(weight, activation)
        elif self.config.metric == ImportanceMetric.HYBRID:
            scores = self._hybrid_importance(
                layer_name, weight, gradient, activation, output_gradient
            )
        else:
            raise ValueError(f"Unknown importance metric: {self.config.metric}")

        # Apply normalization
        if self.config.normalize:
            scores = self._normalize_scores(scores)

        # Update running average
        if self.config.use_running_average:
            scores = self._update_running_average(layer_name, scores)

        # Store scores
        self.importance_scores[layer_name] = scores

        return scores

    def _gradient_importance(self,
                           weight: Tensor,
                           gradient: Optional[Tensor]) -> Tensor:
        """
        Compute gradient-based importance.

        Args:
            weight: Weight tensor
            gradient: Gradient tensor

        Returns:
            Importance scores
        """
        if gradient is None:
            # Fall back to magnitude if no gradient
            return self._magnitude_importance(weight)

        # Gradient magnitude
        importance = torch.abs(gradient)

        # Weight gradient product (change sensitivity)
        importance = importance * torch.abs(weight)

        return importance

    def _magnitude_importance(self, weight: Tensor) -> Tensor:
        """
        Compute magnitude-based importance.

        Args:
            weight: Weight tensor

        Returns:
            Importance scores
        """
        return torch.abs(weight)

    def _taylor_importance(self,
                         weight: Tensor,
                         gradient: Optional[Tensor],
                         output_gradient: Optional[Tensor]) -> Tensor:
        """
        Compute Taylor expansion-based importance.

        Args:
            weight: Weight tensor
            gradient: Weight gradient
            output_gradient: Output gradient

        Returns:
            Importance scores
        """
        if gradient is None:
            return self._magnitude_importance(weight)

        # First-order Taylor importance
        importance = torch.abs(weight * gradient)

        # Second-order term if available
        if self.config.taylor_order >= 2 and output_gradient is not None:
            # Approximate second-order term
            hessian_diag_approx = gradient ** 2
            second_order = 0.5 * torch.abs(weight ** 2 * hessian_diag_approx)
            importance = importance + second_order

        return importance

    def _fisher_importance(self,
                         layer_name: str,
                         weight: Tensor,
                         gradient: Optional[Tensor]) -> Tensor:
        """
        Compute Fisher information-based importance.

        Args:
            layer_name: Layer name
            weight: Weight tensor
            gradient: Weight gradient

        Returns:
            Importance scores
        """
        if gradient is None:
            return self._magnitude_importance(weight)

        # Initialize accumulator if needed
        if layer_name not in self.fisher_accumulator:
            self.fisher_accumulator[layer_name] = torch.zeros_like(weight)

        # Accumulate squared gradients (Fisher information diagonal)
        self.fisher_accumulator[layer_name] += gradient ** 2

        # Compute importance as Fisher-weighted magnitude
        fisher_diag = self.fisher_accumulator[layer_name] / self.config.fisher_samples
        importance = torch.sqrt(fisher_diag + 1e-8) * torch.abs(weight)

        return importance

    def _activation_importance(self,
                             weight: Tensor,
                             activation: Optional[Tensor]) -> Tensor:
        """
        Compute activation-based importance.

        Args:
            weight: Weight tensor
            activation: Activation tensor

        Returns:
            Importance scores
        """
        if activation is None:
            return self._magnitude_importance(weight)

        # Compute activation statistics
        if weight.dim() == 2:  # Linear layer
            # Average activation magnitude
            if activation.dim() == 3:  # batch x seq x dim
                act_importance = torch.abs(activation).mean(dim=(0, 1))
            elif activation.dim() == 2:  # batch x dim
                act_importance = torch.abs(activation).mean(dim=0)
            else:
                act_importance = torch.abs(activation)

            # Expand to weight shape
            if act_importance.numel() == weight.size(1):
                importance = torch.abs(weight) * act_importance.unsqueeze(0)
            else:
                importance = torch.abs(weight)

        elif weight.dim() == 4:  # Conv layer
            # Channel-wise activation importance
            act_importance = torch.abs(activation).mean(dim=(0, 2, 3))
            importance = torch.abs(weight) * act_importance.view(-1, 1, 1, 1)

        else:
            importance = torch.abs(weight)

        return importance

    def _hybrid_importance(self,
                         layer_name: str,
                         weight: Tensor,
                         gradient: Optional[Tensor],
                         activation: Optional[Tensor],
                         output_gradient: Optional[Tensor]) -> Tensor:
        """
        Compute hybrid importance combining multiple metrics.

        Args:
            layer_name: Layer name
            weight: Weight tensor
            gradient: Weight gradient
            activation: Activation tensor
            output_gradient: Output gradient

        Returns:
            Combined importance scores
        """
        importance_dict = {}

        # Compute individual importances
        if 'gradient' in self.config.hybrid_weights:
            importance_dict['gradient'] = self._gradient_importance(weight, gradient)

        if 'magnitude' in self.config.hybrid_weights:
            importance_dict['magnitude'] = self._magnitude_importance(weight)

        if 'taylor' in self.config.hybrid_weights:
            importance_dict['taylor'] = self._taylor_importance(
                weight, gradient, output_gradient
            )

        if 'activation' in self.config.hybrid_weights:
            importance_dict['activation'] = self._activation_importance(weight, activation)

        # Normalize each component
        for key in importance_dict:
            importance_dict[key] = self._normalize_scores(importance_dict[key])

        # Weighted combination
        combined = torch.zeros_like(weight)
        for key, importance in importance_dict.items():
            weight_factor = self.config.hybrid_weights.get(key, 0.0)
            combined += weight_factor * importance

        return combined

    def _normalize_scores(self, scores: Tensor) -> Tensor:
        """
        Normalize importance scores.

        Args:
            scores: Raw importance scores

        Returns:
            Normalized scores
        """
        # Min-max normalization per layer
        min_val = scores.min()
        max_val = scores.max()

        if max_val > min_val:
            scores = (scores - min_val) / (max_val - min_val)
        else:
            scores = torch.ones_like(scores)

        return scores

    def _update_running_average(self, layer_name: str, scores: Tensor) -> Tensor:
        """
        Update running average of importance scores.

        Args:
            layer_name: Layer name
            scores: Current importance scores

        Returns:
            Averaged importance scores
        """
        if layer_name not in self.running_scores:
            self.running_scores[layer_name] = scores.clone()
        else:
            momentum = self.config.averaging_momentum
            self.running_scores[layer_name] = (
                momentum * self.running_scores[layer_name] +
                (1 - momentum) * scores
            )

        return self.running_scores[layer_name]

    def get_layer_importance(self, layer_name: str) -> Optional[float]:
        """
        Get overall importance score for a layer.

        Args:
            layer_name: Layer name

        Returns:
            Layer-level importance score
        """
        if layer_name not in self.importance_scores:
            return None

        scores = self.importance_scores[layer_name]
        return scores.mean().item()

    def get_global_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Get global ranking of layer importances.

        Returns:
            List of (layer_name, importance) tuples, sorted by importance
        """
        layer_importances = []

        for layer_name in self.importance_scores:
            importance = self.get_layer_importance(layer_name)
            if importance is not None:
                layer_importances.append((layer_name, importance))

        # Sort by importance (descending)
        layer_importances.sort(key=lambda x: x[1], reverse=True)

        return layer_importances

    def compute_structured_importance(self,
                                     weight: Tensor,
                                     structure: str = "channel") -> Tensor:
        """
        Compute structured importance scores.

        Args:
            weight: Weight tensor
            structure: Structure type (channel, filter, block)

        Returns:
            Structured importance scores
        """
        base_importance = self._magnitude_importance(weight)

        if structure == "channel":
            if weight.dim() == 4:  # Conv2d
                # Channel-wise importance
                importance = base_importance.mean(dim=(1, 2, 3))
            elif weight.dim() == 2:  # Linear
                # Column-wise importance
                importance = base_importance.mean(dim=0)
            else:
                importance = base_importance

        elif structure == "filter":
            if weight.dim() == 4:  # Conv2d
                # Filter-wise importance
                importance = base_importance.mean(dim=(0, 2, 3))
            elif weight.dim() == 2:  # Linear
                # Row-wise importance
                importance = base_importance.mean(dim=1)
            else:
                importance = base_importance

        elif structure == "block":
            # Block-wise importance (for block-sparse patterns)
            block_size = 16
            if weight.dim() == 2:
                h, w = weight.shape
                h_blocks = (h + block_size - 1) // block_size
                w_blocks = (w + block_size - 1) // block_size
                importance = torch.zeros(h_blocks, w_blocks, device=weight.device)

                for i in range(h_blocks):
                    for j in range(w_blocks):
                        h_start = i * block_size
                        h_end = min((i + 1) * block_size, h)
                        w_start = j * block_size
                        w_end = min((j + 1) * block_size, w)

                        block_importance = base_importance[h_start:h_end, w_start:w_end].mean()
                        importance[i, j] = block_importance
            else:
                importance = base_importance

        else:
            importance = base_importance

        return importance

    def reset_accumulators(self):
        """Reset all accumulators."""
        self.gradient_accumulator.clear()
        self.activation_accumulator.clear()
        self.fisher_accumulator.clear()
        self.running_scores.clear()

    def get_importance_stats(self) -> Dict:
        """
        Get importance computation statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.importance_scores:
            return {}

        all_scores = []
        for scores in self.importance_scores.values():
            all_scores.append(scores.flatten())

        if all_scores:
            all_scores = torch.cat(all_scores)
            stats = {
                'num_layers': len(self.importance_scores),
                'mean_importance': all_scores.mean().item(),
                'std_importance': all_scores.std().item(),
                'min_importance': all_scores.min().item(),
                'max_importance': all_scores.max().item(),
                'sparsity': (all_scores < 0.1).float().mean().item(),
                'has_running_average': len(self.running_scores) > 0,
                'has_fisher_info': len(self.fisher_accumulator) > 0
            }
        else:
            stats = {}

        return stats