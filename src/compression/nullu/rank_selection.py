"""
Adaptive rank selection for Nullu projection.

This module implements various rank selection strategies for SVD-based compression,
including energy-based, gradient-based, and layer-adaptive methods.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RankSelectionMethod(Enum):
    """Rank selection methods."""
    ENERGY = "energy"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    GRADIENT = "gradient"
    HYBRID = "hybrid"


@dataclass
class RankSelectionConfig:
    """Configuration for rank selection."""
    method: RankSelectionMethod = RankSelectionMethod.ENERGY
    energy_threshold: float = 0.95
    min_rank: int = 1
    max_rank_ratio: float = 0.5
    fixed_rank_ratio: float = 0.1
    gradient_weight: float = 0.3
    layer_wise_adaptation: bool = True
    use_importance_scores: bool = False
    importance_scores: Optional[Dict[str, float]] = None


class RankSelector:
    """Adaptive rank selection for SVD compression."""

    def __init__(self, config: RankSelectionConfig):
        """
        Initialize rank selector.

        Args:
            config: Rank selection configuration
        """
        self.config = config
        self.layer_ranks = {}
        self.energy_preserved = {}
        self.rank_history = []

    def select_rank(self,
                   singular_values: Tensor,
                   layer_name: str,
                   layer_type: Optional[str] = None,
                   gradients: Optional[Tensor] = None) -> int:
        """
        Select optimal rank for layer compression.

        Args:
            singular_values: Singular values from SVD
            layer_name: Name of the layer
            layer_type: Type of layer (linear, conv, attention)
            gradients: Optional gradient information

        Returns:
            Selected rank
        """
        if self.config.method == RankSelectionMethod.ENERGY:
            rank = self._energy_based_selection(singular_values)
        elif self.config.method == RankSelectionMethod.FIXED:
            rank = self._fixed_ratio_selection(singular_values)
        elif self.config.method == RankSelectionMethod.ADAPTIVE:
            rank = self._adaptive_selection(singular_values, layer_name, layer_type)
        elif self.config.method == RankSelectionMethod.GRADIENT:
            rank = self._gradient_based_selection(singular_values, gradients)
        elif self.config.method == RankSelectionMethod.HYBRID:
            rank = self._hybrid_selection(singular_values, layer_name, gradients)
        else:
            raise ValueError(f"Unknown rank selection method: {self.config.method}")

        # Apply constraints
        rank = self._apply_constraints(rank, len(singular_values))

        # Store selection
        self.layer_ranks[layer_name] = rank
        self._compute_energy_preserved(singular_values, rank, layer_name)

        logger.debug(f"Layer {layer_name}: selected rank {rank}/{len(singular_values)}, "
                    f"energy preserved: {self.energy_preserved[layer_name]:.4f}")

        return rank

    def _energy_based_selection(self, singular_values: Tensor) -> int:
        """
        Select rank based on energy preservation criterion.

        Args:
            singular_values: Singular values

        Returns:
            Selected rank
        """
        # Compute cumulative energy
        cumsum_energy = torch.cumsum(singular_values ** 2, dim=0)
        total_energy = cumsum_energy[-1]

        if total_energy == 0:
            return 1

        normalized_cumsum = cumsum_energy / total_energy

        # Find minimum rank that preserves target energy
        rank = torch.searchsorted(
            normalized_cumsum,
            self.config.energy_threshold
        ).item() + 1

        return rank

    def _fixed_ratio_selection(self, singular_values: Tensor) -> int:
        """
        Select fixed ratio of ranks.

        Args:
            singular_values: Singular values

        Returns:
            Selected rank
        """
        max_rank = len(singular_values)
        rank = int(max_rank * self.config.fixed_rank_ratio)
        return max(1, rank)

    def _adaptive_selection(self,
                          singular_values: Tensor,
                          layer_name: str,
                          layer_type: Optional[str]) -> int:
        """
        Adaptively select rank based on layer properties.

        Args:
            singular_values: Singular values
            layer_name: Layer name
            layer_type: Layer type

        Returns:
            Selected rank
        """
        # Compute effective rank using entropy
        normalized_s = singular_values / (singular_values.sum() + 1e-10)
        entropy = -(normalized_s * torch.log(normalized_s + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        # Find elbow point in singular value curve
        elbow_rank = self._find_elbow_point(singular_values)

        # Combine criteria
        base_rank = min(effective_rank, elbow_rank)

        # Apply layer-wise adaptation
        if self.config.layer_wise_adaptation:
            depth_factor = self._get_depth_factor(layer_name, layer_type)
            rank = int(base_rank * depth_factor)
        else:
            rank = int(base_rank)

        # Apply importance scores if available
        if self.config.use_importance_scores and self.config.importance_scores:
            if layer_name in self.config.importance_scores:
                importance = self.config.importance_scores[layer_name]
                rank = int(rank * (0.5 + 0.5 * importance))

        return rank

    def _gradient_based_selection(self,
                                 singular_values: Tensor,
                                 gradients: Optional[Tensor]) -> int:
        """
        Select rank based on gradient information.

        Args:
            singular_values: Singular values
            gradients: Gradient information

        Returns:
            Selected rank
        """
        if gradients is None:
            # Fall back to energy-based selection
            return self._energy_based_selection(singular_values)

        # Compute gradient-weighted importance
        grad_norm = torch.norm(gradients.view(-1))
        if grad_norm > 0:
            grad_importance = torch.abs(gradients).mean(dim=0)
            grad_importance = grad_importance / grad_importance.sum()

            # Weight singular values by gradient importance
            weighted_values = singular_values * grad_importance

            # Select rank based on weighted cumulative sum
            cumsum = torch.cumsum(weighted_values, dim=0)
            total = cumsum[-1]

            if total > 0:
                normalized_cumsum = cumsum / total
                rank = torch.searchsorted(
                    normalized_cumsum,
                    self.config.energy_threshold
                ).item() + 1
            else:
                rank = 1
        else:
            rank = self._energy_based_selection(singular_values)

        return rank

    def _hybrid_selection(self,
                         singular_values: Tensor,
                         layer_name: str,
                         gradients: Optional[Tensor]) -> int:
        """
        Hybrid selection combining multiple criteria.

        Args:
            singular_values: Singular values
            layer_name: Layer name
            gradients: Optional gradients

        Returns:
            Selected rank
        """
        # Get ranks from different methods
        energy_rank = self._energy_based_selection(singular_values)
        adaptive_rank = self._adaptive_selection(singular_values, layer_name, None)

        if gradients is not None:
            gradient_rank = self._gradient_based_selection(singular_values, gradients)
            # Weighted combination
            rank = int(
                (1 - self.config.gradient_weight) * 0.5 * (energy_rank + adaptive_rank) +
                self.config.gradient_weight * gradient_rank
            )
        else:
            # Average of energy and adaptive
            rank = int((energy_rank + adaptive_rank) / 2)

        return rank

    def _find_elbow_point(self, singular_values: Tensor) -> int:
        """
        Find elbow point in singular value curve.

        Args:
            singular_values: Singular values

        Returns:
            Elbow point index
        """
        if len(singular_values) < 3:
            return len(singular_values)

        # Log-scale for better elbow detection
        log_values = torch.log(singular_values + 1e-10)

        # Compute second derivative
        first_diff = torch.diff(log_values)
        second_diff = torch.diff(first_diff)

        if len(second_diff) > 0:
            # Find maximum curvature point
            elbow_idx = torch.argmax(torch.abs(second_diff)).item() + 2
        else:
            elbow_idx = len(singular_values) // 2

        return min(elbow_idx, len(singular_values))

    def _get_depth_factor(self, layer_name: str, layer_type: Optional[str]) -> float:
        """
        Get depth-based adjustment factor.

        Args:
            layer_name: Layer name
            layer_type: Layer type

        Returns:
            Depth factor for rank adjustment
        """
        # Parse layer depth from name
        depth_factor = 1.0

        # Check layer type
        if layer_type == "attention":
            depth_factor = 1.1  # Preserve more for attention
        elif layer_type == "mlp":
            depth_factor = 0.9  # Compress more for MLP

        # Check position in network
        if any(keyword in layer_name.lower() for keyword in ['early', 'conv1', 'layer1', 'embed']):
            depth_factor *= 1.2  # Preserve more in early layers
        elif any(keyword in layer_name.lower() for keyword in ['late', 'fc', 'head', 'classifier']):
            depth_factor *= 0.8  # Compress more in late layers
        elif any(keyword in layer_name.lower() for keyword in ['mid', 'layer2', 'layer3']):
            depth_factor *= 1.0  # Standard for middle layers

        return depth_factor

    def _apply_constraints(self, rank: int, max_rank: int) -> int:
        """
        Apply min/max constraints to rank.

        Args:
            rank: Proposed rank
            max_rank: Maximum possible rank

        Returns:
            Constrained rank
        """
        rank = max(self.config.min_rank, rank)
        rank = min(rank, int(max_rank * self.config.max_rank_ratio))
        rank = min(rank, max_rank)
        return rank

    def _compute_energy_preserved(self,
                                 singular_values: Tensor,
                                 rank: int,
                                 layer_name: str):
        """
        Compute and store energy preserved by rank selection.

        Args:
            singular_values: All singular values
            rank: Selected rank
            layer_name: Layer name
        """
        total_energy = (singular_values ** 2).sum().item()
        if total_energy > 0:
            preserved_energy = (singular_values[:rank] ** 2).sum().item()
            self.energy_preserved[layer_name] = preserved_energy / total_energy
        else:
            self.energy_preserved[layer_name] = 1.0

    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics.

        Returns:
            Dictionary of compression statistics
        """
        if not self.layer_ranks:
            return {}

        ranks = list(self.layer_ranks.values())
        energies = list(self.energy_preserved.values())

        return {
            'total_layers': len(self.layer_ranks),
            'average_rank': np.mean(ranks),
            'min_rank': min(ranks),
            'max_rank': max(ranks),
            'average_energy_preserved': np.mean(energies),
            'min_energy_preserved': min(energies),
            'max_energy_preserved': max(energies),
            'layer_ranks': self.layer_ranks,
            'energy_preserved': self.energy_preserved
        }

    def optimize_rank_distribution(self,
                                  target_compression: float,
                                  layer_importances: Dict[str, float]) -> Dict[str, int]:
        """
        Optimize rank distribution across layers for target compression.

        Args:
            target_compression: Target compression ratio
            layer_importances: Importance scores per layer

        Returns:
            Optimized rank assignments
        """
        # Sort layers by importance
        sorted_layers = sorted(
            layer_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        optimized_ranks = {}
        remaining_budget = 1.0 / target_compression

        for layer_name, importance in sorted_layers:
            if layer_name in self.layer_ranks:
                original_rank = self.layer_ranks[layer_name]
                # Allocate rank proportional to importance
                allocated_ratio = importance * remaining_budget
                optimized_rank = int(original_rank * allocated_ratio)
                optimized_rank = max(self.config.min_rank, optimized_rank)
                optimized_ranks[layer_name] = optimized_rank

        return optimized_ranks