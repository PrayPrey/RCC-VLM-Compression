"""
Subspace analysis for null space preservation.

This module analyzes the overlap and orthogonality of null spaces
across compression stages.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class SubspaceMetrics:
    """Metrics for subspace analysis."""
    grassmann_distance: float
    principal_angles: np.ndarray
    subspace_overlap: float
    effective_rank: float
    coherence: float
    null_space_dimension: int


class SubspaceAnalyzer:
    """
    Analyzes subspaces and their relationships during compression.
    """

    def __init__(self):
        """Initialize subspace analyzer."""
        self.subspaces = {}
        self.metrics_history = []

    def extract_subspace(
        self,
        weight_matrix: torch.Tensor,
        rank: Optional[int] = None,
        energy_threshold: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract column and null subspaces from weight matrix.

        Args:
            weight_matrix: Weight matrix to analyze
            rank: Desired rank (optional)
            energy_threshold: Energy preservation threshold

        Returns:
            Column space basis, null space basis
        """
        # Perform SVD
        U, S, V = torch.svd(weight_matrix)

        # Determine rank if not specified
        if rank is None:
            # Use energy threshold
            cumsum_energy = torch.cumsum(S ** 2, dim=0)
            total_energy = cumsum_energy[-1]
            normalized_cumsum = cumsum_energy / total_energy
            rank = torch.searchsorted(normalized_cumsum, energy_threshold).item() + 1

        # Extract column space (range)
        column_space = U[:, :rank]

        # Extract null space
        null_rank = V.size(1) - rank
        if null_rank > 0:
            null_space = V[:, rank:]
        else:
            null_space = torch.empty(V.size(0), 0, device=V.device)

        return column_space, null_space

    def compute_grassmann_distance(
        self,
        subspace1: torch.Tensor,
        subspace2: torch.Tensor
    ) -> float:
        """
        Compute Grassmann distance between two subspaces.

        Args:
            subspace1: First subspace basis (orthonormal columns)
            subspace2: Second subspace basis (orthonormal columns)

        Returns:
            Grassmann distance
        """
        if subspace1.size(1) == 0 or subspace2.size(1) == 0:
            return 1.0  # Maximum distance for empty subspaces

        # Ensure same ambient dimension
        if subspace1.size(0) != subspace2.size(0):
            raise ValueError("Subspaces must have same ambient dimension")

        # Compute principal angles via SVD of X^T Y
        XtY = subspace1.T @ subspace2
        _, S, _ = torch.svd(XtY)

        # Clamp singular values to [0, 1]
        S = torch.clamp(S, 0, 1)

        # Principal angles
        principal_angles = torch.acos(S)

        # Grassmann distance (geodesic)
        distance = torch.norm(principal_angles).item()

        return distance

    def compute_principal_angles(
        self,
        subspace1: torch.Tensor,
        subspace2: torch.Tensor
    ) -> np.ndarray:
        """
        Compute principal angles between two subspaces.

        Args:
            subspace1: First subspace basis
            subspace2: Second subspace basis

        Returns:
            Array of principal angles (in radians)
        """
        if subspace1.size(1) == 0 or subspace2.size(1) == 0:
            return np.array([])

        # Compute X^T Y
        XtY = subspace1.T @ subspace2

        # SVD to get cosines of principal angles
        _, S, _ = torch.svd(XtY)
        S = torch.clamp(S, 0, 1)

        # Convert to angles
        angles = torch.acos(S).cpu().numpy()

        return angles

    def compute_subspace_overlap(
        self,
        subspace1: torch.Tensor,
        subspace2: torch.Tensor
    ) -> float:
        """
        Compute overlap between two subspaces.

        Args:
            subspace1: First subspace basis
            subspace2: Second subspace basis

        Returns:
            Overlap coefficient in [0, 1]
        """
        if subspace1.size(1) == 0 or subspace2.size(1) == 0:
            return 0.0

        # Project subspace2 onto subspace1
        projection = subspace1 @ (subspace1.T @ subspace2)

        # Compute Frobenius norm ratio
        overlap = torch.norm(projection, 'fro') / torch.norm(subspace2, 'fro')

        return overlap.item()

    def compute_effective_rank(self, matrix: torch.Tensor) -> float:
        """
        Compute effective rank using entropy.

        Args:
            matrix: Input matrix

        Returns:
            Effective rank
        """
        _, S, _ = torch.svd(matrix)

        # Normalize singular values
        S_normalized = S / S.sum()

        # Compute entropy
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

        # Effective rank
        effective_rank = torch.exp(entropy).item()

        return effective_rank

    def compute_coherence(
        self,
        subspace: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute mutual coherence of subspace.

        Args:
            subspace: Subspace basis
            reference: Reference basis (optional)

        Returns:
            Coherence value
        """
        if reference is None:
            # Self-coherence
            gram = subspace.T @ subspace
            # Zero out diagonal
            gram = gram - torch.diag(torch.diag(gram))
            coherence = torch.max(torch.abs(gram)).item()
        else:
            # Cross-coherence
            cross_gram = subspace.T @ reference
            coherence = torch.max(torch.abs(cross_gram)).item()

        return coherence

    def analyze_compression_stage(
        self,
        original_weights: Dict[str, torch.Tensor],
        compressed_weights: Dict[str, torch.Tensor],
        stage_name: str
    ) -> Dict[str, SubspaceMetrics]:
        """
        Analyze subspace changes for a compression stage.

        Args:
            original_weights: Original weight tensors
            compressed_weights: Compressed weight tensors
            stage_name: Name of compression stage

        Returns:
            Dictionary of subspace metrics per layer
        """
        metrics = {}

        for layer_name in original_weights:
            if layer_name not in compressed_weights:
                continue

            orig_weight = original_weights[layer_name]
            comp_weight = compressed_weights[layer_name]

            # Skip if not a matrix
            if orig_weight.dim() < 2:
                continue

            # Reshape if needed
            if orig_weight.dim() > 2:
                orig_weight = orig_weight.view(orig_weight.size(0), -1)
                comp_weight = comp_weight.view(comp_weight.size(0), -1)

            # Extract subspaces
            orig_col, orig_null = self.extract_subspace(orig_weight)
            comp_col, comp_null = self.extract_subspace(comp_weight)

            # Compute metrics
            layer_metrics = SubspaceMetrics(
                grassmann_distance=self.compute_grassmann_distance(orig_col, comp_col),
                principal_angles=self.compute_principal_angles(orig_col, comp_col),
                subspace_overlap=self.compute_subspace_overlap(orig_col, comp_col),
                effective_rank=self.compute_effective_rank(comp_weight),
                coherence=self.compute_coherence(comp_col),
                null_space_dimension=comp_null.size(1)
            )

            metrics[layer_name] = layer_metrics

            # Store for history
            self.subspaces[f"{stage_name}_{layer_name}"] = {
                'column_space': comp_col,
                'null_space': comp_null
            }

        self.metrics_history.append({
            'stage': stage_name,
            'metrics': metrics
        })

        return metrics

    def analyze_cascade_orthogonality(
        self,
        stages: List[str]
    ) -> Dict[str, float]:
        """
        Analyze orthogonality between cascade stages.

        Args:
            stages: List of stage names

        Returns:
            Orthogonality metrics between stages
        """
        orthogonality = {}

        for i in range(len(stages) - 1):
            stage1 = stages[i]
            stage2 = stages[i + 1]

            # Get all layer pairs
            stage1_layers = [k for k in self.subspaces if k.startswith(stage1)]
            stage2_layers = [k for k in self.subspaces if k.startswith(stage2)]

            distances = []
            for layer1 in stage1_layers:
                # Find corresponding layer in stage2
                layer_name = layer1.replace(stage1, '')
                layer2 = stage2 + layer_name

                if layer2 in stage2_layers:
                    null1 = self.subspaces[layer1]['null_space']
                    null2 = self.subspaces[layer2]['null_space']

                    if null1.size(1) > 0 and null2.size(1) > 0:
                        # Compute Grassmann distance between null spaces
                        dist = self.compute_grassmann_distance(null1, null2)
                        distances.append(dist)

            if distances:
                orthogonality[f"{stage1}_to_{stage2}"] = np.mean(distances)

        return orthogonality

    def visualize_subspace_evolution(
        self,
        layer_name: str,
        stages: List[str]
    ):
        """
        Visualize how subspaces evolve across compression stages.

        Args:
            layer_name: Name of layer to visualize
            stages: List of stage names
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Collect metrics for the layer across stages
        grassmann_distances = []
        overlaps = []
        effective_ranks = []
        null_dims = []

        for i, stage in enumerate(stages):
            for hist in self.metrics_history:
                if hist['stage'] == stage and layer_name in hist['metrics']:
                    metrics = hist['metrics'][layer_name]
                    if i > 0:
                        grassmann_distances.append(metrics.grassmann_distance)
                        overlaps.append(metrics.subspace_overlap)
                    effective_ranks.append(metrics.effective_rank)
                    null_dims.append(metrics.null_space_dimension)

        # Plot Grassmann distances
        if grassmann_distances:
            axes[0, 0].plot(range(1, len(grassmann_distances) + 1),
                          grassmann_distances, 'b-o')
            axes[0, 0].set_xlabel('Stage Transition')
            axes[0, 0].set_ylabel('Grassmann Distance')
            axes[0, 0].set_title('Subspace Distance Evolution')
            axes[0, 0].grid(True)

        # Plot overlaps
        if overlaps:
            axes[0, 1].plot(range(1, len(overlaps) + 1), overlaps, 'g-s')
            axes[0, 1].set_xlabel('Stage Transition')
            axes[0, 1].set_ylabel('Subspace Overlap')
            axes[0, 1].set_title('Subspace Overlap Evolution')
            axes[0, 1].grid(True)

        # Plot effective ranks
        axes[1, 0].bar(stages[:len(effective_ranks)], effective_ranks)
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Effective Rank')
        axes[1, 0].set_title('Effective Rank per Stage')
        axes[1, 0].grid(True, axis='y')

        # Plot null space dimensions
        axes[1, 1].bar(stages[:len(null_dims)], null_dims, color='orange')
        axes[1, 1].set_xlabel('Stage')
        axes[1, 1].set_ylabel('Null Space Dimension')
        axes[1, 1].set_title('Null Space Growth')
        axes[1, 1].grid(True, axis='y')

        plt.suptitle(f'Subspace Evolution for {layer_name}')
        plt.tight_layout()

        return fig

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive subspace analysis report.

        Returns:
            Analysis report dictionary
        """
        report = {
            'num_stages': len(self.metrics_history),
            'stages': [h['stage'] for h in self.metrics_history],
            'summary_statistics': {}
        }

        # Aggregate statistics across all layers and stages
        all_grassmann = []
        all_overlaps = []
        all_ranks = []

        for hist in self.metrics_history:
            for metrics in hist['metrics'].values():
                if metrics.grassmann_distance > 0:
                    all_grassmann.append(metrics.grassmann_distance)
                all_overlaps.append(metrics.subspace_overlap)
                all_ranks.append(metrics.effective_rank)

        if all_grassmann:
            report['summary_statistics']['grassmann_distance'] = {
                'mean': np.mean(all_grassmann),
                'std': np.std(all_grassmann),
                'min': np.min(all_grassmann),
                'max': np.max(all_grassmann)
            }

        report['summary_statistics']['subspace_overlap'] = {
            'mean': np.mean(all_overlaps),
            'std': np.std(all_overlaps),
            'min': np.min(all_overlaps),
            'max': np.max(all_overlaps)
        }

        report['summary_statistics']['effective_rank'] = {
            'mean': np.mean(all_ranks),
            'std': np.std(all_ranks),
            'min': np.min(all_ranks),
            'max': np.max(all_ranks)
        }

        # Check orthogonality criterion
        if all_grassmann:
            avg_distance = np.mean(all_grassmann)
            report['orthogonality_preserved'] = avg_distance > 0.7  # Threshold

        return report