"""
Grassmann distance computation for null space analysis.

This module provides functions to compute Grassmann distances between subspaces
to analyze the orthogonality of compression stages.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_grassmann_distance(
    subspace1: torch.Tensor,
    subspace2: torch.Tensor,
    method: str = 'projection'
) -> float:
    """
    Compute Grassmann distance between two subspaces.

    Args:
        subspace1: First subspace basis matrix [n, k1]
        subspace2: Second subspace basis matrix [n, k2]
        method: Distance computation method ('projection', 'geodesic', 'chordal')

    Returns:
        Grassmann distance between subspaces
    """
    # Ensure inputs are on the same device
    device = subspace1.device
    subspace2 = subspace2.to(device)

    # Orthonormalize bases if needed
    subspace1 = orthonormalize_basis(subspace1)
    subspace2 = orthonormalize_basis(subspace2)

    if method == 'projection':
        return _projection_distance(subspace1, subspace2)
    elif method == 'geodesic':
        return _geodesic_distance(subspace1, subspace2)
    elif method == 'chordal':
        return _chordal_distance(subspace1, subspace2)
    else:
        raise ValueError(f"Unknown method: {method}")


def _projection_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """
    Compute projection-based Grassmann distance.

    Args:
        U1: First orthonormal basis
        U2: Second orthonormal basis

    Returns:
        Projection distance
    """
    # Compute projection matrix
    P1 = torch.mm(U1, U1.T)
    P2 = torch.mm(U2, U2.T)

    # Frobenius norm of difference
    diff = P1 - P2
    distance = torch.norm(diff, p='fro').item() / np.sqrt(2)

    return distance


def _geodesic_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """
    Compute geodesic Grassmann distance.

    Args:
        U1: First orthonormal basis
        U2: Second orthonormal basis

    Returns:
        Geodesic distance
    """
    # Compute principal angles
    M = torch.mm(U1.T, U2)
    _, s, _ = torch.svd(M)

    # Clamp singular values to avoid numerical issues
    s = torch.clamp(s, -1, 1)

    # Principal angles
    theta = torch.acos(s)

    # Geodesic distance
    distance = torch.norm(theta).item()

    return distance


def _chordal_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """
    Compute chordal Grassmann distance.

    Args:
        U1: First orthonormal basis
        U2: Second orthonormal basis

    Returns:
        Chordal distance
    """
    # Compute principal angles
    M = torch.mm(U1.T, U2)
    _, s, _ = torch.svd(M)

    # Clamp singular values
    s = torch.clamp(s, 0, 1)

    # Chordal distance
    distance = torch.sqrt(torch.sum(1 - s**2)).item()

    return distance


def orthonormalize_basis(basis: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize a basis using QR decomposition.

    Args:
        basis: Basis matrix to orthonormalize

    Returns:
        Orthonormal basis
    """
    if basis.shape[0] < basis.shape[1]:
        logger.warning("Basis has more columns than rows, transposing")
        basis = basis.T

    # QR decomposition
    Q, _ = torch.linalg.qr(basis, mode='reduced')

    return Q


def compute_principal_angles(
    subspace1: torch.Tensor,
    subspace2: torch.Tensor
) -> torch.Tensor:
    """
    Compute principal angles between two subspaces.

    Args:
        subspace1: First subspace basis
        subspace2: Second subspace basis

    Returns:
        Principal angles in radians
    """
    # Orthonormalize
    U1 = orthonormalize_basis(subspace1)
    U2 = orthonormalize_basis(subspace2)

    # Compute SVD of U1^T U2
    M = torch.mm(U1.T, U2)
    _, s, _ = torch.svd(M)

    # Clamp to avoid numerical issues
    s = torch.clamp(s, -1, 1)

    # Principal angles
    angles = torch.acos(s)

    return angles


def subspace_overlap(
    subspace1: torch.Tensor,
    subspace2: torch.Tensor
) -> float:
    """
    Compute overlap between two subspaces.

    Args:
        subspace1: First subspace basis
        subspace2: Second subspace basis

    Returns:
        Overlap measure [0, 1]
    """
    # Orthonormalize
    U1 = orthonormalize_basis(subspace1)
    U2 = orthonormalize_basis(subspace2)

    # Compute trace of U1^T U2 U2^T U1
    M = torch.mm(U1.T, U2)
    overlap = torch.trace(torch.mm(M, M.T)).item()

    # Normalize by dimension
    max_overlap = min(U1.shape[1], U2.shape[1])
    normalized_overlap = overlap / max_overlap if max_overlap > 0 else 0

    return normalized_overlap


class GrassmannAnalyzer:
    """Analyzer for Grassmann distances in compression pipeline."""

    def __init__(self):
        """Initialize the Grassmann analyzer."""
        self.subspaces = {}
        self.distances = {}

    def add_subspace(
        self,
        name: str,
        subspace: torch.Tensor,
        layer_name: Optional[str] = None
    ):
        """
        Add a subspace for analysis.

        Args:
            name: Name identifier for the subspace
            subspace: Subspace basis matrix
            layer_name: Optional layer name for hierarchical storage
        """
        if layer_name:
            if name not in self.subspaces:
                self.subspaces[name] = {}
            self.subspaces[name][layer_name] = orthonormalize_basis(subspace)
        else:
            self.subspaces[name] = orthonormalize_basis(subspace)

    def compute_all_distances(
        self,
        method: str = 'projection'
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute all pairwise Grassmann distances.

        Args:
            method: Distance computation method

        Returns:
            Dictionary of pairwise distances
        """
        names = list(self.subspaces.keys())
        distances = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]

                if isinstance(self.subspaces[name1], dict):
                    # Layer-wise distances
                    layer_distances = {}
                    for layer in self.subspaces[name1]:
                        if layer in self.subspaces[name2]:
                            dist = compute_grassmann_distance(
                                self.subspaces[name1][layer],
                                self.subspaces[name2][layer],
                                method=method
                            )
                            layer_distances[layer] = dist

                    # Average distance
                    if layer_distances:
                        avg_dist = sum(layer_distances.values()) / len(layer_distances)
                        distances[(name1, name2)] = avg_dist
                        distances[f"{name1}_{name2}_layers"] = layer_distances
                else:
                    # Single subspace distance
                    dist = compute_grassmann_distance(
                        self.subspaces[name1],
                        self.subspaces[name2],
                        method=method
                    )
                    distances[(name1, name2)] = dist

        self.distances = distances
        return distances

    def check_orthogonality(
        self,
        threshold: float = 0.7
    ) -> Dict[Tuple[str, str], bool]:
        """
        Check if subspaces are sufficiently orthogonal.

        Args:
            threshold: Minimum distance for orthogonality

        Returns:
            Dictionary indicating orthogonality status
        """
        if not self.distances:
            self.compute_all_distances()

        orthogonality = {}
        for key, distance in self.distances.items():
            if isinstance(key, tuple) and len(key) == 2:
                orthogonality[key] = distance > threshold

        return orthogonality

    def get_overlap_matrix(self) -> np.ndarray:
        """
        Compute overlap matrix between all subspaces.

        Returns:
            Overlap matrix
        """
        names = list(self.subspaces.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    name1, name2 = names[i], names[j]

                    if isinstance(self.subspaces[name1], dict):
                        # Average overlap for layer-wise subspaces
                        overlaps = []
                        for layer in self.subspaces[name1]:
                            if layer in self.subspaces.get(name2, {}):
                                overlap = subspace_overlap(
                                    self.subspaces[name1][layer],
                                    self.subspaces[name2][layer]
                                )
                                overlaps.append(overlap)
                        if overlaps:
                            matrix[i, j] = np.mean(overlaps)
                    else:
                        matrix[i, j] = subspace_overlap(
                            self.subspaces[name1],
                            self.subspaces.get(name2, self.subspaces[name1])
                        )

        return matrix

    def recommend_adjustments(
        self,
        min_distance: float = 0.7
    ) -> List[str]:
        """
        Recommend adjustments based on Grassmann analysis.

        Args:
            min_distance: Minimum acceptable distance

        Returns:
            List of recommendations
        """
        if not self.distances:
            self.compute_all_distances()

        recommendations = []

        for (name1, name2), distance in self.distances.items():
            if isinstance(distance, float) and distance < min_distance:
                recommendations.append(
                    f"Low orthogonality between {name1} and {name2}: "
                    f"distance={distance:.4f}. Consider adjusting compression parameters."
                )

                # Specific recommendations
                if distance < 0.3:
                    recommendations.append(
                        f"  - Critical: Use different compression methods or "
                        f"significantly different parameters"
                    )
                elif distance < 0.5:
                    recommendations.append(
                        f"  - Warning: Increase rank reduction difference or "
                        f"use orthogonalization"
                    )
                else:
                    recommendations.append(
                        f"  - Minor: Fine-tune compression parameters for "
                        f"better orthogonality"
                    )

        if not recommendations:
            recommendations.append("All subspaces show good orthogonality.")

        return recommendations

    def visualize_grassmann_distances(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Visualize Grassmann distances as heatmap.

        Args:
            save_path: Optional path to save figure
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.distances:
            self.compute_all_distances()

        # Create distance matrix
        names = list(self.subspaces.keys())
        n = len(names)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                elif (names[i], names[j]) in self.distances:
                    distance_matrix[i, j] = self.distances[(names[i], names[j])]
                elif (names[j], names[i]) in self.distances:
                    distance_matrix[i, j] = self.distances[(names[j], names[i])]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Heatmap of distances
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=names,
            yticklabels=names,
            ax=ax1,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Grassmann Distance'}
        )
        ax1.set_title('Grassmann Distances Between Compression Stages')
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Stage')

        # Bar plot of average distances
        avg_distances = {}
        for name in names:
            distances = []
            for (n1, n2), d in self.distances.items():
                if isinstance(d, float):
                    if n1 == name or n2 == name:
                        distances.append(d)
            if distances:
                avg_distances[name] = np.mean(distances)

        if avg_distances:
            stages = list(avg_distances.keys())
            values = list(avg_distances.values())
            colors = ['green' if v > 0.7 else 'yellow' if v > 0.5 else 'red' for v in values]

            bars = ax2.bar(stages, values, color=colors)
            ax2.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Good (>0.7)')
            ax2.axhline(y=0.5, color='y', linestyle='--', alpha=0.5, label='Warning (>0.5)')
            ax2.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Critical (>0.3)')

            ax2.set_title('Average Grassmann Distance per Stage')
            ax2.set_xlabel('Compression Stage')
            ax2.set_ylabel('Average Distance')
            ax2.set_ylim(0, 1)
            ax2.legend()

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grassmann distance visualization saved to {save_path}")

        plt.show()

    def visualize_principal_angles(
        self,
        subspace1_name: str,
        subspace2_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize principal angles between two subspaces.

        Args:
            subspace1_name: Name of first subspace
            subspace2_name: Name of second subspace
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        if subspace1_name not in self.subspaces or subspace2_name not in self.subspaces:
            raise ValueError("Subspace not found in analyzer")

        # Compute principal angles
        angles = compute_principal_angles(
            self.subspaces[subspace1_name],
            self.subspaces[subspace2_name]
        )

        angles_deg = angles.cpu().numpy() * 180 / np.pi

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot of angles
        indices = np.arange(len(angles_deg))
        ax1.bar(indices, angles_deg, color='steelblue')
        ax1.set_xlabel('Principal Angle Index')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title(f'Principal Angles: {subspace1_name} vs {subspace2_name}')
        ax1.grid(True, alpha=0.3)

        # Cumulative plot
        cumulative = np.cumsum(angles_deg) / np.sum(angles_deg) * 100
        ax2.plot(indices, cumulative, 'b-', linewidth=2)
        ax2.fill_between(indices, 0, cumulative, alpha=0.3)
        ax2.set_xlabel('Principal Angle Index')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.set_title('Cumulative Principal Angle Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Principal angles visualization saved to {save_path}")

        plt.show()

    def visualize_overlap_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Visualize subspace overlap matrix.

        Args:
            save_path: Optional path to save figure
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        overlap_matrix = self.get_overlap_matrix()
        names = list(self.subspaces.keys())

        plt.figure(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            xticklabels=names,
            yticklabels=names,
            vmin=0,
            vmax=1,
            center=0.5,
            cbar_kws={'label': 'Subspace Overlap'}
        )

        plt.title('Subspace Overlap Matrix')
        plt.xlabel('Compression Stage')
        plt.ylabel('Compression Stage')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Overlap matrix visualization saved to {save_path}")

        plt.show()


# ============= TEST CODE SECTION =============
import unittest
from unittest.mock import Mock, patch


class TestGrassmannDistance(unittest.TestCase):
    """Unit tests for Grassmann distance computation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create orthogonal subspaces
        self.subspace1 = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        self.subspace2 = torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0]
        ])

    def test_orthonormalize_basis(self):
        """Test basis orthonormalization."""
        basis = torch.randn(5, 3)
        ortho_basis = orthonormalize_basis(basis)

        # Check orthonormality
        gram = torch.mm(ortho_basis.T, ortho_basis)
        identity = torch.eye(3)
        self.assertTrue(torch.allclose(gram, identity, atol=1e-6))

    def test_projection_distance(self):
        """Test projection-based Grassmann distance."""
        distance = compute_grassmann_distance(
            self.subspace1, self.subspace2, method='projection'
        )
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)
        self.assertLessEqual(distance, 1)

    def test_geodesic_distance(self):
        """Test geodesic Grassmann distance."""
        distance = compute_grassmann_distance(
            self.subspace1, self.subspace2, method='geodesic'
        )
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)

    def test_chordal_distance(self):
        """Test chordal Grassmann distance."""
        distance = compute_grassmann_distance(
            self.subspace1, self.subspace2, method='chordal'
        )
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)
        self.assertLessEqual(distance, np.sqrt(2))

    def test_principal_angles(self):
        """Test principal angle computation."""
        angles = compute_principal_angles(self.subspace1, self.subspace2)
        self.assertEqual(len(angles), min(self.subspace1.shape[1], self.subspace2.shape[1]))
        self.assertTrue(torch.all(angles >= 0))
        self.assertTrue(torch.all(angles <= np.pi/2))

    def test_subspace_overlap(self):
        """Test subspace overlap computation."""
        overlap = subspace_overlap(self.subspace1, self.subspace1)
        self.assertAlmostEqual(overlap, 1.0, places=5)

        overlap = subspace_overlap(self.subspace1, self.subspace2)
        self.assertLess(overlap, 0.5)


class TestGrassmannAnalyzer(unittest.TestCase):
    """Unit tests for GrassmannAnalyzer."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = GrassmannAnalyzer()

        # Add test subspaces
        self.analyzer.add_subspace('stage1', torch.randn(10, 3))
        self.analyzer.add_subspace('stage2', torch.randn(10, 4))
        self.analyzer.add_subspace('stage3', torch.randn(10, 2))

    def test_add_subspace(self):
        """Test adding subspaces to analyzer."""
        self.assertEqual(len(self.analyzer.subspaces), 3)
        self.assertIn('stage1', self.analyzer.subspaces)

    def test_compute_all_distances(self):
        """Test computing all pairwise distances."""
        distances = self.analyzer.compute_all_distances()

        # Check expected number of distances
        expected_pairs = 3  # C(3,2) = 3
        actual_pairs = sum(1 for k in distances.keys() if isinstance(k, tuple))
        self.assertEqual(actual_pairs, expected_pairs)

        # Check distance properties
        for key, distance in distances.items():
            if isinstance(key, tuple):
                self.assertIsInstance(distance, float)
                self.assertGreaterEqual(distance, 0)

    def test_check_orthogonality(self):
        """Test orthogonality checking."""
        self.analyzer.compute_all_distances()
        orthogonality = self.analyzer.check_orthogonality(threshold=0.5)

        for key, is_orthogonal in orthogonality.items():
            self.assertIsInstance(is_orthogonal, bool)

    def test_get_overlap_matrix(self):
        """Test overlap matrix computation."""
        matrix = self.analyzer.get_overlap_matrix()

        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(np.all(matrix <= 1))
        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(3))

    def test_recommend_adjustments(self):
        """Test recommendation generation."""
        recommendations = self.analyzer.recommend_adjustments(min_distance=0.8)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualizations(self, mock_savefig, mock_show):
        """Test visualization methods."""
        # Test Grassmann distance visualization
        self.analyzer.visualize_grassmann_distances(save_path='test.png')
        mock_savefig.assert_called()

        # Test overlap matrix visualization
        self.analyzer.visualize_overlap_matrix(save_path='test2.png')
        self.assertEqual(mock_savefig.call_count, 2)

        # Test principal angles visualization
        self.analyzer.visualize_principal_angles('stage1', 'stage2', save_path='test3.png')
        self.assertEqual(mock_savefig.call_count, 3)


def run_grassmann_tests():
    """Run all Grassmann analysis tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)