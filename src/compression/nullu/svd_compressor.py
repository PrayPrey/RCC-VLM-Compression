"""
Nullu Projection: SVD-based rank reduction with energy preservation.

This module implements low-rank decomposition using SVD with adaptive rank selection
based on energy preservation thresholds.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass

from ..base import CompressionModule, CompressionConfig, CompressionMetrics

logger = logging.getLogger(__name__)


@dataclass
class NulluConfig(CompressionConfig):
    """Configuration specific to Nullu projection."""
    min_rank: int = 1
    rank_selection_method: str = "energy"  # energy, fixed, adaptive
    fixed_rank_ratio: float = 0.1
    layer_wise_adaptation: bool = True
    use_randomized_svd: bool = True
    power_iterations: int = 2
    oversampling: int = 10
    preserve_null_space: bool = True
    reconstruction_loss_weight: float = 0.1


class LowRankLayer(nn.Module):
    """Low-rank decomposed layer using SVD factorization."""

    def __init__(self, U: Tensor, S: Tensor, V: Tensor,
                 bias: Optional[Tensor] = None,
                 input_dim: int = None,
                 output_dim: int = None):
        """
        Initialize low-rank layer.

        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors
            bias: Bias term (optional)
            input_dim: Input dimension for reshaping
            output_dim: Output dimension for reshaping
        """
        super().__init__()
        self.rank = len(S)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Store factorized components
        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('V', V)

        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through low-rank layer."""
        # Reconstruct weight: W = U @ diag(S) @ V^T
        weight = self.U @ torch.diag(self.S) @ self.V.T

        if weight.dim() == 4:  # Conv2d
            return nn.functional.conv2d(x, weight, self.bias)
        elif weight.dim() == 2:  # Linear
            return nn.functional.linear(x, weight, self.bias)
        else:
            # For efficient computation, we can avoid reconstructing the full weight
            # x @ W^T = x @ V @ diag(S) @ U^T
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.size(-1))

            # Compute in factorized form for efficiency
            out = x_flat @ self.V  # (batch, rank)
            out = out * self.S.unsqueeze(0)  # (batch, rank)
            out = out @ self.U.T  # (batch, output_dim)

            out = out.view(*batch_shape, out.size(-1))

            if self.bias is not None:
                out = out + self.bias

            return out

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio of this layer."""
        original_params = self.output_dim * self.input_dim
        compressed_params = (self.U.numel() + self.S.numel() + self.V.numel())
        return original_params / compressed_params


class NulluProjector(CompressionModule):
    """Nullu projection using SVD-based low-rank approximation."""

    def __init__(self, config: NulluConfig):
        """
        Initialize Nullu compressor.

        Args:
            config: Nullu configuration
        """
        super().__init__(config)
        self.config = config
        self.rank_selections = {}
        self.energy_preserved = {}
        self.null_spaces = {}

    def compress(self, module: nn.Module, **kwargs) -> nn.Module:
        """
        Apply Nullu projection to module.

        Args:
            module: Module to compress
            **kwargs: Additional parameters

        Returns:
            Compressed module with low-rank layers
        """
        compressed_module = self._create_compressed_copy(module)

        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                compressed_child = self._compress_layer(child, name)
                setattr(compressed_module, name, compressed_child)
            elif len(list(child.children())) > 0:
                # Recursively compress nested modules
                compressed_child = self.compress(child, **kwargs)
                setattr(compressed_module, name, compressed_child)
            else:
                # Copy non-compressible layers
                setattr(compressed_module, name, child)

        return compressed_module

    def _compress_layer(self, layer: Union[nn.Linear, nn.Conv2d],
                       layer_name: str) -> nn.Module:
        """
        Compress a single layer using SVD.

        Args:
            layer: Layer to compress
            layer_name: Name of the layer

        Returns:
            Compressed layer
        """
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None

        # Reshape conv weights to 2D for SVD
        if isinstance(layer, nn.Conv2d):
            original_shape = weight.shape
            weight_2d = weight.view(weight.size(0), -1)
        else:
            weight_2d = weight

        # Compute SVD
        if self.config.use_randomized_svd and min(weight_2d.shape) > 100:
            U, S, V = self._randomized_svd(
                weight_2d,
                self.config.power_iterations,
                self.config.oversampling
            )
        else:
            U, S, V = torch.svd(weight_2d)

        # Select rank based on energy preservation
        rank = self._select_rank(S, layer_name)
        self.rank_selections[layer_name] = rank

        # Truncate to selected rank
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:, :rank]

        # Store null space if needed
        if self.config.preserve_null_space:
            null_rank = min(weight_2d.shape) - rank
            if null_rank > 0:
                self.null_spaces[layer_name] = V[:, rank:]

        # Compute energy preserved
        total_energy = (S ** 2).sum().item()
        preserved_energy = (S_r ** 2).sum().item()
        self.energy_preserved[layer_name] = preserved_energy / total_energy

        logger.info(f"Layer {layer_name}: rank {rank}/{min(weight_2d.shape)}, "
                   f"energy preserved: {self.energy_preserved[layer_name]:.4f}")

        # Create low-rank layer
        if isinstance(layer, nn.Conv2d):
            # For conv layers, we need to reshape back
            U_r = U_r.view(original_shape[0], rank)
            V_r = V_r.view(original_shape[1] * original_shape[2] * original_shape[3], rank)

        return LowRankLayer(
            U_r, S_r, V_r, bias,
            input_dim=weight_2d.size(1),
            output_dim=weight_2d.size(0)
        )

    def _select_rank(self, singular_values: Tensor, layer_name: str) -> int:
        """
        Select rank based on energy preservation criterion.

        Args:
            singular_values: Singular values from SVD
            layer_name: Name of the layer

        Returns:
            Selected rank
        """
        if self.config.rank_selection_method == "fixed":
            # Fixed rank ratio
            max_rank = len(singular_values)
            rank = max(
                self.config.min_rank,
                int(max_rank * self.config.fixed_rank_ratio)
            )
            return min(rank, max_rank)

        elif self.config.rank_selection_method == "energy":
            # Energy-based selection
            cumsum_energy = torch.cumsum(singular_values ** 2, dim=0)
            total_energy = cumsum_energy[-1]
            normalized_cumsum = cumsum_energy / total_energy

            # Find minimum rank that preserves target energy
            rank = torch.searchsorted(
                normalized_cumsum,
                self.config.energy_threshold
            ).item() + 1

            # Apply constraints
            rank = max(self.config.min_rank, rank)
            rank = min(rank, int(len(singular_values) * self.config.max_rank_ratio))

            return rank

        elif self.config.rank_selection_method == "adaptive":
            # Adaptive selection based on layer properties
            return self._adaptive_rank_selection(singular_values, layer_name)

        else:
            raise ValueError(f"Unknown rank selection method: {self.config.rank_selection_method}")

    def _adaptive_rank_selection(self, singular_values: Tensor,
                                layer_name: str) -> int:
        """
        Adaptively select rank based on singular value distribution.

        Args:
            singular_values: Singular values
            layer_name: Layer name

        Returns:
            Adaptively selected rank
        """
        # Compute effective rank using entropy
        normalized_s = singular_values / singular_values.sum()
        entropy = -(normalized_s * torch.log(normalized_s + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        # Compute elbow point in singular value curve
        second_derivative = torch.diff(torch.diff(torch.log(singular_values + 1e-10)))
        elbow_point = torch.argmax(torch.abs(second_derivative)).item() + 2

        # Combine criteria
        if self.config.layer_wise_adaptation:
            # Adjust based on layer position (assuming depth from name)
            depth_factor = 1.0
            if 'early' in layer_name or 'conv1' in layer_name:
                depth_factor = 1.2  # Preserve more ranks in early layers
            elif 'late' in layer_name or 'fc' in layer_name:
                depth_factor = 0.8  # Compress more in late layers

            rank = int(min(effective_rank, elbow_point) * depth_factor)
        else:
            rank = int(min(effective_rank, elbow_point))

        # Apply constraints
        rank = max(self.config.min_rank, rank)
        rank = min(rank, int(len(singular_values) * self.config.max_rank_ratio))

        return rank

    def _randomized_svd(self, matrix: Tensor, n_iter: int = 2,
                       oversample: int = 10) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute randomized SVD for large matrices.

        Args:
            matrix: Input matrix
            n_iter: Number of power iterations
            oversample: Oversampling parameter

        Returns:
            U, S, V from SVD decomposition
        """
        m, n = matrix.shape
        rank = min(m, n)
        target_rank = int(rank * self.config.max_rank_ratio) + oversample

        # Generate random projection matrix
        omega = torch.randn(n, target_rank, device=matrix.device, dtype=matrix.dtype)

        # Power iteration for better approximation
        Y = matrix @ omega
        for _ in range(n_iter):
            Y = matrix @ (matrix.T @ Y)

        # QR decomposition
        Q, _ = torch.qr(Y)

        # Project matrix to lower dimension
        B = Q.T @ matrix

        # SVD of smaller matrix
        U_tilde, S, V = torch.svd(B)

        # Recover full U
        U = Q @ U_tilde

        return U, S, V

    def _create_compressed_copy(self, module: nn.Module) -> nn.Module:
        """
        Create a copy of module structure without parameters.

        Args:
            module: Module to copy structure from

        Returns:
            Empty module with same structure
        """
        # Create new instance with same configuration
        module_class = type(module)
        try:
            # Try to create with no arguments
            compressed = module_class()
        except:
            # If that fails, create a simple container
            compressed = nn.Module()

        return compressed

    def compute_metrics(self, original: nn.Module,
                       compressed: nn.Module) -> CompressionMetrics:
        """
        Compute compression metrics.

        Args:
            original: Original module
            compressed: Compressed module

        Returns:
            Compression metrics
        """
        original_params = self.count_parameters(original)
        compressed_params = self.count_parameters(compressed)

        # Average energy preserved
        avg_energy = np.mean(list(self.energy_preserved.values())) \
            if self.energy_preserved else 1.0

        # Compute null space distance if preserved
        null_space_distance = None
        if self.config.preserve_null_space and self.null_spaces:
            distances = []
            for name, null_space in self.null_spaces.items():
                # Simplified null space distance computation
                if null_space.size(1) > 0:
                    distance = 1.0 - (torch.norm(null_space) / null_space.numel()).item()
                    distances.append(distance)
            null_space_distance = np.mean(distances) if distances else 0.0

        return CompressionMetrics(
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / compressed_params,
            energy_preserved=avg_energy,
            null_space_distance=null_space_distance
        )

    def reconstruct_weight(self, layer_name: str) -> Optional[Tensor]:
        """
        Reconstruct full weight from low-rank decomposition.

        Args:
            layer_name: Name of layer to reconstruct

        Returns:
            Reconstructed weight tensor
        """
        if layer_name not in self.rank_selections:
            return None

        # This would need access to the compressed layer
        # Placeholder for reconstruction logic
        return None


# Alias for backward compatibility
NulluCompressor = NulluProjector