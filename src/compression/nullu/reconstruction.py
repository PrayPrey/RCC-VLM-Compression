"""
Weight reconstruction for Nullu projection.

This module handles weight reconstruction from low-rank decomposition,
including null space projection and error correction.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionConfig:
    """Configuration for weight reconstruction."""
    reconstruction_method: str = "direct"  # direct, iterative, corrected
    error_correction: bool = True
    correction_iterations: int = 3
    correction_lr: float = 0.01
    preserve_null_space: bool = True
    null_space_weight: float = 0.1
    use_residual_connection: bool = False
    quantization_aware: bool = False
    quantization_bits: int = 8


class WeightReconstructor:
    """Handles weight reconstruction from low-rank decomposition."""

    def __init__(self, config: ReconstructionConfig):
        """
        Initialize weight reconstructor.

        Args:
            config: Reconstruction configuration
        """
        self.config = config
        self.reconstruction_errors = {}
        self.null_spaces = {}
        self.correction_weights = {}

    def reconstruct(self,
                   U: Tensor,
                   S: Tensor,
                   V: Tensor,
                   layer_name: str,
                   original_shape: Optional[Tuple] = None,
                   null_space: Optional[Tensor] = None) -> Tensor:
        """
        Reconstruct weight matrix from SVD components.

        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors
            layer_name: Name of the layer
            original_shape: Original weight shape (for conv layers)
            null_space: Optional null space basis

        Returns:
            Reconstructed weight tensor
        """
        if self.config.reconstruction_method == "direct":
            weight = self._direct_reconstruction(U, S, V)
        elif self.config.reconstruction_method == "iterative":
            weight = self._iterative_reconstruction(U, S, V, layer_name)
        elif self.config.reconstruction_method == "corrected":
            weight = self._corrected_reconstruction(U, S, V, layer_name)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.config.reconstruction_method}")

        # Apply null space projection if available
        if self.config.preserve_null_space and null_space is not None:
            weight = self._apply_null_space_projection(weight, null_space)
            self.null_spaces[layer_name] = null_space

        # Reshape for conv layers
        if original_shape is not None and len(original_shape) == 4:
            weight = weight.view(original_shape)

        # Apply quantization if enabled
        if self.config.quantization_aware:
            weight = self._quantize_weight(weight)

        return weight

    def _direct_reconstruction(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        """
        Direct reconstruction: W = U @ diag(S) @ V^T.

        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors

        Returns:
            Reconstructed weight
        """
        # Efficient reconstruction
        weight = U @ torch.diag(S) @ V.T
        return weight

    def _iterative_reconstruction(self,
                                 U: Tensor,
                                 S: Tensor,
                                 V: Tensor,
                                 layer_name: str) -> Tensor:
        """
        Iterative reconstruction with refinement.

        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors
            layer_name: Layer name

        Returns:
            Iteratively refined weight
        """
        # Start with direct reconstruction
        weight = self._direct_reconstruction(U, S, V)

        # Iterative refinement
        for iteration in range(self.config.correction_iterations):
            # Compute reconstruction error proxy
            error_proxy = self._compute_error_proxy(weight, U, S, V)

            # Apply correction
            correction = error_proxy * self.config.correction_lr
            weight = weight - correction

            # Re-project to low-rank space
            weight = self._project_to_low_rank(weight, len(S))

        return weight

    def _corrected_reconstruction(self,
                                 U: Tensor,
                                 S: Tensor,
                                 V: Tensor,
                                 layer_name: str) -> Tensor:
        """
        Reconstruction with learned error correction.

        Args:
            U: Left singular vectors
            S: Singular values
            V: Right singular vectors
            layer_name: Layer name

        Returns:
            Corrected weight reconstruction
        """
        # Base reconstruction
        weight = self._direct_reconstruction(U, S, V)

        # Apply learned correction if available
        if layer_name in self.correction_weights:
            correction = self.correction_weights[layer_name]
            weight = weight + correction

        # Store for analysis
        self.reconstruction_errors[layer_name] = 0.0  # Placeholder

        return weight

    def _apply_null_space_projection(self,
                                    weight: Tensor,
                                    null_space: Tensor) -> Tensor:
        """
        Apply null space projection to weight.

        Args:
            weight: Reconstructed weight
            null_space: Null space basis vectors

        Returns:
            Weight with null space projection
        """
        # Project weight onto null space complement
        # P_null = I - V_null @ V_null^T
        null_projection = null_space @ null_space.T
        identity = torch.eye(null_projection.size(0), device=weight.device, dtype=weight.dtype)
        complement_projection = identity - null_projection

        # Apply projection with weighting
        projected_weight = weight @ complement_projection
        null_component = weight @ null_projection

        # Combine with weighting
        final_weight = projected_weight + self.config.null_space_weight * null_component

        return final_weight

    def _compute_error_proxy(self,
                           weight: Tensor,
                           U: Tensor,
                           S: Tensor,
                           V: Tensor) -> Tensor:
        """
        Compute proxy for reconstruction error.

        Args:
            weight: Current weight
            U, S, V: SVD components

        Returns:
            Error proxy tensor
        """
        # Compute difference from perfect reconstruction
        perfect = U @ torch.diag(S) @ V.T
        error = weight - perfect

        # Apply smoothing
        kernel_size = 3
        if weight.dim() == 2 and weight.size(0) > kernel_size and weight.size(1) > kernel_size:
            # Simple averaging for smoothing
            error_smooth = torch.nn.functional.avg_pool2d(
                error.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ).squeeze()
        else:
            error_smooth = error

        return error_smooth

    def _project_to_low_rank(self, weight: Tensor, rank: int) -> Tensor:
        """
        Project weight to low-rank space.

        Args:
            weight: Weight matrix
            rank: Target rank

        Returns:
            Low-rank projected weight
        """
        # Perform SVD
        U, S, V = torch.svd(weight)

        # Truncate to rank
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:, :rank]

        # Reconstruct
        weight_lr = U_r @ torch.diag(S_r) @ V_r.T

        return weight_lr

    def _quantize_weight(self, weight: Tensor) -> Tensor:
        """
        Apply quantization to weight.

        Args:
            weight: Weight tensor

        Returns:
            Quantized weight
        """
        # Simple uniform quantization
        bits = self.config.quantization_bits
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Scale and quantize
        scale = (weight.max() - weight.min()) / (qmax - qmin)
        zero_point = qmin - weight.min() / scale

        weight_q = torch.round(weight / scale + zero_point)
        weight_q = torch.clamp(weight_q, qmin, qmax)

        # Dequantize
        weight_dq = (weight_q - zero_point) * scale

        return weight_dq

    def compute_reconstruction_error(self,
                                    original: Tensor,
                                    reconstructed: Tensor,
                                    layer_name: str) -> Dict[str, float]:
        """
        Compute reconstruction error metrics.

        Args:
            original: Original weight
            reconstructed: Reconstructed weight
            layer_name: Layer name

        Returns:
            Error metrics dictionary
        """
        # Flatten for consistent computation
        orig_flat = original.view(-1)
        recon_flat = reconstructed.view(-1)

        # Compute various error metrics
        mse = torch.mean((orig_flat - recon_flat) ** 2).item()
        mae = torch.mean(torch.abs(orig_flat - recon_flat)).item()

        # Relative error
        orig_norm = torch.norm(orig_flat)
        if orig_norm > 0:
            relative_error = torch.norm(orig_flat - recon_flat) / orig_norm
            relative_error = relative_error.item()
        else:
            relative_error = 0.0

        # Cosine similarity
        if orig_flat.numel() > 0:
            cosine_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0),
                recon_flat.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 1.0

        metrics = {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'cosine_similarity': cosine_sim
        }

        # Store for tracking
        self.reconstruction_errors[layer_name] = metrics

        return metrics

    def learn_correction_weights(self,
                                original_weights: Dict[str, Tensor],
                                reconstructed_weights: Dict[str, Tensor],
                                learning_rate: float = 0.01,
                                iterations: int = 100):
        """
        Learn correction weights to minimize reconstruction error.

        Args:
            original_weights: Original weight tensors
            reconstructed_weights: Reconstructed weight tensors
            learning_rate: Learning rate for correction
            iterations: Number of optimization iterations
        """
        for layer_name in original_weights:
            if layer_name not in reconstructed_weights:
                continue

            original = original_weights[layer_name]
            reconstructed = reconstructed_weights[layer_name]

            # Initialize correction
            correction = torch.zeros_like(original)
            correction.requires_grad = True

            # Simple gradient descent
            optimizer = torch.optim.SGD([correction], lr=learning_rate)

            for _ in range(iterations):
                optimizer.zero_grad()

                # Compute corrected weight
                corrected = reconstructed + correction

                # Compute loss
                loss = torch.mean((corrected - original) ** 2)

                # Backward and update
                loss.backward()
                optimizer.step()

            # Store learned correction
            self.correction_weights[layer_name] = correction.detach()

    def apply_residual_connection(self,
                                 reconstructed: Tensor,
                                 original: Tensor,
                                 alpha: float = 0.1) -> Tensor:
        """
        Apply residual connection between original and reconstructed weights.

        Args:
            reconstructed: Reconstructed weight
            original: Original weight
            alpha: Residual weight factor

        Returns:
            Weight with residual connection
        """
        if self.config.use_residual_connection:
            return reconstructed + alpha * original
        return reconstructed

    def get_reconstruction_stats(self) -> Dict:
        """
        Get reconstruction statistics.

        Returns:
            Dictionary of reconstruction statistics
        """
        if not self.reconstruction_errors:
            return {}

        all_mse = [e['mse'] for e in self.reconstruction_errors.values()]
        all_mae = [e['mae'] for e in self.reconstruction_errors.values()]
        all_relative = [e['relative_error'] for e in self.reconstruction_errors.values()]
        all_cosine = [e['cosine_similarity'] for e in self.reconstruction_errors.values()]

        return {
            'num_layers': len(self.reconstruction_errors),
            'average_mse': np.mean(all_mse),
            'average_mae': np.mean(all_mae),
            'average_relative_error': np.mean(all_relative),
            'average_cosine_similarity': np.mean(all_cosine),
            'max_error': max(all_relative),
            'min_error': min(all_relative),
            'null_spaces_preserved': len(self.null_spaces),
            'corrections_learned': len(self.correction_weights)
        }