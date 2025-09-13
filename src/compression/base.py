"""
Base compression interfaces with gradient preservation for RCC system.

This module provides abstract base classes and common utilities for all compression
techniques in the Recursive Cascade Compression pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionMode(Enum):
    """Compression modes for different stages of the pipeline."""
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    HYBRID = "hybrid"


@dataclass
class CompressionConfig:
    """Configuration for compression operations."""
    target_sparsity: float
    mode: CompressionMode
    preserve_gradients: bool = True
    energy_threshold: float = 0.99
    max_rank_ratio: float = 0.5
    alpha_init: float = 1.0
    alpha_lr: float = 1e-3
    polynomial_decay_power: float = 3.0
    warmup_steps: int = 1000
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 <= self.target_sparsity <= 1, "Sparsity must be in [0, 1]"
        assert 0 < self.energy_threshold <= 1, "Energy threshold must be in (0, 1]"
        assert 0 < self.max_rank_ratio <= 1, "Max rank ratio must be in (0, 1]"
        assert self.alpha_init > 0, "Alpha init must be positive"
        assert self.polynomial_decay_power > 0, "Polynomial decay power must be positive"


@dataclass
class CompressionMetrics:
    """Metrics for compression operations."""
    original_params: int
    compressed_params: int
    compression_ratio: float
    energy_preserved: float
    gradient_norm: Optional[float] = None
    null_space_distance: Optional[float] = None
    inference_speedup: Optional[float] = None
    memory_reduction: Optional[float] = None

    @property
    def parameter_reduction(self) -> float:
        """Calculate parameter reduction percentage."""
        return 100 * (1 - self.compressed_params / self.original_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "original_params": self.original_params,
            "compressed_params": self.compressed_params,
            "compression_ratio": self.compression_ratio,
            "parameter_reduction": self.parameter_reduction,
            "energy_preserved": self.energy_preserved,
            "gradient_norm": self.gradient_norm,
            "null_space_distance": self.null_space_distance,
            "inference_speedup": self.inference_speedup,
            "memory_reduction": self.memory_reduction
        }


class CompressionModule(nn.Module, ABC):
    """Abstract base class for compression modules."""

    def __init__(self, config: CompressionConfig):
        """
        Initialize compression module.

        Args:
            config: Compression configuration
        """
        super().__init__()
        self.config = config
        self._metrics = None
        self._gradient_hooks = []

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    @abstractmethod
    def compress(self, module: nn.Module, **kwargs) -> nn.Module:
        """
        Apply compression to a module.

        Args:
            module: Module to compress
            **kwargs: Additional compression parameters

        Returns:
            Compressed module
        """
        pass

    @abstractmethod
    def compute_metrics(self, original: nn.Module, compressed: nn.Module) -> CompressionMetrics:
        """
        Compute compression metrics.

        Args:
            original: Original module
            compressed: Compressed module

        Returns:
            Compression metrics
        """
        pass

    def register_gradient_hooks(self, module: nn.Module) -> None:
        """
        Register gradient preservation hooks.

        Args:
            module: Module to register hooks on
        """
        if not self.config.preserve_gradients:
            return

        def gradient_preservation_hook(grad: Tensor) -> Tensor:
            """Preserve gradient flow through compressed layers."""
            # Scale gradients to maintain norm
            with torch.no_grad():
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    grad = grad * (1.0 / (1.0 - self.config.target_sparsity + 1e-8))
            return grad

        for param in module.parameters():
            if param.requires_grad:
                hook = param.register_hook(gradient_preservation_hook)
                self._gradient_hooks.append(hook)

    def remove_gradient_hooks(self) -> None:
        """Remove all registered gradient hooks."""
        for hook in self._gradient_hooks:
            hook.remove()
        self._gradient_hooks.clear()

    @staticmethod
    def count_parameters(module: nn.Module, count_nonzero_only: bool = False) -> int:
        """
        Count parameters in a module.

        Args:
            module: Module to count parameters
            count_nonzero_only: If True, count only non-zero parameters

        Returns:
            Number of parameters
        """
        if count_nonzero_only:
            return sum((p != 0).sum().item() for p in module.parameters())
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def compute_energy(matrix: Tensor, k: Optional[int] = None) -> float:
        """
        Compute energy preserved by top-k singular values.

        Args:
            matrix: Input matrix
            k: Number of singular values to keep (None for all)

        Returns:
            Energy preserved ratio
        """
        with torch.no_grad():
            U, S, V = torch.svd(matrix)
            total_energy = (S ** 2).sum().item()

            if k is None or k >= len(S):
                return 1.0

            preserved_energy = (S[:k] ** 2).sum().item()
            return preserved_energy / (total_energy + 1e-8)

    def get_pruning_mask(self, weight: Tensor, sparsity: float,
                        structured: bool = False) -> Tensor:
        """
        Generate pruning mask based on magnitude.

        Args:
            weight: Weight tensor
            sparsity: Target sparsity ratio
            structured: Whether to use structured pruning

        Returns:
            Binary pruning mask
        """
        with torch.no_grad():
            if structured:
                # Structured pruning (channel-wise)
                importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
                threshold_idx = int(sparsity * len(importance))
                threshold = torch.topk(importance, threshold_idx, largest=False)[0].max()
                mask = importance > threshold
                mask = mask.view(-1, *([1] * (weight.dim() - 1)))
                mask = mask.expand_as(weight)
            else:
                # Unstructured pruning (element-wise)
                importance = weight.abs()
                threshold = torch.quantile(importance.flatten(), sparsity)
                mask = importance > threshold

            return mask.float()


class CascadeCompressor(nn.Module):
    """Orchestrates the cascade of compression techniques."""

    def __init__(self,
                 dare_compressor: CompressionModule,
                 nullu_compressor: CompressionModule,
                 alpha_adapter: CompressionModule,
                 config: CompressionConfig):
        """
        Initialize cascade compressor.

        Args:
            dare_compressor: DARE pruning module
            nullu_compressor: Nullu projection module
            alpha_adapter: AlphaEdit weight adapter
            config: Compression configuration
        """
        super().__init__()
        self.dare = dare_compressor
        self.nullu = nullu_compressor
        self.alpha = alpha_adapter
        self.config = config
        self.compression_history = []

    def forward(self, model: nn.Module,
                stage: Optional[str] = None) -> Tuple[nn.Module, CompressionMetrics]:
        """
        Apply cascade compression to model.

        Args:
            model: Model to compress
            stage: Specific stage to apply (None for full cascade)

        Returns:
            Compressed model and metrics
        """
        original_params = self.dare.count_parameters(model)

        if stage == "dare" or stage is None:
            logger.info("Applying DARE pruning...")
            model = self.dare.compress(model)

        if stage == "nullu" or stage is None:
            logger.info("Applying Nullu projection...")
            model = self.nullu.compress(model)

        if stage == "alpha" or stage is None:
            logger.info("Applying AlphaEdit adaptation...")
            model = self.alpha.compress(model)

        compressed_params = self.dare.count_parameters(model, count_nonzero_only=True)

        metrics = CompressionMetrics(
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / compressed_params,
            energy_preserved=self._compute_total_energy(model)
        )

        self.compression_history.append(metrics)
        logger.info(f"Compression complete: {metrics.parameter_reduction:.2f}% reduction")

        return model, metrics

    def _compute_total_energy(self, model: nn.Module) -> float:
        """
        Compute total energy preserved across all layers.

        Args:
            model: Compressed model

        Returns:
            Average energy preserved
        """
        energies = []
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                energy = self.dare.compute_energy(param.data)
                energies.append(energy)

        return np.mean(energies) if energies else 1.0

    def verify_null_space(self, original: nn.Module, compressed: nn.Module) -> float:
        """
        Verify null space preservation between original and compressed models.

        Args:
            original: Original model
            compressed: Compressed model

        Returns:
            Null space distance metric
        """
        distances = []

        for (name1, param1), (name2, param2) in zip(
            original.named_parameters(),
            compressed.named_parameters()
        ):
            if param1.dim() >= 2 and name1 == name2:
                # Compute null space for original
                U1, S1, V1 = torch.svd(param1.data)
                null_dim1 = (S1 < 1e-6).sum().item()

                if null_dim1 > 0:
                    null_space1 = V1[:, -null_dim1:]

                    # Compute null space for compressed
                    U2, S2, V2 = torch.svd(param2.data)
                    null_dim2 = (S2 < 1e-6).sum().item()

                    if null_dim2 > 0:
                        null_space2 = V2[:, -null_dim2:]

                        # Compute subspace distance
                        overlap = torch.mm(null_space1.T, null_space2)
                        distance = 1.0 - torch.norm(overlap).item() / min(null_dim1, null_dim2)
                        distances.append(distance)

        return np.mean(distances) if distances else 0.0