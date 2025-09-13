"""
DARE (Drop And REscale) pruning implementation with polynomial scheduling.

This module implements magnitude-based pruning with rescaling to preserve
expected output values during compression.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass

from ..base import CompressionModule, CompressionConfig, CompressionMetrics, CompressionMode

logger = logging.getLogger(__name__)


@dataclass
class DAREConfig(CompressionConfig):
    """Configuration specific to DARE pruning."""
    rescale_weights: bool = True
    importance_metric: str = "magnitude"  # magnitude, gradient, fisher
    block_size: Optional[int] = None  # For block-sparse patterns
    n_m_sparsity: Optional[Tuple[int, int]] = None  # N:M structured sparsity
    iterative_pruning: bool = True
    pruning_iterations: int = 10
    final_finetune_steps: int = 1000


class DARESparseLayer(nn.Module):
    """Sparse layer wrapper for DARE-pruned weights."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None,
                 mask: Optional[Tensor] = None, rescale_factor: float = 1.0):
        """
        Initialize sparse layer.

        Args:
            weight: Weight tensor
            bias: Bias tensor (optional)
            mask: Pruning mask
            rescale_factor: Weight rescaling factor
        """
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        if mask is not None:
            self.register_buffer('mask', mask)
        else:
            self.mask = None
        self.rescale_factor = rescale_factor

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sparse weights."""
        weight = self.weight
        if self.mask is not None:
            weight = weight * self.mask
        if self.rescale_factor != 1.0:
            weight = weight * self.rescale_factor

        if weight.dim() == 4:  # Conv2d
            return nn.functional.conv2d(x, weight, self.bias)
        elif weight.dim() == 2:  # Linear
            return nn.functional.linear(x, weight, self.bias)
        else:
            raise ValueError(f"Unsupported weight dimension: {weight.dim()}")


class DARECompressor(CompressionModule):
    """DARE pruning with polynomial scheduling and rescaling."""

    def __init__(self, config: DAREConfig):
        """
        Initialize DARE pruner.

        Args:
            config: DARE configuration
        """
        super().__init__(config)
        self.config = config
        self.importance_scores = {}
        self.pruning_history = []
        self.current_step = 0

    def compress(self, module: nn.Module, **kwargs) -> nn.Module:
        """
        Apply DARE pruning to module.

        Args:
            module: Module to prune
            **kwargs: Additional parameters

        Returns:
            Pruned module
        """
        if self.config.iterative_pruning:
            return self._iterative_pruning(module, **kwargs)
        else:
            return self._one_shot_pruning(module, **kwargs)

    def _iterative_pruning(self, module: nn.Module, **kwargs) -> nn.Module:
        """
        Apply iterative magnitude pruning with polynomial scheduling.

        Args:
            module: Module to prune
            **kwargs: Additional parameters

        Returns:
            Iteratively pruned module
        """
        original_params = self.count_parameters(module)
        target_sparsity = self.config.target_sparsity

        for iteration in range(self.config.pruning_iterations):
            current_sparsity = self._polynomial_schedule(
                iteration,
                self.config.pruning_iterations,
                target_sparsity
            )

            logger.info(f"Pruning iteration {iteration + 1}/{self.config.pruning_iterations}, "
                       f"target sparsity: {current_sparsity:.2%}")

            # Compute importance scores
            self._compute_importance_scores(module)

            # Apply pruning
            for name, param in module.named_parameters():
                if self._should_prune(name, param):
                    mask = self._compute_pruning_mask(param, current_sparsity, name)
                    self._apply_mask(param, mask)

            # Log metrics
            current_params = self.count_parameters(module, count_nonzero_only=True)
            actual_sparsity = 1 - (current_params / original_params)
            self.pruning_history.append({
                'iteration': iteration,
                'target_sparsity': current_sparsity,
                'actual_sparsity': actual_sparsity,
                'parameters': current_params
            })

        # Apply rescaling if enabled
        if self.config.rescale_weights:
            self._rescale_weights(module)

        return module

    def _one_shot_pruning(self, module: nn.Module, **kwargs) -> nn.Module:
        """
        Apply one-shot magnitude pruning.

        Args:
            module: Module to prune
            **kwargs: Additional parameters

        Returns:
            Pruned module
        """
        # Compute importance scores
        self._compute_importance_scores(module)

        # Apply pruning
        for name, param in module.named_parameters():
            if self._should_prune(name, param):
                mask = self._compute_pruning_mask(
                    param,
                    self.config.target_sparsity,
                    name
                )
                self._apply_mask(param, mask)

        # Apply rescaling if enabled
        if self.config.rescale_weights:
            self._rescale_weights(module)

        return module

    def _compute_importance_scores(self, module: nn.Module) -> None:
        """
        Compute importance scores for all parameters.

        Args:
            module: Module to compute scores for
        """
        for name, param in module.named_parameters():
            if not self._should_prune(name, param):
                continue

            if self.config.importance_metric == "magnitude":
                score = param.data.abs()
            elif self.config.importance_metric == "gradient":
                if param.grad is not None:
                    score = (param.data.abs() * param.grad.abs())
                else:
                    score = param.data.abs()
            elif self.config.importance_metric == "fisher":
                if param.grad is not None:
                    score = (param.grad ** 2)
                else:
                    score = param.data.abs()
            else:
                raise ValueError(f"Unknown importance metric: {self.config.importance_metric}")

            self.importance_scores[name] = score

    def _compute_pruning_mask(self, param: Tensor, sparsity: float,
                             name: str) -> Tensor:
        """
        Compute pruning mask for a parameter.

        Args:
            param: Parameter tensor
            sparsity: Target sparsity
            name: Parameter name

        Returns:
            Binary pruning mask
        """
        importance = self.importance_scores.get(name, param.data.abs())

        if self.config.mode == CompressionMode.STRUCTURED:
            return self._structured_mask(importance, sparsity)
        elif self.config.mode == CompressionMode.UNSTRUCTURED:
            return self._unstructured_mask(importance, sparsity)
        elif self.config.mode == CompressionMode.HYBRID:
            return self._hybrid_mask(importance, sparsity)
        else:
            raise ValueError(f"Unknown compression mode: {self.config.mode}")

    def _unstructured_mask(self, importance: Tensor, sparsity: float) -> Tensor:
        """
        Compute unstructured pruning mask.

        Args:
            importance: Importance scores
            sparsity: Target sparsity

        Returns:
            Binary mask
        """
        if self.config.n_m_sparsity is not None:
            return self._n_m_structured_mask(importance, *self.config.n_m_sparsity)

        threshold = torch.quantile(importance.flatten(), sparsity)
        mask = (importance > threshold).float()
        return mask

    def _structured_mask(self, importance: Tensor, sparsity: float) -> Tensor:
        """
        Compute structured pruning mask (channel-wise).

        Args:
            importance: Importance scores
            sparsity: Target sparsity

        Returns:
            Binary mask
        """
        if importance.dim() == 4:  # Conv2d weights
            channel_importance = importance.sum(dim=(1, 2, 3))
        elif importance.dim() == 2:  # Linear weights
            channel_importance = importance.sum(dim=1)
        else:
            return self._unstructured_mask(importance, sparsity)

        k = int((1 - sparsity) * len(channel_importance))
        _, top_indices = torch.topk(channel_importance, k)

        mask = torch.zeros_like(importance)
        if importance.dim() == 4:
            mask[top_indices] = 1.0
        elif importance.dim() == 2:
            mask[top_indices, :] = 1.0

        return mask

    def _hybrid_mask(self, importance: Tensor, sparsity: float) -> Tensor:
        """
        Compute hybrid structured/unstructured mask.

        Args:
            importance: Importance scores
            sparsity: Target sparsity

        Returns:
            Binary mask
        """
        # First apply structured pruning (50% of target sparsity)
        structured_sparsity = sparsity * 0.5
        mask = self._structured_mask(importance, structured_sparsity)

        # Then apply unstructured pruning on remaining weights
        remaining_importance = importance * mask
        remaining_sparsity = (sparsity - structured_sparsity) / (1 - structured_sparsity)
        unstructured_mask = self._unstructured_mask(remaining_importance, remaining_sparsity)

        return mask * unstructured_mask

    def _n_m_structured_mask(self, importance: Tensor, n: int, m: int) -> Tensor:
        """
        Compute N:M structured sparsity mask.

        Args:
            importance: Importance scores
            n: Keep n weights
            m: Out of every m weights

        Returns:
            Binary mask
        """
        original_shape = importance.shape
        importance_flat = importance.flatten()

        # Pad if necessary
        pad_len = (m - len(importance_flat) % m) % m
        if pad_len > 0:
            importance_flat = torch.cat([
                importance_flat,
                torch.zeros(pad_len, device=importance_flat.device)
            ])

        # Reshape into blocks of m
        importance_blocked = importance_flat.view(-1, m)

        # Keep top n in each block
        _, indices = torch.topk(importance_blocked, n, dim=1)
        mask = torch.zeros_like(importance_blocked)
        mask.scatter_(1, indices, 1.0)

        # Reshape back
        mask = mask.flatten()[:importance.numel()]
        return mask.view(original_shape)

    def _apply_mask(self, param: Tensor, mask: Tensor) -> None:
        """
        Apply pruning mask to parameter.

        Args:
            param: Parameter to prune
            mask: Pruning mask
        """
        with torch.no_grad():
            param.data.mul_(mask)

    def _rescale_weights(self, module: nn.Module) -> None:
        """
        Rescale weights to preserve expected output.

        Args:
            module: Module with pruned weights
        """
        for name, param in module.named_parameters():
            if not self._should_prune(name, param):
                continue

            with torch.no_grad():
                # Compute density (fraction of non-zero weights)
                density = (param != 0).float().mean().item()

                if density > 0 and density < 1:
                    # Rescale by 1/density to preserve expected value
                    rescale_factor = 1.0 / density
                    param.data.mul_(rescale_factor)

                    logger.debug(f"Rescaled {name} by factor {rescale_factor:.3f}")

    def _should_prune(self, name: str, param: Tensor) -> bool:
        """
        Check if parameter should be pruned.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            Whether to prune this parameter
        """
        # Skip batch norm and bias parameters
        if 'bn' in name or 'norm' in name or 'bias' in name:
            return False

        # Skip 1D parameters
        if param.dim() < 2:
            return False

        # Skip embedding layers
        if 'embedding' in name:
            return False

        return True

    def _polynomial_schedule(self, step: int, total_steps: int,
                           target: float) -> float:
        """
        Polynomial decay schedule for sparsity.

        Args:
            step: Current step
            total_steps: Total steps
            target: Target value

        Returns:
            Scheduled value
        """
        progress = min(step / total_steps, 1.0)
        return target * (progress ** self.config.polynomial_decay_power)

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
        compressed_params = self.count_parameters(compressed, count_nonzero_only=True)

        # Compute average energy preserved
        energy_preserved = []
        for (name, param_orig), (_, param_comp) in zip(
            original.named_parameters(),
            compressed.named_parameters()
        ):
            if self._should_prune(name, param_orig):
                with torch.no_grad():
                    orig_norm = torch.norm(param_orig).item()
                    comp_norm = torch.norm(param_comp).item()
                    if orig_norm > 0:
                        energy_preserved.append(comp_norm / orig_norm)

        avg_energy = np.mean(energy_preserved) if energy_preserved else 1.0

        return CompressionMetrics(
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / compressed_params,
            energy_preserved=avg_energy
        )

    def get_sparsity_pattern(self, module: nn.Module) -> Dict[str, np.ndarray]:
        """
        Get sparsity patterns for visualization.

        Args:
            module: Pruned module

        Returns:
            Dictionary of sparsity patterns
        """
        patterns = {}
        for name, param in module.named_parameters():
            if self._should_prune(name, param):
                with torch.no_grad():
                    pattern = (param != 0).cpu().numpy()
                    patterns[name] = pattern
        return patterns