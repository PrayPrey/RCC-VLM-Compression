"""
Sparsity pattern analysis for DARE pruning.

This module analyzes and manages sparsity patterns created during pruning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SparsityPatternAnalyzer:
    """Analyzes sparsity patterns in pruned networks."""

    def __init__(self):
        """Initialize the sparsity pattern analyzer."""
        self.patterns = {}
        self.statistics = {}

    def analyze_model_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Analyze sparsity patterns across the entire model.

        Args:
            model: The model to analyze

        Returns:
            Dictionary of sparsity statistics
        """
        total_params = 0
        sparse_params = 0
        layer_sparsity = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total if total > 0 else 0

                total_params += total
                sparse_params += zeros
                layer_sparsity[name] = sparsity

                # Analyze structure
                if len(param.shape) >= 2:
                    self._analyze_structure(name, param)

        overall_sparsity = sparse_params / total_params if total_params > 0 else 0

        self.statistics = {
            'overall_sparsity': overall_sparsity,
            'layer_sparsity': layer_sparsity,
            'total_parameters': total_params,
            'sparse_parameters': sparse_params,
            'dense_parameters': total_params - sparse_params
        }

        return self.statistics

    def _analyze_structure(self, layer_name: str, weight: torch.Tensor):
        """
        Analyze structural sparsity patterns.

        Args:
            layer_name: Name of the layer
            weight: Weight tensor
        """
        pattern = {}

        # Row-wise sparsity
        if len(weight.shape) >= 2:
            row_sparsity = (weight.sum(dim=-1) == 0).float().mean().item()
            pattern['row_sparsity'] = row_sparsity

            # Column-wise sparsity
            col_sparsity = (weight.sum(dim=0) == 0).float().mean().item()
            pattern['column_sparsity'] = col_sparsity

            # Block sparsity (4x4 blocks)
            if weight.shape[0] >= 4 and weight.shape[1] >= 4:
                block_sparsity = self._compute_block_sparsity(weight, block_size=4)
                pattern['block_sparsity_4x4'] = block_sparsity

            # Channel-wise sparsity for conv layers
            if len(weight.shape) == 4:  # Conv2d weights
                channel_sparsity = (weight.sum(dim=(1, 2, 3)) == 0).float().mean().item()
                pattern['channel_sparsity'] = channel_sparsity

        self.patterns[layer_name] = pattern

    def _compute_block_sparsity(
        self,
        weight: torch.Tensor,
        block_size: int = 4
    ) -> float:
        """
        Compute block-wise sparsity.

        Args:
            weight: Weight tensor
            block_size: Size of blocks to analyze

        Returns:
            Block sparsity ratio
        """
        h, w = weight.shape[:2]
        h_blocks = h // block_size
        w_blocks = w // block_size

        if h_blocks == 0 or w_blocks == 0:
            return 0.0

        total_blocks = h_blocks * w_blocks
        sparse_blocks = 0

        for i in range(h_blocks):
            for j in range(w_blocks):
                block = weight[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                if torch.all(block == 0):
                    sparse_blocks += 1

        return sparse_blocks / total_blocks if total_blocks > 0 else 0.0

    def compute_connectivity_graph(
        self,
        model: nn.Module
    ) -> Dict[str, List[str]]:
        """
        Compute connectivity graph based on non-zero connections.

        Args:
            model: The model to analyze

        Returns:
            Connectivity graph
        """
        connectivity = {}
        layers = dict(model.named_modules())

        for name, module in layers.items():
            if hasattr(module, 'weight'):
                weight = module.weight
                if weight is not None:
                    # Find connected layers based on non-zero weights
                    connected = []

                    # Simple heuristic: layers are connected if they're sequential
                    # and have compatible dimensions
                    parts = name.split('.')
                    if len(parts) > 1:
                        parent = '.'.join(parts[:-1])
                        if parent in layers:
                            connected.append(parent)

                    connectivity[name] = connected

        return connectivity

    def get_prunable_groups(
        self,
        model: nn.Module,
        min_group_size: int = 2
    ) -> List[List[str]]:
        """
        Identify groups of layers that can be pruned together.

        Args:
            model: The model to analyze
            min_group_size: Minimum size of prunable groups

        Returns:
            List of prunable layer groups
        """
        groups = []
        processed = set()

        for name, module in model.named_modules():
            if name in processed:
                continue

            if isinstance(module, (nn.Conv2d, nn.Linear)):
                group = [name]

                # Find related batch norm and activation layers
                parts = name.split('.')
                if len(parts) > 1:
                    base = '.'.join(parts[:-1])

                    # Look for associated batch norm
                    bn_name = f"{base}.bn"
                    if bn_name in dict(model.named_modules()):
                        group.append(bn_name)

                    # Look for associated activation
                    act_name = f"{base}.act"
                    if act_name in dict(model.named_modules()):
                        group.append(act_name)

                if len(group) >= min_group_size:
                    groups.append(group)
                    processed.update(group)

        return groups

    def visualize_sparsity_heatmap(
        self,
        weight: torch.Tensor,
        max_size: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Create a heatmap visualization of sparsity patterns.

        Args:
            weight: Weight tensor to visualize
            max_size: Maximum size for visualization

        Returns:
            Heatmap array
        """
        # Convert to 2D if needed
        if len(weight.shape) > 2:
            weight = weight.reshape(weight.shape[0], -1)

        # Downsample if too large
        h, w = weight.shape
        if h > max_size[0] or w > max_size[1]:
            stride_h = max(1, h // max_size[0])
            stride_w = max(1, w // max_size[1])
            weight = weight[::stride_h, ::stride_w]

        # Create heatmap (1 for non-zero, 0 for zero)
        heatmap = (weight != 0).float().cpu().numpy()

        return heatmap

    def compute_compression_ratio(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module
    ) -> Dict[str, float]:
        """
        Compute compression ratio between original and pruned models.

        Args:
            original_model: Original unpruned model
            pruned_model: Pruned model

        Returns:
            Compression statistics
        """
        original_params = sum(p.numel() for p in original_model.parameters())

        # Count only non-zero parameters
        pruned_params = 0
        for p in pruned_model.parameters():
            pruned_params += (p != 0).sum().item()

        compression_ratio = 1 - (pruned_params / original_params)

        # Memory estimation (assuming float32)
        original_memory = original_params * 4 / (1024 ** 2)  # MB
        pruned_memory = pruned_params * 4 / (1024 ** 2)  # MB

        return {
            'compression_ratio': compression_ratio,
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'original_memory_mb': original_memory,
            'pruned_memory_mb': pruned_memory,
            'memory_reduction': 1 - (pruned_memory / original_memory)
        }

    def export_sparse_format(
        self,
        model: nn.Module,
        format: str = 'csr'
    ) -> Dict[str, Any]:
        """
        Export model weights in sparse format.

        Args:
            model: Model to export
            format: Sparse format ('csr', 'coo', 'csc')

        Returns:
            Dictionary of sparse tensors
        """
        from scipy import sparse

        sparse_weights = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                weight = param.detach().cpu().numpy()

                if len(weight.shape) >= 2:
                    # Reshape if needed
                    original_shape = weight.shape
                    if len(weight.shape) > 2:
                        weight = weight.reshape(weight.shape[0], -1)

                    # Convert to sparse format
                    if format == 'csr':
                        sparse_weight = sparse.csr_matrix(weight)
                    elif format == 'coo':
                        sparse_weight = sparse.coo_matrix(weight)
                    elif format == 'csc':
                        sparse_weight = sparse.csc_matrix(weight)
                    else:
                        raise ValueError(f"Unknown sparse format: {format}")

                    sparse_weights[name] = {
                        'data': sparse_weight,
                        'original_shape': original_shape,
                        'nnz': sparse_weight.nnz,
                        'sparsity': 1 - (sparse_weight.nnz / weight.size)
                    }

        return sparse_weights