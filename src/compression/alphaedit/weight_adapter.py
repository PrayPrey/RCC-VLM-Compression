"""
AlphaEdit: Adaptive weight scaling with learned importance parameters.

This module implements learnable scaling factors for compressed weights using
Fisher information to guide importance estimation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict

from ..base import CompressionModule, CompressionConfig, CompressionMetrics

logger = logging.getLogger(__name__)


@dataclass
class AlphaEditConfig(CompressionConfig):
    """Configuration specific to AlphaEdit."""
    use_fisher_information: bool = True
    fisher_num_samples: int = 1000
    alpha_learning_rate: float = 1e-3
    alpha_momentum: float = 0.9
    alpha_regularization: float = 1e-4
    min_alpha: float = 0.1
    max_alpha: float = 10.0
    update_frequency: int = 100
    use_layer_wise_alphas: bool = True
    use_channel_wise_alphas: bool = False
    temperature: float = 1.0


class AlphaParameter(nn.Module):
    """Learnable alpha scaling parameter."""

    def __init__(self, shape: Tuple[int, ...], init_value: float = 1.0,
                 min_value: float = 0.1, max_value: float = 10.0):
        """
        Initialize alpha parameter.

        Args:
            shape: Shape of alpha parameter
            init_value: Initial value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

        # Initialize alpha parameter
        self.alpha = nn.Parameter(torch.full(shape, init_value))

        # Statistics for adaptive updates
        self.register_buffer('fisher_info', torch.zeros(shape))
        self.register_buffer('gradient_accumulator', torch.zeros(shape))
        self.register_buffer('update_count', torch.tensor(0))

    def forward(self) -> Tensor:
        """Get clamped alpha values."""
        return torch.clamp(self.alpha, self.min_value, self.max_value)

    def update_fisher(self, gradient: Tensor) -> None:
        """
        Update Fisher information estimate.

        Args:
            gradient: Gradient tensor
        """
        with torch.no_grad():
            # Exponential moving average of squared gradients
            self.fisher_info = 0.9 * self.fisher_info + 0.1 * gradient ** 2

    def update_gradient(self, gradient: Tensor) -> None:
        """
        Accumulate gradients for adaptive updates.

        Args:
            gradient: Gradient tensor
        """
        with torch.no_grad():
            self.gradient_accumulator += gradient
            self.update_count += 1


class AdaptiveWeightLayer(nn.Module):
    """Layer with adaptive weight scaling."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None,
                 alpha: Optional[AlphaParameter] = None,
                 layer_type: str = "linear"):
        """
        Initialize adaptive weight layer.

        Args:
            weight: Weight tensor
            bias: Bias tensor (optional)
            alpha: Alpha scaling parameter
            layer_type: Type of layer (linear, conv2d)
        """
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

        self.alpha = alpha
        self.layer_type = layer_type

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with adaptive scaling."""
        weight = self.weight

        # Apply alpha scaling if available
        if self.alpha is not None:
            alpha_values = self.alpha()
            if alpha_values.numel() == 1:
                # Scalar alpha
                weight = weight * alpha_values
            elif alpha_values.shape[0] == weight.shape[0]:
                # Channel-wise alpha
                if weight.dim() == 4:  # Conv2d
                    alpha_values = alpha_values.view(-1, 1, 1, 1)
                elif weight.dim() == 2:  # Linear
                    alpha_values = alpha_values.view(-1, 1)
                weight = weight * alpha_values
            else:
                # Element-wise alpha
                weight = weight * alpha_values

        # Apply layer operation
        if self.layer_type == "conv2d" and weight.dim() == 4:
            return nn.functional.conv2d(x, weight, self.bias)
        elif self.layer_type == "linear" and weight.dim() == 2:
            return nn.functional.linear(x, weight, self.bias)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")


class AlphaEditor(CompressionModule):
    """AlphaEdit adaptive weight scaling module."""

    def __init__(self, config: AlphaEditConfig):
        """
        Initialize AlphaEdit adapter.

        Args:
            config: AlphaEdit configuration
        """
        super().__init__(config)
        self.config = config
        self.alpha_parameters = nn.ModuleDict()
        self.fisher_information = {}
        self.layer_importance = {}
        self.optimization_history = []

    def compress(self, module: nn.Module, dataloader: Optional[Any] = None,
                **kwargs) -> nn.Module:
        """
        Apply AlphaEdit adaptation to module.

        Args:
            module: Module to adapt
            dataloader: Data loader for Fisher information estimation
            **kwargs: Additional parameters

        Returns:
            Module with adaptive weight scaling
        """
        # Compute Fisher information if enabled
        if self.config.use_fisher_information and dataloader is not None:
            self._compute_fisher_information(module, dataloader)

        # Initialize alpha parameters
        self._initialize_alphas(module)

        # Wrap layers with adaptive scaling
        adapted_module = self._wrap_with_alphas(module)

        # Optimize alpha parameters if data is available
        if dataloader is not None:
            self._optimize_alphas(adapted_module, dataloader)

        return adapted_module

    def _initialize_alphas(self, module: nn.Module) -> None:
        """
        Initialize alpha parameters for each layer.

        Args:
            module: Module to initialize alphas for
        """
        for name, param in module.named_parameters():
            if self._should_adapt(name, param):
                # Determine alpha shape
                if self.config.use_channel_wise_alphas and param.dim() >= 2:
                    alpha_shape = (param.shape[0],)
                elif self.config.use_layer_wise_alphas:
                    alpha_shape = (1,)
                else:
                    alpha_shape = param.shape

                # Initialize based on Fisher information or uniform
                if name in self.fisher_information:
                    # Initialize proportional to Fisher information
                    fisher = self.fisher_information[name]
                    if self.config.use_channel_wise_alphas:
                        fisher_channelwise = fisher.sum(dim=tuple(range(1, fisher.dim())))
                        init_value = 1.0 + 0.1 * torch.tanh(fisher_channelwise).mean().item()
                    else:
                        init_value = 1.0 + 0.1 * torch.tanh(fisher.mean()).item()
                else:
                    init_value = self.config.alpha_init

                # Create alpha parameter
                alpha = AlphaParameter(
                    alpha_shape,
                    init_value,
                    self.config.min_alpha,
                    self.config.max_alpha
                )

                self.alpha_parameters[name] = alpha

    def _wrap_with_alphas(self, module: nn.Module) -> nn.Module:
        """
        Wrap module layers with adaptive scaling.

        Args:
            module: Module to wrap

        Returns:
            Wrapped module
        """
        wrapped_module = self._create_module_copy(module)

        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                # Get corresponding alpha parameter
                weight_name = f"{name}.weight"
                alpha = self.alpha_parameters.get(weight_name, None)

                # Create adaptive layer
                layer_type = "linear" if isinstance(child, nn.Linear) else "conv2d"
                adapted_child = AdaptiveWeightLayer(
                    child.weight.data,
                    child.bias.data if child.bias is not None else None,
                    alpha,
                    layer_type
                )

                setattr(wrapped_module, name, adapted_child)

            elif len(list(child.children())) > 0:
                # Recursively wrap nested modules
                wrapped_child = self._wrap_with_alphas(child)
                setattr(wrapped_module, name, wrapped_child)
            else:
                # Copy non-adaptable layers
                setattr(wrapped_module, name, child)

        return wrapped_module

    def _compute_fisher_information(self, module: nn.Module,
                                   dataloader: Any) -> None:
        """
        Compute Fisher information for weight importance.

        Args:
            module: Module to compute Fisher information for
            dataloader: Data loader for sampling
        """
        logger.info("Computing Fisher information...")

        # Enable gradient computation
        module.train()
        fisher_accumulator = defaultdict(list)

        # Sample batches for Fisher estimation
        num_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples >= self.config.fisher_num_samples:
                break

            # Forward pass
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
            else:
                inputs = batch
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

            outputs = module(inputs)

            # Compute log-likelihood (assuming classification)
            if outputs.dim() == 2:
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                # Sample from output distribution
                sampled_targets = torch.multinomial(
                    torch.exp(log_probs).detach(),
                    num_samples=1
                ).squeeze()
                loss = torch.nn.functional.nll_loss(log_probs, sampled_targets)
            else:
                # For other tasks, use L2 loss as proxy
                loss = 0.5 * (outputs ** 2).mean()

            # Compute gradients
            loss.backward()

            # Accumulate squared gradients (Fisher information)
            for name, param in module.named_parameters():
                if param.grad is not None:
                    fisher_accumulator[name].append(param.grad.data ** 2)

            # Clear gradients
            module.zero_grad()

            num_samples += inputs.size(0)

        # Average Fisher information
        for name, fisher_list in fisher_accumulator.items():
            self.fisher_information[name] = torch.stack(fisher_list).mean(dim=0)

        logger.info(f"Computed Fisher information for {len(self.fisher_information)} parameters")

    def _optimize_alphas(self, module: nn.Module, dataloader: Any,
                        num_epochs: int = 10) -> None:
        """
        Optimize alpha parameters using gradient descent.

        Args:
            module: Module with alpha parameters
            dataloader: Data loader for optimization
            num_epochs: Number of optimization epochs
        """
        logger.info("Optimizing alpha parameters...")

        # Collect all alpha parameters
        alpha_params = []
        for alpha_module in self.alpha_parameters.values():
            alpha_params.extend(alpha_module.parameters())

        if not alpha_params:
            return

        # Create optimizer for alpha parameters
        optimizer = Adam(
            alpha_params,
            lr=self.config.alpha_learning_rate,
            betas=(self.config.alpha_momentum, 0.999)
        )

        # Training loop
        module.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 100:  # Limit optimization steps
                    break

                # Forward pass
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                else:
                    inputs = batch
                    targets = None
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                outputs = module(inputs)

                # Compute loss
                if targets is not None and outputs.dim() == 2:
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                else:
                    # Self-supervised: minimize reconstruction error
                    loss = torch.nn.functional.mse_loss(outputs, inputs)

                # Add regularization
                reg_loss = 0.0
                for alpha_module in self.alpha_parameters.values():
                    alpha_values = alpha_module()
                    # L2 regularization to keep alphas close to 1
                    reg_loss += self.config.alpha_regularization * \
                               ((alpha_values - 1.0) ** 2).mean()

                total_loss = loss + reg_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=1.0)

                # Update alphas
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            self.optimization_history.append({
                'epoch': epoch,
                'loss': avg_loss
            })

            logger.info(f"Alpha optimization epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")

    def _should_adapt(self, name: str, param: Tensor) -> bool:
        """
        Check if parameter should have adaptive scaling.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            Whether to adapt this parameter
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

    def _create_module_copy(self, module: nn.Module) -> nn.Module:
        """
        Create a copy of module structure.

        Args:
            module: Module to copy

        Returns:
            Module copy
        """
        module_class = type(module)
        try:
            copied = module_class()
        except:
            copied = nn.Module()
        return copied

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

        # Add alpha parameters to compressed count
        for alpha_module in self.alpha_parameters.values():
            compressed_params += sum(p.numel() for p in alpha_module.parameters())

        return CompressionMetrics(
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / compressed_params,
            energy_preserved=1.0  # Alphas preserve energy by scaling
        )

    def get_alpha_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about learned alpha values.

        Returns:
            Dictionary of alpha statistics per layer
        """
        stats = {}
        for name, alpha_module in self.alpha_parameters.items():
            alpha_values = alpha_module().detach()
            stats[name] = {
                'mean': alpha_values.mean().item(),
                'std': alpha_values.std().item(),
                'min': alpha_values.min().item(),
                'max': alpha_values.max().item(),
                'num_params': alpha_values.numel()
            }
        return stats


# Alias for backward compatibility
AlphaEditAdapter = AlphaEditor