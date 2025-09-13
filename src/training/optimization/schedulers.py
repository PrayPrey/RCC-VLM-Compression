"""
Learning rate schedulers for training.

This module provides various learning rate scheduling strategies
for optimizing compressed models.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """Cosine scheduler with linear warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        """
        Initialize warmup cosine scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """Linear scheduler with warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        """
        Initialize warmup linear scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            linear_factor = 1 - progress
            return [
                self.min_lr + (base_lr - self.min_lr) * linear_factor
                for base_lr in self.base_lrs
            ]


class PolynomialScheduler(_LRScheduler):
    """Polynomial learning rate scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 total_steps: int,
                 power: float = 1.0,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        """
        Initialize polynomial scheduler.

        Args:
            optimizer: Optimizer instance
            total_steps: Total training steps
            power: Polynomial power
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch >= self.total_steps:
            return [self.min_lr for _ in self.base_lrs]

        decay_factor = (1 - self.last_epoch / self.total_steps) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class CyclicCosineScheduler(_LRScheduler):
    """Cyclic cosine annealing scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps: int,
                 min_lr: float = 0,
                 cycle_mult: float = 1.0,
                 last_epoch: int = -1):
        """
        Initialize cyclic cosine scheduler.

        Args:
            optimizer: Optimizer instance
            cycle_steps: Steps per cycle
            min_lr: Minimum learning rate
            cycle_mult: Cycle length multiplier
            last_epoch: Last epoch number
        """
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult
        self.cycle = 0
        self.cycle_epoch = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch >= self.cycle_steps:
            self.cycle += 1
            self.cycle_epoch = 0
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
        else:
            self.cycle_epoch += 1

        progress = self.cycle_epoch / max(1, self.cycle_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]


class LayerWiseScheduler:
    """Layer-wise learning rate scheduler for fine-tuning."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 layer_decay: float = 0.9,
                 num_layers: Optional[int] = None):
        """
        Initialize layer-wise scheduler.

        Args:
            optimizer: Optimizer instance
            layer_decay: Decay factor per layer
            num_layers: Number of layers
        """
        self.optimizer = optimizer
        self.layer_decay = layer_decay
        self.num_layers = num_layers or len(optimizer.param_groups)

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Apply layer-wise decay
        self._apply_layer_decay()

    def _apply_layer_decay(self):
        """Apply layer-wise learning rate decay."""
        for i, group in enumerate(self.optimizer.param_groups):
            layer_idx = i
            if 'layer_idx' in group:
                layer_idx = group['layer_idx']

            decay_factor = self.layer_decay ** (self.num_layers - layer_idx - 1)
            group['lr'] = self.base_lrs[i] * decay_factor

    def step(self):
        """Step the scheduler (no-op for layer-wise)."""
        pass

    def state_dict(self) -> dict:
        """Get state dict."""
        return {
            'base_lrs': self.base_lrs,
            'layer_decay': self.layer_decay,
            'num_layers': self.num_layers
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict."""
        self.base_lrs = state_dict['base_lrs']
        self.layer_decay = state_dict['layer_decay']
        self.num_layers = state_dict['num_layers']
        self._apply_layer_decay()


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str,
                    num_epochs: int,
                    steps_per_epoch: int,
                    warmup_ratio: float = 0.1,
                    min_lr: float = 0,
                    **kwargs) -> _LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        warmup_ratio: Warmup ratio
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
    elif scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
    elif scheduler_type == "polynomial":
        power = kwargs.get('power', 1.0)
        scheduler = PolynomialScheduler(
            optimizer,
            total_steps=total_steps,
            power=power,
            min_lr=min_lr
        )
    elif scheduler_type == "cyclic":
        cycle_steps = kwargs.get('cycle_steps', steps_per_epoch)
        scheduler = CyclicCosineScheduler(
            optimizer,
            cycle_steps=cycle_steps,
            min_lr=min_lr
        )
    elif scheduler_type == "step":
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == "exponential":
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=min_lr
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using constant LR")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return scheduler


class CompressionAwareScheduler(_LRScheduler):
    """Scheduler that adapts to compression level."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 compression_schedule: List[float],
                 lr_scale_factors: List[float],
                 last_epoch: int = -1):
        """
        Initialize compression-aware scheduler.

        Args:
            optimizer: Optimizer instance
            compression_schedule: Compression levels over time
            lr_scale_factors: LR scaling per compression level
            last_epoch: Last epoch number
        """
        self.compression_schedule = compression_schedule
        self.lr_scale_factors = lr_scale_factors
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates based on compression level."""
        if self.last_epoch >= len(self.compression_schedule):
            compression_idx = -1
        else:
            compression_idx = self.last_epoch

        scale_factor = self.lr_scale_factors[compression_idx]

        return [base_lr * scale_factor for base_lr in self.base_lrs]