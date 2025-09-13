"""
Mixed precision training utilities.

This module provides automatic mixed precision (AMP) support for
efficient training of compressed models.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any, Callable, Union
import logging
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    enabled: bool = True
    dtype: torch.dtype = torch.float16  # float16 or bfloat16
    cache_enabled: bool = True
    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enable_autocast: bool = True
    gradient_accumulation_steps: int = 1


class MixedPrecisionTrainer:
    """Handles mixed precision training."""

    def __init__(self, config: MixedPrecisionConfig):
        """
        Initialize mixed precision trainer.

        Args:
            config: Mixed precision configuration
        """
        self.config = config
        self.scaler = None
        self.autocast_dtype = config.dtype

        if config.enabled and torch.cuda.is_available():
            self.scaler = GradScaler(
                init_scale=config.init_scale,
                growth_factor=config.growth_factor,
                backoff_factor=config.backoff_factor,
                growth_interval=config.growth_interval,
                enabled=True
            )
            logger.info(f"Mixed precision training enabled with {config.dtype}")
        else:
            logger.info("Mixed precision training disabled")

    @contextmanager
    def autocast_context(self):
        """
        Context manager for automatic mixed precision.

        Yields:
            Autocast context
        """
        if self.config.enabled and self.config.enable_autocast:
            with autocast(
                device_type='cuda',
                dtype=self.autocast_dtype,
                enabled=True,
                cache_enabled=self.config.cache_enabled
            ):
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision.

        Args:
            loss: Unscaled loss

        Returns:
            Scaled loss
        """
        if self.scaler is not None and self.config.enabled:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Optimizer step with mixed precision.

        Args:
            optimizer: Optimizer instance

        Returns:
            Whether step was successful (not skipped due to inf/nan)
        """
        if self.scaler is not None and self.config.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()

            # Check if gradients were finite
            scale_before = self.scaler.get_scale()
            self.scaler.update()
            scale_after = self.scaler.get_scale()

            # Step was skipped if scale decreased
            return scale_after >= scale_before
        else:
            optimizer.step()
            return True

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients before clipping.

        Args:
            optimizer: Optimizer instance
        """
        if self.scaler is not None and self.config.enabled:
            self.scaler.unscale_(optimizer)

    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Backward pass with mixed precision.

        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
        """
        if self.scaler is not None and self.config.enabled:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def state_dict(self) -> Dict:
        """
        Get state dict for checkpointing.

        Returns:
            State dictionary
        """
        state = {'config': self.config}
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict):
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary
        """
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])

    def get_grad_scale(self) -> float:
        """
        Get current gradient scale.

        Returns:
            Current scale value
        """
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


class GradientAccumulator:
    """Handles gradient accumulation with mixed precision."""

    def __init__(self,
                 accumulation_steps: int = 1,
                 mixed_precision: Optional[MixedPrecisionTrainer] = None):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of accumulation steps
            mixed_precision: Mixed precision trainer
        """
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.step_count = 0

    def should_step(self) -> bool:
        """
        Check if optimizer should step.

        Returns:
            Whether to perform optimizer step
        """
        return (self.step_count + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for accumulation.

        Args:
            loss: Unscaled loss

        Returns:
            Scaled loss
        """
        # Scale by accumulation steps
        loss = loss / self.accumulation_steps

        # Apply mixed precision scaling
        if self.mixed_precision:
            loss = self.mixed_precision.scale_loss(loss)

        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Perform optimizer step if ready.

        Args:
            optimizer: Optimizer instance

        Returns:
            Whether step was performed
        """
        self.step_count += 1

        if self.should_step():
            if self.mixed_precision:
                success = self.mixed_precision.step(optimizer)
            else:
                optimizer.step()
                success = True

            optimizer.zero_grad()
            return success

        return False


def convert_model_to_mixed_precision(model: nn.Module,
                                    dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Convert model for mixed precision training.

    Args:
        model: Model to convert
        dtype: Target dtype

    Returns:
        Converted model
    """
    # Convert batch norm layers to float32 for stability
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm)):
            module.float()

    # Set autocast dtype for custom modules
    for module in model.modules():
        if hasattr(module, 'set_mixed_precision'):
            module.set_mixed_precision(dtype)

    return model


def optimize_model_for_mixed_precision(model: nn.Module) -> nn.Module:
    """
    Optimize model operations for mixed precision.

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    # Enable TensorFloat-32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True

    # Ensure model is in eval mode during optimization
    model.eval()

    # Apply optimizations
    with torch.no_grad():
        # Fuse operations where possible
        if hasattr(torch.jit, 'fuse'):
            model = torch.jit.fuse(model)

    model.train()
    return model


class DynamicLossScaler:
    """Dynamic loss scaling for mixed precision."""

    def __init__(self,
                 init_scale: float = 2.0 ** 16,
                 scale_factor: float = 2.0,
                 scale_window: int = 2000,
                 min_scale: float = 1.0,
                 max_scale: float = 2.0 ** 24):
        """
        Initialize dynamic loss scaler.

        Args:
            init_scale: Initial scale
            scale_factor: Scale growth/reduction factor
            scale_window: Steps between scale updates
            min_scale: Minimum scale
            max_scale: Maximum scale
        """
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.steps_since_update = 0
        self.found_inf = False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss."""
        return loss * self.scale

    def update_scale(self, found_inf: bool):
        """
        Update scale based on gradient status.

        Args:
            found_inf: Whether inf/nan was found in gradients
        """
        if found_inf:
            # Reduce scale
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self.steps_since_update = 0
            logger.debug(f"Reduced scale to {self.scale}")
        else:
            self.steps_since_update += 1

            if self.steps_since_update >= self.scale_window:
                # Increase scale
                self.scale = min(self.scale * self.scale_factor, self.max_scale)
                self.steps_since_update = 0
                logger.debug(f"Increased scale to {self.scale}")


def check_gradients_finite(model: nn.Module) -> bool:
    """
    Check if all gradients are finite.

    Args:
        model: Model to check

    Returns:
        True if all gradients are finite
    """
    for param in model.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                return False
    return True