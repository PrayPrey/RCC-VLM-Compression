"""
Compression scheduling for cascade pipeline.

This module implements scheduling strategies for multi-stage compression,
including progressive compression, warmup, and adaptive scheduling.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Schedule types for compression."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    STEP = "step"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


@dataclass
class CompressionSchedule:
    """Schedule configuration for a compression stage."""
    stage_name: str
    schedule_type: ScheduleType
    initial_value: float
    final_value: float
    num_steps: int
    warmup_steps: int = 0
    cooldown_steps: int = 0
    step_size: int = 10
    gamma: float = 0.1
    power: float = 1.0
    custom_fn: Optional[Callable] = None


class CompressionScheduler:
    """Manages compression scheduling across pipeline stages."""

    def __init__(self, schedules: List[CompressionSchedule]):
        """
        Initialize compression scheduler.

        Args:
            schedules: List of schedules for each stage
        """
        self.schedules = {s.stage_name: s for s in schedules}
        self.current_steps = {s.stage_name: 0 for s in schedules}
        self.current_values = {}
        self.schedule_history = {s.stage_name: [] for s in schedules}

        # Initialize current values
        for schedule in schedules:
            self.current_values[schedule.stage_name] = schedule.initial_value

    def step(self, stage_name: str) -> float:
        """
        Get next value and advance schedule for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Scheduled value for current step
        """
        if stage_name not in self.schedules:
            raise ValueError(f"No schedule found for stage: {stage_name}")

        schedule = self.schedules[stage_name]
        current_step = self.current_steps[stage_name]

        # Compute scheduled value
        value = self._compute_value(schedule, current_step)

        # Update tracking
        self.current_values[stage_name] = value
        self.current_steps[stage_name] += 1
        self.schedule_history[stage_name].append(value)

        return value

    def get_current_value(self, stage_name: str) -> float:
        """
        Get current value without advancing schedule.

        Args:
            stage_name: Name of the stage

        Returns:
            Current scheduled value
        """
        return self.current_values.get(stage_name, 0.0)

    def reset_stage(self, stage_name: str):
        """
        Reset schedule for a stage.

        Args:
            stage_name: Name of the stage
        """
        if stage_name in self.schedules:
            schedule = self.schedules[stage_name]
            self.current_steps[stage_name] = 0
            self.current_values[stage_name] = schedule.initial_value
            self.schedule_history[stage_name] = []

    def _compute_value(self, schedule: CompressionSchedule, step: int) -> float:
        """
        Compute scheduled value for given step.

        Args:
            schedule: Schedule configuration
            step: Current step

        Returns:
            Scheduled value
        """
        # Handle warmup
        if step < schedule.warmup_steps:
            warmup_ratio = step / max(1, schedule.warmup_steps)
            return schedule.initial_value * warmup_ratio

        # Handle cooldown
        total_steps = schedule.num_steps
        if step >= total_steps - schedule.cooldown_steps:
            cooldown_step = step - (total_steps - schedule.cooldown_steps)
            cooldown_ratio = 1.0 - (cooldown_step / max(1, schedule.cooldown_steps))
            return schedule.final_value * cooldown_ratio

        # Adjust step for main schedule
        adjusted_step = step - schedule.warmup_steps
        adjusted_total = total_steps - schedule.warmup_steps - schedule.cooldown_steps

        if schedule.schedule_type == ScheduleType.LINEAR:
            value = self._linear_schedule(
                schedule.initial_value,
                schedule.final_value,
                adjusted_step,
                adjusted_total
            )
        elif schedule.schedule_type == ScheduleType.COSINE:
            value = self._cosine_schedule(
                schedule.initial_value,
                schedule.final_value,
                adjusted_step,
                adjusted_total
            )
        elif schedule.schedule_type == ScheduleType.EXPONENTIAL:
            value = self._exponential_schedule(
                schedule.initial_value,
                schedule.final_value,
                adjusted_step,
                adjusted_total,
                schedule.gamma
            )
        elif schedule.schedule_type == ScheduleType.POLYNOMIAL:
            value = self._polynomial_schedule(
                schedule.initial_value,
                schedule.final_value,
                adjusted_step,
                adjusted_total,
                schedule.power
            )
        elif schedule.schedule_type == ScheduleType.STEP:
            value = self._step_schedule(
                schedule.initial_value,
                schedule.final_value,
                adjusted_step,
                schedule.step_size,
                schedule.gamma
            )
        elif schedule.schedule_type == ScheduleType.ADAPTIVE:
            value = self._adaptive_schedule(schedule, adjusted_step, adjusted_total)
        elif schedule.schedule_type == ScheduleType.CUSTOM:
            if schedule.custom_fn is not None:
                value = schedule.custom_fn(adjusted_step, adjusted_total)
            else:
                value = schedule.initial_value
        else:
            value = schedule.initial_value

        return value

    def _linear_schedule(self,
                        initial: float,
                        final: float,
                        step: int,
                        total_steps: int) -> float:
        """Linear interpolation schedule."""
        if total_steps <= 0:
            return initial

        ratio = min(1.0, step / total_steps)
        return initial + (final - initial) * ratio

    def _cosine_schedule(self,
                        initial: float,
                        final: float,
                        step: int,
                        total_steps: int) -> float:
        """Cosine annealing schedule."""
        if total_steps <= 0:
            return initial

        ratio = min(1.0, step / total_steps)
        cosine_val = 0.5 * (1 + math.cos(math.pi * ratio))
        return final + (initial - final) * cosine_val

    def _exponential_schedule(self,
                             initial: float,
                             final: float,
                             step: int,
                             total_steps: int,
                             gamma: float) -> float:
        """Exponential decay schedule."""
        if total_steps <= 0:
            return initial

        decay_rate = -math.log(final / initial) / total_steps if final > 0 else gamma
        return initial * math.exp(-decay_rate * step)

    def _polynomial_schedule(self,
                           initial: float,
                           final: float,
                           step: int,
                           total_steps: int,
                           power: float) -> float:
        """Polynomial schedule."""
        if total_steps <= 0:
            return initial

        ratio = min(1.0, step / total_steps)
        poly_val = 1.0 - pow(ratio, power)
        return final + (initial - final) * poly_val

    def _step_schedule(self,
                      initial: float,
                      final: float,
                      step: int,
                      step_size: int,
                      gamma: float) -> float:
        """Step decay schedule."""
        num_steps = step // step_size
        current_value = initial * (gamma ** num_steps)
        return max(final, current_value)

    def _adaptive_schedule(self,
                         schedule: CompressionSchedule,
                         step: int,
                         total_steps: int) -> float:
        """
        Adaptive schedule based on performance feedback.

        Args:
            schedule: Schedule configuration
            step: Current step
            total_steps: Total steps

        Returns:
            Adaptively computed value
        """
        # Start with linear schedule as base
        base_value = self._linear_schedule(
            schedule.initial_value,
            schedule.final_value,
            step,
            total_steps
        )

        # Adapt based on history (simple momentum-based adaptation)
        if len(self.schedule_history[schedule.stage_name]) > 1:
            recent_values = self.schedule_history[schedule.stage_name][-5:]
            momentum = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0

            # Adjust based on momentum
            adaptive_factor = 1.0 + 0.1 * np.sign(momentum) * min(abs(momentum), 1.0)
            base_value *= adaptive_factor

        return np.clip(base_value, schedule.final_value, schedule.initial_value)

    def get_schedule_plot_data(self, stage_name: str) -> Tuple[List[int], List[float]]:
        """
        Get data for plotting schedule.

        Args:
            stage_name: Name of the stage

        Returns:
            Steps and values for plotting
        """
        if stage_name not in self.schedules:
            return [], []

        schedule = self.schedules[stage_name]
        steps = list(range(schedule.num_steps))
        values = []

        # Compute values for all steps
        for step in steps:
            value = self._compute_value(schedule, step)
            values.append(value)

        return steps, values

    def create_cascade_schedule(self,
                               stages: List[str],
                               total_compression: float,
                               distribution: str = "uniform") -> List[CompressionSchedule]:
        """
        Create schedules for cascade compression.

        Args:
            stages: List of stage names
            total_compression: Total target compression
            distribution: How to distribute compression (uniform, exponential, front-loaded)

        Returns:
            List of compression schedules
        """
        num_stages = len(stages)
        schedules = []

        if distribution == "uniform":
            # Equal compression per stage
            per_stage_compression = total_compression ** (1 / num_stages)
            for i, stage in enumerate(stages):
                schedule = CompressionSchedule(
                    stage_name=stage,
                    schedule_type=ScheduleType.COSINE,
                    initial_value=1.0,
                    final_value=per_stage_compression,
                    num_steps=100,
                    warmup_steps=10
                )
                schedules.append(schedule)

        elif distribution == "exponential":
            # Exponentially increasing compression
            for i, stage in enumerate(stages):
                stage_compression = total_compression ** ((i + 1) / num_stages)
                schedule = CompressionSchedule(
                    stage_name=stage,
                    schedule_type=ScheduleType.EXPONENTIAL,
                    initial_value=1.0,
                    final_value=stage_compression,
                    num_steps=100,
                    gamma=0.95
                )
                schedules.append(schedule)

        elif distribution == "front-loaded":
            # More compression in early stages
            compression_ratios = np.linspace(0.6, 0.3, num_stages)
            compression_ratios = compression_ratios / compression_ratios.sum()

            for i, (stage, ratio) in enumerate(zip(stages, compression_ratios)):
                stage_compression = total_compression * ratio
                schedule = CompressionSchedule(
                    stage_name=stage,
                    schedule_type=ScheduleType.POLYNOMIAL,
                    initial_value=1.0,
                    final_value=stage_compression,
                    num_steps=100,
                    power=2.0
                )
                schedules.append(schedule)

        return schedules

    def get_schedule_stats(self) -> Dict:
        """
        Get statistics about schedules.

        Returns:
            Dictionary of schedule statistics
        """
        stats = {}

        for stage_name, schedule in self.schedules.items():
            history = self.schedule_history[stage_name]

            if history:
                stats[stage_name] = {
                    'current_step': self.current_steps[stage_name],
                    'total_steps': schedule.num_steps,
                    'current_value': self.current_values[stage_name],
                    'initial_value': schedule.initial_value,
                    'final_value': schedule.final_value,
                    'mean_value': np.mean(history),
                    'std_value': np.std(history),
                    'min_value': min(history),
                    'max_value': max(history)
                }

        return stats


class AdaptiveScheduler:
    """Adaptive scheduler that adjusts based on performance feedback."""

    def __init__(self,
                 initial_schedule: CompressionSchedule,
                 performance_threshold: float = 0.95,
                 adjustment_factor: float = 0.1):
        """
        Initialize adaptive scheduler.

        Args:
            initial_schedule: Initial schedule configuration
            performance_threshold: Performance threshold for adjustment
            adjustment_factor: Factor for schedule adjustment
        """
        self.schedule = initial_schedule
        self.performance_threshold = performance_threshold
        self.adjustment_factor = adjustment_factor

        self.performance_history = []
        self.adjustment_history = []

    def update(self,
              current_performance: float,
              target_performance: float) -> float:
        """
        Update schedule based on performance.

        Args:
            current_performance: Current model performance
            target_performance: Target performance

        Returns:
            Adjusted compression value
        """
        # Record performance
        self.performance_history.append(current_performance)

        # Calculate performance ratio
        performance_ratio = current_performance / target_performance

        # Determine adjustment
        if performance_ratio < self.performance_threshold:
            # Performance below threshold, reduce compression
            adjustment = -self.adjustment_factor * (1 - performance_ratio)
        elif performance_ratio > 1.0:
            # Performance above target, can increase compression
            adjustment = self.adjustment_factor * min(performance_ratio - 1.0, 0.1)
        else:
            # Performance within acceptable range
            adjustment = 0.0

        # Apply adjustment
        current_value = self.schedule.final_value
        adjusted_value = np.clip(
            current_value * (1 + adjustment),
            0.0,
            1.0
        )

        # Update schedule
        self.schedule.final_value = adjusted_value
        self.adjustment_history.append(adjustment)

        return adjusted_value

    def get_adjustment_stats(self) -> Dict:
        """
        Get adjustment statistics.

        Returns:
            Dictionary of adjustment statistics
        """
        if not self.adjustment_history:
            return {}

        return {
            'num_adjustments': len(self.adjustment_history),
            'mean_adjustment': np.mean(self.adjustment_history),
            'total_adjustment': sum(self.adjustment_history),
            'performance_mean': np.mean(self.performance_history),
            'performance_std': np.std(self.performance_history)
        }