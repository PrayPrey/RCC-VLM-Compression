"""
Performance profiling and bottleneck analysis for compression pipeline.

This module provides detailed performance analysis tools for identifying
bottlenecks and optimization opportunities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import time
import psutil
import GPUtil
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
import json
import logging
from contextlib import contextmanager
from collections import defaultdict
import tracemalloc
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Container for profiling results."""
    operation: str
    duration_ms: float
    memory_mb: float
    gpu_memory_mb: float
    cpu_percent: float
    gpu_percent: float
    flops: Optional[int] = None
    parameters: Optional[int] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiler for ML models."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize performance profiler.

        Args:
            device: Device to profile on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[ProfileResult] = []
        self.memory_snapshots = []

    @contextmanager
    def profile_operation(self, operation_name: str, track_memory: bool = True):
        """
        Context manager for profiling an operation.

        Args:
            operation_name: Name of operation
            track_memory: Whether to track memory usage
        """
        # Start tracking
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            start_gpu_util = self._get_gpu_utilization()
        else:
            start_gpu_memory = 0
            start_gpu_util = 0

        start_cpu_percent = psutil.cpu_percent(interval=None)

        if track_memory:
            tracemalloc.start()

        try:
            yield
        finally:
            # End tracking
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                end_gpu_util = self._get_gpu_utilization()
            else:
                end_gpu_memory = 0
                end_gpu_util = 0

            end_cpu_percent = psutil.cpu_percent(interval=None)

            if track_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_peak_mb = peak / 1024 / 1024
            else:
                memory_peak_mb = end_memory - start_memory

            # Create result
            result = ProfileResult(
                operation=operation_name,
                duration_ms=(end_time - start_time) * 1000,
                memory_mb=memory_peak_mb,
                gpu_memory_mb=end_gpu_memory - start_gpu_memory,
                cpu_percent=(start_cpu_percent + end_cpu_percent) / 2,
                gpu_percent=(start_gpu_util + end_gpu_util) / 2
            )

            self.results.append(result)
            logger.debug(f"Profiled {operation_name}: {result.duration_ms:.2f}ms, "
                       f"{result.memory_mb:.2f}MB")

    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                     num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile model performance.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_iterations: Number of iterations for timing

        Returns:
            Profiling results
        """
        model = model.to(self.device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)

        # Profile forward pass
        with self.profile_operation("forward_pass"):
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Actual timing
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    _ = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

        avg_forward_time = (end_time - start_time) / num_iterations * 1000  # ms

        # Profile backward pass
        dummy_input.requires_grad = True
        model.train()

        with self.profile_operation("backward_pass"):
            output = model(dummy_input)
            if isinstance(output, dict):
                loss = output.get('loss', output.get('logits', list(output.values())[0])).mean()
            else:
                loss = output.mean()

            loss.backward()

        # Calculate FLOPs if possible
        flops = self._calculate_flops(model, dummy_input)

        # Memory footprint
        memory_stats = self._get_memory_stats(model)

        return {
            'avg_forward_time_ms': avg_forward_time,
            'forward_profile': self.results[-2] if len(self.results) >= 2 else None,
            'backward_profile': self.results[-1] if self.results else None,
            'flops': flops,
            'memory_stats': memory_stats,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

    def profile_compression_pipeline(self, pipeline: Any, model: nn.Module,
                                    stages: List[str]) -> pd.DataFrame:
        """
        Profile compression pipeline stages.

        Args:
            pipeline: Compression pipeline
            model: Model to compress
            stages: Compression stages to profile

        Returns:
            DataFrame with profiling results
        """
        results = []

        for stage in stages:
            logger.info(f"Profiling {stage} compression...")

            with self.profile_operation(f"compression_{stage}"):
                if hasattr(pipeline, f'apply_{stage}'):
                    compressed_model = getattr(pipeline, f'apply_{stage}')(model.clone())
                else:
                    compressed_model = model

            # Get compression statistics
            orig_params = sum(p.numel() for p in model.parameters())
            comp_params = sum(p.numel() for p in compressed_model.parameters())

            result = {
                'stage': stage,
                'duration_ms': self.results[-1].duration_ms if self.results else 0,
                'memory_mb': self.results[-1].memory_mb if self.results else 0,
                'original_params': orig_params,
                'compressed_params': comp_params,
                'compression_ratio': 1 - (comp_params / orig_params) if orig_params > 0 else 0
            }

            # Profile compressed model
            profile = self.profile_model(compressed_model, (1, 3, 224, 224), num_iterations=10)
            result['inference_time_ms'] = profile['avg_forward_time_ms']
            result['flops'] = profile['flops']

            results.append(result)
            model = compressed_model  # Use compressed model for next stage

        return pd.DataFrame(results)

    def _calculate_flops(self, model: nn.Module, input: torch.Tensor) -> Optional[int]:
        """Calculate FLOPs for model."""
        try:
            from thop import profile
            flops, params = profile(model, inputs=(input,), verbose=False)
            return int(flops)
        except ImportError:
            logger.debug("THOP not installed, skipping FLOP calculation")
            return None
        except Exception as e:
            logger.debug(f"Failed to calculate FLOPs: {e}")
            return None

    def _get_memory_stats(self, model: nn.Module) -> Dict[str, float]:
        """Get memory statistics for model."""
        stats = {
            'model_size_mb': 0,
            'gradient_size_mb': 0,
            'buffer_size_mb': 0
        }

        # Model parameters
        for p in model.parameters():
            stats['model_size_mb'] += p.numel() * p.element_size() / 1024 / 1024
            if p.grad is not None:
                stats['gradient_size_mb'] += p.grad.numel() * p.grad.element_size() / 1024 / 1024

        # Buffers
        for b in model.buffers():
            stats['buffer_size_mb'] += b.numel() * b.element_size() / 1024 / 1024

        stats['total_size_mb'] = sum(stats.values())

        return stats

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate performance profiling report.

        Args:
            save_path: Optional path to save report

        Returns:
            Report as string
        """
        df = pd.DataFrame([{
            'Operation': r.operation,
            'Duration (ms)': r.duration_ms,
            'Memory (MB)': r.memory_mb,
            'GPU Memory (MB)': r.gpu_memory_mb,
            'CPU %': r.cpu_percent,
            'GPU %': r.gpu_percent
        } for r in self.results])

        report = "# Performance Profiling Report\n\n"
        report += "## Summary Statistics\n\n"
        report += df.describe().to_string() + "\n\n"

        report += "## Detailed Results\n\n"
        report += df.to_string(index=False) + "\n\n"

        report += "## Bottleneck Analysis\n\n"

        # Find bottlenecks
        slowest_op = df.loc[df['Duration (ms)'].idxmax()]
        report += f"- Slowest operation: {slowest_op['Operation']} ({slowest_op['Duration (ms)']:.2f}ms)\n"

        memory_heavy = df.loc[df['Memory (MB)'].idxmax()]
        report += f"- Most memory intensive: {memory_heavy['Operation']} ({memory_heavy['Memory (MB)']:.2f}MB)\n"

        if df['GPU Memory (MB)'].sum() > 0:
            gpu_heavy = df.loc[df['GPU Memory (MB)'].idxmax()]
            report += f"- Highest GPU memory: {gpu_heavy['Operation']} ({gpu_heavy['GPU Memory (MB)']:.2f}MB)\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report

    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize profiling results.

        Args:
            save_path: Optional path to save visualization
        """
        if not self.results:
            logger.warning("No profiling results to visualize")
            return

        df = pd.DataFrame([{
            'Operation': r.operation,
            'Duration (ms)': r.duration_ms,
            'Memory (MB)': r.memory_mb,
            'GPU Memory (MB)': r.gpu_memory_mb
        } for r in self.results])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Duration breakdown
        df.plot(x='Operation', y='Duration (ms)', kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Operation Duration')
        axes[0, 0].set_ylabel('Duration (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Memory usage
        df[['Operation', 'Memory (MB)', 'GPU Memory (MB)']].set_index('Operation').plot(
            kind='bar', ax=axes[0, 1], stacked=False
        )
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Pie chart of time distribution
        axes[1, 0].pie(df['Duration (ms)'], labels=df['Operation'], autopct='%1.1f%%')
        axes[1, 0].set_title('Time Distribution')

        # Resource utilization heatmap
        util_data = df[['CPU %', 'GPU %']].values.T if 'CPU %' in df.columns else np.zeros((2, len(df)))
        sns.heatmap(util_data, annot=True, fmt='.1f', ax=axes[1, 1],
                   xticklabels=df['Operation'], yticklabels=['CPU', 'GPU'],
                   cmap='YlOrRd')
        axes[1, 1].set_title('Resource Utilization (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class LayerProfiler:
    """Profiles individual layers in a model."""

    def __init__(self):
        """Initialize layer profiler."""
        self.layer_stats = defaultdict(lambda: {
            'forward_time': [],
            'backward_time': [],
            'input_shape': None,
            'output_shape': None,
            'parameters': 0,
            'flops': 0
        })

    def profile_layers(self, model: nn.Module, input_shape: Tuple[int, ...]) -> pd.DataFrame:
        """
        Profile individual layers.

        Args:
            model: Model to profile
            input_shape: Input shape

        Returns:
            DataFrame with layer statistics
        """
        hooks = []
        dummy_input = torch.randn(*input_shape)

        def forward_hook(module, input, output, name):
            start = time.perf_counter()

            def backward_hook(grad):
                self.layer_stats[name]['backward_time'].append(
                    (time.perf_counter() - start) * 1000
                )
                return None

            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            self.layer_stats[name]['forward_time'].append(
                (time.perf_counter() - start) * 1000
            )
            self.layer_stats[name]['input_shape'] = input[0].shape if isinstance(input, tuple) else input.shape
            self.layer_stats[name]['output_shape'] = output.shape if isinstance(output, torch.Tensor) else None
            self.layer_stats[name]['parameters'] = sum(p.numel() for p in module.parameters())

        # Register hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: forward_hook(m, i, o, n)
                )
                hooks.append(hook)

        # Forward and backward pass
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Create DataFrame
        data = []
        for name, stats in self.layer_stats.items():
            data.append({
                'Layer': name,
                'Avg Forward (ms)': np.mean(stats['forward_time']) if stats['forward_time'] else 0,
                'Avg Backward (ms)': np.mean(stats['backward_time']) if stats['backward_time'] else 0,
                'Parameters': stats['parameters'],
                'Input Shape': str(stats['input_shape']),
                'Output Shape': str(stats['output_shape'])
            })

        return pd.DataFrame(data).sort_values('Avg Forward (ms)', ascending=False)


class BottleneckAnalyzer:
    """Identifies and analyzes performance bottlenecks."""

    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize bottleneck analyzer.

        Args:
            profiler: Performance profiler instance
        """
        self.profiler = profiler

    def identify_bottlenecks(self, threshold_percentile: float = 75) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percentile: Percentile threshold for bottleneck detection

        Returns:
            List of bottlenecks with recommendations
        """
        if not self.profiler.results:
            return []

        bottlenecks = []

        # Analyze time bottlenecks
        durations = [r.duration_ms for r in self.profiler.results]
        time_threshold = np.percentile(durations, threshold_percentile)

        for result in self.profiler.results:
            if result.duration_ms > time_threshold:
                bottleneck = {
                    'type': 'time',
                    'operation': result.operation,
                    'severity': 'high' if result.duration_ms > np.percentile(durations, 90) else 'medium',
                    'value': result.duration_ms,
                    'recommendation': self._get_time_recommendation(result)
                }
                bottlenecks.append(bottleneck)

        # Analyze memory bottlenecks
        memories = [r.memory_mb for r in self.profiler.results]
        memory_threshold = np.percentile(memories, threshold_percentile)

        for result in self.profiler.results:
            if result.memory_mb > memory_threshold:
                bottleneck = {
                    'type': 'memory',
                    'operation': result.operation,
                    'severity': 'high' if result.memory_mb > np.percentile(memories, 90) else 'medium',
                    'value': result.memory_mb,
                    'recommendation': self._get_memory_recommendation(result)
                }
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _get_time_recommendation(self, result: ProfileResult) -> str:
        """Get recommendation for time bottleneck."""
        recommendations = []

        if 'compression' in result.operation:
            recommendations.append("Consider using incremental compression")
            recommendations.append("Enable GPU acceleration if available")

        if 'forward' in result.operation:
            recommendations.append("Enable mixed precision training")
            recommendations.append("Use gradient checkpointing for large models")

        if result.cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider parallelization")

        return "; ".join(recommendations) if recommendations else "Optimize algorithm efficiency"

    def _get_memory_recommendation(self, result: ProfileResult) -> str:
        """Get recommendation for memory bottleneck."""
        recommendations = []

        if result.memory_mb > 1000:
            recommendations.append("Consider gradient accumulation")
            recommendations.append("Reduce batch size")

        if result.gpu_memory_mb > 0:
            recommendations.append("Enable gradient checkpointing")
            recommendations.append("Use mixed precision training")

        return "; ".join(recommendations) if recommendations else "Optimize memory usage"

    def generate_optimization_plan(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """
        Generate optimization plan based on bottlenecks.

        Args:
            bottlenecks: List of identified bottlenecks

        Returns:
            Optimization plan as string
        """
        plan = "# Performance Optimization Plan\n\n"

        # Group by severity
        high_severity = [b for b in bottlenecks if b['severity'] == 'high']
        medium_severity = [b for b in bottlenecks if b['severity'] == 'medium']

        if high_severity:
            plan += "## High Priority Optimizations\n\n"
            for b in high_severity:
                plan += f"- **{b['operation']}** ({b['type']} bottleneck: {b['value']:.2f})\n"
                plan += f"  - Recommendation: {b['recommendation']}\n\n"

        if medium_severity:
            plan += "## Medium Priority Optimizations\n\n"
            for b in medium_severity:
                plan += f"- **{b['operation']}** ({b['type']} bottleneck: {b['value']:.2f})\n"
                plan += f"  - Recommendation: {b['recommendation']}\n\n"

        # General recommendations
        plan += "## General Recommendations\n\n"
        plan += "1. Profile regularly during development\n"
        plan += "2. Monitor resource utilization in production\n"
        plan += "3. Consider hardware upgrades for persistent bottlenecks\n"
        plan += "4. Implement caching where appropriate\n"
        plan += "5. Use asynchronous operations for I/O-bound tasks\n"

        return plan