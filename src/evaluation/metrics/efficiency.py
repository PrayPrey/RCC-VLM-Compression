"""
Efficiency metrics for model compression evaluation.

This module provides metrics for measuring computational efficiency,
memory usage, and inference speed of compressed models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any, Union
import time
import numpy as np
import logging
from dataclasses import dataclass
import psutil
import GPUtil
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""
    model_size_mb: float
    param_count: int
    compression_ratio: float
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    flops: Optional[int] = None
    mac_operations: Optional[int] = None
    energy_consumption: Optional[float] = None


class EfficiencyProfiler:
    """Profiles model efficiency metrics."""

    def __init__(self,
                 device: str = "cuda",
                 warmup_iterations: int = 10,
                 measurement_iterations: int = 100):
        """
        Initialize efficiency profiler.

        Args:
            device: Device to run profiling on
            warmup_iterations: Warmup iterations before measurement
            measurement_iterations: Number of measurement iterations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations

    def profile_model(self,
                     model: nn.Module,
                     input_shape: Tuple[int, ...],
                     original_model: Optional[nn.Module] = None) -> EfficiencyMetrics:
        """
        Profile model efficiency.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            original_model: Original model for comparison

        Returns:
            Efficiency metrics
        """
        model = model.to(self.device)
        model.eval()

        # Model size and parameters
        model_size = self._get_model_size(model)
        param_count = self._count_parameters(model)

        # Compression ratio
        if original_model is not None:
            original_params = self._count_parameters(original_model)
            compression_ratio = original_params / param_count if param_count > 0 else float('inf')
        else:
            compression_ratio = 1.0

        # Inference time
        inference_time, throughput = self._measure_inference_time(model, input_shape)

        # Memory usage
        memory_usage = self._measure_memory_usage(model, input_shape)

        # FLOPs (optional)
        flops = self._estimate_flops(model, input_shape)

        return EfficiencyMetrics(
            model_size_mb=model_size,
            param_count=param_count,
            compression_ratio=compression_ratio,
            inference_time_ms=inference_time,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=memory_usage,
            flops=flops
        )

    def _get_model_size(self, model: nn.Module) -> float:
        """
        Get model size in MB.

        Args:
            model: Model to measure

        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def _count_parameters(self, model: nn.Module) -> int:
        """
        Count model parameters.

        Args:
            model: Model to count

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in model.parameters())

    def _measure_inference_time(self,
                               model: nn.Module,
                               input_shape: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Measure inference time.

        Args:
            model: Model to measure
            input_shape: Input shape

        Returns:
            Inference time in ms and throughput
        """
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(dummy_input)

        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(self.measurement_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = model(dummy_input)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(times) * 1000  # Convert to ms
        batch_size = input_shape[0] if len(input_shape) > 0 else 1
        throughput = batch_size / np.mean(times)  # Samples per second

        return mean_time, throughput

    def _measure_memory_usage(self,
                            model: nn.Module,
                            input_shape: Tuple[int, ...]) -> float:
        """
        Measure memory usage.

        Args:
            model: Model to measure
            input_shape: Input shape

        Returns:
            Memory usage in MB
        """
        if self.device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create input
            dummy_input = torch.randn(*input_shape).to(self.device)

            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)

            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            # Clear cache
            torch.cuda.empty_cache()

            return peak_memory
        else:
            # CPU memory measurement
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024

            dummy_input = torch.randn(*input_shape)
            with torch.no_grad():
                _ = model(dummy_input)

            memory_after = process.memory_info().rss / 1024 / 1024
            return memory_after - memory_before

    def _estimate_flops(self,
                       model: nn.Module,
                       input_shape: Tuple[int, ...]) -> Optional[int]:
        """
        Estimate FLOPs for model.

        Args:
            model: Model to analyze
            input_shape: Input shape

        Returns:
            Estimated FLOPs
        """
        try:
            from thop import profile
            dummy_input = torch.randn(*input_shape).to(self.device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            return int(flops)
        except ImportError:
            logger.debug("thop not installed, skipping FLOPs calculation")
            return None
        except Exception as e:
            logger.debug(f"Failed to calculate FLOPs: {e}")
            return None


class CompressionEfficiencyAnalyzer:
    """Analyzes efficiency gains from compression."""

    def __init__(self):
        """Initialize compression efficiency analyzer."""
        self.profiler = EfficiencyProfiler()

    def compare_models(self,
                      original_model: nn.Module,
                      compressed_model: nn.Module,
                      input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Compare efficiency of original vs compressed model.

        Args:
            original_model: Original model
            compressed_model: Compressed model
            input_shape: Input shape

        Returns:
            Comparison metrics
        """
        # Profile both models
        original_metrics = self.profiler.profile_model(original_model, input_shape)
        compressed_metrics = self.profiler.profile_model(compressed_model, input_shape, original_model)

        # Calculate improvements
        comparison = {
            'original': self._metrics_to_dict(original_metrics),
            'compressed': self._metrics_to_dict(compressed_metrics),
            'improvements': {
                'size_reduction': 1 - (compressed_metrics.model_size_mb / original_metrics.model_size_mb),
                'param_reduction': 1 - (compressed_metrics.param_count / original_metrics.param_count),
                'speedup': original_metrics.inference_time_ms / compressed_metrics.inference_time_ms,
                'memory_reduction': 1 - (compressed_metrics.memory_usage_mb / original_metrics.memory_usage_mb),
                'throughput_increase': compressed_metrics.throughput_samples_per_sec / original_metrics.throughput_samples_per_sec
            }
        }

        # Add FLOPs comparison if available
        if original_metrics.flops and compressed_metrics.flops:
            comparison['improvements']['flops_reduction'] = 1 - (compressed_metrics.flops / original_metrics.flops)

        return comparison

    def _metrics_to_dict(self, metrics: EfficiencyMetrics) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'model_size_mb': metrics.model_size_mb,
            'param_count': metrics.param_count,
            'compression_ratio': metrics.compression_ratio,
            'inference_time_ms': metrics.inference_time_ms,
            'throughput_samples_per_sec': metrics.throughput_samples_per_sec,
            'memory_usage_mb': metrics.memory_usage_mb,
            'flops': metrics.flops
        }


@contextmanager
def track_memory():
    """
    Context manager to track memory usage.

    Yields:
        Dictionary to store memory stats
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    stats = {}
    yield stats

    if torch.cuda.is_available():
        stats['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        stats['current_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        torch.cuda.empty_cache()


def benchmark_model_latency(model: nn.Module,
                           input_shapes: List[Tuple[int, ...]],
                           batch_sizes: List[int] = [1, 8, 32, 128],
                           device: str = "cuda") -> Dict[str, List[float]]:
    """
    Benchmark model latency across different batch sizes.

    Args:
        model: Model to benchmark
        input_shapes: List of input shapes
        batch_sizes: Batch sizes to test
        device: Device to run on

    Returns:
        Latency measurements
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        latencies = []

        for base_shape in input_shapes:
            # Adjust shape for batch size
            shape = (batch_size,) + base_shape[1:]
            dummy_input = torch.randn(*shape).to(device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Measure
            if device.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.perf_counter()
                    _ = model(dummy_input)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # ms

            latencies.append(np.mean(times))

        results[f'batch_{batch_size}'] = latencies

    return results


def calculate_compression_metrics(original_model: nn.Module,
                                 compressed_model: nn.Module) -> Dict[str, float]:
    """
    Calculate comprehensive compression metrics.

    Args:
        original_model: Original model
        compressed_model: Compressed model

    Returns:
        Compression metrics
    """
    # Count parameters
    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())

    # Count non-zero parameters (for sparse models)
    original_nonzero = sum((p != 0).sum().item() for p in original_model.parameters())
    compressed_nonzero = sum((p != 0).sum().item() for p in compressed_model.parameters())

    # Calculate sizes
    original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / 1024 / 1024
    compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / 1024 / 1024

    metrics = {
        'compression_ratio': original_params / compressed_params if compressed_params > 0 else float('inf'),
        'sparsity': 1 - (compressed_nonzero / original_nonzero) if original_nonzero > 0 else 0,
        'size_reduction_mb': original_size - compressed_size,
        'size_reduction_percent': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'parameter_reduction': original_params - compressed_params,
        'parameter_reduction_percent': (1 - compressed_params / original_params) * 100 if original_params > 0 else 0
    }

    return metrics