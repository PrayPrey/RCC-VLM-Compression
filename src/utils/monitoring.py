"""
Monitoring utilities with W&B integration.

This module provides monitoring and tracking capabilities
for experiments and model training.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import psutil
import GPUtil
from datetime import datetime
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ExperimentMonitor:
    """
    Comprehensive experiment monitoring with W&B.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        resume: bool = False
    ):
        """
        Initialize experiment monitor.

        Args:
            project: W&B project name
            name: Experiment name
            config: Configuration dictionary
            tags: Experiment tags
            notes: Experiment notes
            resume: Whether to resume from previous run
        """
        self.project = project
        self.name = name or f"rcc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize W&B
        self.run = wandb.init(
            project=project,
            name=self.name,
            config=config,
            tags=tags or ["rcc", "compression"],
            notes=notes,
            resume=resume
        )

        # Track system metrics
        self.system_monitor = SystemMonitor()

        # Compression tracking
        self.compression_history = []
        self.performance_history = []

        logger.info(f"Initialized W&B monitoring - Project: {project}, Run: {self.name}")

    def log_compression_stage(
        self,
        stage_name: str,
        metrics: Dict[str, Any],
        model: Optional[torch.nn.Module] = None
    ):
        """
        Log compression stage results.

        Args:
            stage_name: Name of compression stage
            metrics: Stage metrics
            model: Model instance (optional)
        """
        # Add stage prefix to metrics
        stage_metrics = {f"{stage_name}/{k}": v for k, v in metrics.items()}

        # Add to history
        self.compression_history.append({
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })

        # Log to W&B
        wandb.log(stage_metrics)

        # Log model if provided
        if model:
            self.log_model_stats(model, prefix=stage_name)

        logger.info(f"Logged compression stage: {stage_name}")

    def log_training_step(
        self,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ):
        """
        Log training step.

        Args:
            step: Training step
            loss: Loss value
            metrics: Additional metrics
            learning_rate: Current learning rate
        """
        log_dict = {
            'train/loss': loss,
            'train/step': step
        }

        if metrics:
            for k, v in metrics.items():
                log_dict[f'train/{k}'] = v

        if learning_rate:
            log_dict['train/learning_rate'] = learning_rate

        # Add system metrics
        system_stats = self.system_monitor.get_current_stats()
        for k, v in system_stats.items():
            log_dict[f'system/{k}'] = v

        wandb.log(log_dict, step=step)

    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best_metric: Optional[float] = None
    ):
        """
        Log validation results.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            best_metric: Best metric value so far
        """
        val_metrics = {f'val/{k}': v for k, v in metrics.items()}
        val_metrics['epoch'] = epoch

        if best_metric is not None:
            val_metrics['val/best_metric'] = best_metric

        wandb.log(val_metrics)

        # Update performance history
        self.performance_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def log_model_stats(
        self,
        model: torch.nn.Module,
        prefix: str = "model"
    ):
        """
        Log model statistics.

        Args:
            model: Model instance
            prefix: Metric prefix
        """
        stats = self._compute_model_stats(model)

        log_dict = {f'{prefix}/{k}': v for k, v in stats.items()}
        wandb.log(log_dict)

    def _compute_model_stats(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Compute model statistics.

        Args:
            model: Model instance

        Returns:
            Model statistics
        """
        total_params = 0
        trainable_params = 0
        non_zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            non_zero_params += (param != 0).sum().item()

        # Compute model size
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_zero_parameters': non_zero_params,
            'sparsity': 1 - (non_zero_params / total_params),
            'model_size_mb': model_size_mb,
            'compression_ratio': total_params / non_zero_params if non_zero_params > 0 else 0
        }

    def log_images(
        self,
        images: Dict[str, Union[torch.Tensor, np.ndarray, plt.Figure]],
        step: Optional[int] = None
    ):
        """
        Log images to W&B.

        Args:
            images: Dictionary of images
            step: Current step
        """
        wandb_images = {}

        for name, image in images.items():
            if isinstance(image, torch.Tensor):
                # Convert tensor to numpy
                if image.dim() == 4:  # Batch
                    image = image[0]
                if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()

            if isinstance(image, np.ndarray):
                wandb_images[name] = wandb.Image(image)
            elif isinstance(image, plt.Figure):
                wandb_images[name] = wandb.Image(image)

        wandb.log(wandb_images, step=step)

    def log_table(
        self,
        name: str,
        data: List[List[Any]],
        columns: List[str]
    ):
        """
        Log table to W&B.

        Args:
            name: Table name
            data: Table data
            columns: Column names
        """
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        name: str,
        aliases: Optional[List[str]] = None
    ):
        """
        Save checkpoint as W&B artifact.

        Args:
            checkpoint: Checkpoint dictionary
            name: Artifact name
            aliases: Artifact aliases
        """
        artifact = wandb.Artifact(
            name=name,
            type='model',
            metadata=checkpoint.get('metadata', {})
        )

        # Save checkpoint to temporary file
        temp_path = Path(f"/tmp/{name}.pt")
        torch.save(checkpoint, temp_path)

        # Add to artifact
        artifact.add_file(temp_path)

        # Log artifact
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])

        # Clean up
        temp_path.unlink()

        logger.info(f"Saved checkpoint as artifact: {name}")

    def finish(self, summary: Optional[Dict[str, Any]] = None):
        """
        Finish monitoring and close W&B run.

        Args:
            summary: Final summary metrics
        """
        if summary:
            for k, v in summary.items():
                wandb.summary[k] = v

        # Log compression summary
        if self.compression_history:
            final_compression = self.compression_history[-1]['metrics']
            wandb.summary['final_compression_ratio'] = final_compression.get('compression_ratio', 0)
            wandb.summary['final_sparsity'] = final_compression.get('sparsity', 0)

        # Log performance summary
        if self.performance_history:
            best_performance = max(self.performance_history,
                                  key=lambda x: x['metrics'].get('accuracy', 0))
            wandb.summary['best_accuracy'] = best_performance['metrics'].get('accuracy', 0)
            wandb.summary['best_epoch'] = best_performance['epoch']

        wandb.finish()
        logger.info("Experiment monitoring finished")


class SystemMonitor:
    """
    Monitor system resources.
    """

    def __init__(self):
        """Initialize system monitor."""
        self.cpu_percent_history = []
        self.memory_percent_history = []
        self.gpu_memory_history = []

    def get_current_stats(self) -> Dict[str, float]:
        """
        Get current system statistics.

        Returns:
            Dictionary of system stats
        """
        stats = {}

        # CPU stats
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        stats['cpu_count'] = psutil.cpu_count()

        # Memory stats
        memory = psutil.virtual_memory()
        stats['memory_percent'] = memory.percent
        stats['memory_used_gb'] = memory.used / (1024 ** 3)
        stats['memory_available_gb'] = memory.available / (1024 ** 3)

        # GPU stats (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                stats['gpu_memory_percent'] = gpu.memoryUtil * 100
                stats['gpu_memory_used_mb'] = gpu.memoryUsed
                stats['gpu_temperature'] = gpu.temperature
                stats['gpu_utilization'] = gpu.load * 100
        except:
            pass

        # PyTorch CUDA stats
        if torch.cuda.is_available():
            stats['cuda_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['cuda_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)

        return stats

    def log_summary(self) -> Dict[str, Any]:
        """
        Get summary of system usage.

        Returns:
            System usage summary
        """
        current = self.get_current_stats()

        summary = {
            'cpu_average': np.mean(self.cpu_percent_history) if self.cpu_percent_history else current.get('cpu_percent', 0),
            'cpu_peak': max(self.cpu_percent_history) if self.cpu_percent_history else current.get('cpu_percent', 0),
            'memory_average': np.mean(self.memory_percent_history) if self.memory_percent_history else current.get('memory_percent', 0),
            'memory_peak': max(self.memory_percent_history) if self.memory_percent_history else current.get('memory_percent', 0),
        }

        if self.gpu_memory_history:
            summary['gpu_memory_average'] = np.mean(self.gpu_memory_history)
            summary['gpu_memory_peak'] = max(self.gpu_memory_history)

        return summary


def log_compression_cascade(
    stages: List[str],
    metrics: List[Dict[str, float]],
    project: str = "rcc-compression"
):
    """
    Log complete compression cascade to W&B.

    Args:
        stages: List of stage names
        metrics: List of metrics per stage
        project: W&B project name
    """
    with wandb.init(project=project, name="cascade_summary") as run:
        # Create summary table
        data = []
        for stage, stage_metrics in zip(stages, metrics):
            data.append([
                stage,
                stage_metrics.get('compression_ratio', 0),
                stage_metrics.get('sparsity', 0),
                stage_metrics.get('performance', 0),
                stage_metrics.get('inference_time_ms', 0)
            ])

        table = wandb.Table(
            columns=['Stage', 'Compression Ratio', 'Sparsity', 'Performance', 'Inference (ms)'],
            data=data
        )

        run.log({'cascade_summary': table})

        # Log final metrics
        final_metrics = metrics[-1] if metrics else {}
        for k, v in final_metrics.items():
            wandb.summary[f'final_{k}'] = v

        logger.info("Logged compression cascade to W&B")