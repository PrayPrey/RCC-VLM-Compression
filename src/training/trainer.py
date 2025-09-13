"""
Main training pipeline for RCC compression with mixed precision and distributed support.

This module orchestrates the complete training workflow including compression,
distillation, and evaluation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from typing import Dict, Optional, List, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm
import wandb
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Model configuration
    model_type: str = "clip"
    model_name: str = "openai/clip-vit-base-patch32"

    # Compression configuration
    compression_target: float = 0.995
    compression_stages: List[str] = field(default_factory=lambda: ["dare", "nullu", "alpha"])

    # Training configuration
    num_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Distillation configuration
    use_distillation: bool = True
    temperature: float = 4.0
    alpha_distill: float = 0.7
    feature_distill_weight: float = 0.1
    attention_distill_weight: float = 0.1

    # Optimization configuration
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 1
    eval_frequency: int = 1
    resume_from: Optional[str] = None

    # Logging
    log_frequency: int = 100
    use_wandb: bool = True
    project_name: str = "rcc-compression"
    experiment_name: str = "rcc-vit-base"

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # Evaluation
    eval_batch_size: int = 256
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "retrieval", "compression"])

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


class RCCTrainer:
    """Main trainer for RCC compression pipeline."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize RCC trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Set seeds for reproducibility
        self._set_seeds(config.seed)

        # Initialize distributed training if enabled
        if config.distributed:
            self._init_distributed()

        # Initialize logging
        self._init_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_history = defaultdict(list)

        # Components (to be initialized)
        self.model = None
        self.teacher_model = None
        self.compressor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_distributed(self) -> None:
        """Initialize distributed training."""
        dist.init_process_group(
            backend=self.config.backend,
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f'cuda:{self.config.local_rank}')
        logger.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")

    def _init_logging(self) -> None:
        """Initialize logging and tracking."""
        if self.config.use_wandb and (not self.config.distributed or self.config.rank == 0):
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
            logger.info("Initialized Weights & Biases logging")

    def initialize_models(self, model: nn.Module,
                         teacher_model: Optional[nn.Module] = None) -> None:
        """
        Initialize student and teacher models.

        Args:
            model: Student model to compress
            teacher_model: Teacher model for distillation
        """
        self.model = model.to(self.device)

        if teacher_model is not None and self.config.use_distillation:
            self.teacher_model = teacher_model.to(self.device)
            self.teacher_model.eval()
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        # Wrap model for distributed training
        if self.config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank
            )

        logger.info(f"Initialized models on {self.device}")

    def initialize_compression(self, compressor: nn.Module) -> None:
        """
        Initialize compression module.

        Args:
            compressor: Compression module (cascade compressor)
        """
        self.compressor = compressor
        logger.info("Initialized compression module")

    def initialize_data(self, train_dataset: Any, val_dataset: Any) -> None:
        """
        Initialize data loaders.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.config.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False
            )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )

        logger.info(f"Initialized data loaders: {len(train_dataset)} train, {len(val_dataset)} val")

    def initialize_optimization(self) -> None:
        """Initialize optimizer and scheduler."""
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Create optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Create scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs

        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps
            )
        elif self.config.scheduler == "constant":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        # Initialize mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = GradScaler()

        logger.info(f"Initialized optimizer: {self.config.optimizer} with LR {self.config.learning_rate}")

    def apply_compression(self) -> Dict[str, Any]:
        """
        Apply compression to the model.

        Returns:
            Compression metrics
        """
        logger.info("Applying compression pipeline...")

        compression_metrics = {}

        # Apply compression stages sequentially
        for stage in self.config.compression_stages:
            logger.info(f"Applying {stage} compression...")

            # Get the underlying model (unwrap DDP if needed)
            model_to_compress = self.model.module if hasattr(self.model, 'module') else self.model

            # Apply compression stage
            compressed_model, metrics = self.compressor(model_to_compress, stage=stage)

            # Update model
            if hasattr(self.model, 'module'):
                self.model.module = compressed_model
            else:
                self.model = compressed_model

            compression_metrics[stage] = metrics.to_dict()

        # Log compression results
        total_reduction = compression_metrics.get('alpha', {}).get('parameter_reduction', 0)
        logger.info(f"Compression complete: {total_reduction:.2f}% parameter reduction")

        return compression_metrics

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()

        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_loader)

        # Set epoch for distributed sampler
        if self.config.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(**batch)
                loss = outputs.get('loss', 0)

                # Add distillation loss if enabled
                if self.config.use_distillation and self.teacher_model is not None:
                    distill_loss = self._compute_distillation_loss(batch, outputs)
                    loss = (1 - self.config.alpha_distill) * loss + \
                           self.config.alpha_distill * distill_loss
                    epoch_metrics['distill_loss'] += distill_loss.item()

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Update metrics
            epoch_metrics['loss'] += loss.item() * self.config.gradient_accumulation_steps
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.config.gradient_accumulation_steps,
                'lr': epoch_metrics['lr']
            })

            # Log metrics
            if self.global_step % self.config.log_frequency == 0:
                self._log_metrics({
                    'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                    'train/lr': epoch_metrics['lr'],
                    'train/epoch': self.epoch,
                    'train/step': self.global_step
                })

        # Average metrics
        for key in epoch_metrics:
            if key != 'lr':
                epoch_metrics[key] /= num_batches

        return dict(epoch_metrics)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        eval_metrics = defaultdict(float)
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)

                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', 0)

                eval_metrics['loss'] += loss.item()

                # Compute additional metrics based on outputs
                if 'logits' in outputs:
                    # Classification metrics
                    predictions = outputs['logits'].argmax(dim=-1)
                    if 'labels' in batch:
                        accuracy = (predictions == batch['labels']).float().mean()
                        eval_metrics['accuracy'] += accuracy.item()

                if 'image_embeds' in outputs and 'text_embeds' in outputs:
                    # Retrieval metrics
                    retrieval_metrics = self._compute_retrieval_metrics(
                        outputs['image_embeds'],
                        outputs['text_embeds']
                    )
                    for k, v in retrieval_metrics.items():
                        eval_metrics[k] += v

        # Average metrics
        for key in eval_metrics:
            eval_metrics[key] /= num_batches

        return dict(eval_metrics)

    def _compute_distillation_loss(self, batch: Dict[str, Tensor],
                                  student_outputs: Dict[str, Tensor]) -> Tensor:
        """
        Compute knowledge distillation loss.

        Args:
            batch: Input batch
            student_outputs: Student model outputs

        Returns:
            Distillation loss
        """
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)

        loss = 0.0

        # Response distillation (KL divergence)
        if 'logits' in student_outputs and 'logits' in teacher_outputs:
            student_logits = student_outputs['logits'] / self.config.temperature
            teacher_logits = teacher_outputs['logits'] / self.config.temperature

            loss += nn.functional.kl_div(
                torch.log_softmax(student_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (self.config.temperature ** 2)

        # Feature distillation (MSE)
        if self.config.feature_distill_weight > 0:
            if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
                feature_loss = nn.functional.mse_loss(
                    student_outputs['hidden_states'],
                    teacher_outputs['hidden_states']
                )
                loss += self.config.feature_distill_weight * feature_loss

        return loss

    def _compute_retrieval_metrics(self, image_embeds: Tensor,
                                  text_embeds: Tensor) -> Dict[str, float]:
        """
        Compute image-text retrieval metrics.

        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings

        Returns:
            Retrieval metrics (R@1, R@5, R@10)
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        similarity = torch.matmul(image_embeds, text_embeds.t())

        # Image-to-text retrieval
        _, indices = similarity.topk(10, dim=1)
        labels = torch.arange(len(similarity), device=similarity.device).unsqueeze(1)

        i2t_r1 = (indices[:, :1] == labels).any(dim=1).float().mean().item()
        i2t_r5 = (indices[:, :5] == labels).any(dim=1).float().mean().item()
        i2t_r10 = (indices == labels).any(dim=1).float().mean().item()

        # Text-to-image retrieval
        _, indices = similarity.t().topk(10, dim=1)

        t2i_r1 = (indices[:, :1] == labels).any(dim=1).float().mean().item()
        t2i_r5 = (indices[:, :5] == labels).any(dim=1).float().mean().item()
        t2i_r10 = (indices == labels).any(dim=1).float().mean().item()

        return {
            'i2t_r1': i2t_r1,
            'i2t_r5': i2t_r5,
            'i2t_r10': i2t_r10,
            't2i_r1': t2i_r1,
            't2i_r5': t2i_r5,
            't2i_r10': t2i_r10
        }

    def train(self) -> None:
        """Run the complete training pipeline."""
        logger.info("Starting training...")

        # Apply compression first
        if self.compressor is not None:
            compression_metrics = self.apply_compression()
            self._log_metrics({'compression': compression_metrics})

            # Re-initialize optimizer after compression
            self.initialize_optimization()

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            self.training_history['train'].append(train_metrics)

            logger.info(f"Epoch {epoch + 1} - Train loss: {train_metrics['loss']:.4f}")

            # Evaluate
            if (epoch + 1) % self.config.eval_frequency == 0:
                eval_metrics = self.evaluate()
                self.training_history['eval'].append(eval_metrics)

                logger.info(f"Epoch {epoch + 1} - Eval loss: {eval_metrics['loss']:.4f}")

                # Check for best model
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self.save_checkpoint(best=True)

            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint()

        logger.info("Training complete!")

    def save_checkpoint(self, best: bool = False) -> None:
        """
        Save training checkpoint.

        Args:
            best: Whether this is the best model so far
        """
        if self.config.distributed and self.config.rank != 0:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = "best" if best else f"epoch_{self.epoch + 1}"
        checkpoint_path = checkpoint_dir / f"checkpoint_{suffix}.pt"

        # Get model state dict (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') \
                     else self.model.state_dict()

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

        if self.config.mixed_precision and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler if using mixed precision
        if self.config.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', defaultdict(list))

        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {self.epoch})")

    def _move_batch_to_device(self, batch: Union[Dict, Tuple, Tensor]) -> Union[Dict, Tuple, Tensor]:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to wandb and console."""
        if self.config.use_wandb and (not self.config.distributed or self.config.rank == 0):
            wandb.log(metrics, step=self.global_step)

    def analyze_compression_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact of compression on model performance.

        Returns:
            Analysis results including performance drops and efficiency gains
        """
        analysis = {
            'compression_ratio': 0.0,
            'performance_retention': {},
            'efficiency_metrics': {},
            'layer_analysis': {}
        }

        # Calculate compression ratio
        if hasattr(self.model, 'get_compression_statistics'):
            stats = self.model.get_compression_statistics()
            analysis['compression_ratio'] = stats.get('compression_ratio', 0.0)
            analysis['layer_analysis'] = stats.get('layer_statistics', {})

        # Evaluate performance retention
        if self.eval_loader is not None:
            eval_metrics = self.evaluate()
            if hasattr(self, 'baseline_metrics'):
                for key in eval_metrics:
                    if key in self.baseline_metrics:
                        retention = eval_metrics[key] / self.baseline_metrics[key]
                        analysis['performance_retention'][key] = retention

        # Measure efficiency metrics
        analysis['efficiency_metrics'] = self._measure_efficiency()

        return analysis

    def _measure_efficiency(self) -> Dict[str, float]:
        """
        Measure efficiency metrics like inference speed and memory usage.

        Returns:
            Efficiency metrics
        """
        import time
        import torch.cuda

        metrics = {}

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        # Measure inference time
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self.model(dummy_input)

            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(100):
                _ = self.model(dummy_input)

            torch.cuda.synchronize()
            end_time = time.time()

            metrics['inference_time_ms'] = (end_time - start_time) / 100 * 1000

        # Measure memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = self.model(dummy_input)

            metrics['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Calculate FLOPs if available
        try:
            from thop import profile
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            metrics['flops'] = flops
            metrics['params'] = params
        except ImportError:
            pass

        return metrics

    def visualize_training_progress(self, save_path: Optional[str] = None) -> None:
        """
        Visualize training progress including loss curves and metrics.

        Args:
            save_path: Optional path to save the visualization
        """
        import matplotlib.pyplot as plt

        if not self.training_history['train']:
            logger.warning("No training history to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        train_losses = [m['loss'] for m in self.training_history['train']]
        epochs = range(1, len(train_losses) + 1)

        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
        if self.training_history['eval']:
            eval_losses = [m['loss'] for m in self.training_history['eval']]
            eval_epochs = range(1, len(eval_losses) + 1)
            axes[0, 0].plot(eval_epochs, eval_losses, 'r-', label='Eval Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        if 'learning_rate' in self.training_history['train'][0]:
            lrs = [m['learning_rate'] for m in self.training_history['train']]
            axes[0, 1].plot(epochs, lrs, 'g-')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True, alpha=0.3)

        # Compression metrics
        if hasattr(self, 'compression_history') and self.compression_history:
            stages = list(self.compression_history.keys())
            compressions = [self.compression_history[s].get('compression_ratio', 0) for s in stages]
            axes[1, 0].bar(stages, compressions, color='steelblue')
            axes[1, 0].set_xlabel('Compression Stage')
            axes[1, 0].set_ylabel('Compression Ratio')
            axes[1, 0].set_title('Compression per Stage')
            axes[1, 0].grid(True, alpha=0.3)

        # Performance metrics
        if self.training_history['eval'] and 'accuracy' in self.training_history['eval'][0]:
            accuracies = [m['accuracy'] for m in self.training_history['eval']]
            axes[1, 1].plot(eval_epochs, accuracies, 'purple', marker='o')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Evaluation Accuracy')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training visualization saved to {save_path}")

        plt.show()


# ============= TEST CODE SECTION =============
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestTrainingConfig(unittest.TestCase):
    """Unit tests for TrainingConfig."""

    def test_config_creation(self):
        """Test configuration dataclass creation."""
        config = TrainingConfig(
            model_type="clip",
            num_epochs=10,
            batch_size=64
        )

        self.assertEqual(config.model_type, "clip")
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.compression_target, 0.995)  # Default value

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TrainingConfig()

        self.assertEqual(config.compression_stages, ["dare", "nullu", "alpha"])
        self.assertTrue(config.use_distillation)
        self.assertEqual(config.temperature, 4.0)


class TestRCCTrainer(unittest.TestCase):
    """Unit tests for RCCTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TrainingConfig(
            num_epochs=2,
            batch_size=32,
            distributed=False,
            use_wandb=False
        )

        # Mock model
        self.mock_model = Mock()
        self.mock_model.parameters = Mock(return_value=[torch.randn(10, 10)])
        self.mock_model.to = Mock(return_value=self.mock_model)

        # Mock data loaders
        self.mock_train_loader = Mock()
        self.mock_eval_loader = Mock()

    @patch('src.training.trainer.wandb')
    def test_trainer_initialization(self, mock_wandb):
        """Test trainer initialization."""
        with patch.object(RCCTrainer, 'setup_distributed'):
            with patch.object(RCCTrainer, 'initialize_model', return_value=self.mock_model):
                with patch.object(RCCTrainer, 'initialize_data_loaders',
                                return_value=(self.mock_train_loader, self.mock_eval_loader)):
                    trainer = RCCTrainer(self.config)

                    self.assertIsNotNone(trainer.model)
                    self.assertIsNotNone(trainer.train_loader)
                    self.assertIsNotNone(trainer.eval_loader)
                    self.assertEqual(trainer.epoch, 0)
                    self.assertEqual(trainer.global_step, 0)

    def test_distillation_loss(self):
        """Test distillation loss computation."""
        trainer = Mock()
        trainer.config = self.config

        student_outputs = {
            'logits': torch.randn(32, 10),
            'hidden_states': torch.randn(32, 512)
        }
        teacher_outputs = {
            'logits': torch.randn(32, 10),
            'hidden_states': torch.randn(32, 512)
        }

        # Manually call the method
        loss = RCCTrainer.compute_distillation_loss(
            trainer,
            student_outputs,
            teacher_outputs
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss

    def test_retrieval_metrics(self):
        """Test retrieval metrics computation."""
        trainer = Mock()

        image_embeds = torch.randn(10, 512)
        text_embeds = torch.randn(10, 512)

        metrics = RCCTrainer._compute_retrieval_metrics(
            trainer,
            image_embeds,
            text_embeds
        )

        self.assertIn('i2t_r1', metrics)
        self.assertIn('t2i_r5', metrics)
        self.assertIsInstance(metrics['i2t_r1'], float)
        self.assertGreaterEqual(metrics['i2t_r1'], 0)
        self.assertLessEqual(metrics['i2t_r1'], 1)

    def test_analyze_compression_impact(self):
        """Test compression impact analysis."""
        trainer = Mock()
        trainer.model = Mock()
        trainer.model.get_compression_statistics = Mock(return_value={
            'compression_ratio': 0.95,
            'layer_statistics': {'layer1': 0.9, 'layer2': 0.85}
        })
        trainer.eval_loader = self.mock_eval_loader
        trainer.evaluate = Mock(return_value={'accuracy': 0.92, 'loss': 0.5})
        trainer.baseline_metrics = {'accuracy': 0.95, 'loss': 0.4}
        trainer._measure_efficiency = Mock(return_value={'inference_time_ms': 5.0})

        analysis = RCCTrainer.analyze_compression_impact(trainer)

        self.assertEqual(analysis['compression_ratio'], 0.95)
        self.assertIn('accuracy', analysis['performance_retention'])
        self.assertAlmostEqual(analysis['performance_retention']['accuracy'], 0.92/0.95, places=5)
        self.assertEqual(analysis['efficiency_metrics']['inference_time_ms'], 5.0)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_training_progress(self, mock_savefig, mock_show):
        """Test training visualization."""
        trainer = Mock()
        trainer.training_history = {
            'train': [
                {'loss': 1.0, 'learning_rate': 0.001},
                {'loss': 0.8, 'learning_rate': 0.0008},
                {'loss': 0.6, 'learning_rate': 0.0005}
            ],
            'eval': [
                {'loss': 0.9, 'accuracy': 0.7},
                {'loss': 0.7, 'accuracy': 0.8},
                {'loss': 0.5, 'accuracy': 0.85}
            ]
        }
        trainer.compression_history = {
            'dare': {'compression_ratio': 0.9},
            'nullu': {'compression_ratio': 0.5},
            'alpha': {'compression_ratio': 0.1}
        }

        RCCTrainer.visualize_training_progress(trainer, save_path='test.png')

        mock_savefig.assert_called_once()
        mock_show.assert_called_once()


class TestIntegrationTraining(unittest.TestCase):
    """Integration tests for training pipeline."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_training_step(self, mock_cuda):
        """Test single training step."""
        config = TrainingConfig(
            batch_size=8,
            distributed=False,
            use_wandb=False,
            mixed_precision=False,
            device='cpu'
        )

        # Create mock trainer
        trainer = Mock()
        trainer.config = config
        trainer.device = torch.device('cpu')
        trainer.global_step = 0

        # Mock model
        model = Mock()
        model.train = Mock()
        outputs = {
            'loss': torch.tensor(1.0),
            'logits': torch.randn(8, 10)
        }
        model.return_value = outputs
        trainer.model = model

        # Mock optimizer and scaler
        trainer.optimizer = Mock()
        trainer.scaler = None

        # Mock batch
        batch = {
            'images': torch.randn(8, 3, 224, 224),
            'labels': torch.randint(0, 10, (8,))
        }

        # Run training step
        trainer._move_batch_to_device = Mock(return_value=batch)
        metrics = RCCTrainer.train_step(trainer, batch)

        self.assertIn('loss', metrics)
        trainer.optimizer.step.assert_called_once()


def run_trainer_tests():
    """Run all trainer tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)