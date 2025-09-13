"""
Teacher-student setup for knowledge distillation.

This module implements the teacher-student framework for model compression
with various distillation strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import logging
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class TeacherStudentConfig:
    """Configuration for teacher-student distillation."""
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3  # Weight for task loss
    feature_matching: bool = True
    attention_matching: bool = True
    hidden_states_matching: bool = False
    layer_mapping: Optional[Dict[int, int]] = None  # Map student to teacher layers
    distance_metric: str = "mse"  # mse, cosine, kl
    freeze_teacher: bool = True


class TeacherStudentPair:
    """
    Manages teacher-student model pair for distillation.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: TeacherStudentConfig
    ):
        """
        Initialize teacher-student pair.

        Args:
            teacher: Teacher model (usually larger)
            student: Student model (compressed)
            config: Distillation configuration
        """
        self.teacher = teacher
        self.student = student
        self.config = config

        # Freeze teacher if specified
        if config.freeze_teacher:
            self.freeze_teacher()

        # Set up layer mapping
        self.layer_mapping = config.layer_mapping or self._create_default_mapping()

        # Hooks for intermediate features
        self.teacher_features = {}
        self.student_features = {}
        self._register_hooks()

    def freeze_teacher(self):
        """Freeze teacher model parameters."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        logger.info("Teacher model frozen")

    def _create_default_mapping(self) -> Dict[int, int]:
        """
        Create default layer mapping (assumes same architecture).

        Returns:
            Layer mapping dictionary
        """
        # Get number of layers in each model
        teacher_layers = self._count_layers(self.teacher)
        student_layers = self._count_layers(self.student)

        if student_layers == teacher_layers:
            # Direct 1-to-1 mapping
            return {i: i for i in range(student_layers)}
        else:
            # Proportional mapping
            mapping = {}
            for i in range(student_layers):
                teacher_idx = int(i * teacher_layers / student_layers)
                mapping[i] = teacher_idx
            return mapping

    def _count_layers(self, model: nn.Module) -> int:
        """
        Count the number of layers in a model.

        Args:
            model: Model to count layers

        Returns:
            Number of layers
        """
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.TransformerEncoderLayer)):
                count += 1
        return count

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        if not (self.config.feature_matching or self.config.attention_matching):
            return

        # Register teacher hooks
        for name, module in self.teacher.named_modules():
            if self._should_capture(name, module):
                module.register_forward_hook(
                    lambda m, inp, out, n=name: self._save_teacher_feature(n, out)
                )

        # Register student hooks
        for name, module in self.student.named_modules():
            if self._should_capture(name, module):
                module.register_forward_hook(
                    lambda m, inp, out, n=name: self._save_student_feature(n, out)
                )

    def _should_capture(self, name: str, module: nn.Module) -> bool:
        """
        Determine if features should be captured from this module.

        Args:
            name: Module name
            module: Module instance

        Returns:
            Whether to capture features
        """
        # Capture from attention layers
        if self.config.attention_matching and 'attention' in name.lower():
            return True

        # Capture from projection layers
        if self.config.feature_matching and any(x in name.lower() for x in ['proj', 'fc', 'linear']):
            return True

        # Capture from encoder layers
        if self.config.hidden_states_matching and 'encoder' in name.lower():
            return True

        return False

    def _save_teacher_feature(self, name: str, output):
        """Save teacher feature."""
        if isinstance(output, tuple):
            output = output[0]
        self.teacher_features[name] = output.detach()

    def _save_student_feature(self, name: str, output):
        """Save student feature."""
        if isinstance(output, tuple):
            output = output[0]
        self.student_features[name] = output

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute various distillation losses.

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels (optional)

        Returns:
            Dictionary of losses
        """
        losses = {}

        # KL divergence loss (main distillation loss)
        T = self.config.temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T ** 2)
        losses['kl_loss'] = kl_loss

        # Task loss (if labels provided)
        if labels is not None:
            task_loss = F.cross_entropy(student_logits, labels)
            losses['task_loss'] = task_loss

        # Feature matching loss
        if self.config.feature_matching and self.student_features:
            feature_loss = self._compute_feature_loss()
            losses['feature_loss'] = feature_loss

        # Attention matching loss
        if self.config.attention_matching:
            attention_loss = self._compute_attention_loss()
            if attention_loss is not None:
                losses['attention_loss'] = attention_loss

        # Hidden states matching loss
        if self.config.hidden_states_matching:
            hidden_loss = self._compute_hidden_states_loss()
            if hidden_loss is not None:
                losses['hidden_loss'] = hidden_loss

        # Combine losses
        total_loss = self.config.alpha * kl_loss
        if 'task_loss' in losses:
            total_loss += self.config.beta * losses['task_loss']
        if 'feature_loss' in losses:
            total_loss += 0.1 * losses['feature_loss']
        if 'attention_loss' in losses:
            total_loss += 0.1 * losses['attention_loss']

        losses['total_loss'] = total_loss

        return losses

    def _compute_feature_loss(self) -> torch.Tensor:
        """
        Compute feature matching loss.

        Returns:
            Feature loss
        """
        total_loss = 0.0
        count = 0

        for student_name, student_feat in self.student_features.items():
            # Find corresponding teacher feature
            teacher_name = self._find_teacher_layer(student_name)
            if teacher_name in self.teacher_features:
                teacher_feat = self.teacher_features[teacher_name]

                # Compute distance
                if self.config.distance_metric == "mse":
                    loss = F.mse_loss(student_feat, teacher_feat)
                elif self.config.distance_metric == "cosine":
                    loss = 1 - F.cosine_similarity(
                        student_feat.flatten(start_dim=1),
                        teacher_feat.flatten(start_dim=1)
                    ).mean()
                else:
                    loss = F.mse_loss(student_feat, teacher_feat)

                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

    def _compute_attention_loss(self) -> Optional[torch.Tensor]:
        """
        Compute attention matching loss.

        Returns:
            Attention loss or None
        """
        attention_pairs = []

        for student_name, student_feat in self.student_features.items():
            if 'attention' not in student_name.lower():
                continue

            teacher_name = self._find_teacher_layer(student_name)
            if teacher_name in self.teacher_features:
                attention_pairs.append((
                    student_feat,
                    self.teacher_features[teacher_name]
                ))

        if not attention_pairs:
            return None

        total_loss = 0.0
        for student_attn, teacher_attn in attention_pairs:
            # Normalize attention weights
            if len(student_attn.shape) > 2:
                student_attn = student_attn.mean(dim=1)  # Average over heads
            if len(teacher_attn.shape) > 2:
                teacher_attn = teacher_attn.mean(dim=1)

            # Compute attention loss
            loss = F.mse_loss(student_attn, teacher_attn)
            total_loss += loss

        return total_loss / len(attention_pairs)

    def _compute_hidden_states_loss(self) -> Optional[torch.Tensor]:
        """
        Compute hidden states matching loss.

        Returns:
            Hidden states loss or None
        """
        hidden_pairs = []

        for student_name, student_feat in self.student_features.items():
            if 'hidden' not in student_name.lower() and 'encoder' not in student_name.lower():
                continue

            teacher_name = self._find_teacher_layer(student_name)
            if teacher_name in self.teacher_features:
                hidden_pairs.append((
                    student_feat,
                    self.teacher_features[teacher_name]
                ))

        if not hidden_pairs:
            return None

        total_loss = 0.0
        for student_hidden, teacher_hidden in hidden_pairs:
            # Project if dimensions don't match
            if student_hidden.shape != teacher_hidden.shape:
                # Use adaptive pooling or linear projection
                if len(student_hidden.shape) == 3:  # Sequence data
                    student_hidden = F.adaptive_avg_pool1d(
                        student_hidden.transpose(1, 2),
                        teacher_hidden.shape[1]
                    ).transpose(1, 2)

            loss = F.mse_loss(student_hidden, teacher_hidden)
            total_loss += loss

        return total_loss / len(hidden_pairs)

    def _find_teacher_layer(self, student_layer_name: str) -> str:
        """
        Find corresponding teacher layer for a student layer.

        Args:
            student_layer_name: Student layer name

        Returns:
            Teacher layer name
        """
        # Simple heuristic: replace 'student' with 'teacher' if present
        teacher_name = student_layer_name.replace('student', 'teacher')

        # If not found, try to match by structure
        if teacher_name not in self.teacher_features:
            for t_name in self.teacher_features.keys():
                if student_layer_name.split('.')[-1] == t_name.split('.')[-1]:
                    return t_name

        return teacher_name

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step with distillation.

        Args:
            inputs: Input batch
            optimizer: Optimizer for student model

        Returns:
            Dictionary of losses
        """
        # Clear previous features
        self.teacher_features.clear()
        self.student_features.clear()

        # Forward pass through teacher (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        # Forward pass through student
        student_outputs = self.student(**inputs)

        # Compute losses
        losses = self.compute_distillation_loss(
            student_outputs.get('logits', student_outputs),
            teacher_outputs.get('logits', teacher_outputs),
            inputs.get('labels')
        )

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}

        return loss_dict

    def evaluate(
        self,
        dataloader,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Evaluate student model with teacher comparison.

        Args:
            dataloader: Evaluation dataloader
            device: Device to use

        Returns:
            Evaluation metrics
        """
        self.student.eval()
        self.teacher.eval()

        metrics = {
            'student_loss': 0.0,
            'teacher_loss': 0.0,
            'distillation_loss': 0.0,
            'student_accuracy': 0.0,
            'teacher_accuracy': 0.0,
        }

        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get predictions
                teacher_outputs = self.teacher(**batch)
                student_outputs = self.student(**batch)

                teacher_logits = teacher_outputs.get('logits', teacher_outputs)
                student_logits = student_outputs.get('logits', student_outputs)

                # Compute metrics
                if 'labels' in batch:
                    labels = batch['labels']

                    # Losses
                    student_loss = F.cross_entropy(student_logits, labels)
                    teacher_loss = F.cross_entropy(teacher_logits, labels)

                    # Accuracies
                    student_preds = student_logits.argmax(dim=-1)
                    teacher_preds = teacher_logits.argmax(dim=-1)

                    student_acc = (student_preds == labels).float().mean()
                    teacher_acc = (teacher_preds == labels).float().mean()

                    # Update metrics
                    batch_size = labels.size(0)
                    metrics['student_loss'] += student_loss.item() * batch_size
                    metrics['teacher_loss'] += teacher_loss.item() * batch_size
                    metrics['student_accuracy'] += student_acc.item() * batch_size
                    metrics['teacher_accuracy'] += teacher_acc.item() * batch_size
                    total_samples += batch_size

                # Distillation loss
                T = self.config.temperature
                dist_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2)
                metrics['distillation_loss'] += dist_loss.item()

        # Average metrics
        if total_samples > 0:
            for key in ['student_loss', 'teacher_loss', 'student_accuracy', 'teacher_accuracy']:
                metrics[key] /= total_samples

        metrics['distillation_loss'] /= len(dataloader)
        metrics['accuracy_gap'] = metrics['teacher_accuracy'] - metrics['student_accuracy']

        self.student.train()

        return metrics