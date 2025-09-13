"""
Knowledge distillation loss functions.

This module implements various knowledge distillation losses for
training compressed models with teacher supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    Combines task loss with distillation loss from teacher model.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        loss_type: str = 'kl'
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss (1-alpha for task loss)
            loss_type: Type of distillation loss ('kl', 'mse', 'cosine')
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type

    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation and task loss.

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Ground truth targets (optional)
            task_loss_fn: Task-specific loss function (optional)

        Returns:
            Total loss and component losses
        """
        losses = {}

        # Distillation loss
        if self.loss_type == 'kl':
            distill_loss = self._kl_divergence_loss(
                student_outputs,
                teacher_outputs
            )
        elif self.loss_type == 'mse':
            distill_loss = F.mse_loss(student_outputs, teacher_outputs)
        elif self.loss_type == 'cosine':
            distill_loss = self._cosine_loss(student_outputs, teacher_outputs)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        losses['distillation'] = distill_loss.item()

        # Task loss if targets provided
        total_loss = self.alpha * distill_loss

        if targets is not None and task_loss_fn is not None:
            task_loss = task_loss_fn(student_outputs, targets)
            losses['task'] = task_loss.item()
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        losses['total'] = total_loss.item()

        return total_loss, losses

    def _kl_divergence_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss with temperature scaling.

        Args:
            student_outputs: Student logits
            teacher_outputs: Teacher logits

        Returns:
            KL divergence loss
        """
        # Soften probabilities
        student_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=-1)

        # KL divergence
        loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss

    def _cosine_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            student_outputs: Student embeddings
            teacher_outputs: Teacher embeddings

        Returns:
            Cosine loss
        """
        # Normalize
        student_norm = F.normalize(student_outputs, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_outputs, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.sum(student_norm * teacher_norm, dim=-1)

        # Loss is 1 - similarity
        loss = (1 - similarity).mean()

        return loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss.

    Matches intermediate features between student and teacher.
    """

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ):
        """
        Initialize feature distillation loss.

        Args:
            feature_weights: Weights for different feature layers
            normalize: Whether to normalize features before matching
        """
        super().__init__()
        self.feature_weights = feature_weights or {}
        self.normalize = normalize

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute feature matching loss.

        Args:
            student_features: Dictionary of student features
            teacher_features: Dictionary of teacher features

        Returns:
            Total loss and per-layer losses
        """
        total_loss = 0
        layer_losses = {}

        for layer_name in student_features:
            if layer_name not in teacher_features:
                continue

            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]

            # Normalize if requested
            if self.normalize:
                student_feat = F.normalize(student_feat, p=2, dim=-1)
                teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

            # Compute MSE loss
            loss = F.mse_loss(student_feat, teacher_feat)

            # Apply weight
            weight = self.feature_weights.get(layer_name, 1.0)
            weighted_loss = weight * loss

            total_loss += weighted_loss
            layer_losses[f"feature_{layer_name}"] = loss.item()

        return total_loss, layer_losses


class AttentionDistillationLoss(nn.Module):
    """
    Attention map distillation loss.

    Matches attention patterns between student and teacher.
    """

    def __init__(
        self,
        normalize_attention: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize attention distillation loss.

        Args:
            normalize_attention: Whether to normalize attention maps
            temperature: Temperature for attention softening
        """
        super().__init__()
        self.normalize_attention = normalize_attention
        self.temperature = temperature

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention matching loss.

        Args:
            student_attention: Student attention maps [B, H, N, N]
            teacher_attention: Teacher attention maps [B, H, N, N]

        Returns:
            Attention distillation loss
        """
        # Apply temperature if needed
        if self.temperature != 1.0:
            student_attention = student_attention / self.temperature
            teacher_attention = teacher_attention / self.temperature

        # Normalize attention maps
        if self.normalize_attention:
            student_attention = F.softmax(student_attention, dim=-1)
            teacher_attention = F.softmax(teacher_attention, dim=-1)

        # Compute KL divergence
        loss = F.kl_div(
            torch.log(student_attention + 1e-8),
            teacher_attention,
            reduction='batchmean'
        )

        return loss


class RelationalDistillationLoss(nn.Module):
    """
    Relational knowledge distillation loss.

    Matches relationships between samples in student and teacher.
    """

    def __init__(
        self,
        distance_metric: str = 'euclidean'
    ):
        """
        Initialize relational distillation loss.

        Args:
            distance_metric: Distance metric for relationships
        """
        super().__init__()
        self.distance_metric = distance_metric

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relational matching loss.

        Args:
            student_embeddings: Student embeddings [B, D]
            teacher_embeddings: Teacher embeddings [B, D]

        Returns:
            Relational distillation loss
        """
        # Compute pairwise distances
        student_distances = self._compute_distances(student_embeddings)
        teacher_distances = self._compute_distances(teacher_embeddings)

        # Normalize distances
        student_distances = F.normalize(student_distances, p=2, dim=-1)
        teacher_distances = F.normalize(teacher_distances, p=2, dim=-1)

        # Match distance matrices
        loss = F.mse_loss(student_distances, teacher_distances)

        return loss

    def _compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances.

        Args:
            embeddings: Embeddings [B, D]

        Returns:
            Distance matrix [B, B]
        """
        if self.distance_metric == 'euclidean':
            # Compute pairwise Euclidean distances
            distances = torch.cdist(embeddings, embeddings, p=2)
        elif self.distance_metric == 'cosine':
            # Compute cosine distances
            norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
            similarities = torch.mm(norm_embeddings, norm_embeddings.T)
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances


class MultiModalDistillationLoss(nn.Module):
    """
    Multi-modal distillation loss for vision-language models.

    Handles distillation for both vision and language modalities.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        vision_weight: float = 0.5,
        text_weight: float = 0.5,
        cross_modal_weight: float = 0.3
    ):
        """
        Initialize multi-modal distillation loss.

        Args:
            temperature: Temperature for softening
            vision_weight: Weight for vision distillation
            text_weight: Weight for text distillation
            cross_modal_weight: Weight for cross-modal alignment
        """
        super().__init__()
        self.temperature = temperature
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.cross_modal_weight = cross_modal_weight

    def forward(
        self,
        student_vision: torch.Tensor,
        student_text: torch.Tensor,
        teacher_vision: torch.Tensor,
        teacher_text: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-modal distillation loss.

        Args:
            student_vision: Student vision embeddings
            student_text: Student text embeddings
            teacher_vision: Teacher vision embeddings
            teacher_text: Teacher text embeddings

        Returns:
            Total loss and component losses
        """
        losses = {}

        # Vision distillation
        vision_loss = F.mse_loss(student_vision, teacher_vision)
        losses['vision'] = vision_loss.item()

        # Text distillation
        text_loss = F.mse_loss(student_text, teacher_text)
        losses['text'] = text_loss.item()

        # Cross-modal alignment
        student_similarity = torch.mm(student_vision, student_text.T) / self.temperature
        teacher_similarity = torch.mm(teacher_vision, teacher_text.T) / self.temperature

        cross_modal_loss = F.kl_div(
            F.log_softmax(student_similarity, dim=-1),
            F.softmax(teacher_similarity, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        losses['cross_modal'] = cross_modal_loss.item()

        # Combined loss
        total_loss = (
            self.vision_weight * vision_loss +
            self.text_weight * text_loss +
            self.cross_modal_weight * cross_modal_loss
        )

        losses['total'] = total_loss.item()

        return total_loss, losses