"""
Attention transfer for knowledge distillation.

This module implements attention transfer mechanisms to improve
student model performance by matching attention patterns from the teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class AttentionTransferConfig:
    """Configuration for attention transfer."""
    transfer_type: str = "activation"  # activation, gradient, flow
    attention_loss_weight: float = 0.1
    spatial_attention: bool = True
    channel_attention: bool = True
    multi_scale: bool = False
    normalize_attention: bool = True
    temperature: float = 1.0
    layer_weights: Optional[List[float]] = None


class AttentionTransfer(nn.Module):
    """
    Implements various attention transfer methods for distillation.
    """

    def __init__(self, config: AttentionTransferConfig):
        """
        Initialize attention transfer module.

        Args:
            config: Attention transfer configuration
        """
        super().__init__()
        self.config = config

        # Attention extractors
        self.teacher_attentions = []
        self.student_attentions = []

        # Adaptation layers for dimension matching
        self.adaptation_layers = nn.ModuleDict()

    def extract_attention_maps(
        self,
        features: torch.Tensor,
        attention_type: str = "spatial"
    ) -> torch.Tensor:
        """
        Extract attention maps from features.

        Args:
            features: Feature tensor (B, C, H, W) or (B, N, D)
            attention_type: Type of attention to extract

        Returns:
            Attention maps
        """
        if attention_type == "spatial":
            return self._extract_spatial_attention(features)
        elif attention_type == "channel":
            return self._extract_channel_attention(features)
        elif attention_type == "combined":
            spatial = self._extract_spatial_attention(features)
            channel = self._extract_channel_attention(features)
            return spatial * channel.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def _extract_spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial attention maps.

        Args:
            features: Feature tensor

        Returns:
            Spatial attention maps
        """
        if len(features.shape) == 4:  # CNN features (B, C, H, W)
            # Sum over channel dimension
            attention = features.sum(dim=1, keepdim=True)  # (B, 1, H, W)

            if self.config.normalize_attention:
                # Normalize to [0, 1]
                B = attention.size(0)
                attention = attention.view(B, -1)
                attention = F.softmax(attention / self.config.temperature, dim=-1)
                attention = attention.view(B, 1, features.size(2), features.size(3))

        elif len(features.shape) == 3:  # Transformer features (B, N, D)
            # Use norm of feature vectors as attention
            attention = features.norm(dim=-1, keepdim=True)  # (B, N, 1)

            if self.config.normalize_attention:
                attention = F.softmax(attention / self.config.temperature, dim=1)

        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")

        return attention

    def _extract_channel_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract channel attention.

        Args:
            features: Feature tensor

        Returns:
            Channel attention
        """
        if len(features.shape) == 4:  # CNN features (B, C, H, W)
            # Global average pooling
            attention = features.mean(dim=(2, 3))  # (B, C)

            if self.config.normalize_attention:
                attention = F.softmax(attention / self.config.temperature, dim=-1)

        elif len(features.shape) == 3:  # Transformer features (B, N, D)
            # Average over sequence dimension
            attention = features.mean(dim=1)  # (B, D)

            if self.config.normalize_attention:
                attention = F.softmax(attention / self.config.temperature, dim=-1)

        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")

        return attention

    def compute_activation_attention_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute activation-based attention transfer loss.

        Args:
            student_features: List of student feature maps
            teacher_features: List of teacher feature maps

        Returns:
            Attention transfer loss
        """
        total_loss = 0.0
        num_pairs = min(len(student_features), len(teacher_features))

        for i in range(num_pairs):
            student_feat = student_features[i]
            teacher_feat = teacher_features[i]

            # Extract attention maps
            if self.config.spatial_attention:
                student_spatial = self._extract_spatial_attention(student_feat)
                teacher_spatial = self._extract_spatial_attention(teacher_feat)

                # Match dimensions if needed
                if student_spatial.shape != teacher_spatial.shape:
                    student_spatial = self._match_dimensions(
                        student_spatial, teacher_spatial.shape
                    )

                # Compute loss
                spatial_loss = F.mse_loss(student_spatial, teacher_spatial)
                total_loss += spatial_loss

            if self.config.channel_attention:
                student_channel = self._extract_channel_attention(student_feat)
                teacher_channel = self._extract_channel_attention(teacher_feat)

                # Match dimensions if needed
                if student_channel.shape != teacher_channel.shape:
                    student_channel = self._match_dimensions(
                        student_channel, teacher_channel.shape
                    )

                # Compute loss
                channel_loss = F.mse_loss(student_channel, teacher_channel)
                total_loss += channel_loss

        return total_loss / num_pairs

    def compute_gradient_attention_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_gradients: List[torch.Tensor],
        teacher_gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute gradient-based attention transfer loss.

        Args:
            student_features: Student feature maps
            teacher_features: Teacher feature maps
            student_gradients: Student gradients
            teacher_gradients: Teacher gradients

        Returns:
            Gradient attention loss
        """
        total_loss = 0.0
        num_pairs = min(len(student_features), len(teacher_features))

        for i in range(num_pairs):
            # Compute gradient attention
            student_grad_att = torch.abs(student_features[i] * student_gradients[i])
            teacher_grad_att = torch.abs(teacher_features[i] * teacher_gradients[i])

            # Pool to get attention maps
            if len(student_grad_att.shape) == 4:
                student_grad_att = student_grad_att.mean(dim=1)
                teacher_grad_att = teacher_grad_att.mean(dim=1)

            # Match dimensions
            if student_grad_att.shape != teacher_grad_att.shape:
                student_grad_att = self._match_dimensions(
                    student_grad_att, teacher_grad_att.shape
                )

            # Normalize
            if self.config.normalize_attention:
                student_grad_att = F.normalize(student_grad_att.flatten(1), dim=1)
                teacher_grad_att = F.normalize(teacher_grad_att.flatten(1), dim=1)

            # Compute loss
            loss = F.mse_loss(student_grad_att, teacher_grad_att)
            total_loss += loss

        return total_loss / num_pairs

    def compute_flow_attention_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute flow-based attention transfer loss.

        Args:
            student_features: Student feature maps
            teacher_features: Teacher feature maps

        Returns:
            Flow attention loss
        """
        total_loss = 0.0
        num_layers = min(len(student_features), len(teacher_features)) - 1

        for i in range(num_layers):
            # Compute flow between consecutive layers
            student_flow = self._compute_flow(
                student_features[i], student_features[i + 1]
            )
            teacher_flow = self._compute_flow(
                teacher_features[i], teacher_features[i + 1]
            )

            # Match dimensions
            if student_flow.shape != teacher_flow.shape:
                student_flow = self._match_dimensions(
                    student_flow, teacher_flow.shape
                )

            # Compute loss
            loss = F.mse_loss(student_flow, teacher_flow)
            total_loss += loss

        return total_loss / max(num_layers, 1)

    def _compute_flow(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow between two feature maps.

        Args:
            features1: First feature map
            features2: Second feature map

        Returns:
            Flow map
        """
        # Ensure same spatial dimensions
        if features1.shape != features2.shape:
            if len(features1.shape) == 4:
                features2 = F.adaptive_avg_pool2d(
                    features2,
                    (features1.size(2), features1.size(3))
                )
            elif len(features1.shape) == 3:
                features2 = F.adaptive_avg_pool1d(
                    features2.transpose(1, 2),
                    features1.size(1)
                ).transpose(1, 2)

        # Compute flow as difference
        flow = features2 - features1

        # Apply attention pooling
        if len(flow.shape) == 4:
            flow = flow.mean(dim=1)
        elif len(flow.shape) == 3:
            flow = flow.mean(dim=-1)

        return flow

    def _match_dimensions(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Match tensor dimensions to target shape.

        Args:
            tensor: Input tensor
            target_shape: Target shape

        Returns:
            Reshaped tensor
        """
        if len(tensor.shape) == len(target_shape):
            if len(tensor.shape) == 4:  # Spatial dimensions
                return F.adaptive_avg_pool2d(tensor, target_shape[2:])
            elif len(tensor.shape) == 3:  # Sequence dimensions
                return F.adaptive_avg_pool1d(
                    tensor.transpose(1, 2),
                    target_shape[1]
                ).transpose(1, 2)
            elif len(tensor.shape) == 2:  # Feature dimensions
                # Use linear projection
                if f"{tensor.shape}_{target_shape}" not in self.adaptation_layers:
                    self.adaptation_layers[f"{tensor.shape}_{target_shape}"] = nn.Linear(
                        tensor.shape[-1], target_shape[-1]
                    ).to(tensor.device)
                return self.adaptation_layers[f"{tensor.shape}_{target_shape}"](tensor)

        return tensor

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_gradients: Optional[List[torch.Tensor]] = None,
        teacher_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.

        Args:
            student_features: Student feature maps
            teacher_features: Teacher feature maps
            student_gradients: Student gradients (optional)
            teacher_gradients: Teacher gradients (optional)

        Returns:
            Total attention transfer loss
        """
        if self.config.transfer_type == "activation":
            loss = self.compute_activation_attention_loss(
                student_features, teacher_features
            )
        elif self.config.transfer_type == "gradient":
            if student_gradients is None or teacher_gradients is None:
                raise ValueError("Gradients required for gradient attention transfer")
            loss = self.compute_gradient_attention_loss(
                student_features, teacher_features,
                student_gradients, teacher_gradients
            )
        elif self.config.transfer_type == "flow":
            loss = self.compute_flow_attention_loss(
                student_features, teacher_features
            )
        else:
            raise ValueError(f"Unknown transfer type: {self.config.transfer_type}")

        return loss * self.config.attention_loss_weight


class MultiHeadAttentionTransfer(nn.Module):
    """
    Specialized attention transfer for multi-head attention layers.
    """

    def __init__(
        self,
        num_heads_student: int,
        num_heads_teacher: int,
        head_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Initialize multi-head attention transfer.

        Args:
            num_heads_student: Number of student attention heads
            num_heads_teacher: Number of teacher attention heads
            head_mapping: Mapping from student to teacher heads
        """
        super().__init__()
        self.num_heads_student = num_heads_student
        self.num_heads_teacher = num_heads_teacher

        # Create head mapping
        if head_mapping is None:
            self.head_mapping = self._create_default_mapping()
        else:
            self.head_mapping = head_mapping

    def _create_default_mapping(self) -> Dict[int, int]:
        """
        Create default head mapping.

        Returns:
            Head mapping dictionary
        """
        if self.num_heads_student == self.num_heads_teacher:
            # Direct mapping
            return {i: i for i in range(self.num_heads_student)}
        elif self.num_heads_student < self.num_heads_teacher:
            # Map to evenly spaced teacher heads
            step = self.num_heads_teacher / self.num_heads_student
            return {i: int(i * step) for i in range(self.num_heads_student)}
        else:
            # Map multiple student heads to same teacher head
            step = self.num_heads_student / self.num_heads_teacher
            return {i: min(int(i / step), self.num_heads_teacher - 1)
                   for i in range(self.num_heads_student)}

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-head attention transfer loss.

        Args:
            student_attention: Student attention weights (B, H_s, N, N)
            teacher_attention: Teacher attention weights (B, H_t, N, N)

        Returns:
            Attention transfer loss
        """
        B, H_s, N, _ = student_attention.shape
        total_loss = 0.0

        for s_head, t_head in self.head_mapping.items():
            student_head_att = student_attention[:, s_head]  # (B, N, N)
            teacher_head_att = teacher_attention[:, t_head]  # (B, N, N)

            # Compute KL divergence between attention distributions
            student_log = torch.log(student_head_att + 1e-8)
            loss = F.kl_div(student_log, teacher_head_att, reduction='batchmean')

            total_loss += loss

        return total_loss / len(self.head_mapping)