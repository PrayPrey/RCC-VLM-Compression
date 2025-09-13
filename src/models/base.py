"""
Abstract base classes for model wrappers.

This module provides the foundational interfaces for all model wrappers
in the RCC compression system.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base configuration for models."""
    model_name: str
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    compress_vision: bool = True
    compress_text: bool = True
    preserve_embeddings: bool = True
    preserve_final_layers: bool = True
    cache_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class BaseModelWrapper(nn.Module, ABC):
    """
    Abstract base class for all model wrappers.

    This class defines the interface that all model wrappers must implement
    for compatibility with the RCC compression pipeline.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model wrapper.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Model components (to be set by subclasses)
        self.vision_encoder = None
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None

        # Compression tracking
        self.compression_applied = False
        self.compression_stats = {}

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model."""
        pass

    @abstractmethod
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text: Optional[Union[str, List[str], torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images (B, C, H, W)
            text: Input text (strings or token ids)
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing model outputs
        """
        pass

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to features.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Image features (B, D)
        """
        pass

    @abstractmethod
    def encode_text(self, text: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text to features.

        Args:
            text: Input text

        Returns:
            Text features (B, D)
        """
        pass

    def get_compressible_modules(self) -> List[Tuple[str, nn.Module]]:
        """
        Get list of modules that can be compressed.

        Returns:
            List of (name, module) tuples
        """
        compressible = []

        for name, module in self.named_modules():
            if self._should_compress(name, module):
                compressible.append((name, module))

        return compressible

    def _should_compress(self, name: str, module: nn.Module) -> bool:
        """
        Determine if a module should be compressed.

        Args:
            name: Module name
            module: Module instance

        Returns:
            Whether to compress this module
        """
        # Skip if compression disabled for modality
        if 'vision' in name and not self.config.compress_vision:
            return False
        if 'text' in name and not self.config.compress_text:
            return False

        # Skip embeddings if configured
        if self.config.preserve_embeddings and 'embed' in name.lower():
            return False

        # Skip final layers if configured
        if self.config.preserve_final_layers:
            if any(x in name.lower() for x in ['final', 'head', 'output', 'logit']):
                return False

        # Compress linear and conv layers
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Skip small layers
            if hasattr(module, 'weight'):
                if module.weight.numel() < 1000:  # Skip tiny layers
                    return False
            return True

        return False

    def apply_compression_mask(self, masks: Dict[str, torch.Tensor]) -> None:
        """
        Apply compression masks to model weights.

        Args:
            masks: Dictionary mapping module names to masks
        """
        for name, module in self.named_modules():
            if name in masks:
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.data *= masks[name].to(module.weight.device)
                    logger.debug(f"Applied mask to {name}")

        self.compression_applied = True

    def get_parameter_groups(self, base_lr: float = 1e-4) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimization with layer-wise learning rates.

        Args:
            base_lr: Base learning rate

        Returns:
            Parameter groups for optimizer
        """
        # Separate parameters by type
        embed_params = []
        encoder_params = []
        head_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if 'embed' in name.lower():
                embed_params.append(param)
            elif 'encoder' in name or 'transformer' in name:
                encoder_params.append(param)
            elif 'head' in name or 'classifier' in name:
                head_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if embed_params:
            param_groups.append({
                'params': embed_params,
                'lr': base_lr * 0.1,  # Lower LR for embeddings
                'name': 'embeddings'
            })

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': base_lr,
                'name': 'encoder'
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr * 2.0,  # Higher LR for heads
                'name': 'heads'
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })

        return param_groups

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute similarity between image and text features.

        Args:
            image_features: Image features (B1, D)
            text_features: Text features (B2, D)
            temperature: Temperature scaling factor

        Returns:
            Similarity matrix (B1, B2)
        """
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.matmul(image_features, text_features.T) / temperature

        return similarity

    def get_attention_maps(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract attention maps from the model.

        Returns:
            Dictionary of attention maps or None if not available
        """
        attention_maps = {}

        # Extract from vision encoder
        if hasattr(self, 'vision_encoder') and self.vision_encoder is not None:
            if hasattr(self.vision_encoder, 'get_attention_weights'):
                attention_maps['vision'] = self.vision_encoder.get_attention_weights()

        # Extract from text encoder
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            if hasattr(self.text_encoder, 'get_attention_weights'):
                attention_maps['text'] = self.text_encoder.get_attention_weights()

        return attention_maps if attention_maps else None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone parameters.

        Args:
            freeze: Whether to freeze parameters
        """
        modules_to_freeze = []

        if hasattr(self, 'vision_encoder') and self.vision_encoder is not None:
            modules_to_freeze.append(self.vision_encoder)

        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            modules_to_freeze.append(self.text_encoder)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = not freeze

        logger.info(f"{'Froze' if freeze else 'Unfroze'} backbone parameters")

    def get_compression_statistics(self) -> Dict[str, Any]:
        """
        Get compression statistics for the model.

        Returns:
            Dictionary of compression statistics
        """
        stats = {
            'compression_applied': self.compression_applied,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

        if self.compression_applied:
            # Count non-zero parameters
            non_zero = sum((p != 0).sum().item() for p in self.parameters())
            stats['non_zero_parameters'] = non_zero
            stats['sparsity'] = 1.0 - (non_zero / stats['total_parameters'])

            # Add stored compression stats
            stats.update(self.compression_stats)

        return stats

    def save_compressed(self, save_path: str) -> None:
        """
        Save compressed model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'compression_stats': self.get_compression_statistics(),
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Saved compressed model to {save_path}")

    def load_compressed(self, checkpoint_path: str) -> None:
        """
        Load compressed model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if 'compression_stats' in checkpoint:
            self.compression_stats = checkpoint['compression_stats']
            self.compression_applied = True

        logger.info(f"Loaded compressed model from {checkpoint_path}")


class VisionLanguageModelWrapper(BaseModelWrapper):
    """
    Base class for vision-language models with additional VL-specific methods.
    """

    @abstractmethod
    def generate_caption(
        self,
        images: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for images.

        Args:
            images: Input images
            max_length: Maximum caption length
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            List of generated captions
        """
        pass

    @abstractmethod
    def compute_retrieval_scores(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image-text retrieval scores.

        Args:
            images: Input images
            texts: Input texts

        Returns:
            image_to_text_scores: Scores for retrieving text given image
            text_to_image_scores: Scores for retrieving image given text
        """
        pass

    def compute_contrastive_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.

        Args:
            image_features: Image features (B, D)
            text_features: Text features (B, D)
            temperature: Temperature for scaling

        Returns:
            Contrastive loss value
        """
        batch_size = image_features.shape[0]

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / temperature

        # Create labels (diagonal is positive)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross entropy loss in both directions
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2