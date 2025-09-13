"""
CLIP model wrapper with compression hooks.

This module provides interfaces for applying RCC compression to CLIP models
while preserving multimodal alignment.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Any, Union
import numpy as np
from transformers import CLIPModel, CLIPProcessor, CLIPConfig, CLIPTokenizer
from dataclasses import dataclass
import logging
from pathlib import Path
from ..base import BaseModelWrapper, ModelConfig, VisionLanguageModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class CLIPCompressionConfig:
    """Configuration for CLIP model compression."""
    model_name: str = "openai/clip-vit-base-patch32"
    compress_vision: bool = True
    compress_text: bool = True
    preserve_embeddings: bool = True
    preserve_final_layers: bool = True
    vision_compression_ratio: float = 0.995
    text_compression_ratio: float = 0.995
    maintain_alignment: bool = True
    alignment_loss_weight: float = 0.1
    temperature: float = 0.07


class CompressedCLIPModel(nn.Module):
    """Compressed CLIP model with RCC optimizations."""

    def __init__(self, config: CLIPCompressionConfig):
        """
        Initialize compressed CLIP model.

        Args:
            config: CLIP compression configuration
        """
        super().__init__()
        self.config = config

        # Load base CLIP model
        self.base_model = CLIPModel.from_pretrained(config.model_name)
        self.processor = CLIPProcessor.from_pretrained(config.model_name)

        # Store original dimensions
        self.vision_dim = self.base_model.config.vision_config.hidden_size
        self.text_dim = self.base_model.config.text_config.hidden_size
        self.projection_dim = self.base_model.config.projection_dim

        # Compression tracking
        self.compression_applied = False
        self.compression_metrics = {}

    def forward(self,
                input_ids: Optional[Tensor] = None,
                pixel_values: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                return_loss: bool = False,
                return_dict: bool = True) -> Dict[str, Tensor]:
        """
        Forward pass through compressed CLIP model.

        Args:
            input_ids: Text input token IDs
            pixel_values: Image pixel values
            attention_mask: Attention mask for text
            position_ids: Position IDs for text
            return_loss: Whether to return contrastive loss
            return_dict: Whether to return dictionary output

        Returns:
            Model outputs including embeddings and optional loss
        """
        outputs = {}

        # Process vision inputs
        if pixel_values is not None:
            vision_outputs = self.base_model.vision_model(
                pixel_values=pixel_values,
                return_dict=True
            )
            image_embeds = vision_outputs.last_hidden_state
            image_embeds = self.base_model.visual_projection(image_embeds[:, 0, :])
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            outputs['image_embeds'] = image_embeds

        # Process text inputs
        if input_ids is not None:
            text_outputs = self.base_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True
            )
            text_embeds = text_outputs.last_hidden_state
            text_embeds = self.base_model.text_projection(text_embeds[:, 0, :])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            outputs['text_embeds'] = text_embeds

        # Compute contrastive loss if requested
        if return_loss and 'image_embeds' in outputs and 'text_embeds' in outputs:
            logit_scale = self.base_model.logit_scale.exp()
            logits_per_image = torch.matmul(outputs['image_embeds'],
                                           outputs['text_embeds'].t()) * logit_scale
            logits_per_text = logits_per_image.t()

            # Compute contrastive loss
            labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
            loss_i = nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            outputs['loss'] = loss
            outputs['logits_per_image'] = logits_per_image
            outputs['logits_per_text'] = logits_per_text

        return outputs

    def get_compression_hooks(self) -> Dict[str, List[str]]:
        """
        Get layer names for compression hooks.

        Returns:
            Dictionary mapping compression stages to layer names
        """
        hooks = {
            'vision_encoder': [],
            'text_encoder': [],
            'projections': [],
            'skip': []
        }

        # Vision encoder layers
        if self.config.compress_vision:
            for i, layer in enumerate(self.base_model.vision_model.encoder.layers):
                if not (self.config.preserve_final_layers and
                       i >= len(self.base_model.vision_model.encoder.layers) - 2):
                    hooks['vision_encoder'].append(f'vision_model.encoder.layers.{i}')

        # Text encoder layers
        if self.config.compress_text:
            for i, layer in enumerate(self.base_model.text_model.encoder.layers):
                if not (self.config.preserve_final_layers and
                       i >= len(self.base_model.text_model.encoder.layers) - 2):
                    hooks['text_encoder'].append(f'text_model.encoder.layers.{i}')

        # Projection layers (usually keep these)
        if not self.config.preserve_embeddings:
            hooks['projections'].extend([
                'visual_projection',
                'text_projection'
            ])
        else:
            hooks['skip'].extend([
                'visual_projection',
                'text_projection',
                'vision_model.embeddings',
                'text_model.embeddings'
            ])

        return hooks

    def apply_compression(self, compressor: Any) -> None:
        """
        Apply compression using provided compressor.

        Args:
            compressor: Compression module (cascade compressor)
        """
        if self.compression_applied:
            logger.warning("Compression already applied to model")
            return

        hooks = self.get_compression_hooks()

        # Compress vision encoder
        if self.config.compress_vision:
            logger.info("Compressing vision encoder...")
            for layer_name in hooks['vision_encoder']:
                layer = self._get_layer_by_name(layer_name)
                if layer is not None:
                    compressed_layer = compressor(layer)
                    self._set_layer_by_name(layer_name, compressed_layer)

        # Compress text encoder
        if self.config.compress_text:
            logger.info("Compressing text encoder...")
            for layer_name in hooks['text_encoder']:
                layer = self._get_layer_by_name(layer_name)
                if layer is not None:
                    compressed_layer = compressor(layer)
                    self._set_layer_by_name(layer_name, compressed_layer)

        self.compression_applied = True
        logger.info("Compression applied successfully")

    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """
        Get layer by hierarchical name.

        Args:
            name: Dot-separated layer name

        Returns:
            Layer module or None
        """
        parts = name.split('.')
        module = self.base_model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(part)]
            else:
                return None

        return module

    def _set_layer_by_name(self, name: str, new_layer: nn.Module) -> None:
        """
        Set layer by hierarchical name.

        Args:
            name: Dot-separated layer name
            new_layer: New layer module
        """
        parts = name.split('.')
        parent = self.base_model

        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            elif part.isdigit() and hasattr(parent, '__getitem__'):
                parent = parent[int(part)]

        if hasattr(parent, parts[-1]):
            setattr(parent, parts[-1], new_layer)
        elif parts[-1].isdigit() and hasattr(parent, '__setitem__'):
            parent[int(parts[-1])] = new_layer

    def compute_alignment_loss(self, image_embeds: Tensor,
                              text_embeds: Tensor) -> Tensor:
        """
        Compute multimodal alignment preservation loss.

        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings

        Returns:
            Alignment loss
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logit_scale = self.base_model.logit_scale.exp()
        similarity = torch.matmul(image_embeds, text_embeds.t()) * logit_scale

        # Alignment loss (contrastive)
        labels = torch.arange(len(similarity), device=similarity.device)
        loss = nn.functional.cross_entropy(similarity, labels)

        return loss

    def get_compression_statistics(self) -> Dict[str, Any]:
        """
        Get compression statistics for the model.

        Returns:
            Dictionary of compression metrics
        """
        stats = {
            'original_params': sum(p.numel() for p in self.base_model.parameters()),
            'compressed_params': sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
            'vision_encoder_params': sum(
                p.numel() for name, p in self.named_parameters()
                if 'vision_model' in name
            ),
            'text_encoder_params': sum(
                p.numel() for name, p in self.named_parameters()
                if 'text_model' in name
            ),
            'compression_applied': self.compression_applied
        }

        if stats['original_params'] > 0:
            stats['compression_ratio'] = (
                1 - stats['compressed_params'] / stats['original_params']
            ) * 100

        return stats


class CLIPWrapper(VisionLanguageModelWrapper):
    """Wrapper for CLIP model compression workflow."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda", **kwargs):
        """
        Initialize CLIP wrapper.

        Args:
            model_name: CLIP model name
            device: Device to use
            **kwargs: Additional configuration
        """
        config = ModelConfig(
            model_name=model_name,
            device=device,
            **kwargs
        )
        super().__init__(config)

        # Initialize CLIP compression config
        self.compression_config = CLIPCompressionConfig(
            model_name=model_name,
            compress_vision=config.compress_vision,
            compress_text=config.compress_text,
            preserve_embeddings=config.preserve_embeddings,
            preserve_final_layers=config.preserve_final_layers
        )

        # Load model components
        self.load_model()
        self.model.to(self.device)

    def prepare_inputs(self, images: List[Any] = None,
                      texts: List[str] = None) -> Dict[str, Tensor]:
        """
        Prepare inputs for CLIP model.

        Args:
            images: List of PIL images or tensors
            texts: List of text strings

        Returns:
            Dictionary of prepared inputs
        """
        inputs = {}

        if images is not None:
            image_inputs = self.model.processor(
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs['pixel_values'] = image_inputs['pixel_values'].to(self.device)

        if texts is not None:
            text_inputs = self.model.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs['input_ids'] = text_inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = text_inputs['attention_mask'].to(self.device)

        return inputs

    def encode_image(self, images: List[Any]) -> Tensor:
        """
        Encode images to embeddings.

        Args:
            images: List of images

        Returns:
            Image embeddings
        """
        inputs = self.prepare_inputs(images=images)
        with torch.no_grad():
            outputs = self.model(pixel_values=inputs['pixel_values'])
            return outputs['image_embeds']

    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts

        Returns:
            Text embeddings
        """
        inputs = self.prepare_inputs(texts=texts)
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            return outputs['text_embeds']

    def compute_similarity(self, images: List[Any],
                          texts: List[str]) -> Tensor:
        """
        Compute similarity between images and texts.

        Args:
            images: List of images
            texts: List of texts

        Returns:
            Similarity matrix
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)

        # Normalize and compute cosine similarity
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(image_embeds, text_embeds.t())
        return similarity

    def load_model(self) -> None:
        """
        Load the CLIP model and its components.
        """
        self.model = CompressedCLIPModel(self.compression_config)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name)

        # Set model components for base class
        self.vision_encoder = self.model.base_model.vision_model
        self.text_encoder = self.model.base_model.text_model

        logger.info(f"Loaded CLIP model: {self.config.model_name}")

    def forward(self, images: Optional[torch.Tensor] = None,
                text: Optional[Union[str, List[str], torch.Tensor]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP model.

        Args:
            images: Input images
            text: Input text
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        # Prepare inputs
        inputs = {}
        if images is not None:
            inputs['pixel_values'] = images

        if text is not None:
            if isinstance(text, (str, list)):
                text_inputs = self.tokenizer(text, padding=True, truncation=True,
                                            return_tensors="pt")
                inputs['input_ids'] = text_inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = text_inputs['attention_mask'].to(self.device)
            else:
                inputs['input_ids'] = text

        return self.model(**inputs, **kwargs)

    def generate_caption(self, images: torch.Tensor, max_length: int = 50,
                        temperature: float = 1.0, **kwargs) -> List[str]:
        """
        Generate captions for images (using zero-shot classification).

        Args:
            images: Input images
            max_length: Maximum caption length (not used for CLIP)
            temperature: Temperature for scaling
            **kwargs: Additional arguments

        Returns:
            List of "captions" (actually top classification)
        """
        # CLIP doesn't generate captions, return top classification instead
        logger.warning("CLIP doesn't support caption generation, returning top classification")

        # Use predefined templates for zero-shot classification
        templates = [
            "a photo of a {}",
            "an image showing {}",
            "a picture containing {}"
        ]

        # Common object categories
        categories = kwargs.get('categories', ['object', 'person', 'animal', 'scene', 'text'])
        texts = [template.format(cat) for template in templates[:1] for cat in categories]

        similarity = self.compute_similarity(images, texts)
        top_indices = similarity.argmax(dim=1)

        captions = [texts[idx] for idx in top_indices]
        return captions

    def compute_retrieval_scores(self, images: torch.Tensor,
                                texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image-text retrieval scores.

        Args:
            images: Input images
            texts: Input texts

        Returns:
            image_to_text_scores, text_to_image_scores
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)

        # Compute similarity scores
        similarity = self.compute_similarity(image_embeds, text_embeds)

        # Return both directions
        return similarity, similarity.T

    def save_compressed(self, path: str) -> None:
        """
        Save compressed model.

        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'compression_stats': self.model.get_compression_statistics()
        }, path)
        logger.info(f"Compressed model saved to {path}")

    def load_compressed(self, path: str) -> None:
        """
        Load compressed model.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Compressed model loaded from {path}")