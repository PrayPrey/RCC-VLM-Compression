"""
BLIP model wrapper for compression.

This module provides a wrapper for BLIP models with compression support
and unified interface.
"""

import torch
import torch.nn as nn
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval,
    BlipConfig
)
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import logging
from ..base import VisionLanguageModelWrapper, ModelConfig

logger = logging.getLogger(__name__)


class BLIPWrapper(VisionLanguageModelWrapper):
    """
    Wrapper for BLIP models with compression support.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        task: str = "captioning",  # 'captioning' or 'retrieval'
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        compress_vision: bool = True,
        compress_text: bool = True,
        max_length: int = 50
    ):
        """
        Initialize BLIP model wrapper.

        Args:
            model_name: Name of the BLIP model
            task: Task type ('captioning' or 'retrieval')
            device: Device to load model on
            dtype: Data type for model
            compress_vision: Whether to compress vision encoder
            compress_text: Whether to compress text encoder
            max_length: Maximum caption length
        """
        super().__init__()

        self.model_name = model_name
        self.task = task
        self.device = device
        self.dtype = dtype
        self.compress_vision = compress_vision
        self.compress_text = compress_text
        self.max_length = max_length

        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load BLIP model and processor."""
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)

            if self.task == "captioning":
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype
                )
            elif self.task == "retrieval":
                self.model = BlipForImageTextRetrieval.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype
                )
            else:
                raise ValueError(f"Unknown task: {self.task}")

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded BLIP model: {self.model_name} for {self.task}")

        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise

    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image]]
    ) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: Input images (tensor or PIL images)

        Returns:
            Image embeddings
        """
        # Process images
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL images if needed
            if len(images.shape) == 4:  # Batch of tensors
                pil_images = []
                for img in images:
                    if img.shape[0] == 3:  # CHW format
                        img = img.permute(1, 2, 0)
                    img = (img * 255).byte().cpu().numpy()
                    pil_images.append(Image.fromarray(img))
                images = pil_images

        # Process with BLIP processor
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features
        with torch.no_grad():
            if hasattr(self.model, 'vision_model'):
                outputs = self.model.vision_model(**inputs)
                image_embeds = outputs.last_hidden_state.mean(dim=1)  # Pool
            else:
                # For conditional generation model
                image_embeds = self.model.get_image_features(**inputs)

        return image_embeds

    def encode_text(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: Input texts

        Returns:
            Text embeddings
        """
        # Process texts
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get text features
        with torch.no_grad():
            if hasattr(self.model, 'text_model'):
                outputs = self.model.text_model(**inputs)
                text_embeds = outputs.last_hidden_state.mean(dim=1)  # Pool
            else:
                # For conditional generation model
                text_embeds = self.model.get_text_features(**inputs)

        return text_embeds

    def generate_caption(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        num_beams: int = 3,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Generate captions for images.

        Args:
            images: Input images
            num_beams: Number of beams for beam search
            max_length: Maximum caption length

        Returns:
            Generated captions
        """
        if self.task != "captioning":
            raise ValueError("Caption generation only available for captioning task")

        # Process images
        if isinstance(images, torch.Tensor):
            # Convert to PIL if needed
            if len(images.shape) == 4:
                pil_images = []
                for img in images:
                    if img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    img = (img * 255).byte().cpu().numpy()
                    pil_images.append(Image.fromarray(img))
                images = pil_images

        # Process inputs
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate captions
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length or self.max_length
            )

        # Decode captions
        captions = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        return captions

    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between image and text embeddings.

        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings

        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = nn.functional.normalize(text_embeds, p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(image_embeds, text_embeds.T)

        return similarity

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images
            texts: Input texts
            labels: Ground truth labels
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary of outputs
        """
        outputs = {}

        if images is not None:
            outputs['image_embeds'] = self.encode_image(images)

        if texts is not None:
            outputs['text_embeds'] = self.encode_text(texts)

        if images is not None and texts is not None:
            outputs['similarity'] = self.compute_similarity(
                outputs['image_embeds'],
                outputs['text_embeds']
            )

            if return_loss and labels is not None:
                # Compute contrastive loss
                outputs['loss'] = self._compute_contrastive_loss(
                    outputs['similarity'],
                    labels
                )

        return outputs

    def _compute_contrastive_loss(
        self,
        similarity: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            similarity: Similarity matrix
            labels: Ground truth labels

        Returns:
            Contrastive loss
        """
        # Image-to-text loss
        loss_i2t = nn.functional.cross_entropy(
            similarity,
            labels
        )

        # Text-to-image loss
        loss_t2i = nn.functional.cross_entropy(
            similarity.T,
            labels
        )

        # Average losses
        loss = (loss_i2t + loss_t2i) / 2

        return loss

    def get_compressible_layers(self) -> List[nn.Module]:
        """
        Get layers that can be compressed.

        Returns:
            List of compressible layers
        """
        compressible = []

        for name, module in self.model.named_modules():
            if self._should_compress(name, module):
                compressible.append(module)

        return compressible

    def _should_compress(self, name: str, module: nn.Module) -> bool:
        """
        Check if a layer should be compressed.

        Args:
            name: Layer name
            module: Layer module

        Returns:
            Whether to compress
        """
        # Skip embeddings and final layers
        if any(skip in name for skip in ['embed', 'final', 'head', 'lm_head']):
            return False

        # Check modality
        if 'vision' in name and not self.compress_vision:
            return False

        if 'text' in name and not self.compress_text:
            return False

        # Compress linear and conv layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Skip small layers
            if hasattr(module, 'weight'):
                if module.weight.numel() < 1000:  # Skip very small layers
                    return False
            return True

        return False

    def apply_compression_masks(
        self,
        masks: Dict[str, torch.Tensor]
    ):
        """
        Apply compression masks to model.

        Args:
            masks: Dictionary of masks per layer
        """
        for name, module in self.model.named_modules():
            if name in masks:
                if hasattr(module, 'weight'):
                    module.weight.data *= masks[name].to(module.weight.device)
                    logger.debug(f"Applied mask to {name}")

    def get_compression_statistics(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Compression statistics
        """
        total_params = 0
        compressed_params = 0
        vision_params = 0
        text_params = 0
        vision_compressed = 0
        text_compressed = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total = param.numel()
                non_zero = (param != 0).sum().item()

                total_params += total
                compressed_params += non_zero

                if 'vision' in name:
                    vision_params += total
                    vision_compressed += non_zero
                elif 'text' in name:
                    text_params += total
                    text_compressed += non_zero

        return {
            'total_parameters': total_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': 1 - (compressed_params / total_params) if total_params > 0 else 0,
            'vision_parameters': vision_params,
            'vision_compressed': vision_compressed,
            'vision_compression_ratio': 1 - (vision_compressed / vision_params) if vision_params > 0 else 0,
            'text_parameters': text_params,
            'text_compressed': text_compressed,
            'text_compression_ratio': 1 - (text_compressed / text_params) if text_params > 0 else 0
        }


class BLIPCompressionConfig:
    """Configuration for BLIP model compression."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        compress_vision: bool = True,
        compress_text: bool = True,
        vision_compression_ratio: float = 0.995,
        text_compression_ratio: float = 0.995,
        preserve_embeddings: bool = True,
        preserve_final_layers: bool = True
    ):
        """
        Initialize BLIP compression configuration.

        Args:
            model_name: BLIP model name
            compress_vision: Whether to compress vision encoder
            compress_text: Whether to compress text encoder
            vision_compression_ratio: Target compression for vision
            text_compression_ratio: Target compression for text
            preserve_embeddings: Whether to preserve embedding layers
            preserve_final_layers: Whether to preserve final layers
        """
        self.model_name = model_name
        self.compress_vision = compress_vision
        self.compress_text = compress_text
        self.vision_compression_ratio = vision_compression_ratio
        self.text_compression_ratio = text_compression_ratio
        self.preserve_embeddings = preserve_embeddings
        self.preserve_final_layers = preserve_final_layers


class CaptionDecoder(nn.Module):
    """
    Caption decoder module for BLIP.
    """

    def __init__(self, model: nn.Module, tokenizer=None):
        """
        Initialize caption decoder.

        Args:
            model: BLIP model
            tokenizer: BLIP tokenizer (optional)
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        image_embeds: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generate captions from image embeddings.

        Args:
            image_embeds: Image embeddings
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            **kwargs: Additional generation arguments

        Returns:
            Generated captions
        """
        # Generate token IDs
        output_ids = self.model.generate(
            inputs=image_embeds,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )

        # Decode to text if tokenizer available
        if self.tokenizer:
            captions = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
        else:
            captions = [str(ids.tolist()) for ids in output_ids]

        return captions


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module for BLIP.
    """

    def __init__(self, config):
        """
        Initialize multimodal fusion.

        Args:
            config: BLIP configuration
        """
        super().__init__()
        self.config = config

        # Get dimensions from config
        if hasattr(config, 'vision_config'):
            vision_dim = config.vision_config.hidden_size
        else:
            vision_dim = 768  # Default

        if hasattr(config, 'text_config'):
            text_dim = config.text_config.hidden_size
        else:
            text_dim = 768  # Default

        hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else 768

        # Cross-attention layers
        self.vision_to_text_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True
        )

        self.text_to_vision_attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse vision and text features.

        Args:
            vision_features: Vision features (B, N, D_v)
            text_features: Text features (B, M, D_t)
            vision_mask: Vision attention mask
            text_mask: Text attention mask

        Returns:
            Fused features (B, D)
        """
        # Cross-attention: vision attends to text
        vision_attended, _ = self.text_to_vision_attention(
            vision_features,
            text_features,
            text_features,
            key_padding_mask=text_mask
        )

        # Cross-attention: text attends to vision
        text_attended, _ = self.vision_to_text_attention(
            text_features,
            vision_features,
            vision_features,
            key_padding_mask=vision_mask
        )

        # Pool features
        vision_pooled = vision_attended.mean(dim=1)  # (B, D_v)
        text_pooled = text_attended.mean(dim=1)  # (B, D_t)

        # Concatenate and project
        fused = torch.cat([vision_pooled, text_pooled], dim=-1)  # (B, D_v + D_t)
        fused = self.fusion_projection(fused)  # (B, D)

        # Final projection
        output = self.output_projection(fused)  # (B, D)

        return output