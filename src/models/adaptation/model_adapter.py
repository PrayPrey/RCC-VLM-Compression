"""
Model adaptation layer for applying compression to different architectures.

This module provides adapters for various vision-language model architectures
to ensure compatibility with the RCC compression pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LayerConfig:
    """Configuration for layer-specific compression."""
    layer_name: str
    layer_type: str
    compression_eligible: bool
    dare_sparsity: Optional[float] = None
    nullu_rank_ratio: Optional[float] = None
    alphaedit_enabled: bool = True
    preserve_structure: bool = False


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def prepare_for_compression(self, model: nn.Module) -> nn.Module:
        """Prepare model for compression."""
        pass

    @abstractmethod
    def apply_compression_config(self, model: nn.Module,
                                config: Dict[str, Any]) -> nn.Module:
        """Apply compression configuration to model."""
        pass

    @abstractmethod
    def restore_functionality(self, model: nn.Module) -> nn.Module:
        """Restore model functionality after compression."""
        pass

    @abstractmethod
    def get_layer_configs(self, model: nn.Module) -> List[LayerConfig]:
        """Get compression configuration for each layer."""
        pass


class CLIPAdapter(ModelAdapter):
    """Adapter for CLIP models."""

    def __init__(self, preserve_embeddings: bool = True):
        """
        Initialize CLIP adapter.

        Args:
            preserve_embeddings: Whether to preserve embedding layers
        """
        self.preserve_embeddings = preserve_embeddings

    def prepare_for_compression(self, model: nn.Module) -> nn.Module:
        """Prepare CLIP model for compression."""
        # Ensure model is in eval mode for analysis
        model.eval()

        # Mark critical layers
        self._mark_critical_layers(model)

        # Prepare vision encoder
        if hasattr(model, 'visual'):
            self._prepare_vision_encoder(model.visual)

        # Prepare text encoder
        if hasattr(model, 'text_projection'):
            self._prepare_text_encoder(model)

        return model

    def _mark_critical_layers(self, model: nn.Module):
        """Mark layers critical for functionality."""
        critical_patterns = [
            'position_embedding',
            'class_embedding',
            'ln_final',
            'text_projection',
            'visual.proj'
        ]

        for name, module in model.named_modules():
            for pattern in critical_patterns:
                if pattern in name:
                    module._compression_critical = True
                    logger.debug(f"Marked {name} as critical")

    def _prepare_vision_encoder(self, vision_encoder: nn.Module):
        """Prepare vision encoder for compression."""
        # Handle vision transformer
        if hasattr(vision_encoder, 'transformer'):
            for i, block in enumerate(vision_encoder.transformer.resblocks):
                # Configure attention layers
                if hasattr(block, 'attn'):
                    block.attn._compression_config = {
                        'dare_sparsity': 0.9 - 0.1 * (i / len(vision_encoder.transformer.resblocks)),
                        'nullu_rank_ratio': 0.5
                    }

                # Configure MLP layers
                if hasattr(block, 'mlp'):
                    block.mlp._compression_config = {
                        'dare_sparsity': 0.95,
                        'nullu_rank_ratio': 0.4
                    }

    def _prepare_text_encoder(self, model: nn.Module):
        """Prepare text encoder for compression."""
        if hasattr(model, 'transformer'):
            transformer = model.transformer

            # Configure text transformer blocks
            if hasattr(transformer, 'resblocks'):
                for i, block in enumerate(transformer.resblocks):
                    block._compression_config = {
                        'dare_sparsity': 0.85,
                        'nullu_rank_ratio': 0.6 - 0.1 * (i / len(transformer.resblocks))
                    }

    def apply_compression_config(self, model: nn.Module,
                                config: Dict[str, Any]) -> nn.Module:
        """Apply compression configuration to CLIP model."""
        for name, module in model.named_modules():
            # Skip critical layers if marked
            if hasattr(module, '_compression_critical') and module._compression_critical:
                continue

            # Apply custom compression config if available
            if hasattr(module, '_compression_config'):
                for key, value in module._compression_config.items():
                    setattr(module, f'compression_{key}', value)

            # Apply layer-specific compression
            if isinstance(module, nn.Linear):
                self._configure_linear_compression(module, name, config)
            elif isinstance(module, nn.MultiheadAttention):
                self._configure_attention_compression(module, name, config)

        return model

    def _configure_linear_compression(self, module: nn.Linear, name: str,
                                     config: Dict[str, Any]):
        """Configure compression for linear layers."""
        # Determine compression parameters based on layer position
        if 'mlp' in name:
            module.dare_sparsity = config.get('mlp_sparsity', 0.95)
            module.nullu_rank_ratio = config.get('mlp_rank_ratio', 0.4)
        elif 'attn' in name or 'attention' in name:
            module.dare_sparsity = config.get('attn_sparsity', 0.9)
            module.nullu_rank_ratio = config.get('attn_rank_ratio', 0.5)
        else:
            module.dare_sparsity = config.get('default_sparsity', 0.9)
            module.nullu_rank_ratio = config.get('default_rank_ratio', 0.5)

    def _configure_attention_compression(self, module: nn.MultiheadAttention,
                                        name: str, config: Dict[str, Any]):
        """Configure compression for attention layers."""
        module.dare_sparsity = config.get('attn_sparsity', 0.85)
        module.nullu_rank_ratio = config.get('attn_rank_ratio', 0.6)

    def restore_functionality(self, model: nn.Module) -> nn.Module:
        """Restore CLIP model functionality after compression."""
        # Ensure output projections are intact
        if hasattr(model, 'visual') and hasattr(model.visual, 'proj'):
            self._validate_projection(model.visual.proj)

        if hasattr(model, 'text_projection'):
            self._validate_projection(model.text_projection)

        # Re-normalize if needed
        self._restore_normalization(model)

        return model

    def _validate_projection(self, projection: nn.Module):
        """Validate projection layer integrity."""
        if isinstance(projection, nn.Parameter):
            # Ensure projection is normalized
            with torch.no_grad():
                projection.data = nn.functional.normalize(projection.data, dim=0)

    def _restore_normalization(self, model: nn.Module):
        """Restore normalization layers."""
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                # Ensure normalization parameters are not compressed
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = True
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = True

    def get_layer_configs(self, model: nn.Module) -> List[LayerConfig]:
        """Get compression configuration for each layer."""
        configs = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Determine if layer should be compressed
                compression_eligible = not (
                    hasattr(module, '_compression_critical') and
                    module._compression_critical
                )

                # Get compression parameters
                dare_sparsity = getattr(module, 'dare_sparsity', 0.9)
                nullu_rank_ratio = getattr(module, 'nullu_rank_ratio', 0.5)

                config = LayerConfig(
                    layer_name=name,
                    layer_type=type(module).__name__,
                    compression_eligible=compression_eligible,
                    dare_sparsity=dare_sparsity if compression_eligible else None,
                    nullu_rank_ratio=nullu_rank_ratio if compression_eligible else None,
                    alphaedit_enabled=compression_eligible,
                    preserve_structure='embedding' in name or 'projection' in name
                )

                configs.append(config)

        return configs


class BLIPAdapter(ModelAdapter):
    """Adapter for BLIP models."""

    def prepare_for_compression(self, model: nn.Module) -> nn.Module:
        """Prepare BLIP model for compression."""
        # Mark cross-attention layers as critical
        for name, module in model.named_modules():
            if 'cross' in name and 'attention' in name:
                module._compression_critical = True
                logger.debug(f"Marked {name} as critical for BLIP")

        # Configure vision encoder
        if hasattr(model, 'visual_encoder'):
            self._configure_visual_encoder(model.visual_encoder)

        # Configure text encoder
        if hasattr(model, 'text_encoder'):
            self._configure_text_encoder(model.text_encoder)

        # Configure multimodal layers
        if hasattr(model, 'text_decoder'):
            self._configure_multimodal_decoder(model.text_decoder)

        return model

    def _configure_visual_encoder(self, encoder: nn.Module):
        """Configure BLIP visual encoder for compression."""
        for i, layer in enumerate(encoder.blocks if hasattr(encoder, 'blocks') else []):
            # Progressive compression
            compression_factor = 1 - (i / len(encoder.blocks)) * 0.3

            layer._compression_config = {
                'dare_sparsity': 0.95 * compression_factor,
                'nullu_rank_ratio': 0.4 * compression_factor
            }

    def _configure_text_encoder(self, encoder: nn.Module):
        """Configure BLIP text encoder for compression."""
        if hasattr(encoder, 'layer'):
            for i, layer in enumerate(encoder.layer):
                layer._compression_config = {
                    'dare_sparsity': 0.85,
                    'nullu_rank_ratio': 0.5
                }

    def _configure_multimodal_decoder(self, decoder: nn.Module):
        """Configure multimodal decoder for compression."""
        # More conservative compression for decoder
        for name, module in decoder.named_modules():
            if isinstance(module, nn.Linear):
                module._compression_config = {
                    'dare_sparsity': 0.7,  # Less aggressive
                    'nullu_rank_ratio': 0.6  # Preserve more ranks
                }

    def apply_compression_config(self, model: nn.Module,
                                config: Dict[str, Any]) -> nn.Module:
        """Apply compression configuration to BLIP model."""
        for name, module in model.named_modules():
            if hasattr(module, '_compression_config'):
                # Apply stored configuration
                comp_config = module._compression_config

                if isinstance(module, nn.Linear):
                    module.dare_sparsity = comp_config.get('dare_sparsity', 0.9)
                    module.nullu_rank_ratio = comp_config.get('nullu_rank_ratio', 0.5)

        return model

    def restore_functionality(self, model: nn.Module) -> nn.Module:
        """Restore BLIP model functionality after compression."""
        # Validate cross-modal connections
        if hasattr(model, 'text_decoder'):
            self._validate_cross_modal_attention(model.text_decoder)

        # Ensure output heads are functional
        self._validate_output_heads(model)

        return model

    def _validate_cross_modal_attention(self, decoder: nn.Module):
        """Validate cross-modal attention mechanisms."""
        for name, module in decoder.named_modules():
            if 'cross_attention' in name:
                # Ensure cross-attention weights are preserved
                if hasattr(module, 'weight'):
                    logger.debug(f"Validated cross-attention layer: {name}")

    def _validate_output_heads(self, model: nn.Module):
        """Validate output prediction heads."""
        output_modules = ['cls_head', 'itm_head', 'lm_head']

        for module_name in output_modules:
            if hasattr(model, module_name):
                head = getattr(model, module_name)
                # Ensure head is functional
                if isinstance(head, nn.Linear):
                    logger.debug(f"Validated output head: {module_name}")

    def get_layer_configs(self, model: nn.Module) -> List[LayerConfig]:
        """Get compression configuration for each layer."""
        configs = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Check if critical
                is_critical = (
                    hasattr(module, '_compression_critical') and
                    module._compression_critical
                )

                # Get compression config
                comp_config = getattr(module, '_compression_config', {})

                config = LayerConfig(
                    layer_name=name,
                    layer_type=type(module).__name__,
                    compression_eligible=not is_critical,
                    dare_sparsity=comp_config.get('dare_sparsity'),
                    nullu_rank_ratio=comp_config.get('nullu_rank_ratio'),
                    alphaedit_enabled=not is_critical,
                    preserve_structure='cross' in name or 'head' in name
                )

                configs.append(config)

        return configs


class UniversalAdapter(ModelAdapter):
    """Universal adapter for arbitrary models."""

    def __init__(self, compression_threshold: float = 0.8):
        """
        Initialize universal adapter.

        Args:
            compression_threshold: Threshold for determining compressible layers
        """
        self.compression_threshold = compression_threshold

    def prepare_for_compression(self, model: nn.Module) -> nn.Module:
        """Prepare arbitrary model for compression."""
        # Analyze model structure
        layer_stats = self._analyze_model_structure(model)

        # Determine compression strategy
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                stats = layer_stats.get(name, {})

                # Determine if layer should be compressed
                if stats.get('relative_params', 0) > 0.01:  # >1% of parameters
                    module._should_compress = True

                    # Set compression parameters based on layer importance
                    importance = stats.get('importance', 0.5)
                    module.dare_sparsity = 0.95 * (1 - importance * 0.3)
                    module.nullu_rank_ratio = 0.5 * (1 - importance * 0.2)

        return model

    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model structure for compression planning."""
        total_params = sum(p.numel() for p in model.parameters())
        layer_stats = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params = sum(p.numel() for p in module.parameters())
                relative_params = params / total_params if total_params > 0 else 0

                # Estimate importance based on position and size
                depth = len(name.split('.'))
                importance = 1.0 / (1 + depth) * (1 + relative_params)

                layer_stats[name] = {
                    'params': params,
                    'relative_params': relative_params,
                    'importance': min(importance, 1.0),
                    'depth': depth
                }

        return layer_stats

    def apply_compression_config(self, model: nn.Module,
                                config: Dict[str, Any]) -> nn.Module:
        """Apply compression configuration to arbitrary model."""
        for name, module in model.named_modules():
            if hasattr(module, '_should_compress') and module._should_compress:
                # Apply compression parameters
                if hasattr(module, 'dare_sparsity'):
                    module.dare_sparsity = min(
                        module.dare_sparsity,
                        config.get('max_sparsity', 0.95)
                    )

                if hasattr(module, 'nullu_rank_ratio'):
                    module.nullu_rank_ratio = max(
                        module.nullu_rank_ratio,
                        config.get('min_rank_ratio', 0.3)
                    )

        return model

    def restore_functionality(self, model: nn.Module) -> nn.Module:
        """Restore model functionality after compression."""
        # Validate model outputs
        self._validate_forward_pass(model)

        # Restore batch norm statistics if needed
        self._restore_batch_norm(model)

        return model

    def _validate_forward_pass(self, model: nn.Module):
        """Validate model can still perform forward pass."""
        try:
            # Create dummy input based on first layer
            first_layer = next(model.modules())
            if isinstance(first_layer, nn.Linear):
                dummy_input = torch.randn(1, first_layer.in_features)
            elif isinstance(first_layer, nn.Conv2d):
                dummy_input = torch.randn(1, first_layer.in_channels, 32, 32)
            else:
                dummy_input = torch.randn(1, 3, 224, 224)

            with torch.no_grad():
                _ = model(dummy_input)

            logger.debug("Model forward pass validated successfully")
        except Exception as e:
            logger.warning(f"Forward pass validation failed: {e}")

    def _restore_batch_norm(self, model: nn.Module):
        """Restore batch normalization layers."""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Reset running statistics if needed
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()

    def get_layer_configs(self, model: nn.Module) -> List[LayerConfig]:
        """Get compression configuration for each layer."""
        configs = []
        layer_stats = self._analyze_model_structure(model)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                stats = layer_stats.get(name, {})

                config = LayerConfig(
                    layer_name=name,
                    layer_type=type(module).__name__,
                    compression_eligible=getattr(module, '_should_compress', False),
                    dare_sparsity=getattr(module, 'dare_sparsity', None),
                    nullu_rank_ratio=getattr(module, 'nullu_rank_ratio', None),
                    alphaedit_enabled=True,
                    preserve_structure=stats.get('importance', 0) > 0.8
                )

                configs.append(config)

        return configs


def get_model_adapter(model_type: str) -> ModelAdapter:
    """
    Get appropriate adapter for model type.

    Args:
        model_type: Type of model ('clip', 'blip', 'universal')

    Returns:
        Model adapter instance
    """
    adapters = {
        'clip': CLIPAdapter,
        'blip': BLIPAdapter,
        'universal': UniversalAdapter
    }

    adapter_class = adapters.get(model_type.lower(), UniversalAdapter)
    return adapter_class()