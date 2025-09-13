"""
Model registry for managing different model types.

This module provides a factory pattern for creating and managing
different vision-language models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing model classes and instances.
    """

    _models: Dict[str, Type[nn.Module]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[nn.Module],
        default_config: Optional[Dict[str, Any]] = None
    ):
        """
        Register a model class.

        Args:
            name: Model name
            model_class: Model class
            default_config: Default configuration
        """
        cls._models[name] = model_class

        if default_config:
            cls._configs[name] = default_config

        logger.info(f"Registered model: {name}")

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance.

        Args:
            name: Model name
            config: Model configuration
            **kwargs: Additional arguments

        Returns:
            Model instance
        """
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered. "
                           f"Available models: {list(cls._models.keys())}")

        model_class = cls._models[name]

        # Merge configurations
        final_config = {}

        # Start with default config if available
        if name in cls._configs:
            final_config.update(cls._configs[name])

        # Override with provided config
        if config:
            final_config.update(config)

        # Override with kwargs
        final_config.update(kwargs)

        # Create model instance
        try:
            model = model_class(**final_config)
            logger.info(f"Created model: {name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            raise

    @classmethod
    def list_models(cls) -> list:
        """
        List registered models.

        Returns:
            List of model names
        """
        return list(cls._models.keys())

    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """
        Get default configuration for a model.

        Args:
            name: Model name

        Returns:
            Default configuration
        """
        if name not in cls._configs:
            return {}
        return cls._configs[name].copy()


def register_default_models():
    """Register default models."""
    # Import model wrappers
    from .clip.model_wrapper import CLIPModelWrapper
    from .blip.model_wrapper import BLIPModelWrapper

    # Register CLIP models
    ModelRegistry.register(
        'clip-vit-base-patch32',
        CLIPModelWrapper,
        {
            'model_name': 'openai/clip-vit-base-patch32',
            'compress_vision': True,
            'compress_text': True
        }
    )

    ModelRegistry.register(
        'clip-vit-large-patch14',
        CLIPModelWrapper,
        {
            'model_name': 'openai/clip-vit-large-patch14',
            'compress_vision': True,
            'compress_text': True
        }
    )

    # Register BLIP models
    ModelRegistry.register(
        'blip-base',
        BLIPModelWrapper,
        {
            'model_name': 'Salesforce/blip-image-captioning-base',
            'compress_vision': True,
            'compress_text': True
        }
    )

    ModelRegistry.register(
        'blip-large',
        BLIPModelWrapper,
        {
            'model_name': 'Salesforce/blip-image-captioning-large',
            'compress_vision': True,
            'compress_text': True
        }
    )


class ModelFactory:
    """
    Factory for creating models with compression support.
    """

    @staticmethod
    def create_model(
        model_type: str,
        model_name: str,
        compression_config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        Create a model with compression configuration.

        Args:
            model_type: Type of model ('clip', 'blip')
            model_name: Specific model name
            compression_config: Compression configuration
            device: Device to load model on

        Returns:
            Model instance
        """
        # Ensure models are registered
        if not ModelRegistry.list_models():
            register_default_models()

        # Map model type and name to registry key
        if model_type == 'clip':
            if 'base' in model_name:
                registry_key = 'clip-vit-base-patch32'
            elif 'large' in model_name:
                registry_key = 'clip-vit-large-patch14'
            else:
                registry_key = model_name
        elif model_type == 'blip':
            if 'base' in model_name:
                registry_key = 'blip-base'
            elif 'large' in model_name:
                registry_key = 'blip-large'
            else:
                registry_key = model_name
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create model
        config = {
            'model_name': model_name,
            'device': device
        }

        if compression_config:
            config.update(compression_config)

        model = ModelRegistry.create(registry_key, config)

        return model

    @staticmethod
    def load_compressed_model(
        checkpoint_path: str,
        model_type: str,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        Load a compressed model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model_type: Type of model
            device: Device to load on

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model configuration
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # Try to infer from state dict
            model_config = {
                'model_type': model_type,
                'model_name': checkpoint.get('model_name', 'unknown')
            }

        # Create model
        model = ModelFactory.create_model(
            model_type=model_config['model_type'],
            model_name=model_config['model_name'],
            device=device
        )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded compressed model from {checkpoint_path}")

        return model

    @staticmethod
    def save_compressed_model(
        model: nn.Module,
        save_path: str,
        model_config: Dict[str, Any],
        compression_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Save a compressed model.

        Args:
            model: Model to save
            save_path: Path to save to
            model_config: Model configuration
            compression_stats: Compression statistics
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
        }

        if compression_stats:
            checkpoint['compression_stats'] = compression_stats

        # Add metadata
        import datetime
        checkpoint['timestamp'] = datetime.datetime.now().isoformat()
        checkpoint['torch_version'] = torch.__version__

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        logger.info(f"Saved compressed model to {save_path}")


class ModelWrapper(nn.Module):
    """
    Base class for model wrappers with compression support.
    """

    def __init__(
        self,
        model: nn.Module,
        compress_vision: bool = True,
        compress_text: bool = True
    ):
        """
        Initialize model wrapper.

        Args:
            model: Base model
            compress_vision: Whether to compress vision encoder
            compress_text: Whether to compress text encoder
        """
        super().__init__()
        self.model = model
        self.compress_vision = compress_vision
        self.compress_text = compress_text

    def get_compressible_layers(self) -> list:
        """
        Get list of layers that can be compressed.

        Returns:
            List of compressible layers
        """
        compressible = []

        for name, module in self.model.named_modules():
            # Check if layer should be compressed
            if self._should_compress(name, module):
                compressible.append((name, module))

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
        if 'embed' in name or 'final' in name or 'head' in name:
            return False

        # Check modality
        if 'vision' in name and not self.compress_vision:
            return False

        if 'text' in name and not self.compress_text:
            return False

        # Compress linear and conv layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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
                    module.weight.data *= masks[name]
                    logger.debug(f"Applied mask to {name}")

    def get_compression_statistics(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Compression statistics
        """
        total_params = 0
        compressed_params = 0
        layer_stats = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total = param.numel()
                non_zero = (param != 0).sum().item()

                total_params += total
                compressed_params += non_zero

                layer_stats[name] = {
                    'total': total,
                    'non_zero': non_zero,
                    'sparsity': 1 - (non_zero / total)
                }

        return {
            'total_parameters': total_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': 1 - (compressed_params / total_params),
            'layer_statistics': layer_stats
        }