"""
Cascade compression pipeline orchestration.

This module manages the three-stage compression cascade with checkpointing,
validation, and rollback capabilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import copy

from ..base import ICompressionMethod, CompressionConfig
from ..dare.pruner import DAREPruner
from ..nullu.svd_compressor import NulluCompressor
from ..alphaedit.weight_adapter import AlphaEditAdapter

logger = logging.getLogger(__name__)


class RCCPipeline:
    """
    Orchestrates the three-stage compression cascade.

    This class manages the sequential application of DARE, Nullu, and AlphaEdit
    compression methods with validation and rollback capabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        stages: Optional[List[Dict]] = None,
        performance_threshold: float = 0.95,
        rollback_enabled: bool = True,
        checkpoint_dir: str = "./checkpoints/cascade"
    ):
        """
        Initialize the RCC pipeline.

        Args:
            model: The model to compress
            stages: List of stage configurations
            performance_threshold: Minimum performance retention
            rollback_enabled: Whether to enable rollback on failure
            checkpoint_dir: Directory for saving checkpoints
        """
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.stages = stages or self._default_stages()
        self.performance_threshold = performance_threshold
        self.rollback_enabled = rollback_enabled
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints = []
        self.compression_stats = {}
        self.null_spaces = {}

    def _default_stages(self) -> List[Dict]:
        """Get default compression stages configuration."""
        return [
            {
                'name': 'dare',
                'method': 'DAREPruner',
                'parameters': {
                    'target_sparsity': 0.98,
                    'mode': 'hybrid',
                    'importance_metric': 'magnitude',
                    'iterative_pruning': True,
                    'pruning_iterations': 10
                },
                'validation': {
                    'min_performance': 0.95,
                    'max_sparsity': 0.98
                }
            },
            {
                'name': 'nullu',
                'method': 'NulluCompressor',
                'parameters': {
                    'energy_threshold': 0.99,
                    'max_rank_ratio': 0.1,
                    'rank_selection_method': 'energy',
                    'layer_wise_adaptation': True
                },
                'validation': {
                    'min_performance': 0.93,
                    'energy_threshold': 0.95
                }
            },
            {
                'name': 'alphaedit',
                'method': 'AlphaEditAdapter',
                'parameters': {
                    'alpha_init': 1.0,
                    'alpha_learning_rate': 0.001,
                    'use_fisher_information': True,
                    'use_layer_wise_alphas': True
                },
                'validation': {
                    'min_performance': 0.95,
                    'convergence_threshold': 0.001
                }
            }
        ]

    def run_pipeline(
        self,
        dataloader: Optional[Any] = None,
        validator: Optional[Any] = None
    ) -> nn.Module:
        """
        Run the complete compression pipeline.

        Args:
            dataloader: DataLoader for validation
            validator: Evaluator for performance validation

        Returns:
            Compressed model
        """
        logger.info("Starting RCC compression pipeline")

        # Save initial checkpoint
        self.save_checkpoint("initial", self.model.state_dict(), {})

        # Run compression stages
        for i, stage_config in enumerate(self.stages):
            logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage_config['name']}")

            try:
                # Apply compression stage
                self.model = self.run_stage(stage_config, dataloader)

                # Validate if validator provided
                if validator:
                    metrics = self.validate_stage(
                        stage_config['name'],
                        validator,
                        dataloader
                    )

                    # Check performance threshold
                    if not self._check_validation(metrics, stage_config['validation']):
                        if self.rollback_enabled:
                            logger.warning(f"Stage {stage_config['name']} failed validation, rolling back")
                            self.rollback_to_checkpoint(f"stage_{i-1}" if i > 0 else "initial")
                        else:
                            logger.warning(f"Stage {stage_config['name']} failed validation, continuing anyway")

                # Save checkpoint
                self.save_checkpoint(
                    f"stage_{i}_{stage_config['name']}",
                    self.model.state_dict(),
                    self.compression_stats.get(stage_config['name'], {})
                )

            except Exception as e:
                logger.error(f"Stage {stage_config['name']} failed: {e}")
                if self.rollback_enabled:
                    self.rollback_to_checkpoint(f"stage_{i-1}" if i > 0 else "initial")
                raise

        # Analyze null space overlap
        if len(self.null_spaces) > 1:
            overlap_metrics = self.analyze_null_space_overlap()
            logger.info(f"Null space overlap analysis: {overlap_metrics}")

        # Get final statistics
        summary = self.get_pipeline_summary()
        logger.info(f"Pipeline complete. Summary: {summary}")

        return self.model

    def run_stage(
        self,
        stage_config: Dict,
        dataloader: Optional[Any] = None
    ) -> nn.Module:
        """
        Run a single compression stage.

        Args:
            stage_config: Stage configuration
            dataloader: Optional dataloader for compression

        Returns:
            Compressed model
        """
        method_name = stage_config['method']
        parameters = stage_config['parameters']

        # Create compression method instance
        if method_name == 'DAREPruner':
            from ..dare.pruner import DAREPruner, DAREConfig
            config = DAREConfig(**parameters)
            compressor = DAREPruner(config)
        elif method_name == 'NulluCompressor':
            from ..nullu.svd_compressor import NulluCompressor, NulluConfig
            config = NulluConfig(**parameters)
            compressor = NulluCompressor(config)
        elif method_name == 'AlphaEditAdapter':
            from ..alphaedit.weight_adapter import AlphaEditAdapter, AlphaEditConfig
            config = AlphaEditConfig(**parameters)
            compressor = AlphaEditAdapter(config)
        else:
            raise ValueError(f"Unknown compression method: {method_name}")

        # Apply compression
        compressed_model, metrics = compressor.compress(self.model, dataloader)

        # Store compression statistics
        self.compression_stats[stage_config['name']] = metrics

        # Store null space if available
        if hasattr(compressor, 'get_null_spaces'):
            self.null_spaces[stage_config['name']] = compressor.get_null_spaces()

        return compressed_model

    def save_checkpoint(
        self,
        stage_name: str,
        model_state: Dict,
        metrics: Dict
    ) -> str:
        """
        Save a checkpoint for the current stage.

        Args:
            stage_name: Name of the stage
            model_state: Model state dictionary
            metrics: Compression metrics

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'stage_name': stage_name,
            'stage_index': len(self.checkpoints),
            'model_state': model_state,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'compression_stats': self.compression_stats.copy()
        }

        checkpoint_path = self.checkpoint_dir / f"{stage_name}_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        self.checkpoints.append(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

    def rollback_to_checkpoint(self, stage_name: str) -> bool:
        """
        Rollback to a previous checkpoint.

        Args:
            stage_name: Name of the stage to rollback to

        Returns:
            Success status
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_checkpoint.pt"

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.compression_stats = checkpoint.get('compression_stats', {})

        logger.info(f"Rolled back to checkpoint: {stage_name}")
        return True

    def validate_stage(
        self,
        stage_name: str,
        validator: Any,
        dataloader: Any
    ) -> Dict[str, float]:
        """
        Validate model performance after a compression stage.

        Args:
            stage_name: Name of the stage
            validator: Evaluator instance
            dataloader: Validation dataloader

        Returns:
            Validation metrics
        """
        logger.info(f"Validating stage: {stage_name}")

        # Run evaluation
        metrics = validator.evaluate(self.model, dataloader)

        # Calculate retention relative to original
        baseline_metrics = validator.evaluate(self.original_model, dataloader)

        retention = {}
        for key in metrics:
            if key in baseline_metrics and baseline_metrics[key] > 0:
                retention[f"{key}_retention"] = metrics[key] / baseline_metrics[key]

        metrics.update(retention)

        logger.info(f"Validation metrics for {stage_name}: {metrics}")
        return metrics

    def analyze_null_space_overlap(self) -> Dict[str, float]:
        """
        Analyze null space overlap between compression stages.

        Returns:
            Overlap metrics including Grassmann distances
        """
        if len(self.null_spaces) < 2:
            return {}

        from ...analysis.null_space.grassmann import compute_grassmann_distance

        overlap_metrics = {}
        stage_names = list(self.null_spaces.keys())

        for i in range(len(stage_names) - 1):
            stage1 = stage_names[i]
            stage2 = stage_names[i + 1]

            # Compute Grassmann distance for each layer
            distances = {}
            for layer_name in self.null_spaces[stage1]:
                if layer_name in self.null_spaces[stage2]:
                    dist = compute_grassmann_distance(
                        self.null_spaces[stage1][layer_name],
                        self.null_spaces[stage2][layer_name]
                    )
                    distances[layer_name] = dist

            # Average distance
            avg_distance = sum(distances.values()) / len(distances) if distances else 0
            overlap_metrics[f"{stage1}_to_{stage2}_distance"] = avg_distance

            # Check orthogonality threshold
            overlap_metrics[f"{stage1}_to_{stage2}_orthogonal"] = avg_distance > 0.7

        return overlap_metrics

    def get_pipeline_summary(self) -> Dict:
        """
        Get comprehensive pipeline execution summary.

        Returns:
            Summary dictionary with compression statistics
        """
        # Calculate total compression
        original_params = sum(p.numel() for p in self.original_model.parameters())
        compressed_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Calculate effective parameters (non-zero)
        effective_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                effective_params += (p != 0).sum().item()

        summary = {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'effective_parameters': effective_params,
            'total_compression_ratio': 1 - (effective_params / original_params),
            'stages_completed': len(self.compression_stats),
            'stage_metrics': self.compression_stats,
            'checkpoints_saved': len(self.checkpoints)
        }

        # Add null space overlap if available
        if len(self.null_spaces) > 1:
            summary['null_space_overlap'] = self.analyze_null_space_overlap()

        return summary

    def _check_validation(
        self,
        metrics: Dict[str, float],
        criteria: Dict[str, Any]
    ) -> bool:
        """
        Check if metrics meet validation criteria.

        Args:
            metrics: Computed metrics
            criteria: Validation criteria

        Returns:
            Whether validation passed
        """
        for key, threshold in criteria.items():
            if key == 'min_performance':
                # Check performance retention
                perf_keys = [k for k in metrics if 'retention' in k]
                if perf_keys:
                    avg_retention = sum(metrics[k] for k in perf_keys) / len(perf_keys)
                    if avg_retention < threshold:
                        logger.warning(f"Performance retention {avg_retention:.4f} < {threshold}")
                        return False
            elif key in metrics:
                if key.startswith('min_') and metrics[key] < threshold:
                    logger.warning(f"{key}: {metrics[key]:.4f} < {threshold}")
                    return False
                elif key.startswith('max_') and metrics[key] > threshold:
                    logger.warning(f"{key}: {metrics[key]:.4f} > {threshold}")
                    return False

        return True


class CascadeCompressor:
    """
    Alternative interface for cascade compression matching the base specification.
    """

    def __init__(
        self,
        dare_compressor: Any,
        nullu_compressor: Any,
        alpha_adapter: Any,
        config: CompressionConfig
    ):
        """
        Initialize cascade compressor with individual components.

        Args:
            dare_compressor: DARE pruning instance
            nullu_compressor: Nullu SVD compressor instance
            alpha_adapter: AlphaEdit adapter instance
            config: Base compression configuration
        """
        self.dare = dare_compressor
        self.nullu = nullu_compressor
        self.alpha = alpha_adapter
        self.config = config

    def compress(self, model: nn.Module, dataloader: Any = None) -> Tuple[nn.Module, Dict]:
        """
        Apply cascade compression.

        Args:
            model: Model to compress
            dataloader: Optional dataloader for adaptive compression

        Returns:
            Compressed model and metrics
        """
        metrics = {}

        # Stage 1: DARE pruning
        model, dare_metrics = self.dare.compress(model)
        metrics['dare'] = dare_metrics

        # Stage 2: Nullu projection
        model, nullu_metrics = self.nullu.compress(model)
        metrics['nullu'] = nullu_metrics

        # Stage 3: AlphaEdit adaptation
        if dataloader:
            model, alpha_metrics = self.alpha.compress(model, dataloader)
            metrics['alphaedit'] = alpha_metrics

        # Calculate total compression
        metrics['total_compression'] = self._calculate_total_compression(metrics)

        return model, metrics

    def _calculate_total_compression(self, metrics: Dict) -> float:
        """Calculate total compression achieved."""
        compression = 1.0

        if 'dare' in metrics and 'sparsity' in metrics['dare']:
            compression *= (1 - metrics['dare']['sparsity'])

        if 'nullu' in metrics and 'rank_reduction' in metrics['nullu']:
            compression *= (1 - metrics['nullu']['rank_reduction'])

        return 1 - compression