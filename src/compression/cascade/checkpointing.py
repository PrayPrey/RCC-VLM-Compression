"""
Checkpoint management for compression pipeline.

This module handles saving, loading, and managing checkpoints during
the multi-stage compression process, including rollback capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any, Union
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for compression checkpoint."""
    stage_name: str
    stage_index: int
    timestamp: str
    compression_ratio: float
    performance_metrics: Dict[str, float]
    compression_stats: Dict[str, Any]
    config: Dict[str, Any]
    previous_checkpoint: Optional[str] = None
    is_best: bool = False


class CheckpointManager:
    """Manages checkpoints for compression pipeline."""

    def __init__(self,
                 checkpoint_dir: str = "./checkpoints",
                 max_checkpoints: int = 10,
                 save_best_only: bool = False,
                 metric_for_best: str = "accuracy"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoint
            metric_for_best: Metric to determine best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_for_best = metric_for_best

        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = None
        self.checkpoint_history = []

        # Load existing checkpoints
        self._load_checkpoint_index()

    def save_checkpoint(self,
                       model: nn.Module,
                       stage_name: str,
                       stage_index: int,
                       compression_stats: Dict[str, Any],
                       performance_metrics: Dict[str, float],
                       config: Dict[str, Any],
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       additional_state: Optional[Dict] = None) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            stage_name: Name of compression stage
            stage_index: Index of compression stage
            compression_stats: Compression statistics
            performance_metrics: Performance metrics
            config: Configuration used
            optimizer: Optional optimizer state
            additional_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{stage_name}_{stage_index}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Determine if this is the best checkpoint
        is_best = self._is_best_checkpoint(performance_metrics)

        # Create metadata
        metadata = CheckpointMetadata(
            stage_name=stage_name,
            stage_index=stage_index,
            timestamp=timestamp,
            compression_ratio=compression_stats.get('compression_ratio', 0.0),
            performance_metrics=performance_metrics,
            compression_stats=compression_stats,
            config=config,
            previous_checkpoint=self.checkpoints[-1] if self.checkpoints else None,
            is_best=is_best
        )

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': asdict(metadata),
            'config': config,
            'compression_stats': compression_stats,
            'performance_metrics': performance_metrics
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if additional_state is not None:
            checkpoint['additional_state'] = additional_state

        # Save checkpoint
        if not self.save_best_only or is_best:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Update checkpoint list
            self.checkpoints.append(str(checkpoint_name))
            self.checkpoint_history.append(metadata)

            # Update best checkpoint
            if is_best:
                self._update_best_checkpoint(checkpoint_path, metadata)

            # Manage checkpoint limit
            self._cleanup_old_checkpoints()

            # Save checkpoint index
            self._save_checkpoint_index()

        return str(checkpoint_path)

    def load_checkpoint(self,
                       checkpoint_path: Optional[str] = None,
                       stage_name: Optional[str] = None,
                       load_best: bool = False) -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Specific checkpoint path
            stage_name: Load latest checkpoint for stage
            load_best: Load the best checkpoint

        Returns:
            Checkpoint dictionary
        """
        if load_best and self.best_checkpoint:
            checkpoint_path = self.checkpoint_dir / self.best_checkpoint
        elif stage_name:
            checkpoint_path = self._get_latest_stage_checkpoint(stage_name)
        elif checkpoint_path is None:
            # Load latest checkpoint
            if self.checkpoints:
                checkpoint_path = self.checkpoint_dir / self.checkpoints[-1]
            else:
                raise ValueError("No checkpoints available")
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

        return checkpoint

    def rollback_to_checkpoint(self,
                              checkpoint_path: Optional[str] = None,
                              stage_name: Optional[str] = None) -> Dict:
        """
        Rollback to a previous checkpoint.

        Args:
            checkpoint_path: Specific checkpoint to rollback to
            stage_name: Rollback to latest checkpoint of stage

        Returns:
            Loaded checkpoint dictionary
        """
        # Load the checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path, stage_name)

        # Remove all checkpoints after this one
        if checkpoint_path:
            checkpoint_name = Path(checkpoint_path).name
        else:
            checkpoint_name = self._get_latest_stage_checkpoint(stage_name).name

        if checkpoint_name in self.checkpoints:
            rollback_index = self.checkpoints.index(checkpoint_name)
            removed_checkpoints = self.checkpoints[rollback_index + 1:]

            # Delete the removed checkpoint files
            for cp_name in removed_checkpoints:
                cp_path = self.checkpoint_dir / cp_name
                if cp_path.exists():
                    cp_path.unlink()
                    logger.info(f"Removed checkpoint: {cp_path}")

            # Update checkpoint list
            self.checkpoints = self.checkpoints[:rollback_index + 1]
            self.checkpoint_history = self.checkpoint_history[:rollback_index + 1]

            # Save updated index
            self._save_checkpoint_index()

        logger.info(f"Rolled back to checkpoint: {checkpoint_name}")
        return checkpoint

    def get_checkpoint_history(self) -> List[CheckpointMetadata]:
        """
        Get checkpoint history.

        Returns:
            List of checkpoint metadata
        """
        return self.checkpoint_history

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        if self.best_checkpoint:
            return self.checkpoint_dir / self.best_checkpoint
        return None

    def compare_checkpoints(self,
                           checkpoint1: str,
                           checkpoint2: str) -> Dict:
        """
        Compare two checkpoints.

        Args:
            checkpoint1: First checkpoint path
            checkpoint2: Second checkpoint path

        Returns:
            Comparison dictionary
        """
        cp1 = self.load_checkpoint(checkpoint1)
        cp2 = self.load_checkpoint(checkpoint2)

        comparison = {
            'checkpoint1': checkpoint1,
            'checkpoint2': checkpoint2,
            'compression_ratio_diff': (
                cp2['compression_stats']['compression_ratio'] -
                cp1['compression_stats']['compression_ratio']
            ),
            'performance_diff': {}
        }

        # Compare performance metrics
        for metric in cp1['performance_metrics']:
            if metric in cp2['performance_metrics']:
                diff = cp2['performance_metrics'][metric] - cp1['performance_metrics'][metric]
                comparison['performance_diff'][metric] = diff

        return comparison

    def export_checkpoint(self,
                         checkpoint_path: str,
                         export_path: str,
                         include_metadata: bool = True):
        """
        Export checkpoint to a different location.

        Args:
            checkpoint_path: Source checkpoint path
            export_path: Destination path
            include_metadata: Whether to include metadata
        """
        checkpoint = self.load_checkpoint(checkpoint_path)

        if not include_metadata:
            # Export only model weights
            export_data = {'model_state_dict': checkpoint['model_state_dict']}
        else:
            export_data = checkpoint

        torch.save(export_data, export_path)
        logger.info(f"Exported checkpoint to: {export_path}")

    def _is_best_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """
        Determine if current checkpoint is the best.

        Args:
            metrics: Performance metrics

        Returns:
            True if this is the best checkpoint
        """
        if self.metric_for_best not in metrics:
            return False

        current_metric = metrics[self.metric_for_best]

        if self.best_metric is None:
            return True

        # Assume higher is better (can be customized)
        return current_metric > self.best_metric

    def _update_best_checkpoint(self,
                               checkpoint_path: Path,
                               metadata: CheckpointMetadata):
        """
        Update best checkpoint tracking.

        Args:
            checkpoint_path: Path to new best checkpoint
            metadata: Checkpoint metadata
        """
        # Copy to best checkpoint file
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        shutil.copy2(checkpoint_path, best_path)

        self.best_checkpoint = checkpoint_path.name
        self.best_metric = metadata.performance_metrics.get(self.metric_for_best)

        logger.info(f"New best checkpoint: {self.best_checkpoint} "
                   f"({self.metric_for_best}: {self.best_metric:.4f})")

    def _get_latest_stage_checkpoint(self, stage_name: str) -> Optional[Path]:
        """
        Get latest checkpoint for a specific stage.

        Args:
            stage_name: Stage name

        Returns:
            Path to latest checkpoint for stage
        """
        stage_checkpoints = [
            cp for cp in self.checkpoint_history
            if cp.stage_name == stage_name
        ]

        if stage_checkpoints:
            latest = stage_checkpoints[-1]
            # Find corresponding checkpoint file
            for cp_name in reversed(self.checkpoints):
                if stage_name in cp_name:
                    return self.checkpoint_dir / cp_name

        return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Keep best checkpoint and recent ones
            checkpoints_to_remove = []

            while len(self.checkpoints) > self.max_checkpoints:
                # Find oldest non-best checkpoint
                for i, cp_name in enumerate(self.checkpoints):
                    if cp_name != self.best_checkpoint:
                        checkpoints_to_remove.append(cp_name)
                        self.checkpoints.pop(i)
                        self.checkpoint_history.pop(i)
                        break

            # Delete checkpoint files
            for cp_name in checkpoints_to_remove:
                cp_path = self.checkpoint_dir / cp_name
                if cp_path.exists():
                    cp_path.unlink()
                    logger.info(f"Removed old checkpoint: {cp_path}")

    def _save_checkpoint_index(self):
        """Save checkpoint index to file."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        index_data = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': self.best_checkpoint,
            'best_metric': self.best_metric,
            'history': [asdict(meta) for meta in self.checkpoint_history]
        }

        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _load_checkpoint_index(self):
        """Load checkpoint index from file."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        if index_path.exists():
            with open(index_path, 'r') as f:
                index_data = json.load(f)

            self.checkpoints = index_data.get('checkpoints', [])
            self.best_checkpoint = index_data.get('best_checkpoint')
            self.best_metric = index_data.get('best_metric')

            # Reconstruct history
            history_data = index_data.get('history', [])
            self.checkpoint_history = [
                CheckpointMetadata(**meta) for meta in history_data
            ]

            logger.info(f"Loaded {len(self.checkpoints)} checkpoints from index")