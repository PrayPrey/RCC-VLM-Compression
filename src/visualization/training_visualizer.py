"""
Comprehensive visualization module for training and compression metrics.

This module provides visualization tools for training progress, compression
effects, and performance comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """Visualizes training progress and metrics."""

    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize training visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self,
                            history: Dict[str, List[float]],
                            title: str = "Training Progress",
                            save_name: Optional[str] = None) -> None:
        """
        Plot training and validation curves.

        Args:
            history: Dictionary with metric histories
            title: Plot title
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            ax = axes[0, 0]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Accuracy curves
        if 'train_acc' in history and 'val_acc' in history:
            ax = axes[0, 1]
            epochs = range(1, len(history['train_acc']) + 1)
            ax.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
            ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Learning rate schedule
        if 'learning_rate' in history:
            ax = axes[1, 0]
            epochs = range(1, len(history['learning_rate']) + 1)
            ax.plot(epochs, history['learning_rate'], 'g-')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Compression ratio over time
        if 'compression_ratio' in history:
            ax = axes[1, 1]
            epochs = range(1, len(history['compression_ratio']) + 1)
            ax.plot(epochs, history['compression_ratio'], 'm-')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Compression Ratio')
            ax.set_title('Compression Ratio Progress')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")

        plt.show()

    def plot_compression_effects(self,
                                original_metrics: Dict[str, float],
                                compressed_metrics: Dict[str, float],
                                save_name: Optional[str] = None) -> None:
        """
        Visualize compression effects on model metrics.

        Args:
            original_metrics: Metrics from original model
            compressed_metrics: Metrics from compressed model
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Compression Effects Analysis", fontsize=16)

        # Performance comparison
        ax = axes[0]
        metrics = list(set(original_metrics.keys()) & set(compressed_metrics.keys()))
        x = np.arange(len(metrics))
        width = 0.35

        original_values = [original_metrics[m] for m in metrics]
        compressed_values = [compressed_metrics[m] for m in metrics]

        ax.bar(x - width/2, original_values, width, label='Original', color='blue', alpha=0.7)
        ax.bar(x + width/2, compressed_values, width, label='Compressed', color='red', alpha=0.7)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Retention rates
        ax = axes[1]
        retention_rates = []
        retention_labels = []

        for metric in metrics:
            if original_metrics[metric] > 0:
                retention = (compressed_metrics[metric] / original_metrics[metric]) * 100
                retention_rates.append(retention)
                retention_labels.append(metric)

        colors = ['green' if r >= 95 else 'orange' if r >= 90 else 'red' for r in retention_rates]
        bars = ax.bar(range(len(retention_rates)), retention_rates, color=colors, alpha=0.7)

        ax.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
        ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Retention Rate (%)')
        ax.set_title('Performance Retention')
        ax.set_xticks(range(len(retention_labels)))
        ax.set_xticklabels(retention_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Efficiency gains
        ax = axes[2]
        efficiency_metrics = {
            'Size Reduction': (1 - compressed_metrics.get('model_size', 1) /
                             original_metrics.get('model_size', 1)) * 100,
            'Speedup': (original_metrics.get('inference_time', 1) /
                       compressed_metrics.get('inference_time', 1)),
            'Memory Saving': (1 - compressed_metrics.get('memory_usage', 1) /
                            original_metrics.get('memory_usage', 1)) * 100
        }

        ax.barh(list(efficiency_metrics.keys()), list(efficiency_metrics.values()),
                color=['purple', 'cyan', 'magenta'], alpha=0.7)
        ax.set_xlabel('Improvement')
        ax.set_title('Efficiency Gains')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved compression effects to {save_path}")

        plt.show()

    def create_interactive_dashboard(self,
                                    history: Dict[str, List[float]],
                                    save_name: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard with Plotly.

        Args:
            history: Training history
            save_name: Filename to save HTML

        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves',
                          'Learning Rate', 'Compression Progress'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        epochs = list(range(1, len(history.get('train_loss', [1])) + 1))

        # Loss curves
        if 'train_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'],
                          mode='lines', name='Train Loss',
                          line=dict(color='blue')),
                row=1, col=1
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'],
                          mode='lines', name='Val Loss',
                          line=dict(color='red')),
                row=1, col=1
            )

        # Accuracy curves
        if 'train_acc' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_acc'],
                          mode='lines', name='Train Acc',
                          line=dict(color='blue')),
                row=1, col=2
            )
        if 'val_acc' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_acc'],
                          mode='lines', name='Val Acc',
                          line=dict(color='red')),
                row=1, col=2
            )

        # Learning rate
        if 'learning_rate' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['learning_rate'],
                          mode='lines', name='LR',
                          line=dict(color='green')),
                row=2, col=1
            )

        # Compression ratio
        if 'compression_ratio' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['compression_ratio'],
                          mode='lines', name='Compression',
                          line=dict(color='purple')),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Training Dashboard",
            showlegend=True,
            height=800,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")

        if save_name:
            save_path = self.save_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved interactive dashboard to {save_path}")

        return fig


class CompressionVisualizer:
    """Visualizes compression-specific metrics and effects."""

    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize compression visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_layer_wise_compression(self,
                                   layer_stats: Dict[str, Dict[str, float]],
                                   save_name: Optional[str] = None) -> None:
        """
        Plot layer-wise compression statistics.

        Args:
            layer_stats: Dictionary with layer-wise statistics
            save_name: Filename to save plot
        """
        layers = list(layer_stats.keys())
        metrics = list(next(iter(layer_stats.values())).keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Layer-wise Compression Analysis", fontsize=16)

        # Compression ratios per layer
        if 'compression_ratio' in metrics:
            ax = axes[0, 0]
            values = [layer_stats[l].get('compression_ratio', 0) for l in layers]
            bars = ax.bar(range(len(layers)), values, color='blue', alpha=0.7)

            # Color code by compression level
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 100:
                    bar.set_color('darkgreen')
                elif val > 50:
                    bar.set_color('green')
                elif val > 10:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax.set_xlabel('Layer')
            ax.set_ylabel('Compression Ratio')
            ax.set_title('Compression Ratio by Layer')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l[:10] for l in layers], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Sparsity per layer
        if 'sparsity' in metrics:
            ax = axes[0, 1]
            values = [layer_stats[l].get('sparsity', 0) * 100 for l in layers]
            ax.bar(range(len(layers)), values, color='purple', alpha=0.7)
            ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% target')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Sparsity (%)')
            ax.set_title('Sparsity by Layer')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l[:10] for l in layers], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Rank reduction per layer
        if 'rank_reduction' in metrics:
            ax = axes[1, 0]
            values = [layer_stats[l].get('rank_reduction', 0) * 100 for l in layers]
            ax.bar(range(len(layers)), values, color='orange', alpha=0.7)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Rank Reduction (%)')
            ax.set_title('Rank Reduction by Layer')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l[:10] for l in layers], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Performance impact per layer
        if 'performance_impact' in metrics:
            ax = axes[1, 1]
            values = [layer_stats[l].get('performance_impact', 0) for l in layers]
            colors = ['green' if v < 0.05 else 'orange' if v < 0.1 else 'red' for v in values]
            ax.bar(range(len(layers)), values, color=colors, alpha=0.7)
            ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='5% threshold')
            ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Performance Impact')
            ax.set_title('Performance Impact by Layer')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l[:10] for l in layers], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved layer-wise analysis to {save_path}")

        plt.show()

    def plot_compression_trajectory(self,
                                   trajectory: List[Dict[str, float]],
                                   save_name: Optional[str] = None) -> None:
        """
        Plot compression trajectory over stages.

        Args:
            trajectory: List of metrics at each compression stage
            save_name: Filename to save plot
        """
        stages = list(range(len(trajectory)))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Compression Trajectory Analysis", fontsize=16)

        # Model size reduction
        ax = axes[0, 0]
        sizes = [t.get('model_size', 100) for t in trajectory]
        ax.plot(stages, sizes, 'b-o', linewidth=2, markersize=8)
        ax.fill_between(stages, sizes, alpha=0.3)
        ax.set_xlabel('Compression Stage')
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size Reduction')
        ax.grid(True, alpha=0.3)

        # Performance retention
        ax = axes[0, 1]
        performance = [t.get('accuracy', 100) for t in trajectory]
        ax.plot(stages, performance, 'r-o', linewidth=2, markersize=8)
        ax.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% target')
        ax.fill_between(stages, performance, alpha=0.3)
        ax.set_xlabel('Compression Stage')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Performance Retention')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compression-Performance trade-off
        ax = axes[1, 0]
        compression_ratios = [t.get('compression_ratio', 1) for t in trajectory]
        ax.scatter(compression_ratios, performance, c=stages, cmap='viridis', s=100)
        for i, stage in enumerate(stages):
            ax.annotate(f'Stage {stage}', (compression_ratios[i], performance[i]),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Compression-Performance Trade-off')
        ax.grid(True, alpha=0.3)

        # Efficiency metrics
        ax = axes[1, 1]
        inference_speedup = [t.get('speedup', 1) for t in trajectory]
        memory_reduction = [t.get('memory_reduction', 0) * 100 for t in trajectory]

        ax2 = ax.twinx()
        line1 = ax.plot(stages, inference_speedup, 'g-o', label='Speedup', linewidth=2)
        line2 = ax2.plot(stages, memory_reduction, 'm-s', label='Memory Reduction', linewidth=2)

        ax.set_xlabel('Compression Stage')
        ax.set_ylabel('Speedup Factor', color='g')
        ax2.set_ylabel('Memory Reduction (%)', color='m')
        ax.set_title('Efficiency Improvements')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved compression trajectory to {save_path}")

        plt.show()

    def create_3d_compression_surface(self,
                                     x_param: np.ndarray,
                                     y_param: np.ndarray,
                                     z_performance: np.ndarray,
                                     x_label: str = "Sparsity",
                                     y_label: str = "Rank Reduction",
                                     z_label: str = "Accuracy",
                                     save_name: Optional[str] = None) -> go.Figure:
        """
        Create 3D surface plot of compression parameters vs performance.

        Args:
            x_param: X-axis parameter values
            y_param: Y-axis parameter values
            z_performance: Z-axis performance values
            x_label: Label for X-axis
            y_label: Label for Y-axis
            z_label: Label for Z-axis
            save_name: Filename to save HTML

        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Surface(x=x_param, y=y_param, z=z_performance)])

        fig.update_layout(
            title="Compression Parameter Space",
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700
        )

        if save_name:
            save_path = self.save_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved 3D surface to {save_path}")

        return fig


def create_comprehensive_report(history: Dict[str, List[float]],
                               compression_stats: Dict[str, Any],
                               save_dir: str = "./reports") -> None:
    """
    Create comprehensive visual report of training and compression.

    Args:
        history: Training history
        compression_stats: Compression statistics
        save_dir: Directory to save report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizers
    train_viz = TrainingVisualizer(save_dir / "training")
    comp_viz = CompressionVisualizer(save_dir / "compression")

    # Generate all visualizations
    train_viz.plot_training_curves(history, save_name="training_curves")
    train_viz.create_interactive_dashboard(history, save_name="dashboard")

    if 'layer_stats' in compression_stats:
        comp_viz.plot_layer_wise_compression(
            compression_stats['layer_stats'],
            save_name="layer_analysis"
        )

    if 'trajectory' in compression_stats:
        comp_viz.plot_compression_trajectory(
            compression_stats['trajectory'],
            save_name="compression_trajectory"
        )

    logger.info(f"Comprehensive report saved to {save_dir}")