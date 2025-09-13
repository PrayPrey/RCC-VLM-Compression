"""
Centralized logging utilities.

This module provides logging configuration and utilities
for the RCC compression system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import wandb
from logging.handlers import RotatingFileHandler
import colorlog


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    colorize: bool = True,
    wandb_project: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        level: Logging level
        log_dir: Directory for log files
        console: Enable console logging
        file: Enable file logging
        colorize: Use colored console output
        wandb_project: W&B project name for logging

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("rcc")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    if colorize and console:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rcc_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Also create a JSON log for structured logging
        json_file = log_dir / f"rcc_{timestamp}.json"
        json_handler = JSONLogHandler(json_file)
        logger.addHandler(json_handler)

    # W&B logging
    if wandb_project:
        wandb_handler = WandBLogHandler(wandb_project)
        logger.addHandler(wandb_handler)

    logger.info(f"Logging initialized - Level: {level}")

    return logger


class JSONLogHandler(logging.Handler):
    """
    Handler for structured JSON logging.
    """

    def __init__(self, filename: str):
        """
        Initialize JSON log handler.

        Args:
            filename: Path to JSON log file
        """
        super().__init__()
        self.filename = filename
        self.file = open(filename, 'a')

    def emit(self, record):
        """Emit a log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics

        if hasattr(record, 'stage'):
            log_entry['stage'] = record.stage

        if hasattr(record, 'epoch'):
            log_entry['epoch'] = record.epoch

        self.file.write(json.dumps(log_entry) + '\n')
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()
        super().close()


class WandBLogHandler(logging.Handler):
    """
    Handler for logging to Weights & Biases.
    """

    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize W&B log handler.

        Args:
            project: W&B project name
            config: Configuration to log
        """
        super().__init__()
        self.project = project

        # Initialize W&B
        wandb.init(project=project, config=config)

    def emit(self, record):
        """Emit a log record to W&B."""
        # Only log INFO and above to W&B
        if record.levelno >= logging.INFO:
            # Check for metrics in the record
            if hasattr(record, 'metrics'):
                wandb.log(record.metrics, step=getattr(record, 'step', None))
            else:
                # Log as text
                wandb.log({
                    'log': f"{record.levelname}: {record.getMessage()}"
                })


class MetricsLogger:
    """
    Specialized logger for metrics during training.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.metrics_history = []

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        stage: Optional[str] = None
    ):
        """
        Log metrics with context.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch
            stage: Training stage (train/val/test)
        """
        # Create log record with extra fields
        extra = {
            'metrics': metrics,
            'step': step,
            'epoch': epoch,
            'stage': stage
        }

        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'step': step,
            'epoch': epoch,
            'stage': stage
        })

        # Format message
        message_parts = []
        if epoch is not None:
            message_parts.append(f"Epoch {epoch}")
        if step is not None:
            message_parts.append(f"Step {step}")
        if stage:
            message_parts.append(f"[{stage.upper()}]")

        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        message_parts.append(metric_str)

        message = " - ".join(message_parts)

        self.logger.info(message, extra=extra)

    def log_compression_stats(
        self,
        original_params: int,
        compressed_params: int,
        compression_ratio: float,
        performance_retention: float,
        stage_name: str
    ):
        """
        Log compression statistics.

        Args:
            original_params: Original parameter count
            compressed_params: Compressed parameter count
            compression_ratio: Compression ratio achieved
            performance_retention: Performance retention percentage
            stage_name: Compression stage name
        """
        stats = {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'performance_retention': performance_retention,
            'parameter_reduction': 1 - (compressed_params / original_params)
        }

        self.logger.info(
            f"Compression Stage: {stage_name} | "
            f"Params: {original_params:,} → {compressed_params:,} | "
            f"Ratio: {compression_ratio:.2f}x | "
            f"Performance: {performance_retention:.2%}",
            extra={'metrics': stats, 'stage': stage_name}
        )

    def save_metrics_history(self, filepath: str):
        """
        Save metrics history to file.

        Args:
            filepath: Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self.logger.info(f"Saved metrics history to {filepath}")


class ProgressLogger:
    """
    Logger for tracking progress through compression pipeline.
    """

    def __init__(self, logger: logging.Logger, total_stages: int):
        """
        Initialize progress logger.

        Args:
            logger: Base logger
            total_stages: Total number of stages
        """
        self.logger = logger
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_times = []
        self.start_time = None

    def start_stage(self, stage_name: str):
        """
        Log start of a stage.

        Args:
            stage_name: Name of the stage
        """
        self.current_stage += 1
        self.start_time = datetime.now()

        self.logger.info(
            f"[{self.current_stage}/{self.total_stages}] "
            f"Starting stage: {stage_name}"
        )

    def end_stage(self, stage_name: str, success: bool = True):
        """
        Log end of a stage.

        Args:
            stage_name: Name of the stage
            success: Whether stage completed successfully
        """
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.stage_times.append(elapsed)

            status = "✓ Completed" if success else "✗ Failed"
            self.logger.info(
                f"[{self.current_stage}/{self.total_stages}] "
                f"{status} stage: {stage_name} (Time: {elapsed:.2f}s)"
            )

    def summary(self):
        """Log pipeline summary."""
        if self.stage_times:
            total_time = sum(self.stage_times)
            avg_time = total_time / len(self.stage_times)

            self.logger.info(
                f"Pipeline Summary: {self.current_stage}/{self.total_stages} stages completed | "
                f"Total time: {total_time:.2f}s | "
                f"Average per stage: {avg_time:.2f}s"
            )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"rcc.{name}")