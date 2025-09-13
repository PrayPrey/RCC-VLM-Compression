"""
Reproducibility utilities for ensuring consistent results.

This module provides functions to set random seeds and configure
deterministic behavior across different frameworks.
"""

import torch
import numpy as np
import random
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all frameworks.

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"Random seed set to {seed}")


def configure_deterministic_mode(
    enabled: bool = True,
    warn_only: bool = False
):
    """
    Configure PyTorch for deterministic behavior.

    Args:
        enabled: Whether to enable deterministic mode
        warn_only: Only warn about non-deterministic operations
    """
    if enabled:
        # Enable deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for CUBLAS
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        logger.info("Deterministic mode enabled")
    else:
        # Disable deterministic mode for better performance
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        logger.info("Deterministic mode disabled (better performance)")


def get_reproducibility_info() -> dict:
    """
    Get information about reproducibility settings.

    Returns:
        Dictionary with reproducibility information
    """
    info = {
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
        'torch_deterministic': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'torch_seed': torch.initial_seed(),
        'numpy_seed_state': np.random.get_state()[1][0],
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()

    return info


class ReproducibilityManager:
    """
    Manager for handling reproducibility across experiments.
    """

    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = True,
        benchmark: bool = False
    ):
        """
        Initialize reproducibility manager.

        Args:
            seed: Random seed
            deterministic: Enable deterministic mode
            benchmark: Enable CUDNN benchmark (conflicts with deterministic)
        """
        self.seed = seed
        self.deterministic = deterministic
        self.benchmark = benchmark

        # Store initial state
        self.initial_state = self._get_state()

    def __enter__(self):
        """Enter context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.restore()

    def setup(self):
        """Setup reproducibility settings."""
        set_seed(self.seed)

        if self.deterministic:
            configure_deterministic_mode(True)
        elif self.benchmark:
            torch.backends.cudnn.benchmark = True

        logger.info(f"Reproducibility setup: seed={self.seed}, "
                   f"deterministic={self.deterministic}, "
                   f"benchmark={self.benchmark}")

    def restore(self):
        """Restore original settings."""
        if self.initial_state:
            # Restore CUDNN settings
            torch.backends.cudnn.deterministic = self.initial_state['cudnn_deterministic']
            torch.backends.cudnn.benchmark = self.initial_state['cudnn_benchmark']

    def _get_state(self) -> dict:
        """Get current reproducibility state."""
        return {
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'torch_seed': torch.initial_seed(),
        }

    def log_info(self):
        """Log reproducibility information."""
        info = get_reproducibility_info()
        logger.info("Reproducibility information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """
    Worker initialization function for DataLoader.

    Ensures different random seeds for different workers.

    Args:
        worker_id: Worker ID
        base_seed: Base seed value
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_reproducibility_info(
    path: str,
    additional_info: Optional[dict] = None
):
    """
    Save reproducibility information to file.

    Args:
        path: Path to save information
        additional_info: Additional information to save
    """
    import json

    info = get_reproducibility_info()

    if additional_info:
        info.update(additional_info)

    # Add timestamp
    from datetime import datetime
    info['timestamp'] = datetime.now().isoformat()

    # Add environment info
    info['torch_version'] = torch.__version__
    info['numpy_version'] = np.__version__
    info['python_version'] = os.sys.version

    with open(path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Reproducibility info saved to {path}")


def verify_reproducibility(
    func,
    num_runs: int = 3,
    seed: int = 42,
    **kwargs
) -> bool:
    """
    Verify that a function produces reproducible results.

    Args:
        func: Function to test
        num_runs: Number of runs to verify
        seed: Random seed to use
        **kwargs: Arguments to pass to function

    Returns:
        Whether results are reproducible
    """
    results = []

    for i in range(num_runs):
        set_seed(seed)
        result = func(**kwargs)
        results.append(result)

    # Check if all results are the same
    if isinstance(results[0], torch.Tensor):
        # Compare tensors
        reproducible = all(
            torch.allclose(results[0], result)
            for result in results[1:]
        )
    elif isinstance(results[0], (int, float)):
        # Compare numbers
        reproducible = all(
            abs(results[0] - result) < 1e-6
            for result in results[1:]
        )
    else:
        # Simple equality check
        reproducible = all(
            results[0] == result
            for result in results[1:]
        )

    if reproducible:
        logger.info(f"Function {func.__name__} is reproducible")
    else:
        logger.warning(f"Function {func.__name__} is NOT reproducible")

    return reproducible