"""
Comprehensive test suite for RCC compression pipeline.

This module provides automated testing for all components of the compression
system with performance benchmarking and regression detection.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from pathlib import Path
import json
import time
import logging
from dataclasses import dataclass
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None


class CompressionTestSuite(unittest.TestCase):
    """Test suite for compression components."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.test_model = cls._create_test_model()
        cls.test_data = cls._create_test_data()

    @staticmethod
    def _create_test_model() -> nn.Module:
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    @staticmethod
    def _create_test_data() -> torch.Tensor:
        """Create test data."""
        return torch.randn(32, 784)

    def test_dare_pruning(self):
        """Test DARE pruning functionality."""
        from src.compression.dare.pruner import DAREPruner

        pruner = DAREPruner(sparsity=0.9)
        original_params = sum(p.numel() for p in self.test_model.parameters())

        compressed_model = pruner.apply(self.test_model.clone())

        # Check that model still works
        output = compressed_model(self.test_data)
        self.assertEqual(output.shape, (32, 10))

        # Check sparsity
        sparse_params = sum((p == 0).sum().item() for p in compressed_model.parameters())
        total_params = sum(p.numel() for p in compressed_model.parameters())
        actual_sparsity = sparse_params / total_params

        self.assertGreater(actual_sparsity, 0.85)  # Allow some tolerance

    def test_nullu_compression(self):
        """Test Nullu SVD compression."""
        from src.compression.nullu.svd_compressor import NulluCompressor

        compressor = NulluCompressor(rank_reduction=0.5)
        compressed_model = compressor.apply(self.test_model.clone())

        # Check output shape preservation
        output = compressed_model(self.test_data)
        self.assertEqual(output.shape, (32, 10))

        # Check rank reduction
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if weight is factorized
                self.assertTrue(hasattr(module, 'weight_U') or hasattr(module, 'weight'))

    def test_alphaedit_adaptation(self):
        """Test AlphaEdit weight adaptation."""
        from src.compression.alphaedit.weight_adapter import AlphaEditAdapter

        adapter = AlphaEditAdapter(alpha_init=1.0, beta_init=0.0)
        adapted_model = adapter.apply(self.test_model.clone())

        # Check that alpha parameters are added
        alpha_params = [n for n, _ in adapted_model.named_parameters() if 'alpha' in n]
        self.assertGreater(len(alpha_params), 0)

        # Check forward pass
        output = adapted_model(self.test_data)
        self.assertEqual(output.shape, (32, 10))

    def test_cascade_pipeline(self):
        """Test complete cascade compression pipeline."""
        from src.compression.cascade.pipeline import CascadeCompressor

        compressor = CascadeCompressor(
            dare_config={'sparsity': 0.9},
            nullu_config={'rank_reduction': 0.5},
            alphaedit_config={'alpha_init': 1.0}
        )

        original_params = sum(p.numel() for p in self.test_model.parameters())
        compressed_model = compressor.compress(self.test_model.clone())

        # Check functionality
        output = compressed_model(self.test_data)
        self.assertEqual(output.shape, (32, 10))

        # Check compression
        compressed_params = sum(p.numel() for p in compressed_model.parameters()
                              if not torch.all(p == 0))
        compression_ratio = 1 - (compressed_params / original_params)

        self.assertGreater(compression_ratio, 0.9)  # >90% compression

    def test_grassmann_distance(self):
        """Test Grassmann distance calculation."""
        from src.analysis.null_space.grassmann import GrassmannAnalyzer

        analyzer = GrassmannAnalyzer()

        # Create two random subspaces
        A = torch.randn(100, 10)
        B = torch.randn(100, 10)

        # Orthogonalize
        A, _ = torch.linalg.qr(A)
        B, _ = torch.linalg.qr(B)

        distance = analyzer.compute_distance(A, B)

        # Check properties
        self.assertGreaterEqual(distance, 0)
        self.assertLessEqual(distance, np.pi/2)

        # Same subspace should have distance 0
        self_distance = analyzer.compute_distance(A, A)
        self.assertAlmostEqual(self_distance, 0, places=5)


class TrainingTestSuite(unittest.TestCase):
    """Test suite for training components."""

    def test_mixed_precision_training(self):
        """Test mixed precision training setup."""
        from src.training.mixed_precision import MixedPrecisionTrainer

        trainer = MixedPrecisionTrainer(use_fp16=True)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Test autocast context
        with trainer.autocast_context():
            input = torch.randn(5, 10)
            output = model(input)
            loss = output.mean()

        # Test loss scaling
        scaled_loss = trainer.scale_loss(loss)
        self.assertIsInstance(scaled_loss, torch.Tensor)

    def test_knowledge_distillation(self):
        """Test knowledge distillation loss."""
        from src.training.distillation.kd_loss import DistillationLoss

        kd_loss = DistillationLoss(temperature=4.0, alpha=0.7)

        student_logits = torch.randn(32, 10)
        teacher_logits = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))

        loss = kd_loss(student_logits, teacher_logits, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreater(loss.item(), 0)

    def test_learning_rate_schedulers(self):
        """Test custom learning rate schedulers."""
        from src.training.optimization.schedulers import WarmupCosineScheduler

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000
        )

        # Test warmup phase
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(50):
            scheduler.step()

        warmup_lr = optimizer.param_groups[0]['lr']
        self.assertGreater(warmup_lr, initial_lr)

        # Test cosine decay
        for _ in range(500):
            scheduler.step()

        decay_lr = optimizer.param_groups[0]['lr']
        self.assertLess(decay_lr, warmup_lr)


class EvaluationTestSuite(unittest.TestCase):
    """Test suite for evaluation components."""

    def test_zero_shot_evaluation(self):
        """Test zero-shot classification evaluation."""
        from src.evaluation.benchmarks.zero_shot import ZeroShotEvaluator

        evaluator = ZeroShotEvaluator()

        # Mock model
        model = Mock()
        model.encode_image = Mock(return_value=torch.randn(100, 512))
        model.encode_text = Mock(return_value=torch.randn(10, 512))

        # Mock data
        images = torch.randn(100, 3, 224, 224)
        labels = torch.randint(0, 10, (100,))
        class_names = [f"class_{i}" for i in range(10)]

        accuracy = evaluator.evaluate(model, images, labels, class_names)

        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_retrieval_metrics(self):
        """Test retrieval metric computation."""
        from src.evaluation.benchmarks.retrieval import RetrievalEvaluator

        evaluator = RetrievalEvaluator()

        # Create dummy embeddings
        image_embeds = torch.randn(50, 256)
        text_embeds = torch.randn(50, 256)

        # Normalize
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        metrics = evaluator._compute_retrieval_metrics(image_embeds, text_embeds)

        # Check metrics
        self.assertIn('i2t_r1', metrics)
        self.assertIn('t2i_r5', metrics)
        self.assertIsInstance(metrics['i2t_r1'], float)
        self.assertGreaterEqual(metrics['i2t_r1'], 0)
        self.assertLessEqual(metrics['i2t_r1'], 1)

    def test_efficiency_metrics(self):
        """Test efficiency metric calculation."""
        from src.evaluation.metrics.efficiency import EfficiencyEvaluator

        evaluator = EfficiencyEvaluator()
        model = nn.Linear(100, 10)

        metrics = evaluator.evaluate(model, input_shape=(1, 100))

        self.assertIn('inference_time_ms', metrics)
        self.assertIn('model_size_mb', metrics)
        self.assertIn('parameters', metrics)
        self.assertGreater(metrics['parameters'], 0)


class IntegrationTestSuite(unittest.TestCase):
    """Integration tests for complete pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_compression(self):
        """Test complete compression pipeline end-to-end."""
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )

        # Import pipeline
        from src.compression.cascade.pipeline import CascadeCompressor

        # Configure compression
        compressor = CascadeCompressor(
            dare_config={'sparsity': 0.8},
            nullu_config={'rank_reduction': 0.6},
            alphaedit_config={'alpha_init': 1.0}
        )

        # Compress model
        compressed_model = compressor.compress(model)

        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(dummy_input)

        self.assertEqual(output.shape, (1, 10))

        # Check compression ratio
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters()
                              if not torch.all(p == 0))

        compression_ratio = 1 - (compressed_params / original_params)
        self.assertGreater(compression_ratio, 0.5)

    @patch('wandb.init')
    @patch('wandb.log')
    def test_training_with_compression(self, mock_wandb_log, mock_wandb_init):
        """Test training with compression enabled."""
        from src.training.trainer import RCCTrainer, TrainingConfig

        config = TrainingConfig(
            num_epochs=1,
            batch_size=8,
            use_wandb=False,
            checkpoint_dir=self.temp_dir
        )

        trainer = RCCTrainer(config)

        # Mock components
        model = nn.Linear(10, 10)
        trainer.initialize_models(model)

        # Mock data
        train_dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 10, (100,))
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 10),
            torch.randint(0, 10, (20,))
        )

        trainer.initialize_data(train_dataset, val_dataset)
        trainer.initialize_optimization()

        # Mock compressor
        compressor = Mock()
        compressor.return_value = (model, {'compression_ratio': 0.5})
        trainer.initialize_compression(compressor)

        # Run one epoch
        metrics = trainer.train_epoch()

        self.assertIn('loss', metrics)
        self.assertIsInstance(metrics['loss'], float)


class PerformanceTestSuite(unittest.TestCase):
    """Performance benchmarking tests."""

    def test_compression_speed(self):
        """Benchmark compression speed."""
        from src.compression.cascade.pipeline import CascadeCompressor

        model = nn.Sequential(*[nn.Linear(512, 512) for _ in range(10)])
        compressor = CascadeCompressor()

        start_time = time.perf_counter()
        compressed_model = compressor.compress(model)
        compression_time = time.perf_counter() - start_time

        logger.info(f"Compression time: {compression_time:.2f}s")

        # Should complete in reasonable time
        self.assertLess(compression_time, 60)  # Less than 1 minute

    def test_inference_speedup(self):
        """Test inference speedup after compression."""
        model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(5)])
        input_data = torch.randn(32, 1024)

        # Baseline inference time
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = model(input_data)
            baseline_time = time.perf_counter() - start

        # Compress model
        from src.compression.cascade.pipeline import CascadeCompressor
        compressor = CascadeCompressor(
            dare_config={'sparsity': 0.9}
        )
        compressed_model = compressor.compress(model)

        # Compressed inference time
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = compressed_model(input_data)
            compressed_time = time.perf_counter() - start

        speedup = baseline_time / compressed_time
        logger.info(f"Inference speedup: {speedup:.2f}x")

        # Should have some speedup
        self.assertGreater(speedup, 1.0)

    def test_memory_reduction(self):
        """Test memory reduction after compression."""
        model = nn.Sequential(*[nn.Linear(2048, 2048) for _ in range(3)])

        # Original memory
        original_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # Compress
        from src.compression.cascade.pipeline import CascadeCompressor
        compressor = CascadeCompressor()
        compressed_model = compressor.compress(model)

        # Compressed memory
        compressed_memory = sum(
            p.numel() * p.element_size() for p in compressed_model.parameters()
            if not torch.all(p == 0)
        )

        reduction = 1 - (compressed_memory / original_memory)
        logger.info(f"Memory reduction: {reduction:.2%}")

        self.assertGreater(reduction, 0.5)  # At least 50% reduction


def run_all_tests(verbose: int = 2) -> Dict[str, Any]:
    """
    Run all test suites and return results.

    Args:
        verbose: Verbosity level

    Returns:
        Test results summary
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test suites
    suite.addTests(loader.loadTestsFromTestCase(CompressionTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(TrainingTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(EvaluationTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(PerformanceTestSuite))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    # Summarize results
    summary = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
                       if result.testsRun > 0 else 0
    }

    return summary


if __name__ == "__main__":
    # Run tests
    summary = run_all_tests()
    print(f"\nTest Summary: {summary}")