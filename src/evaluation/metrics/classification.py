"""
Classification metrics for evaluating compressed models.

This module implements zero-shot classification evaluation for vision-language models,
particularly focused on ImageNet and similar benchmarks.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float
    f1_score: float
    per_class_accuracy: Dict[int, float]
    confusion_matrix: np.ndarray
    support: Dict[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'accuracy': self.accuracy,
            'top5_accuracy': self.top5_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_per_class_accuracy': np.mean(list(self.per_class_accuracy.values()))
        }


class ZeroShotClassifier:
    """Zero-shot classification evaluator for CLIP-like models."""

    def __init__(self, model: nn.Module, class_names: List[str],
                 templates: Optional[List[str]] = None,
                 device: str = 'cuda'):
        """
        Initialize zero-shot classifier.

        Args:
            model: Vision-language model with encode_image and encode_text methods
            class_names: List of class names
            templates: List of prompt templates (e.g., "a photo of a {}")
            device: Device to run evaluation on
        """
        self.model = model
        self.class_names = class_names
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Default templates for ImageNet-style evaluation
        if templates is None:
            self.templates = [
                "a photo of a {}",
                "a bad photo of a {}",
                "a sculpture of a {}",
                "a photo of the hard to see {}",
                "a low resolution photo of the {}",
                "a rendering of a {}",
                "graffiti of a {}",
                "a bad photo of the {}",
                "a cropped photo of the {}",
                "a tattoo of a {}",
                "the embroidered {}",
                "a photo of a hard to see {}",
                "a bright photo of a {}",
                "a photo of a clean {}",
                "a photo of a dirty {}",
                "a dark photo of the {}",
                "a drawing of a {}",
                "a photo of my {}",
                "the plastic {}",
                "a photo of the cool {}",
                "a close-up photo of a {}",
                "a black and white photo of the {}",
                "a painting of the {}",
                "a painting of a {}",
                "a pixelated photo of the {}",
                "a sculpture of the {}",
                "a bright photo of the {}",
                "a cropped photo of a {}",
                "a plastic {}",
                "a photo of the dirty {}",
                "a jpeg corrupted photo of a {}",
                "a blurry photo of the {}",
                "a photo of the {}",
                "a good photo of the {}",
                "a rendering of the {}",
                "a {} in a video game",
                "a photo of one {}",
                "a doodle of a {}",
                "a close-up photo of the {}",
                "the origami {}",
                "the {} in a video game",
                "a sketch of a {}",
                "a doodle of the {}",
                "a origami {}",
                "a low resolution photo of a {}",
                "the toy {}",
                "a rendition of the {}",
                "a photo of the clean {}",
                "a photo of a large {}",
                "a rendition of a {}",
                "a photo of a nice {}",
                "a photo of a weird {}",
                "a blurry photo of a {}",
                "a cartoon {}",
                "art of a {}",
                "a sketch of the {}",
                "a embroidered {}",
                "a pixelated photo of a {}",
                "itap of the {}",
                "a jpeg corrupted photo of the {}",
                "a good photo of a {}",
                "a plushie {}",
                "a photo of the nice {}",
                "a photo of the small {}",
                "a photo of the weird {}",
                "the cartoon {}",
                "art of the {}",
                "a drawing of the {}",
                "a photo of the large {}",
                "a black and white photo of a {}",
                "the plushie {}",
                "a dark photo of a {}",
                "itap of a {}",
                "graffiti of the {}",
                "a toy {}",
                "itap of my {}",
                "a photo of a cool {}",
                "a photo of a small {}",
                "a tattoo of the {}"
            ]
        else:
            self.templates = templates

        # Precompute text features
        self.text_features = self._compute_text_features()

    def _compute_text_features(self) -> Tensor:
        """
        Compute text features for all classes using templates.

        Returns:
            Text features tensor of shape (num_classes, feature_dim)
        """
        logger.info("Computing text features for zero-shot classification...")

        all_text_features = []

        with torch.no_grad():
            for class_name in tqdm(self.class_names, desc="Encoding classes"):
                # Generate prompts for this class
                texts = [template.format(class_name) for template in self.templates]

                # Encode texts
                if hasattr(self.model, 'encode_text'):
                    text_features = self.model.encode_text(texts)
                else:
                    # Assume model returns dict with text_embeds
                    text_inputs = self.model.processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    outputs = self.model(**text_inputs)
                    text_features = outputs['text_embeds']

                # Average across templates
                text_features = text_features.mean(dim=0, keepdim=True)

                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                all_text_features.append(text_features)

        return torch.cat(all_text_features, dim=0)

    def classify_batch(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Classify a batch of images.

        Args:
            images: Batch of images

        Returns:
            Predictions and logits
        """
        with torch.no_grad():
            # Encode images
            if hasattr(self.model, 'encode_image'):
                image_features = self.model.encode_image(images)
            else:
                outputs = self.model(pixel_values=images)
                image_features = outputs['image_embeds']

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity to text features
            logits = image_features @ self.text_features.T

            # Get predictions
            predictions = logits.argmax(dim=-1)

        return predictions, logits

    def evaluate(self, dataloader: DataLoader,
                num_samples: Optional[int] = None) -> ClassificationMetrics:
        """
        Evaluate classification performance.

        Args:
            dataloader: Data loader with images and labels
            num_samples: Maximum number of samples to evaluate

        Returns:
            Classification metrics
        """
        logger.info("Evaluating zero-shot classification...")

        all_predictions = []
        all_labels = []
        all_logits = []
        samples_processed = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_samples and samples_processed >= num_samples:
                break

            # Get images and labels
            if isinstance(batch, (tuple, list)):
                images, labels = batch[0], batch[1]
            else:
                images = batch['images']
                labels = batch['labels']

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Classify
            predictions, logits = self.classify_batch(images)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

            samples_processed += len(images)

        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_labels, all_logits)

        return metrics

    def _compute_metrics(self, predictions: Tensor, labels: Tensor,
                        logits: Tensor) -> ClassificationMetrics:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted class indices
            labels: True class indices
            logits: Classification logits

        Returns:
            Classification metrics
        """
        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        # Basic metrics
        accuracy = accuracy_score(labels_np, predictions_np)

        # Top-5 accuracy
        top5_predictions = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1)[1]
        top5_correct = (top5_predictions == labels.unsqueeze(-1)).any(dim=-1)
        top5_accuracy = top5_correct.float().mean().item()

        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, predictions_np, average='macro', zero_division=0
        )

        # Per-class accuracy
        per_class_accuracy = {}
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = labels_np == label
            if mask.any():
                per_class_accuracy[int(label)] = accuracy_score(
                    labels_np[mask],
                    predictions_np[mask]
                )

        # Confusion matrix
        conf_matrix = confusion_matrix(labels_np, predictions_np)

        # Support per class
        support_dict = {int(label): int(count)
                       for label, count in zip(unique_labels, support)}

        return ClassificationMetrics(
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=conf_matrix,
            support=support_dict
        )


class ImageNetEvaluator:
    """Specialized evaluator for ImageNet zero-shot classification."""

    # ImageNet class names (abbreviated for space, full list would be loaded from file)
    IMAGENET_CLASSES = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        # ... (would include all 1000 classes)
    ]

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize ImageNet evaluator.

        Args:
            model: Vision-language model
            device: Device to run evaluation on
        """
        # Load full ImageNet class names from file or use default list
        self.classifier = ZeroShotClassifier(
            model,
            self.IMAGENET_CLASSES,
            device=device
        )

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on ImageNet.

        Args:
            dataloader: ImageNet data loader

        Returns:
            Evaluation metrics
        """
        metrics = self.classifier.evaluate(dataloader)

        # Format results for ImageNet
        results = {
            'imagenet/top1_accuracy': metrics.accuracy * 100,
            'imagenet/top5_accuracy': metrics.top5_accuracy * 100,
            'imagenet/mean_per_class_accuracy': np.mean(list(metrics.per_class_accuracy.values())) * 100
        }

        logger.info(f"ImageNet Zero-shot Results:")
        logger.info(f"  Top-1 Accuracy: {results['imagenet/top1_accuracy']:.2f}%")
        logger.info(f"  Top-5 Accuracy: {results['imagenet/top5_accuracy']:.2f}%")

        return results


def evaluate_classification(model: nn.Module,
                           dataloader: DataLoader,
                           class_names: List[str],
                           device: str = 'cuda') -> Dict[str, float]:
    """
    Convenience function for classification evaluation.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        class_names: List of class names
        device: Device to use

    Returns:
        Evaluation metrics
    """
    classifier = ZeroShotClassifier(model, class_names, device=device)
    metrics = classifier.evaluate(dataloader)
    return metrics.to_dict()


# ============= TEST CODE SECTION =============
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestClassificationMetrics(unittest.TestCase):
    """Unit tests for classification metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictions = torch.tensor([0, 1, 2, 1, 0, 2])
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])
        self.logits = torch.randn(6, 3)

    def test_classification_metrics_creation(self):
        """Test ClassificationMetrics dataclass creation."""
        metrics = ClassificationMetrics(
            accuracy=0.8,
            top5_accuracy=0.95,
            precision=0.82,
            recall=0.81,
            f1_score=0.815,
            per_class_accuracy={0: 0.85, 1: 0.80, 2: 0.75},
            confusion_matrix=np.eye(3),
            support={0: 100, 1: 100, 2: 100}
        )

        self.assertEqual(metrics.accuracy, 0.8)
        self.assertEqual(metrics.top5_accuracy, 0.95)
        self.assertIn('accuracy', metrics.to_dict())

    def test_zero_shot_classifier_init(self):
        """Test ZeroShotClassifier initialization."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()

        with patch.object(ZeroShotClassifier, '_compute_text_features', return_value=torch.randn(10, 512)):
            classifier = ZeroShotClassifier(
                model=mock_model,
                class_names=['cat', 'dog'],
                device='cpu'
            )

            self.assertEqual(len(classifier.class_names), 2)
            self.assertIsNotNone(classifier.text_features)
            mock_model.eval.assert_called_once()

    def test_classify_batch(self):
        """Test batch classification."""
        mock_model = Mock()
        mock_model.encode_image = Mock(return_value=torch.randn(4, 512))

        with patch.object(ZeroShotClassifier, '_compute_text_features', return_value=torch.randn(10, 512)):
            classifier = ZeroShotClassifier(
                model=mock_model,
                class_names=['class_' + str(i) for i in range(10)],
                device='cpu'
            )

            images = torch.randn(4, 3, 224, 224)
            predictions, logits = classifier.classify_batch(images)

            self.assertEqual(predictions.shape, (4,))
            self.assertEqual(logits.shape, (4, 10))

    def test_compute_metrics(self):
        """Test metric computation."""
        mock_model = Mock()

        with patch.object(ZeroShotClassifier, '_compute_text_features', return_value=torch.randn(3, 512)):
            classifier = ZeroShotClassifier(
                model=mock_model,
                class_names=['class_0', 'class_1', 'class_2'],
                device='cpu'
            )

            metrics = classifier._compute_metrics(
                self.predictions,
                self.labels,
                self.logits
            )

            self.assertIsInstance(metrics.accuracy, float)
            self.assertIsInstance(metrics.top5_accuracy, float)
            self.assertIsInstance(metrics.confusion_matrix, np.ndarray)

    def test_imagenet_evaluator(self):
        """Test ImageNet evaluator."""
        mock_model = Mock()
        mock_dataloader = Mock()

        with patch.object(ZeroShotClassifier, '__init__', return_value=None):
            with patch.object(ZeroShotClassifier, 'evaluate') as mock_evaluate:
                mock_metrics = Mock()
                mock_metrics.accuracy = 0.75
                mock_metrics.top5_accuracy = 0.92
                mock_metrics.per_class_accuracy = {i: 0.7 + i*0.01 for i in range(10)}
                mock_evaluate.return_value = mock_metrics

                evaluator = ImageNetEvaluator(mock_model)
                evaluator.classifier = ZeroShotClassifier(mock_model, ['test'], device='cpu')
                results = evaluator.evaluate(mock_dataloader)

                self.assertIn('imagenet/top1_accuracy', results)
                self.assertIn('imagenet/top5_accuracy', results)
                self.assertEqual(results['imagenet/top1_accuracy'], 75.0)
                self.assertEqual(results['imagenet/top5_accuracy'], 92.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for classification evaluation."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_end_to_end_evaluation(self, mock_cuda):
        """Test complete evaluation pipeline."""
        # Create mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.encode_image = Mock(return_value=torch.randn(32, 512))
        mock_model.encode_text = Mock(return_value=torch.randn(10, 512))

        # Create mock dataloader
        mock_dataloader = []
        for _ in range(3):
            images = torch.randn(32, 3, 224, 224)
            labels = torch.randint(0, 10, (32,))
            mock_dataloader.append((images, labels))

        # Run evaluation
        with patch.object(ZeroShotClassifier, '_compute_text_features', return_value=torch.randn(10, 512)):
            results = evaluate_classification(
                model=mock_model,
                dataloader=mock_dataloader,
                class_names=['class_' + str(i) for i in range(10)],
                device='cpu'
            )

            self.assertIn('accuracy', results)
            self.assertIn('top5_accuracy', results)
            self.assertIn('precision', results)
            self.assertIn('recall', results)
            self.assertIn('f1_score', results)


def run_tests():
    """Run all classification tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)