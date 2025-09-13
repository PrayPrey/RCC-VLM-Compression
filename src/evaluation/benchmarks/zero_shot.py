"""
Zero-shot evaluation benchmark for compressed models.

This module implements zero-shot classification evaluation
for vision-language models on various datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any, Union
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot evaluation."""
    batch_size: int = 256
    num_workers: int = 4
    use_templates: bool = True
    aggregate_templates: str = "mean"  # mean, max, or first
    device: str = "cuda"
    fp16: bool = False
    verbose: bool = True


class ZeroShotEvaluator:
    """Zero-shot classification evaluator."""

    def __init__(self,
                 model: nn.Module,
                 config: ZeroShotConfig):
        """
        Initialize zero-shot evaluator.

        Args:
            model: Model to evaluate (should have encode_image and encode_text methods)
            config: Evaluation configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Text features cache
        self.text_features_cache = {}

    def evaluate(self,
                dataloader: DataLoader,
                class_names: List[str],
                templates: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            dataloader: DataLoader for evaluation
            class_names: List of class names
            templates: Optional text templates

        Returns:
            Dictionary of metrics
        """
        # Encode text features for all classes
        text_features = self.encode_text_classes(class_names, templates)

        # Evaluate on images
        all_predictions = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not self.config.verbose):
                # Get images and labels
                if isinstance(batch, dict):
                    images = batch.get('images', batch.get('image'))
                    labels = batch.get('labels', batch.get('label'))
                else:
                    images, labels = batch

                # Move to device
                images = images.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                # Encode images
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    image_features = self.encode_images(images)

                    # Compute similarity
                    logits = self.compute_similarity(image_features, text_features)

                # Get predictions
                predictions = logits.argmax(dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu())

        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_labels, all_logits)

        return metrics

    def encode_text_classes(self,
                           class_names: List[str],
                           templates: Optional[List[str]] = None) -> torch.Tensor:
        """
        Encode text features for all classes.

        Args:
            class_names: List of class names
            templates: Optional text templates

        Returns:
            Text features tensor [num_classes, feature_dim]
        """
        # Check cache
        cache_key = f"{','.join(class_names[:5])}_{len(class_names)}"
        if cache_key in self.text_features_cache:
            return self.text_features_cache[cache_key]

        if templates is None:
            templates = ["a photo of a {}."]

        all_text_features = []

        with torch.no_grad():
            for class_name in tqdm(class_names, desc="Encoding text", disable=not self.config.verbose):
                # Generate prompts for this class
                prompts = [template.format(class_name) for template in templates]

                # Encode prompts
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    text_features = self.encode_texts(prompts)

                # Aggregate features across templates
                if self.config.aggregate_templates == "mean":
                    text_features = text_features.mean(dim=0, keepdim=True)
                elif self.config.aggregate_templates == "max":
                    text_features = text_features.max(dim=0, keepdim=True)[0]
                else:  # first
                    text_features = text_features[0:1]

                all_text_features.append(text_features)

        # Stack all class features
        text_features = torch.cat(all_text_features, dim=0)

        # Normalize
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Cache features
        self.text_features_cache[cache_key] = text_features

        return text_features

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images.

        Args:
            images: Image tensor

        Returns:
            Image features
        """
        if hasattr(self.model, 'encode_image'):
            features = self.model.encode_image(images)
        elif hasattr(self.model, 'get_image_features'):
            features = self.model.get_image_features(images)
        else:
            # Fallback: assume model returns features directly
            features = self.model(images)

        # Normalize
        features = F.normalize(features, p=2, dim=-1)

        return features

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts.

        Args:
            texts: List of text strings

        Returns:
            Text features
        """
        if hasattr(self.model, 'encode_text'):
            features = self.model.encode_text(texts)
        elif hasattr(self.model, 'get_text_features'):
            features = self.model.get_text_features(texts)
        else:
            # Fallback: assume model has a tokenizer
            if hasattr(self.model, 'tokenizer'):
                tokens = self.model.tokenizer(texts, return_tensors='pt', padding=True)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                features = self.model(**tokens)
            else:
                raise NotImplementedError("Model must have encode_text or tokenizer method")

        return features

    def compute_similarity(self,
                         image_features: torch.Tensor,
                         text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between image and text features.

        Args:
            image_features: Image features [batch_size, feature_dim]
            text_features: Text features [num_classes, feature_dim]

        Returns:
            Similarity scores [batch_size, num_classes]
        """
        # Cosine similarity
        logits = image_features @ text_features.T

        # Scale by temperature if available
        if hasattr(self.model, 'logit_scale'):
            logit_scale = self.model.logit_scale.exp()
            logits = logits * logit_scale

        return logits

    def compute_metrics(self,
                       predictions: List[int],
                       labels: Optional[List[int]],
                       logits: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted labels
            labels: True labels (optional)
            logits: Raw logits

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        if labels is not None and len(labels) > 0:
            # Accuracy
            metrics['accuracy'] = accuracy_score(labels, predictions)

            # Top-k accuracy
            all_logits = torch.cat(logits, dim=0)
            labels_tensor = torch.tensor(labels)

            for k in [5, 10]:
                if all_logits.size(1) >= k:
                    top_k_preds = all_logits.topk(k, dim=1)[1]
                    correct_k = top_k_preds.eq(labels_tensor.view(-1, 1).expand_as(top_k_preds))
                    metrics[f'top{k}_accuracy'] = correct_k.any(dim=1).float().mean().item()

            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                labels, predictions, average='macro', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1

            # Class-wise accuracy
            unique_labels = np.unique(labels)
            class_accuracies = []
            for label in unique_labels:
                mask = np.array(labels) == label
                if mask.sum() > 0:
                    class_acc = (np.array(predictions)[mask] == label).mean()
                    class_accuracies.append(class_acc)

            metrics['mean_class_accuracy'] = np.mean(class_accuracies)

        # Confidence metrics
        all_logits = torch.cat(logits, dim=0)
        probs = F.softmax(all_logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]

        metrics['mean_confidence'] = max_probs.mean().item()
        metrics['std_confidence'] = max_probs.std().item()

        # Entropy
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        metrics['mean_entropy'] = entropy.item()

        return metrics

    def evaluate_multiple_datasets(self,
                                  datasets: Dict[str, Tuple[DataLoader, List[str]]]) -> Dict[str, Dict]:
        """
        Evaluate on multiple datasets.

        Args:
            datasets: Dictionary of dataset_name -> (dataloader, class_names)

        Returns:
            Dictionary of results per dataset
        """
        results = {}

        for dataset_name, (dataloader, class_names) in datasets.items():
            logger.info(f"Evaluating on {dataset_name}")
            metrics = self.evaluate(dataloader, class_names)
            results[dataset_name] = metrics

            # Log results
            logger.info(f"{dataset_name} results:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value:.4f}")

        return results

    def compare_models(self,
                      original_model: nn.Module,
                      compressed_model: nn.Module,
                      dataloader: DataLoader,
                      class_names: List[str]) -> Dict[str, Any]:
        """
        Compare original and compressed models.

        Args:
            original_model: Original model
            compressed_model: Compressed model
            dataloader: Evaluation dataloader
            class_names: Class names

        Returns:
            Comparison results
        """
        # Evaluate original model
        self.model = original_model
        original_metrics = self.evaluate(dataloader, class_names)

        # Evaluate compressed model
        self.model = compressed_model
        compressed_metrics = self.evaluate(dataloader, class_names)

        # Compute differences
        comparison = {
            'original': original_metrics,
            'compressed': compressed_metrics,
            'differences': {}
        }

        for metric in original_metrics:
            if metric in compressed_metrics:
                diff = compressed_metrics[metric] - original_metrics[metric]
                comparison['differences'][metric] = diff
                comparison['differences'][f'{metric}_relative'] = (
                    diff / original_metrics[metric] if original_metrics[metric] != 0 else 0
                )

        # Retention rate
        if 'accuracy' in original_metrics and 'accuracy' in compressed_metrics:
            retention = compressed_metrics['accuracy'] / original_metrics['accuracy']
            comparison['accuracy_retention'] = retention

        return comparison


def create_zero_shot_evaluator(model: nn.Module,
                              config: Optional[Dict] = None) -> ZeroShotEvaluator:
    """
    Create zero-shot evaluator.

    Args:
        model: Model to evaluate
        config: Configuration dictionary

    Returns:
        ZeroShotEvaluator instance
    """
    if config is None:
        config = {}

    eval_config = ZeroShotConfig(**config)
    return ZeroShotEvaluator(model, eval_config)