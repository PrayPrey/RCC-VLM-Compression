"""
Image-text retrieval benchmark for vision-language models.

This module provides evaluation for image-to-text and text-to-image
retrieval tasks on various datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: float
    mean_rank: float
    mean_reciprocal_rank: float


class RetrievalEvaluator:
    """Evaluates vision-language models on retrieval tasks."""

    def __init__(self,
                 device: str = "cuda",
                 batch_size: int = 128,
                 use_fp16: bool = True):
        """
        Initialize retrieval evaluator.

        Args:
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            use_fp16: Whether to use mixed precision
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

    def evaluate(self,
                model: nn.Module,
                dataloader: Any,
                task: str = "both") -> Dict[str, RetrievalMetrics]:
        """
        Evaluate model on retrieval tasks.

        Args:
            model: Vision-language model
            dataloader: DataLoader with image-text pairs
            task: "image_to_text", "text_to_image", or "both"

        Returns:
            Dictionary with retrieval metrics
        """
        model = model.to(self.device)
        model.eval()

        # Extract all features
        image_features, text_features, indices = self._extract_features(model, dataloader)

        results = {}

        if task in ["image_to_text", "both"]:
            logger.info("Evaluating image-to-text retrieval...")
            i2t_metrics = self._compute_retrieval_metrics(
                image_features, text_features, indices
            )
            results["image_to_text"] = i2t_metrics

        if task in ["text_to_image", "both"]:
            logger.info("Evaluating text-to-image retrieval...")
            t2i_metrics = self._compute_retrieval_metrics(
                text_features, image_features, indices
            )
            results["text_to_image"] = t2i_metrics

        return results

    def _extract_features(self,
                         model: nn.Module,
                         dataloader: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract image and text features from the model.

        Args:
            model: Vision-language model
            dataloader: DataLoader

        Returns:
            Image features, text features, and indices
        """
        image_features_list = []
        text_features_list = []
        indices_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch['images'].to(self.device)
                texts = batch['texts']
                batch_indices = batch.get('indices', np.arange(len(images)))

                # Use automatic mixed precision if enabled
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    # Extract image features
                    if hasattr(model, 'encode_image'):
                        image_feats = model.encode_image(images)
                    else:
                        image_feats = model.visual_encoder(images)

                    # Extract text features
                    if hasattr(model, 'encode_text'):
                        text_feats = model.encode_text(texts)
                    else:
                        text_feats = model.text_encoder(texts)

                    # Normalize features
                    image_feats = F.normalize(image_feats, p=2, dim=-1)
                    text_feats = F.normalize(text_feats, p=2, dim=-1)

                image_features_list.append(image_feats.cpu().numpy())
                text_features_list.append(text_feats.cpu().numpy())
                indices_list.append(batch_indices)

        # Concatenate all features
        image_features = np.concatenate(image_features_list, axis=0)
        text_features = np.concatenate(text_features_list, axis=0)
        indices = np.concatenate(indices_list, axis=0)

        return image_features, text_features, indices

    def _compute_retrieval_metrics(self,
                                  query_features: np.ndarray,
                                  gallery_features: np.ndarray,
                                  indices: np.ndarray) -> RetrievalMetrics:
        """
        Compute retrieval metrics.

        Args:
            query_features: Query features
            gallery_features: Gallery features
            indices: Ground truth indices

        Returns:
            Retrieval metrics
        """
        # Compute similarity matrix
        similarities = cosine_similarity(query_features, gallery_features)

        # Get rankings
        ranks = []
        reciprocal_ranks = []

        for i, sim_row in enumerate(similarities):
            # Sort similarities in descending order
            sorted_indices = np.argsort(-sim_row)

            # Find rank of correct match
            correct_index = indices[i]
            rank = np.where(sorted_indices == correct_index)[0][0] + 1
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)

        ranks = np.array(ranks)

        # Calculate metrics
        metrics = RetrievalMetrics(
            recall_at_1=np.mean(ranks <= 1) * 100,
            recall_at_5=np.mean(ranks <= 5) * 100,
            recall_at_10=np.mean(ranks <= 10) * 100,
            median_rank=np.median(ranks),
            mean_rank=np.mean(ranks),
            mean_reciprocal_rank=np.mean(reciprocal_ranks)
        )

        return metrics

    def evaluate_cross_modal(self,
                           model: nn.Module,
                           image_features: torch.Tensor,
                           text_features: torch.Tensor,
                           k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate cross-modal retrieval with pre-computed features.

        Args:
            model: Vision-language model (for similarity computation)
            image_features: Pre-computed image features
            text_features: Pre-computed text features
            k_values: K values for recall@k

        Returns:
            Cross-modal retrieval metrics
        """
        n_samples = min(len(image_features), len(text_features))
        image_features = image_features[:n_samples]
        text_features = text_features[:n_samples]

        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix
        similarities = torch.matmul(image_features, text_features.t())

        # Image-to-text retrieval
        i2t_ranks = self._get_ranks(similarities)
        i2t_metrics = self._calculate_recall_metrics(i2t_ranks, k_values)

        # Text-to-image retrieval
        t2i_ranks = self._get_ranks(similarities.t())
        t2i_metrics = self._calculate_recall_metrics(t2i_ranks, k_values)

        return {
            'image_to_text': i2t_metrics,
            'text_to_image': t2i_metrics,
            'mean_similarity': similarities.mean().item(),
            'std_similarity': similarities.std().item()
        }

    def _get_ranks(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Get ranking positions for each query.

        Args:
            similarities: Similarity matrix

        Returns:
            Rank positions
        """
        batch_size = similarities.shape[0]
        ranks = torch.zeros(batch_size)

        for i in range(batch_size):
            # Sort similarities for query i
            sorted_indices = torch.argsort(similarities[i], descending=True)
            # Find position of correct match (diagonal element)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0][0] + 1
            ranks[i] = rank

        return ranks

    def _calculate_recall_metrics(self,
                                 ranks: torch.Tensor,
                                 k_values: List[int]) -> Dict[str, float]:
        """
        Calculate recall@k metrics.

        Args:
            ranks: Rank positions
            k_values: K values for recall

        Returns:
            Recall metrics
        """
        metrics = {}

        for k in k_values:
            recall_k = (ranks <= k).float().mean().item() * 100
            metrics[f'recall@{k}'] = recall_k

        metrics['median_rank'] = ranks.median().item()
        metrics['mean_rank'] = ranks.mean().item()
        metrics['mrr'] = (1.0 / ranks).mean().item()

        return metrics


class ContrastiveRetrievalEvaluator:
    """Evaluator specifically for contrastive vision-language models."""

    def __init__(self,
                 temperature: float = 0.07,
                 device: str = "cuda"):
        """
        Initialize contrastive retrieval evaluator.

        Args:
            temperature: Temperature for contrastive loss
            device: Device to run evaluation on
        """
        self.temperature = temperature
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def evaluate_with_hard_negatives(self,
                                    model: nn.Module,
                                    dataloader: Any,
                                    num_hard_negatives: int = 10) -> Dict[str, Any]:
        """
        Evaluate with hard negative mining.

        Args:
            model: Vision-language model
            dataloader: DataLoader
            num_hard_negatives: Number of hard negatives to consider

        Returns:
            Evaluation metrics with hard negatives
        """
        model = model.to(self.device)
        model.eval()

        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating with hard negatives"):
                images = batch['images'].to(self.device)
                texts = batch['texts']

                # Get features
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)

                # Normalize
                image_features = F.normalize(image_features, p=2, dim=-1)
                text_features = F.normalize(text_features, p=2, dim=-1)

                # Compute similarities
                logits = torch.matmul(image_features, text_features.t()) / self.temperature

                # Find hard negatives
                batch_size = logits.shape[0]
                labels = torch.arange(batch_size).to(self.device)

                # For each positive pair, find hardest negatives
                hard_negatives_i2t = []
                hard_negatives_t2i = []

                for i in range(batch_size):
                    # Image-to-text hard negatives
                    neg_scores_i2t = logits[i].clone()
                    neg_scores_i2t[i] = -float('inf')  # Exclude positive
                    _, hard_neg_indices = torch.topk(neg_scores_i2t, num_hard_negatives)
                    hard_negatives_i2t.append(hard_neg_indices)

                    # Text-to-image hard negatives
                    neg_scores_t2i = logits[:, i].clone()
                    neg_scores_t2i[i] = -float('inf')  # Exclude positive
                    _, hard_neg_indices = torch.topk(neg_scores_t2i, num_hard_negatives)
                    hard_negatives_t2i.append(hard_neg_indices)

                # Calculate metrics considering hard negatives
                i2t_acc = self._calculate_accuracy_with_hard_negatives(
                    logits, labels, hard_negatives_i2t, axis=1
                )
                t2i_acc = self._calculate_accuracy_with_hard_negatives(
                    logits.t(), labels, hard_negatives_t2i, axis=1
                )

                all_metrics.append({
                    'i2t_accuracy': i2t_acc,
                    't2i_accuracy': t2i_acc
                })

        # Aggregate metrics
        avg_metrics = {
            'i2t_accuracy_hard': np.mean([m['i2t_accuracy'] for m in all_metrics]),
            't2i_accuracy_hard': np.mean([m['t2i_accuracy'] for m in all_metrics])
        }

        return avg_metrics

    def _calculate_accuracy_with_hard_negatives(self,
                                               logits: torch.Tensor,
                                               labels: torch.Tensor,
                                               hard_negatives: List[torch.Tensor],
                                               axis: int) -> float:
        """
        Calculate accuracy considering hard negatives.

        Args:
            logits: Similarity logits
            labels: Ground truth labels
            hard_negatives: List of hard negative indices
            axis: Axis for argmax

        Returns:
            Accuracy score
        """
        batch_size = logits.shape[0]
        correct = 0

        for i in range(batch_size):
            if axis == 1:
                scores = logits[i]
            else:
                scores = logits[:, i]

            # Consider only positive and hard negatives
            relevant_indices = torch.cat([labels[i:i+1], hard_negatives[i]])
            relevant_scores = scores[relevant_indices]

            # Check if positive has highest score
            if torch.argmax(relevant_scores) == 0:
                correct += 1

        return correct / batch_size


def evaluate_retrieval_comprehensive(model: nn.Module,
                                    test_loader: Any,
                                    device: str = "cuda") -> Dict[str, Any]:
    """
    Comprehensive retrieval evaluation.

    Args:
        model: Vision-language model
        test_loader: Test DataLoader
        device: Device to use

    Returns:
        Comprehensive retrieval metrics
    """
    # Standard retrieval
    evaluator = RetrievalEvaluator(device=device)
    standard_metrics = evaluator.evaluate(model, test_loader, task="both")

    # Contrastive evaluation with hard negatives
    contrastive_eval = ContrastiveRetrievalEvaluator(device=device)
    hard_neg_metrics = contrastive_eval.evaluate_with_hard_negatives(
        model, test_loader, num_hard_negatives=10
    )

    # Combine all metrics
    results = {
        'standard_retrieval': standard_metrics,
        'hard_negative_evaluation': hard_neg_metrics,
        'summary': {
            'i2t_r@1': standard_metrics['image_to_text'].recall_at_1,
            'i2t_r@5': standard_metrics['image_to_text'].recall_at_5,
            't2i_r@1': standard_metrics['text_to_image'].recall_at_1,
            't2i_r@5': standard_metrics['text_to_image'].recall_at_5,
            'mean_rank': (standard_metrics['image_to_text'].mean_rank +
                         standard_metrics['text_to_image'].mean_rank) / 2
        }
    }

    return results