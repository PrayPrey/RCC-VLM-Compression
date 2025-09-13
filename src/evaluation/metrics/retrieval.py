"""
Retrieval metrics for image-text matching evaluation.

This module provides metrics for evaluating cross-modal retrieval performance
including Recall@K, Mean Reciprocal Rank, and NDCG.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Compute retrieval metrics for image-text matching."""

    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Initialize retrieval metrics.

        Args:
            k_values: List of K values for Recall@K computation
        """
        self.k_values = k_values

    def compute_recall_at_k(self, similarity_matrix: torch.Tensor,
                           k: int = 5) -> Tuple[float, float]:
        """
        Compute Recall@K for both image-to-text and text-to-image.

        Args:
            similarity_matrix: [N, N] similarity matrix
            k: Number of top results to consider

        Returns:
            image_to_text_recall, text_to_image_recall
        """
        batch_size = similarity_matrix.shape[0]

        # Image-to-text retrieval
        i2t_ranks = []
        for i in range(batch_size):
            # Get similarity scores for this image
            scores = similarity_matrix[i]
            # Sort in descending order
            sorted_indices = torch.argsort(scores, descending=True)
            # Find rank of correct match (diagonal element)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            i2t_ranks.append(rank)

        # Text-to-image retrieval
        t2i_ranks = []
        similarity_matrix_t = similarity_matrix.t()
        for i in range(batch_size):
            scores = similarity_matrix_t[i]
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            t2i_ranks.append(rank)

        # Calculate Recall@K
        i2t_recall = sum(r < k for r in i2t_ranks) / batch_size
        t2i_recall = sum(r < k for r in t2i_ranks) / batch_size

        return i2t_recall, t2i_recall

    def compute_mrr(self, similarity_matrix: torch.Tensor) -> Tuple[float, float]:
        """
        Compute Mean Reciprocal Rank.

        Args:
            similarity_matrix: [N, N] similarity matrix

        Returns:
            image_to_text_mrr, text_to_image_mrr
        """
        batch_size = similarity_matrix.shape[0]

        # Image-to-text MRR
        i2t_mrr = 0.0
        for i in range(batch_size):
            scores = similarity_matrix[i]
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            i2t_mrr += 1.0 / (rank + 1)
        i2t_mrr /= batch_size

        # Text-to-image MRR
        t2i_mrr = 0.0
        similarity_matrix_t = similarity_matrix.t()
        for i in range(batch_size):
            scores = similarity_matrix_t[i]
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            t2i_mrr += 1.0 / (rank + 1)
        t2i_mrr /= batch_size

        return i2t_mrr, t2i_mrr

    def compute_ndcg(self, similarity_matrix: torch.Tensor,
                     k: int = 10) -> Tuple[float, float]:
        """
        Compute Normalized Discounted Cumulative Gain.

        Args:
            similarity_matrix: [N, N] similarity matrix
            k: Truncation parameter

        Returns:
            image_to_text_ndcg, text_to_image_ndcg
        """
        batch_size = similarity_matrix.shape[0]

        def dcg_at_k(scores: torch.Tensor, relevant_idx: int, k: int) -> float:
            """Calculate DCG@K for a single query."""
            sorted_indices = torch.argsort(scores, descending=True)[:k]

            dcg = 0.0
            for i, idx in enumerate(sorted_indices):
                if idx == relevant_idx:
                    # Relevance is 1 for correct match, 0 otherwise
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
                    break
            return dcg

        # Image-to-text NDCG
        i2t_ndcg = 0.0
        for i in range(batch_size):
            scores = similarity_matrix[i]
            dcg = dcg_at_k(scores, i, k)
            # IDCG for binary relevance is 1/log2(2) = 1
            idcg = 1.0
            i2t_ndcg += dcg / idcg
        i2t_ndcg /= batch_size

        # Text-to-image NDCG
        t2i_ndcg = 0.0
        similarity_matrix_t = similarity_matrix.t()
        for i in range(batch_size):
            scores = similarity_matrix_t[i]
            dcg = dcg_at_k(scores, i, k)
            idcg = 1.0
            t2i_ndcg += dcg / idcg
        t2i_ndcg /= batch_size

        return i2t_ndcg, t2i_ndcg

    def compute_all_metrics(self, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        """
        Compute all retrieval metrics.

        Args:
            similarity_matrix: [N, N] similarity matrix

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Recall@K for different K values
        for k in self.k_values:
            i2t_recall, t2i_recall = self.compute_recall_at_k(similarity_matrix, k)
            metrics[f'i2t_recall@{k}'] = i2t_recall
            metrics[f't2i_recall@{k}'] = t2i_recall
            metrics[f'mean_recall@{k}'] = (i2t_recall + t2i_recall) / 2

        # Mean Reciprocal Rank
        i2t_mrr, t2i_mrr = self.compute_mrr(similarity_matrix)
        metrics['i2t_mrr'] = i2t_mrr
        metrics['t2i_mrr'] = t2i_mrr
        metrics['mean_mrr'] = (i2t_mrr + t2i_mrr) / 2

        # NDCG
        i2t_ndcg, t2i_ndcg = self.compute_ndcg(similarity_matrix, k=10)
        metrics['i2t_ndcg@10'] = i2t_ndcg
        metrics['t2i_ndcg@10'] = t2i_ndcg
        metrics['mean_ndcg@10'] = (i2t_ndcg + t2i_ndcg) / 2

        return metrics

    def compute_rsum(self, metrics: Dict[str, float]) -> float:
        """
        Compute R-sum (sum of recall values).

        Args:
            metrics: Dictionary containing recall metrics

        Returns:
            R-sum value
        """
        rsum = 0.0
        for k in [1, 5, 10]:
            rsum += metrics.get(f'i2t_recall@{k}', 0.0)
            rsum += metrics.get(f't2i_recall@{k}', 0.0)
        return rsum


class ContrastiveMetrics:
    """Metrics for contrastive learning evaluation."""

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive metrics.

        Args:
            temperature: Temperature for scaling
        """
        self.temperature = temperature

    def compute_nce_accuracy(self, image_embeds: torch.Tensor,
                            text_embeds: torch.Tensor) -> float:
        """
        Compute NCE accuracy for contrastive learning.

        Args:
            image_embeds: [N, D] image embeddings
            text_embeds: [N, D] text embeddings

        Returns:
            NCE accuracy
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature

        # Ground truth: diagonal elements should be maximum
        labels = torch.arange(len(logits), device=logits.device)

        # Image-to-text accuracy
        i2t_pred = logits.argmax(dim=1)
        i2t_acc = (i2t_pred == labels).float().mean()

        # Text-to-image accuracy
        t2i_pred = logits.argmax(dim=0)
        t2i_acc = (t2i_pred == labels).float().mean()

        return (i2t_acc + t2i_acc) / 2

    def compute_alignment_uniformity(self, image_embeds: torch.Tensor,
                                    text_embeds: torch.Tensor) -> Tuple[float, float]:
        """
        Compute alignment and uniformity metrics.

        Args:
            image_embeds: [N, D] image embeddings
            text_embeds: [N, D] text embeddings

        Returns:
            alignment_loss, uniformity_loss
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Alignment: distance between positive pairs
        alignment = (image_embeds - text_embeds).norm(dim=1).mean()

        # Uniformity: log of average pairwise Gaussian potential
        def uniformity(x):
            sq_dist = torch.cdist(x, x) ** 2
            # Exclude diagonal
            mask = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
            sq_dist = sq_dist[mask]
            return torch.exp(-2 * sq_dist).mean().log()

        image_uniformity = uniformity(image_embeds)
        text_uniformity = uniformity(text_embeds)
        avg_uniformity = (image_uniformity + text_uniformity) / 2

        return alignment.item(), avg_uniformity.item()


def evaluate_retrieval(model, dataloader, device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate model on retrieval tasks.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use

    Returns:
        Dictionary of retrieval metrics
    """
    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            texts = batch['texts']

            # Get embeddings
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(texts)

            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_embeds, all_text_embeds.t())

    # Compute metrics
    retrieval_metrics = RetrievalMetrics()
    metrics = retrieval_metrics.compute_all_metrics(similarity_matrix)

    # Add contrastive metrics
    contrastive_metrics = ContrastiveMetrics()
    metrics['nce_accuracy'] = contrastive_metrics.compute_nce_accuracy(
        all_image_embeds, all_text_embeds
    )

    alignment, uniformity = contrastive_metrics.compute_alignment_uniformity(
        all_image_embeds, all_text_embeds
    )
    metrics['alignment'] = alignment
    metrics['uniformity'] = uniformity

    return metrics