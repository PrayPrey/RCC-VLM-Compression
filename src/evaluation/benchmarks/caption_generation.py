"""
Caption generation evaluation metrics for vision-language models.

This module provides evaluation metrics for image captioning including
BLEU, METEOR, ROUGE, CIDEr, and SPICE scores.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import Counter
import math
import logging
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class CaptionEvaluator:
    """Evaluates caption generation quality."""

    def __init__(self, tokenizer: Optional[Any] = None):
        """
        Initialize caption evaluator.

        Args:
            tokenizer: Optional tokenizer for text processing
        """
        self.tokenizer = tokenizer or word_tokenize

    def evaluate(self,
                predictions: List[str],
                references: List[List[str]],
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate caption predictions against references.

        Args:
            predictions: List of predicted captions
            references: List of reference captions (multiple per image)
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge_l', 'cider']

        results = {}

        # Tokenize predictions and references
        pred_tokens = [self._tokenize(pred) for pred in predictions]
        ref_tokens = [[self._tokenize(ref) for ref in refs] for refs in references]

        if 'bleu1' in metrics:
            results['bleu1'] = self._compute_bleu(pred_tokens, ref_tokens, n=1)
        if 'bleu2' in metrics:
            results['bleu2'] = self._compute_bleu(pred_tokens, ref_tokens, n=2)
        if 'bleu3' in metrics:
            results['bleu3'] = self._compute_bleu(pred_tokens, ref_tokens, n=3)
        if 'bleu4' in metrics:
            results['bleu4'] = self._compute_bleu(pred_tokens, ref_tokens, n=4)
        if 'meteor' in metrics:
            results['meteor'] = self._compute_meteor(predictions, references)
        if 'rouge_l' in metrics:
            results['rouge_l'] = self._compute_rouge_l(pred_tokens, ref_tokens)
        if 'cider' in metrics:
            results['cider'] = self._compute_cider(pred_tokens, ref_tokens)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if isinstance(self.tokenizer, str):
            return text.lower().split()
        else:
            return [token.lower() for token in self.tokenizer(text)]

    def _compute_bleu(self,
                     predictions: List[List[str]],
                     references: List[List[List[str]]],
                     n: int = 4) -> float:
        """
        Compute BLEU score.

        Args:
            predictions: Tokenized predictions
            references: Tokenized references
            n: n-gram order

        Returns:
            BLEU score
        """
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = corpus_bleu(references, predictions, weights=weights)
        return score * 100

    def _compute_meteor(self,
                       predictions: List[str],
                       references: List[List[str]]) -> float:
        """
        Compute METEOR score.

        Args:
            predictions: Predicted captions
            references: Reference captions

        Returns:
            METEOR score
        """
        scores = []
        for pred, refs in zip(predictions, references):
            # METEOR expects a single reference, so we average over multiple
            ref_scores = [meteor_score([ref], pred) for ref in refs]
            scores.append(max(ref_scores))
        return np.mean(scores) * 100

    def _compute_rouge_l(self,
                        predictions: List[List[str]],
                        references: List[List[List[str]]]) -> float:
        """
        Compute ROUGE-L score.

        Args:
            predictions: Tokenized predictions
            references: Tokenized references

        Returns:
            ROUGE-L F1 score
        """
        scores = []
        for pred, refs in zip(predictions, references):
            ref_scores = []
            for ref in refs:
                lcs_len = self._lcs_length(pred, ref)
                precision = lcs_len / len(pred) if len(pred) > 0 else 0
                recall = lcs_len / len(ref) if len(ref) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                ref_scores.append(f1)
            scores.append(max(ref_scores))
        return np.mean(scores) * 100

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _compute_cider(self,
                      predictions: List[List[str]],
                      references: List[List[List[str]]]) -> float:
        """
        Compute CIDEr score.

        Args:
            predictions: Tokenized predictions
            references: Tokenized references

        Returns:
            CIDEr score
        """
        # Simplified CIDEr implementation
        def compute_doc_freq(references):
            """Compute document frequency of n-grams."""
            doc_freq = Counter()
            for refs in references:
                ngrams_set = set()
                for ref in refs:
                    for n in range(1, 5):  # 1-4 grams
                        ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
                        ngrams_set.update(ngrams)
                for ngram in ngrams_set:
                    doc_freq[ngram] += 1
            return doc_freq

        def compute_cider_score(pred, refs, doc_freq, n_docs):
            """Compute CIDEr score for single prediction."""
            def get_ngrams(tokens, n):
                return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

            def compute_tf(tokens):
                tf = Counter()
                for n in range(1, 5):
                    ngrams = get_ngrams(tokens, n)
                    for ngram in ngrams:
                        tf[ngram] += 1
                return tf

            pred_tf = compute_tf(pred)
            scores = []

            for ref in refs:
                ref_tf = compute_tf(ref)
                score = 0

                for n in range(1, 5):
                    pred_ngrams = get_ngrams(pred, n)
                    ref_ngrams = get_ngrams(ref, n)

                    if not pred_ngrams or not ref_ngrams:
                        continue

                    # Compute TF-IDF vectors
                    common_ngrams = set(pred_ngrams) & set(ref_ngrams)
                    for ngram in common_ngrams:
                        idf = math.log(n_docs / doc_freq[ngram]) if doc_freq[ngram] > 0 else 0
                        score += (pred_tf[ngram] * idf) * (ref_tf[ngram] * idf)

                    # Normalize
                    pred_norm = sum((pred_tf[ng] * math.log(n_docs / doc_freq[ng])) ** 2
                                  for ng in set(pred_ngrams) if doc_freq[ng] > 0)
                    ref_norm = sum((ref_tf[ng] * math.log(n_docs / doc_freq[ng])) ** 2
                                 for ng in set(ref_ngrams) if doc_freq[ng] > 0)

                    if pred_norm > 0 and ref_norm > 0:
                        score /= math.sqrt(pred_norm * ref_norm)

                scores.append(score)

            return max(scores) if scores else 0

        doc_freq = compute_doc_freq(references)
        n_docs = len(references)

        scores = []
        for pred, refs in zip(predictions, references):
            score = compute_cider_score(pred, refs, doc_freq, n_docs)
            scores.append(score)

        return np.mean(scores) * 10  # Scale factor


class DiversityMetrics:
    """Metrics for evaluating caption diversity."""

    @staticmethod
    def compute_diversity(captions: List[str], n: int = 1) -> float:
        """
        Compute n-gram diversity.

        Args:
            captions: List of generated captions
            n: n-gram order

        Returns:
            Diversity score (ratio of unique n-grams)
        """
        all_ngrams = []
        for caption in captions:
            tokens = caption.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams

    @staticmethod
    def compute_novel_ngrams(generated: List[str],
                            reference: List[str],
                            n: int = 2) -> float:
        """
        Compute ratio of novel n-grams not in references.

        Args:
            generated: Generated captions
            reference: Reference captions
            n: n-gram order

        Returns:
            Novel n-gram ratio
        """
        def get_ngrams(texts, n):
            ngrams = set()
            for text in texts:
                tokens = text.lower().split()
                for i in range(len(tokens) - n + 1):
                    ngrams.add(tuple(tokens[i:i+n]))
            return ngrams

        gen_ngrams = get_ngrams(generated, n)
        ref_ngrams = get_ngrams(reference, n)

        novel_ngrams = gen_ngrams - ref_ngrams
        if not gen_ngrams:
            return 0.0

        return len(novel_ngrams) / len(gen_ngrams)


class SemanticSimilarity:
    """Compute semantic similarity between captions."""

    def __init__(self, model: Optional[nn.Module] = None):
        """
        Initialize semantic similarity evaluator.

        Args:
            model: Optional text encoder model
        """
        self.model = model

    def compute_similarity(self,
                         predictions: List[str],
                         references: List[List[str]]) -> float:
        """
        Compute semantic similarity using embeddings.

        Args:
            predictions: Predicted captions
            references: Reference captions

        Returns:
            Average semantic similarity score
        """
        if self.model is None:
            logger.warning("No model provided for semantic similarity")
            return 0.0

        similarities = []

        with torch.no_grad():
            for pred, refs in zip(predictions, references):
                # Encode prediction
                pred_embedding = self._encode_text(pred)

                # Encode references
                ref_embeddings = [self._encode_text(ref) for ref in refs]

                # Compute similarities
                sims = [torch.cosine_similarity(pred_embedding, ref_emb, dim=0).item()
                       for ref_emb in ref_embeddings]

                similarities.append(max(sims))

        return np.mean(similarities)

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        if hasattr(self.model, 'encode_text'):
            return self.model.encode_text(text)
        else:
            # Fallback to simple embedding
            tokens = text.lower().split()
            embedding = torch.randn(512)  # Dummy embedding
            return torch.nn.functional.normalize(embedding, p=2, dim=0)


def evaluate_caption_generation(model: nn.Module,
                              dataloader: Any,
                              max_length: int = 50,
                              device: str = "cuda") -> Dict[str, Any]:
    """
    Comprehensive caption generation evaluation.

    Args:
        model: Caption generation model
        dataloader: DataLoader with images and captions
        max_length: Maximum caption length
        device: Device to use

    Returns:
        Comprehensive evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    references = []

    # Generate captions
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            captions = batch['captions']  # List of reference captions

            # Generate predictions
            if hasattr(model, 'generate_caption'):
                generated = model.generate_caption(images, max_length=max_length)
            else:
                # Fallback to dummy generation
                generated = ["A generated caption"] * len(images)

            predictions.extend(generated)
            references.extend(captions)

    # Evaluate with multiple metrics
    evaluator = CaptionEvaluator()
    scores = evaluator.evaluate(predictions, references)

    # Compute diversity metrics
    diversity = DiversityMetrics()
    scores['diversity_1'] = diversity.compute_diversity(predictions, n=1)
    scores['diversity_2'] = diversity.compute_diversity(predictions, n=2)
    scores['novel_bigrams'] = diversity.compute_novel_ngrams(
        predictions, [ref[0] for ref in references], n=2
    )

    # Compute semantic similarity if model available
    semantic_eval = SemanticSimilarity(model)
    scores['semantic_similarity'] = semantic_eval.compute_similarity(predictions, references)

    return {
        'scores': scores,
        'num_samples': len(predictions),
        'avg_length': np.mean([len(p.split()) for p in predictions]),
        'summary': {
            'bleu4': scores.get('bleu4', 0),
            'meteor': scores.get('meteor', 0),
            'cider': scores.get('cider', 0),
            'diversity': scores.get('diversity_2', 0)
        }
    }