"""Offline relevance metrics on graded judgments (0 to 3)."""

from __future__ import annotations

import numpy as np

RELEVANT = 2  # grade at or above which a result counts as relevant for MRR and recall


def dcg(grades, k: int = 10) -> float:
    g = np.asarray(grades, dtype=np.float64)[:k]
    if g.size == 0:
        return 0.0
    return float(np.sum((2.0 ** g - 1.0) / np.log2(np.arange(2, g.size + 2))))


def ndcg(ranked_grades, all_grades, k: int = 10) -> float:
    """NDCG@k with exponential gain. The ideal ranking comes from every known grade for the query, not only the
    ones this system retrieved, so a system cannot score well by retrieving little."""
    ideal = dcg(sorted(all_grades, reverse=True), k)
    return dcg(ranked_grades, k) / ideal if ideal > 0 else 0.0


def reciprocal_rank(ranked_grades, threshold: int = RELEVANT) -> float:
    for i, g in enumerate(ranked_grades):
        if g >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(ranked_grades, n_relevant: int, k: int = 50, threshold: int = RELEVANT) -> float:
    """Relevant results in the top k over min(k, relevant products that exist)."""
    if n_relevant <= 0:
        return 0.0
    hit = int(np.sum(np.asarray(ranked_grades[:k]) >= threshold))
    return hit / min(k, n_relevant)
