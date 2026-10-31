"""F5 semantic_similarity — pgvector cosine similarity between prefix_vec and candidate.embedding.

Gate: len(prefix) >= 3 AND cosine_sim >= 0.65.
Formula: (cosine_sim - 0.65) / 0.35.  Range [0.0, 1.0].

Note: pgvector `<=>` returns cosine DISTANCE (0=identical..2=opposite).
We work with cosine SIMILARITY = 1 - distance.
"""
from __future__ import annotations

import math

name = "semantic_similarity"
weight = 0.10

_THRESHOLD = 0.65


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity. Returns in [-1, 1]; typically clamped to [0, 1] for sentence embeddings."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute(cand, ctx) -> float:
    """Return F5 raw score in [0.0, 1.0]."""
    if len(ctx.prefix) < 3:
        return 0.0
    if ctx.prefix_vec is None or cand.embedding is None:
        return 0.0
    cos = _cosine_similarity(ctx.prefix_vec, cand.embedding)
    # Runtime assertion: cosine for ko-sroberta is [0, 1] in practice; catch sign-flip bugs.
    if not (0.0 <= cos <= 1.0 + 1e-6):
        import logging

        logging.getLogger(__name__).warning("F5 cosine out of [0,1]: %.3f", cos)
    if cos < _THRESHOLD:
        return 0.0
    return (cos - _THRESHOLD) / (1.0 - _THRESHOLD)
