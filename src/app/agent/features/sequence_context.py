"""F4 sequence_context — bigram transition probability from previous_phrase.

Gate: previous_phrase is not None AND bigram_total >= 2.
Formula: min(1.0, ratio * 1.2) where ratio = bigram_counts[phrase] / bigram_total.
"""
from __future__ import annotations

name = "sequence_context"
weight = 0.10

_NOISE_THRESHOLD = 2   # mirror F3 pattern


def compute(cand: object, ctx: object) -> float:
    if ctx.previous_phrase is None:  # type: ignore[attr-defined]
        return 0.0
    if ctx.bigram_total < _NOISE_THRESHOLD:  # type: ignore[attr-defined]
        return 0.0
    cnt = ctx.bigram_counts.get(cand.phrase, 0)  # type: ignore[attr-defined]
    if cnt == 0:
        return 0.0
    ratio = cnt / ctx.bigram_total  # type: ignore[attr-defined]
    return min(1.0, ratio * 1.2)
