"""F3 time_context — weight 0.15.

Measures how strongly a candidate phrase is associated with the current
KST time bucket (morning / afternoon / evening / night).

Score formula:
  - If the phrase has fewer than 3 total observations across all buckets → 0.0
  - Otherwise: min(1.0, (same_bucket_count / total_count) * 1.5)
"""
from __future__ import annotations

name = "time_context"
weight = 0.15


def compute(cand: object, ctx: object) -> float:
    """Return time_context score in [0.0, 1.0]."""
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    buckets: dict = ctx.time_bucket_counts.get(phrase, {})  # type: ignore[attr-defined]
    total: int = ctx.time_bucket_total.get(phrase, 0)  # type: ignore[attr-defined]
    same: int = buckets.get(ctx.time_bucket_kst, 0)  # type: ignore[attr-defined]

    if total < 3:
        return 0.0
    return min(1.0, (same / total) * 1.5)
