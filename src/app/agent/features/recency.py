"""F6 — recency feature (weight 0.10).

Pure-Python, no I/O.  Exponential decay with a 24-hour half-life.
"""
from __future__ import annotations

import math
from datetime import UTC

name = "recency"
weight = 0.10


def compute(cand: object, ctx: object) -> float:  # type: ignore[override]
    """Return recency score in (0.0, 1.0].

    Score = exp(-ln(2) * hours_since_last_use / 24)
    Half-life: 24 hours.
    """
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    last_at_map: dict = ctx.last_at  # type: ignore[attr-defined]
    now = ctx.now  # type: ignore[attr-defined]

    last = last_at_map.get(phrase)
    if last is None:
        return 0.0

    # Normalise to UTC — DB stores naive UTC datetimes
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    hours = max(0.0, (now - last).total_seconds() / 3600.0)
    return math.exp(-math.log(2) * hours / 24)
