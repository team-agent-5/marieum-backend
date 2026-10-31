"""F2 — patient_frequency feature (weight 0.20).

Pure-Python, no I/O.  Log-scale normalisation with a 30-count reference point.
"""
from __future__ import annotations

import math

name = "patient_frequency"
weight = 0.20


def compute(cand: object, ctx: object) -> float:  # type: ignore[override]
    """Return patient_frequency score in [0.0, 1.0]."""
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    freq_counts: dict = ctx.freq_counts  # type: ignore[attr-defined]
    cnt = freq_counts.get(phrase, 0)
    if cnt == 0:
        return 0.0
    return min(1.0, math.log1p(cnt) / math.log1p(30))
