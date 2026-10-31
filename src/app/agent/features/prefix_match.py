"""F1 — prefix_match feature (weight 0.35).

Pure-Python implementation.  No DB I/O — safe to unit-test in isolation.

Scoring rules:
  - len(prefix) == 0          → 0.0
  - len(prefix) <= 2          → startswith score only
  - len(prefix) >= 3          → max(startswith_score, trigram_score)
"""
from __future__ import annotations

name = "prefix_match"
weight = 0.35


def _trigrams(s: str) -> set[str]:
    """
    Return the set of overlapping 3-char trigrams for *s*.

    Follows pg_trgm conventions:
      - lowercase the string
      - pad with two leading spaces and one trailing space
      - slide a 3-char window over the padded string
    """
    padded = "  " + s.lower() + " "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _startswith_score(prefix: str, phrase: str) -> float:
    if phrase.startswith(prefix):
        return min(1.0, len(prefix) / max(len(phrase), 1) * 1.5)
    return 0.0


def _trigram_score(prefix: str, phrase: str) -> float:
    pt = _trigrams(prefix)
    ct = _trigrams(phrase)
    overlap = len(pt & ct)
    return overlap / max(len(pt), 1) * 0.5


def compute(cand: object, ctx: object) -> float:  # type: ignore[override]
    """Return prefix_match score in [0.0, 1.0]."""
    prefix: str = ctx.prefix  # type: ignore[attr-defined]
    phrase: str = cand.phrase  # type: ignore[attr-defined]

    if len(prefix) == 0:
        return 0.0

    sw_score = _startswith_score(prefix, phrase)

    if len(prefix) <= 2:
        return sw_score

    # len >= 3: return best of startswith OR trigram
    tg_score = _trigram_score(prefix, phrase)
    return max(sw_score, tg_score)
