"""F7 preference_bonus — NOT included in base score sum; added as separate bonus.

Max bonus: 0.15.  Min: 0.0.

Logic (in order):
  1. Exact match against preference content → 0.15
  2. Exact match in boost_phrases list → 0.15
  3. Regex match (only for preference_type == 'topic_preference') → 0.12
  4. No match → 0.0

Note: patient_preference has no weight column in v4 (per data-model.md); weight is 1.0.
"""
from __future__ import annotations

import re

name = "preference_bonus"
weight = 0.0  # applied as bonus outside base sum, not as a weighted feature


def compute(cand: object, ctx: object) -> float:
    """Return preference bonus to ADD to base score. MAX across all prefs. Range [0.0, 0.15]."""
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    prefs: list[dict] = ctx.prefs  # type: ignore[attr-defined]

    best = 0.0
    for p in prefs:
        boost_phrases: list = p.get("boost_phrases") or []
        content: str = p.get("content") or ""

        # exact match in content or in boost_phrases
        if phrase == content or phrase in boost_phrases:
            best = max(best, 0.15)
            continue

        # regex fallback: content treated as regex if preference_type == 'topic_preference'
        if p.get("preference_type") == "topic_preference":
            try:
                if re.search(content, phrase):
                    best = max(best, 0.12)
            except re.error:
                continue

    return best
