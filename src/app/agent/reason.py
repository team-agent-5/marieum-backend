"""reason_label priority resolver — threshold-gated ladder per design §4.1.

Design agent-personalization.md §4.1 ladder (first match wins):
  1. preference_bonus >= 0.12            -> "registered_preference"
  2. time_context    >= 0.5              -> "time_context"
  3. recency         >= 0.5              -> "recency"
  4. patient_frequency >= 0.3            -> "patient_frequency"
  5. sequence_context >= 0.4             -> "sequence_context"
  6. semantic_similarity >= 0.286        -> "semantic_similarity"
     (raw cosine >= 0.75 maps to normalized score >= (0.75-0.65)/0.35 = 0.286)
  7. otherwise                            -> None (no personalization signal)

`cold_start` label is NOT emitted here — that classification belongs to the ranker
(`ctx.freq_total == 0`) which already flows through `audit_payload.cold_start` and
drives `fallback_hint` at the API layer. Returning None lets the UI either hide the
reason badge or fall back to a generic "match" display.
"""
from __future__ import annotations

from .contracts import CandidateScore


def pick_reason_label(c: CandidateScore) -> str | None:
    """Return single reason_label per threshold-gated ladder, or None if no signal."""
    s = c.scores
    if c.bonus >= 0.12:
        return "registered_preference"
    if s.time_context >= 0.5:
        return "time_context"
    if s.recency >= 0.5:
        return "recency"
    if s.patient_frequency >= 0.3:
        return "patient_frequency"
    if s.sequence_context >= 0.4:
        return "sequence_context"
    if s.semantic_similarity >= 0.286:   # raw cosine >= 0.75
        return "semantic_similarity"
    return None
