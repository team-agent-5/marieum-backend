"""Feature protocol — all ranking features must satisfy this interface."""
from __future__ import annotations

from typing import Protocol

from app.agent.contracts import PoolCandidate, RankingContext


class Feature(Protocol):
    name: str
    weight: float

    def compute(self, cand: PoolCandidate, ctx: RankingContext) -> float:
        ...
