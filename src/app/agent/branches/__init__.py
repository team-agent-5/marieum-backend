"""Confidence 분기별 ranker 전략.

``ranker.rank()`` 가 ``confidence_router.classify_branch()`` 로 분기 결정 후
본 서브패키지의 ``<branch>.strategy()`` 를 디스패치한다.

디스패치 패턴::

    from app.agent.branches import (
        high_strategy,
        low_strategy,
        mid_strategy,
        retry_strategy,
    )
    from app.agent.confidence_router import classify_branch

    branch = classify_branch(input.asr_confidence)
    if branch == "retry":
        output = await retry_strategy(input, ctx, session)
    elif branch == "low":
        output = await low_strategy(input, ctx, session)
    elif branch == "mid":
        output = await mid_strategy(input, ctx, session)
    else:  # "high"
        output = await high_strategy(input, ctx, session)
"""
from __future__ import annotations

from .high import strategy as high_strategy
from .low import strategy as low_strategy
from .mid import strategy as mid_strategy
from .retry import strategy as retry_strategy

__all__ = [
    "high_strategy",
    "low_strategy",
    "mid_strategy",
    "retry_strategy",
]
