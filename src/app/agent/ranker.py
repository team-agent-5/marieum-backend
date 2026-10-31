"""Ranker orchestrator (Phase B1 재구성).

Confidence 4-단계 분기 라우팅:
    classify_branch(input.asr_confidence) → retry / low / mid / high

지연 최적화:
    * ``high`` / ``retry`` 분기 → ``build_ranking_context`` **skip** (DB 쿼리 0건,
      목표 지연 ~85ms, 설계 §2.4).
    * ``low`` / ``mid`` 분기 → ``build_ranking_context`` 호출 후 branches 디스패치.

호환성:
    * ``branches/*.strategy()`` 는 ``ctx: RankingContext | None`` 수용 (retry/high
      에서 ctx 를 사용하지 않음 — 시그니처는 통일, 호출측 None 허용).
    * 기존 ``WEIGHTS`` dict / accuracy_mode LLM rerank 로직은 Phase B1 에서
      ``branches/low.py`` 로 이관 완료 (ranker.py 에서 삭제).

설계 참조:
    * ``marieum-plan/design/agent-algorithm.md`` §2 (분기 로직)
    * ``backend/docs/agent-rework-plan.md`` §2 Phase B1
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .branches import (
    high_strategy,
    low_strategy,
    mid_strategy,
    retry_strategy,
)
from .confidence_router import classify_branch
from .context import build_ranking_context
from .contracts import AgentInput, AgentOutput

log = logging.getLogger(__name__)


async def rank(input: AgentInput, session: AsyncSession) -> AgentOutput:
    """4-분기 라우팅으로 후보 ranking.

    분기 결정:
        * ``conf < 0.40`` → ``retry`` (api/phrase.py 가 422 로 선 차단해야 정상)
        * ``0.40 ≤ conf < 0.90`` → ``low``  (6-feature 가중합, ctx 필요)
        * ``0.90 ≤ conf < 0.98`` → ``mid``  (verbatim + 매칭 2개, ctx 필요)
        * ``conf ≥ 0.98`` → ``high`` (verbatim 단독, ctx skip — ~85ms 지연 목표)

    Args:
        input: AgentInput. ``asr_confidence`` 로 분기 결정.
        session: AsyncSession — low/mid 분기에서 DB 조회.

    Returns:
        AgentOutput. ``branch`` 필드에 결정된 분기 enum 포함.
    """
    branch = classify_branch(input.asr_confidence)

    # ── 고신뢰·재질의: ctx 빌드 skip (DB 쿼리 0건, 지연 최적화) ────────────────
    if branch == "retry":
        # api/phrase.py 가 422 로 선 차단하지만, 안전망으로 호출돼도 즉시 빈 응답.
        return await retry_strategy(input, ctx=None, session=session)
    if branch == "high":
        # ranker/pool/context 전부 skip. verbatim 단독 1개 반환 (설계 §2.4).
        return await high_strategy(input, ctx=None, session=session)

    # ── 저신뢰·중신뢰: ctx 빌드 필요 (phrase_log 집계, rejected_counts 포함) ──
    ctx = await build_ranking_context(session, input)

    if branch == "low":
        return await low_strategy(input, ctx, session)

    # mid
    return await mid_strategy(input, ctx, session)
