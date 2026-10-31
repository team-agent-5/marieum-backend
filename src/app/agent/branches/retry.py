"""재질의 분기 전략 (``conf < RETRY_CONFIDENCE``, 기본 0.60 — ML 보고서 2026-11-01 반영).

**원칙**: ``api/phrase.py`` 가 422 ``ASR_LOW_CONFIDENCE`` 로 미리 차단하므로
본 전략은 **원래 호출되지 않는다**.

안전망: 만일 라우팅 실수로 호출돼도 ``suggestions=[]`` 를 즉시 반환하여
지연을 소모하지 않고 ranker 계층에서 재질의 분기를 명시한다.

설계 참조:
    * ``marieum-plan/design/agent-algorithm.md`` §2.1 (재질의 분기)
    * ``backend/docs/agent-rework-plan.md`` §2 Phase A4 retry.py
"""
from __future__ import annotations

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.contracts import AgentInput, AgentOutput, RankingContext

log = logging.getLogger(__name__)


async def strategy(
    input: AgentInput,
    ctx: RankingContext | None,
    session: AsyncSession,
) -> AgentOutput:
    """재질의 분기 — 호출돼선 안 되나, 호출 시 즉시 빈 응답 반환.

    Args:
        input: AgentInput (asr_confidence < 0.40 인 경우).
        ctx: RankingContext | None — ranker.py 가 재질의 분기에서 ctx 빌드를 skip
            하므로 보통 ``None`` 이 들어온다 (지연 최적화).
        session: AsyncSession (사용하지 않음).

    Returns:
        suggestions 가 비어 있는 AgentOutput. ``branch="retry"``.
        audit_payload 에 ``retry_fallback`` 플래그 기록.
    """
    t0 = time.perf_counter()

    log.warning(
        "retry.strategy() invoked — api/phrase.py should have returned 422 ASR_LOW_CONFIDENCE "
        "before reaching ranker. asr_confidence=%s, patient_id=%s",
        input.asr_confidence,
        input.patient_id,
    )

    latency_ms = max(1, int((time.perf_counter() - t0) * 1000))

    audit = {
        "branch": "retry",
        "retry_fallback": True,
        "reason": "ranker_reached_despite_retry_branch",
        "asr_confidence": input.asr_confidence,
        "final_transcript": input.final_transcript,
        "candidates": [],
        "pool_size": 0,
        "llm_called": False,
    }

    return AgentOutput(
        suggestions=[],
        latency_ms=latency_ms,
        cold_start=False,
        accuracy_mode_used=False,
        branch="retry",
        audit_payload=audit,
    )
