"""고신뢰 분기 전략 (``conf ≥ 0.98``).

**사용자 확정 Q15**: 후보 **1개** (verbatim 단독).

**핵심 전략** (설계 §2.4):
    * ranker·pool·context 조회 **생략** — 즉시 반환.
    * 교정 LLM **생략** — final_transcript 그대로.
    * ASR final 의 공백 제거 길이 ≥ ``VERBATIM_MIN_LEN`` 체크. 미달이면
      ``suggestions=[]`` 로 자연 degrade (ranker 상위에서 재질의 처리).
    * ``source="asr_verbatim"``, ``reason_label="verbatim"`` 고정.
    * ``score=None`` — ranking skip 하므로 점수 없음.
    * ``cold_start=False`` — phrase_log 조회 자체 skip.

**Q20 (고신뢰 verbatim 라벨 노출)**:
    * 설계 TBD: F6 매칭 → "등록된 선호 ⭐", F1 매칭 → "자주 쓰는 말" 추천
    * **1차 MVP**: ``reason_label="verbatim"`` 고정, phrase_log 조회 skip.

**지연 목표**: ~85ms (DB 쿼리 없음, 순수 메모리 처리).

phrase_log 쓰기는 ``/phrase/selection`` 수신 시 수행 (현재 설계 그대로).

설계 참조:
    * ``marieum-plan/design/agent-algorithm.md`` §2.4 (고신뢰 분기), §3.9 (verbatim)
    * ``backend/docs/agent-rework-plan.md`` §2 Phase A4 high.py
"""
from __future__ import annotations

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.contracts import AgentInput, AgentOutput, RankingContext

log = logging.getLogger(__name__)


# 공백 제거 후 최소 길이 (설계 §2.4). 미달 시 재질의 성격이므로 빈 응답.
VERBATIM_MIN_LEN: int = 1


async def strategy(
    input: AgentInput,
    ctx: RankingContext | None,
    session: AsyncSession,
) -> AgentOutput:
    """고신뢰 분기 — ASR verbatim 단독 1개 즉시 반환.

    Args:
        input: AgentInput (asr_confidence ≥ 0.98).
        ctx: RankingContext | None — Phase B1 재구성 후 ranker.py 가
            ``build_ranking_context`` 호출을 skip 하므로 보통 ``None`` 이 들어온다.
            (지연 ~85ms 달성 조건, 설계 §2.4.)
        session: AsyncSession (사용하지 않음).

    Returns:
        AgentOutput with ``branch="high"``, 1개 또는 0개 suggestions.
        verbatim 공백 체크 실패 시 빈 응답 — api/phrase.py 가 재질의로 처리.
    """
    t0 = time.perf_counter()

    final = input.final_transcript or ""
    verbatim_text = final.strip()

    if len(verbatim_text) < VERBATIM_MIN_LEN:
        # 공백만 들어온 edge case — ranker 상위에서 재질의 처리하도록 빈 응답.
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        log.warning(
            "high.strategy() empty verbatim after strip (final=%r, patient_id=%s)",
            final,
            input.patient_id,
        )
        return AgentOutput(
            suggestions=[],
            latency_ms=latency_ms,
            cold_start=False,
            accuracy_mode_used=False,
            branch="high",
            audit_payload={
                "branch": "high",
                "verbatim_empty": True,
                "final_transcript": input.final_transcript,
                "asr_confidence": input.asr_confidence,
                "llm_called": False,
                "candidates": [],
            },
        )

    # 1차 MVP: reason_label="verbatim" 고정 (Q20 보류 — F6/F1 매칭 분기 없음).
    suggestion = {
        "rank": 1,
        "text": verbatim_text,
        "reason_label": "verbatim",
        "reason_display": None,
        "score": None,  # ranking skip → 점수 없음
        "source": "asr_verbatim",
    }

    latency_ms = max(1, int((time.perf_counter() - t0) * 1000))

    audit = {
        "branch": "high",
        "verbatim_text": verbatim_text,
        "verbatim_source": "asr_verbatim",
        "asr_confidence": input.asr_confidence,
        "final_transcript": input.final_transcript,
        "llm_called": False,
        "ranker_skipped": True,
        "pool_size": 0,
        "candidates": [
            {
                "phrase": verbatim_text,
                "source": "asr_verbatim",
                "scores": None,
                "base": None,
                "bonus": None,
                "total": None,
            }
        ],
    }

    return AgentOutput(
        suggestions=[suggestion],
        latency_ms=latency_ms,
        cold_start=False,
        accuracy_mode_used=False,
        branch="high",
        audit_payload=audit,
    )
