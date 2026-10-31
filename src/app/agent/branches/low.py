"""저신뢰 분기 전략 (``0.40 ≤ conf < 0.90``).

**핵심 원칙** (설계 §2.2):
    1. ASR ``final_transcript`` 는 **참고용** — ranking 결정적 소스 아님.
    2. phrase_log 패턴 (F2 time_context + F4 sequence_context + F1 patient_frequency)
       지배. F3 recency·F5 semantic·F6 preference 보조. prefix_match 는 ASR 참고용.
    3. 저신뢰 분기 전용 가중치 (설계 §4.1 옵션 B)::

           prefix=0.25, freq=0.25, time=0.20, seq=0.15, sem=0.08, rec=0.07
           preference_bonus 가산 (+0.15 max)

    4. 상위 3개 중 ``total ≥ MIN_SCORE (0.08)`` 통과 건만 노출. empty 금지.
    5. Cold-start (``ctx.freq_total == 0``) 대응: 설계 §C.1·C.2·S-G8 에 따라
       verbatim 1개로 축소 가능 (현재는 catalog seed 로 자연 fallback).
    6. 저신뢰 cold-start 교정 LLM: D25 추천대로 **skip** (환각·지연·비용 절감).
       cold-start 가 아니면 범용 교정 LLM 호출 (Phase A5 완성 전까지 TODO 플레이스홀더).

후보 수:
    * 매칭 충분 → 3개
    * 저신뢰 cold-start → 축소 가능 (설계 S-G8)
    * MIN_SCORE 미달 → catalog seed 1건 보강 (현재 pool.py 가 catalog 전수 포함)

설계 참조:
    * ``marieum-plan/design/agent-algorithm.md`` §2.2 (저신뢰 분기), §4.1 (가중치)
    * ``backend/docs/agent-rework-plan.md`` §2 Phase A4 low.py
"""
from __future__ import annotations

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.contracts import (
    AgentInput,
    AgentOutput,
    CandidateScore,
    FeatureScores,
    RankingContext,
)
from app.agent.features.patient_frequency import compute as f_patient_frequency
from app.agent.features.preference import compute as f_preference
from app.agent.features.prefix_match import compute as f_prefix_match
from app.agent.features.recency import compute as f_recency
from app.agent.features.rejected_penalty import compute as f_rejected_penalty
from app.agent.features.semantic_similarity import compute as f_semantic_similarity
from app.agent.features.sequence_context import compute as f_sequence_context
from app.agent.features.time_context import compute as f_time_context
from app.agent.pool import gather_candidates
from app.agent.reason import pick_reason_label

log = logging.getLogger(__name__)


# 설계 §4.1 옵션 B — 저신뢰 분기 전용 가중치 (합 1.00).
# Phase B 에서 settings.ranker_weights_low 로 외부화 가능.
WEIGHTS_LOW: dict[str, float] = {
    "prefix_match": 0.25,
    "patient_frequency": 0.25,
    "time_context": 0.20,
    "sequence_context": 0.15,
    "semantic_similarity": 0.08,
    "recency": 0.07,
}

# 저신뢰 분기 MIN_SCORE (D4, 설계 §2.2) — 통과 못 하면 catalog fallback.
MIN_SCORE: float = 0.08

# empty 금지 보장용 최소 노출 (설계 §2.2).
# MIN_SCORE 통과 0건이면 pool 상위 1건이라도 노출.
MIN_CANDIDATES: int = 1


async def strategy(
    input: AgentInput,
    ctx: RankingContext,
    session: AsyncSession,
) -> AgentOutput:
    """저신뢰 분기 — phrase_log 패턴 기반 top-3.

    Args:
        input: AgentInput (0.40 ≤ asr_confidence < 0.90).
        ctx: RankingContext (``build_ranking_context`` 결과).
        session: AsyncSession (pool.gather_candidates 에 전달).

    Returns:
        AgentOutput with ``branch="low"``, 1~3개 suggestions,
        audit_payload 에 분기별 가중치·feature 점수·LLM 호출 여부 포함.
    """
    t0 = time.perf_counter()

    pool = await gather_candidates(session, ctx)

    scored: list[CandidateScore] = []
    for cand in pool:
        scores = FeatureScores(
            prefix_match=f_prefix_match(cand, ctx),
            patient_frequency=f_patient_frequency(cand, ctx),
            time_context=f_time_context(cand, ctx),
            sequence_context=f_sequence_context(cand, ctx),
            semantic_similarity=f_semantic_similarity(cand, ctx),
            recency=f_recency(cand, ctx),
            preference_bonus=f_preference(cand, ctx),
            # Phase W1 — A3 rejected_penalty wiring (설계 §6.2, context.py Query E).
            # ctx.rejected_counts 가 비어 있으면 0.0 반환. 음수(or 0) 값이므로
            # 가중합에 add. 24h 창·반감기 24h 지수감쇠 공식.
            rejected_penalty=f_rejected_penalty(cand, ctx),
        )
        base = (
            WEIGHTS_LOW["prefix_match"] * scores.prefix_match
            + WEIGHTS_LOW["patient_frequency"] * scores.patient_frequency
            + WEIGHTS_LOW["time_context"] * scores.time_context
            + WEIGHTS_LOW["sequence_context"] * scores.sequence_context
            + WEIGHTS_LOW["semantic_similarity"] * scores.semantic_similarity
            + WEIGHTS_LOW["recency"] * scores.recency
        )
        bonus = scores.preference_bonus
        # rejected_penalty 는 음수 또는 0 (Phase A3 공식, 설계 §6.2).
        total = min(1.15, max(0.0, base + bonus + scores.rejected_penalty))
        if total <= 0.0:
            continue
        scored.append(
            CandidateScore(
                phrase=cand.phrase,
                source=cand.source,
                scores=scores,
                base=base,
                bonus=bonus,
                total=total,
                reason_label=None,  # label 은 아래 top-3 확정 후 할당.
            )
        )

    scored.sort(key=lambda s: s.total, reverse=True)

    cold_start = ctx.freq_total == 0

    # 저신뢰 분기 — LLM 교정 **완전 생략** (설계 #1 결정, 2026-10-31).
    # 사유:
    #   1. 저신뢰는 ASR 자체 신뢰 어려워 LLM 입력 품질 저하 → 환각 리스크 증가.
    #   2. phrase_log 패턴 (F2+F4+F1) 로 충분 — 환자 과거 발화가 더 안정적 신호.
    #   3. 비용·지연 절감 (중신뢰 종결어미 보정은 유지 — 범위 외).
    accuracy_mode_used = False
    llm_called = False
    accuracy_fallback_reason: str | None = None
    llm_latency_ms: int | None = None

    # MIN_SCORE 필터링 — empty 금지 보장.
    filtered = [s for s in scored if s.total >= MIN_SCORE]
    if len(filtered) < MIN_CANDIDATES and len(scored) > 0:
        # catalog seed fallback — pool 상위 1건이라도 노출.
        filtered = scored[:MIN_CANDIDATES]

    # Cold-start 축소 (설계 S-G8): phrase_log 0건 + MIN_SCORE 통과 0건이면
    # 1개만 노출해 환자 재녹음·수동 선택 유도. 현재는 catalog seed 로 자연 보강됨.
    top3 = filtered[:3]

    for s in top3:
        s.reason_label = pick_reason_label(s)

    suggestions = [
        {
            "rank": i + 1,
            "text": s.phrase,
            "reason_label": s.reason_label,
            "reason_display": None,  # API layer maps label → display
            "score": round(s.total, 4),
            "source": s.source,
        }
        for i, s in enumerate(top3)
    ]

    latency_ms = int((time.perf_counter() - t0) * 1000)

    audit = {
        "branch": "low",
        "weights": WEIGHTS_LOW,
        "min_score": MIN_SCORE,
        "cold_start": cold_start,
        # W3 — 설계 #1 결정 반영 (저신뢰 범용 교정 LLM 생략, 2026-10-31).
        "low_branch_llm": "skipped_by_design",
        "cold_start_llm_skipped": cold_start,
        "llm_called": llm_called,
        "accuracy_mode_used": accuracy_mode_used,
        "accuracy_fallback_reason": accuracy_fallback_reason,
        "llm_latency_ms": llm_latency_ms,
        "pool_size": len(pool),
        "scored_size": len(scored),
        "filtered_size": len(filtered),
        "asr_confidence": input.asr_confidence,
        "final_transcript": input.final_transcript,
        "candidates": [
            {
                "phrase": s.phrase,
                "source": s.source,
                "scores": s.scores.model_dump(),
                "base": s.base,
                "bonus": s.bonus,
                "total": s.total,
            }
            for s in top3
        ],
    }

    return AgentOutput(
        suggestions=suggestions,
        latency_ms=latency_ms,
        cold_start=cold_start,
        accuracy_mode_used=accuracy_mode_used,
        branch="low",
        audit_payload=audit,
    )
