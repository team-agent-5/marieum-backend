"""Accuracy-mode orchestrator — RAG + LLM + 3-level fallback chain.

설계 참조:
  _design/design/agent-mode-c-architecture.md  §4·§7·§11
  _design/design/agent-mode-c-decisions.md     (전체, 특히 §3·§11)
  _design/design/agent-mode-c-api-contract.md  (response shape)

격리 조건:
  app.agent.streaming  — import 금지
  app.agent.ranking    — import 금지
"""
from __future__ import annotations

import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from app.agent.confidence_router import classify_branch
from app.agent.llm_assist import (
    LLMResult,
    reconstruct_low,
    rerank_mid,
    rerank_proactive,
)
from app.agent.rag_retriever import (
    fetch_patient_phrase_universe,
    RetrievalSource,
    RetrievedCandidate,
    _nfc,
    retrieve_for_high,
    retrieve_for_low,
    retrieve_for_mid,
)
from app.schemas_v2 import (
    CandidatesModeResponse,
    CandidateV2,
    MLHeardPhrase,
    ProactiveModeResponse,
    ProactiveSuggestion,
    RetryModeResponse,
    SingleModeResponse,
    SuggestResponseV2,
    VerbatimPhrase,
)
from app.services.embedding import embed_text
from app.services.time_based_fallback import fallback_candidates

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _compute_time_bucket(now: datetime | None = None) -> str:
    """KST 기준 시간대 분류 (morning/afternoon/evening/night)."""
    from zoneinfo import ZoneInfo

    kst = (now or datetime.now(UTC)).astimezone(ZoneInfo("Asia/Seoul"))
    hour = kst.hour
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "night"


async def _is_cold_start(session: AsyncSession, patient_id: UUID) -> bool:
    """phrase_log 5건 미만이면 cold-start."""
    cnt = (
        await session.execute(
            text(
                "SELECT COUNT(*) FROM phrase_log "
                "WHERE patient_id = :pid AND deleted_at IS NULL AND selected_phrase IS NOT NULL"
            ),
            {"pid": str(patient_id)},
        )
    ).scalar() or 0
    return int(cnt) < 5


async def _fetch_recent_history(
    session: AsyncSession, patient_id: UUID, limit: int = 10
) -> list[str]:
    """최근 10개 phrase (중복 포함, 시간 역순)."""
    rows = (
        await session.execute(
            text(
                "SELECT selected_phrase FROM phrase_log "
                "WHERE patient_id = :pid AND deleted_at IS NULL AND selected_phrase IS NOT NULL "
                "ORDER BY created_at DESC LIMIT :lim"
            ),
            {"pid": str(patient_id), "lim": limit},
        )
    ).scalars().all()
    return list(rows)


async def _fetch_prefs_meta(session: AsyncSession, patient_id: UUID) -> list[dict]:
    """patient_preference에서 content + type만. embedding/boost_phrases 제외."""
    rows = (
        await session.execute(
            text(
                "SELECT content, preference_type FROM patient_preference "
                "WHERE patient_id = :pid AND deleted_at IS NULL"
            ),
            {"pid": str(patient_id)},
        )
    ).mappings().all()
    return [{"content": r["content"], "preference_type": r["preference_type"]} for r in rows]


def _source_from_retrieval(candidate: RetrievedCandidate) -> str:
    """RetrievedCandidate.sources → CandidateSourceV2 문자열."""
    if RetrievalSource.CATALOG_COLD_START in candidate.sources:
        return "catalog_fallback"
    return "ranker_fallback"


# ---------------------------------------------------------------------------
# 分기별 처리 함수
# ---------------------------------------------------------------------------


def _base_kwargs(
    trace_id: str,
    conf: float | None,
    cold_start: bool,
    now: datetime,
    t0: float,
) -> dict:
    return dict(
        trace_id=trace_id,
        asr_confidence=conf,
        cold_start=cold_start,
        generated_at=now,
        generation_latency_ms=int((time.perf_counter() - t0) * 1000),
    )


async def _handle_high(
    session: AsyncSession,
    patient_id: UUID,
    ml_text: str,
    time_bucket: str,
    cold_start: bool,
    history: list[str],
    prefs_meta: list[dict],
    trace_id: str,
    conf: float | None,
    now: datetime,
    t0: float,
) -> ProactiveModeResponse:
    """高 분기 (conf ≥ 0.98) — proactive 1개 제안."""
    # 1. RAG retrieve
    try:
        retrieved = await retrieve_for_high(
            session, patient_id, ml_text, time_bucket, cold_start
        )
    except Exception as exc:
        log.warning("retrieve_for_high failed: %s", type(exc).__name__)
        retrieved = []
        # asyncpg: DB 에러 후 트랜잭션 상태 복구
        with contextlib.suppress(Exception):
            await session.rollback()
            # SET LOCAL이 rollback으로 사라지므로 복구
            await session.execute(text("SET LOCAL ivfflat.probes = 2"))

    # Cold-start 高: retrieved 비면 time_tags 기반 catalog phrase inject.
    # verbatim과 동일 phrase는 제외.
    if cold_start and not retrieved:
        tb_phrases = fallback_candidates(time_bucket, limit=8)
        retrieved = [
            RetrievedCandidate(phrase=_nfc(p), sources=[RetrievalSource.CATALOG_COLD_START])
            for p in tb_phrases if _nfc(p) != _nfc(ml_text)
        ]

    # allowed_set = retrieved ∪ history ∪ 환자 전체 발화 universe (180일).
    # universe는 LLM 입력엔 안 들어가고 가드만 완화 — history 10개에 밀려난
    # 환자 선호 phrase도 통과시킴. verbatim 중복은 llm_assist의 verbatim-skip 가드가 처리.
    universe = await fetch_patient_phrase_universe(session, patient_id)
    allowed_set: set[str] = (
        {c.phrase for c in retrieved}
        | {_nfc(h) for h in history if h}
        | universe
    )

    retrieved_as_dicts = [
        {"phrase": c.phrase, "cosine_sim": getattr(c, "cosine_sim", 0.0)}
        for c in retrieved
    ]

    llm_result: LLMResult = await rerank_proactive(
        verbatim=ml_text,
        patient_history=history,
        time_bucket=time_bucket,
        prefs_meta=prefs_meta,
        retrieved_candidates=retrieved_as_dicts,
        allowed_set=allowed_set,
    )

    # 4. 성공
    if llm_result.output is not None:
        from app.agent.llm_assist import _LLMProactiveSuggestion  # local import

        out: _LLMProactiveSuggestion = llm_result.output  # type: ignore[assignment]
        return ProactiveModeResponse(
            mode="proactive",
            branch="high",
            verbatim=VerbatimPhrase(text=ml_text),
            suggestion=ProactiveSuggestion(
                phrase=out.phrase,
                reason_display=out.reason_display,
                modal_text=out.modal_text,
                confidence=out.confidence,
            ),
            accuracy_fallback_reason=None,
            **_base_kwargs(trace_id, conf, cold_start, now, t0),
        )

    # 5. LLM 실패 — verbatim만 반환 (suggestion=None)
    log.info(
        "high_branch: LLM fallback, reason=%s", llm_result.fallback_reason
    )
    return ProactiveModeResponse(
        mode="proactive",
        branch="high",
        verbatim=VerbatimPhrase(text=ml_text),
        suggestion=None,
        accuracy_fallback_reason=llm_result.fallback_reason,
        **_base_kwargs(trace_id, conf, cold_start, now, t0),
    )


async def _handle_mid(
    session: AsyncSession,
    patient_id: UUID,
    ml_text: str,
    time_bucket: str,
    cold_start: bool,
    history: list[str],
    prefs_meta: list[dict],
    trace_id: str,
    conf: float | None,
    now: datetime,
    t0: float,
) -> CandidatesModeResponse:
    """中 분기 (0.90 ≤ conf < 0.98) — 3후보 (verbatim + AI 2개)."""
    # 1. pre-embed ml_text (orchestrator 1회 수행)
    ml_vec: list[float] | None = None
    if ml_text:
        try:
            ml_vec = await embed_text(ml_text)
        except Exception as exc:
            log.warning("mid_branch: embed_text failed: %s", type(exc).__name__)

    # 2. RAG retrieve_for_mid
    try:
        retrieved = await retrieve_for_mid(
            session, patient_id, ml_text, ml_vec, time_bucket, cold_start
        )
    except Exception as exc:
        log.warning("retrieve_for_mid failed: %s", type(exc).__name__)
        retrieved = []
        with contextlib.suppress(Exception):
            await session.rollback()
            await session.execute(text("SET LOCAL ivfflat.probes = 2"))

    # allowed_set = verbatim ∪ retrieved ∪ history ∪ universe.
    ml_nfc = _nfc(ml_text) if ml_text else ""
    universe = await fetch_patient_phrase_universe(session, patient_id)
    allowed_set: set[str] = (
        ({ml_nfc} if ml_nfc else set())
        | {c.phrase for c in retrieved}
        | {_nfc(h) for h in history if h}
        | universe
    )

    # 4. retrieved → dicts for LLM
    retrieved_as_dicts = [
        {
            "phrase": c.phrase,
            "cnt_90d": c.cnt_90d,
            "cosine_sim": c.cosine_sim,
            "bucket_cnt": c.bucket_cnt,
        }
        for c in retrieved
    ]

    # 5. LLM 호출
    llm_result: LLMResult = await rerank_mid(
        verbatim=ml_text,
        patient_history=history,
        time_bucket=time_bucket,
        prefs_meta=prefs_meta,
        retrieved_candidates=retrieved_as_dicts,
        allowed_set=allowed_set,
    )

    # 6. 성공 → 3개 candidates 조립
    if llm_result.output is not None:
        from app.agent.llm_assist import _LLMMidResult  # local import

        out: _LLMMidResult = llm_result.output  # type: ignore[assignment]
        cands_output = out.candidates  # 2개 _LLMMidCandidate
        # LLM이 2개 이상 반환 보장됐지만 defensive
        c1_phrase = cands_output[0].phrase if len(cands_output) > 0 else (retrieved[0].phrase if retrieved else ml_text)
        c1_reason = cands_output[0].reason_display if len(cands_output) > 0 else "자주 쓰시는 말"
        c2_phrase = cands_output[1].phrase if len(cands_output) > 1 else (retrieved[1].phrase if len(retrieved) > 1 else ml_text)
        c2_reason = cands_output[1].reason_display if len(cands_output) > 1 else "이 시간대 빈도 표현"

        candidates = [
            CandidateV2(
                rank=1,
                text=ml_text,
                reason_display="방금 들린 그대로",
                score=None,
                source="asr_verbatim",
                reason_label=None,
            ),
            CandidateV2(
                rank=2,
                text=c1_phrase,
                reason_display=c1_reason,
                score=None,
                source="llm_inference",
                reason_label=None,
            ),
            CandidateV2(
                rank=3,
                text=c2_phrase,
                reason_display=c2_reason,
                score=None,
                source="llm_inference",
                reason_label=None,
            ),
        ]
        return CandidatesModeResponse(
            mode="candidates",
            branch="mid",
            candidates=candidates,
            accuracy_fallback_reason=None,
            **_base_kwargs(trace_id, conf, cold_start, now, t0),
        )

    # 7. LLM 실패 — L1: ML 1개 + retrieved top-2
    log.info("mid_branch: LLM fallback, reason=%s", llm_result.fallback_reason)

    candidates_l1: list[CandidateV2] = [
        CandidateV2(
            rank=1,
            text=ml_text,
            reason_display="방금 들린 그대로",
            score=None,
            source="asr_verbatim",
            reason_label=None,
        )
    ]

    for idx, r in enumerate(retrieved[:2], start=2):
        candidates_l1.append(
            CandidateV2(
                rank=idx,
                text=r.phrase,
                reason_display="자주 쓰시는 말",
                score=None,  # prescore 0~1 범위 초과 가능
                source=_source_from_retrieval(r),  # type: ignore[arg-type]
                reason_label=None,
            )
        )

    # L2: retrieved가 부족하면 time_based_fallback 보충
    if len(candidates_l1) < 3:
        tb_phrases = fallback_candidates(time_bucket, limit=3 - len(candidates_l1))  # type: ignore[arg-type]
        existing_texts = {c.text for c in candidates_l1}
        for phrase in tb_phrases:
            if phrase not in existing_texts and len(candidates_l1) < 3:
                candidates_l1.append(
                    CandidateV2(
                        rank=len(candidates_l1) + 1,
                        text=phrase,
                        reason_display="이 시간대에 흔히 요청되는 말",
                        score=None,
                        source="catalog_fallback",
                        reason_label=None,
                    )
                )

    return CandidatesModeResponse(
        mode="candidates",
        branch="mid",
        candidates=candidates_l1,
        accuracy_fallback_reason=llm_result.fallback_reason,
        **_base_kwargs(trace_id, conf, cold_start, now, t0),
    )


async def _handle_low(
    session: AsyncSession,
    patient_id: UUID,
    ml_text: str,
    time_bucket: str,
    cold_start: bool,
    history: list[str],
    prefs_meta: list[dict],
    trace_id: str,
    conf: float | None,
    now: datetime,
    t0: float,
    previous_phrase: str | None = None,
) -> SingleModeResponse:
    """低 분기 (0.40 ≤ conf < 0.90) — AI 단일 후보 + ASR ml_heard.

    ml_text는 LLM 입력엔 안 들어가지만 (구조적 블로킹 유지) 응답에는 ml_heard로 포함.
    환자가 과거에 쓴 적 없는 새 발화일 때 유일한 단서를 UI에 넘긴다.
    """
    # ml_heard: 2자 이상이고 비어있지 않을 때만 노출 (1자 noise는 null)
    ml_stripped = ml_text.strip() if ml_text else ""
    ml_heard: MLHeardPhrase | None = (
        MLHeardPhrase(text=ml_stripped, confidence=conf or 0.0)
        if len(ml_stripped) >= 2 else None
    )

    # 1. RAG retrieve_for_low (ML text 없이)
    try:
        retrieved = await retrieve_for_low(
            session, patient_id, time_bucket, previous_phrase, cold_start
        )
    except Exception as exc:
        log.warning("retrieve_for_low failed: %s", type(exc).__name__)
        retrieved = []
        with contextlib.suppress(Exception):
            await session.rollback()
            await session.execute(text("SET LOCAL ivfflat.probes = 2"))

    # allowed_set = retrieved ∪ history ∪ universe. ml_text는 구조적 제외 (低 분기 설계).
    universe = await fetch_patient_phrase_universe(session, patient_id)
    allowed_set: set[str] = (
        {c.phrase for c in retrieved}
        | {_nfc(h) for h in history if h}
        | universe
    )

    # 3. retrieved → dicts for LLM
    retrieved_as_dicts = [
        {
            "phrase": c.phrase,
            "cnt_90d": c.cnt_90d,
            "cosine_sim": c.cosine_sim,
            "bucket_cnt": c.bucket_cnt,
            "prescore": c.prescore,
        }
        for c in retrieved
    ]

    # 4. LLM 호출 — ml_verbatim 안 넘김 (구조적 블로킹)
    llm_result: LLMResult = await reconstruct_low(
        patient_history=history,
        time_bucket=time_bucket,
        prefs_meta=prefs_meta,
        retrieved_candidates=retrieved_as_dicts,
        allowed_set=allowed_set,
    )

    # 5. 성공
    if llm_result.output is not None:
        from app.agent.llm_assist import _LLMReconstructedCandidate  # local import

        out: _LLMReconstructedCandidate = llm_result.output  # type: ignore[assignment]
        return SingleModeResponse(
            mode="single",
            branch="low",
            ml_heard=ml_heard,
            candidate=CandidateV2(
                rank=1,
                text=out.phrase,
                reason_display=out.reason_display,
                score=None,
                source="llm_reconstruction",
                reason_label=None,
            ),
            accuracy_fallback_reason=None,
            **_base_kwargs(trace_id, conf, cold_start, now, t0),
        )

    # 6. L1 fallback — retrieved top-1
    log.info("low_branch: LLM fallback, reason=%s", llm_result.fallback_reason)

    if retrieved:
        top = retrieved[0]
        return SingleModeResponse(
            mode="single",
            branch="low",
            ml_heard=ml_heard,
            candidate=CandidateV2(
                rank=1,
                text=top.phrase,
                reason_display="자주 쓰시는 말",
                score=None,  # prescore는 0~1 범위 초과 가능 — score 필드에 넣지 않음
                source=_source_from_retrieval(top),  # type: ignore[arg-type]
                reason_label=None,
            ),
            accuracy_fallback_reason=llm_result.fallback_reason,
            **_base_kwargs(trace_id, conf, cold_start, now, t0),
        )

    # 7. L2 fallback — time_based_fallback 1개
    tb_phrases = fallback_candidates(time_bucket, limit=1)  # type: ignore[arg-type]
    if tb_phrases:
        return SingleModeResponse(
            mode="single",
            branch="low",
            ml_heard=ml_heard,
            candidate=CandidateV2(
                rank=1,
                text=tb_phrases[0],
                reason_display="이 시간대에 흔히 요청되는 말",
                score=None,
                source="catalog_fallback",
                reason_label=None,
            ),
            accuracy_fallback_reason="time_based_fallback",
            **_base_kwargs(trace_id, conf, cold_start, now, t0),
        )

    # 8. 최악 hardcoded fallback
    return SingleModeResponse(
        mode="single",
        branch="low",
        ml_heard=ml_heard,
        candidate=CandidateV2(
            rank=1,
            text="다시 말씀해 주세요",
            reason_display="잠시 후 다시 시도해 주세요",
            score=None,
            source="catalog_fallback",
            reason_label=None,
        ),
        accuracy_fallback_reason="all_fallbacks_exhausted",
        **_base_kwargs(trace_id, conf, cold_start, now, t0),
    )


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


async def suggest(
    patient_id: UUID,
    ml_text: str,
    asr_confidence: float | None,
    trace_id: str,
    session: AsyncSession,
    accuracy_mode: bool = True,
) -> SuggestResponseV2:
    """정확도 모드 orchestration — RAG + LLM + fallback 체인.

    흐름:
    1. classify_branch(asr_confidence) → 分기 결정
    2. retry → RetryModeResponse 즉시 반환
    3. phrase_log count query → cold_start 판정 (<5건)
    4. 분기별 RAG retrieval (rag_retriever.retrieve_for_*)
    5. allowed_set 구성 (patient_log | catalog | prefs)
    6. 분기별 LLM 호출 (llm_assist.rerank_* / reconstruct_*)
    7. LLM 성공 → 응답 조립
    8. LLM 실패 → L1 fallback (retrieved 결과 활용)
    9. L1 실패 → L2 fallback (time_based_fallback.fallback_candidates)
    10. 전부 실패 → 하드코딩 "다시 말씀해 주세요" (SingleModeResponse로)

    모든 분기 공통:
    - trace_id 유지
    - generation_latency_ms = perf_counter 전체 시간
    """
    t0 = time.perf_counter()
    now = datetime.now(UTC)

    # D3 fallback: None → 1.0 (high 분기)
    conf: float = asr_confidence if asr_confidence is not None else 1.0
    branch = classify_branch(conf)

    # ── retry 분기 — 즉시 반환 ───────────────────────────────────────────────
    if branch == "retry":
        return RetryModeResponse(
            mode="retry",
            branch="retry",
            message="잘 못 들었어요. 다시 말씀해 주세요.",
            trace_id=trace_id,
            asr_confidence=conf,
            cold_start=False,
            generated_at=now,
            accuracy_fallback_reason=None,
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── 공통 준비 (cold_start, history, prefs) ─────────────────────────────
    time_bucket = _compute_time_bucket(now)

    # C4 — pgvector IVFFlat probes 세션 레벨 1회 세팅. 분기 안에서 rollback 발생 시 재호출.
    with contextlib.suppress(Exception):
        await session.execute(text("SET LOCAL ivfflat.probes = 2"))

    try:
        cold_start = await _is_cold_start(session, patient_id)
    except Exception as exc:
        log.warning("_is_cold_start failed: %s — defaulting to False", type(exc).__name__)
        cold_start = False

    try:
        history = await _fetch_recent_history(session, patient_id)
    except Exception as exc:
        log.warning("_fetch_recent_history failed: %s", type(exc).__name__)
        history = []

    try:
        prefs_meta = await _fetch_prefs_meta(session, patient_id)
    except Exception as exc:
        log.warning("_fetch_prefs_meta failed: %s", type(exc).__name__)
        prefs_meta = []

    # ── 분기별 처리 ──────────────────────────────────────────────────────────
    try:
        if branch == "high":
            return await _handle_high(
                session, patient_id, ml_text, time_bucket, cold_start,
                history, prefs_meta, trace_id, conf, now, t0,
            )

        if branch == "mid":
            return await _handle_mid(
                session, patient_id, ml_text, time_bucket, cold_start,
                history, prefs_meta, trace_id, conf, now, t0,
            )

        # branch == "low"
        return await _handle_low(
            session, patient_id, ml_text, time_bucket, cold_start,
            history, prefs_meta, trace_id, conf, now, t0,
        )

    except Exception as exc:
        # 최악 상황: 전체 fallback → "다시 말씀해 주세요"
        log.exception("orchestrator.suggest unhandled exception: %s", type(exc).__name__)
        _ml_stripped_exc = ml_text.strip() if ml_text else ""
        _ml_heard_exc = (
            MLHeardPhrase(text=_ml_stripped_exc, confidence=conf or 0.0)
            if len(_ml_stripped_exc) >= 2 else None
        )
        return SingleModeResponse(
            mode="single",
            branch="low",
            ml_heard=_ml_heard_exc,
            candidate=CandidateV2(
                rank=1,
                text="다시 말씀해 주세요",
                reason_display="잠시 후 다시 시도해 주세요",
                score=None,
                source="catalog_fallback",
                reason_label=None,
            ),
            trace_id=trace_id,
            asr_confidence=conf,
            cold_start=cold_start,
            generated_at=now,
            accuracy_fallback_reason="unhandled_error",
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )
