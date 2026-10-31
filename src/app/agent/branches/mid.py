"""중신뢰 분기 전략 (``0.90 ≤ conf < 0.98``).

**D1 결정** (설계 §2.3, 사용자 확정 2026-10-31):
    후보 **3개** 반환 — verbatim 1위 + phrase_log 매칭 상위 2개.

**핵심 전략**:
    1. 1위 **verbatim**: ``features/verbatim.py`` (Phase A3) + ``llm_assist.correct_endings()``
       (Phase A5) 로 종결어미 보정. LLM 실패 시 ``final_transcript`` 그대로 fallback.
       교정 LLM timeout **0.5s** (D5).
    2. 2·3위 **phrase_log 매칭**: F1 patient_frequency + F2 time_context + F5
       semantic_similarity + F6 preference 가중합 (설계 §4.2).
       가중치::

           freq=0.30, time=0.25, sem=0.25, prefix=0.20
           preference_bonus 가산

    3. **중복 제거**: verbatim 과 2·3위 매칭 phrase 가 동일하면 다음 상위로 대체
       (Q22 결정안: 대체).
    4. **매칭 부족 fallback** (설계 §2.3b): phrase_log 2개 못 뽑으면 preset
       (phrase_catalog) semantic match 로 채워 **3개 보장**.

**교정 LLM**:
    * timeout **0.5s** (D5 — 전체 지연 예산 보호)
    * 실패 시 ``verbatim = final_transcript`` 로 degrade
    * audit_log 에 ``accuracy_fallback_reason="llm_corrective_timeout"`` 등 기록

Phase A3·A5 가 아직 미완성이므로 ``features.verbatim`` /
``llm_assist.corrective_rerank_with_llm`` 호출은 **TODO 플레이스홀더** 로 둔다.
Phase B 통합 시 활성화.

설계 참조:
    * ``marieum-plan/design/agent-algorithm.md`` §2.3 (중신뢰 분기), §3.9 (verbatim), §4.2 (가중치)
    * ``backend/docs/agent-rework-plan.md`` §2 Phase A4 mid.py
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

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
from app.agent.features.semantic_similarity import compute as f_semantic_similarity
from app.agent.features.time_context import compute as f_time_context
from app.agent.llm_assist import correct_endings
from app.agent.pool import gather_candidates

log = logging.getLogger(__name__)


# 설계 §4.2 — 중신뢰 2위 매칭용 가중치 (합 1.00).
# verbatim 1위는 가중치 skip (교정 LLM 결과 그대로).
WEIGHTS_MID_MATCH: dict[str, float] = {
    "patient_frequency": 0.30,
    "time_context": 0.25,
    "semantic_similarity": 0.25,
    "prefix_match": 0.20,
}

# 2위 노출 임계 (설계 §4.2) — 통과 못 하면 preset fallback 또는 축소.
MID_MATCH_MIN_SCORE: float = 0.15

# verbatim 최소 길이 (공백 제거 후). 미달 시 재질의로 자연 degrade — ranker 상위에서 처리.
VERBATIM_MIN_LEN: int = 1

# 교정 LLM timeout (D5, 설계 §5.1 지연 예산).
LLM_CORRECTIVE_TIMEOUT_S: float = 0.5


def _compute_verbatim_text(final_transcript: str | None) -> str | None:
    """ASR final_transcript 를 verbatim 후보 텍스트로 변환.

    Phase A3 ``features/verbatim.py`` 완성 전까지의 간이 구현:
        * None/공백 → None
        * strip 후 ``VERBATIM_MIN_LEN`` 이상이면 그대로 반환
    """
    if not final_transcript:
        return None
    stripped = final_transcript.strip()
    if len(stripped) < VERBATIM_MIN_LEN:
        return None
    return stripped


# Hallucination guard — 교정 LLM 결과가 ASR 원본에서 과하게 이탈했는지 검사.
# 설계 §5.1 환각 가드 4종 중 (1) 단어 집합·(2) 길이 ±50% 를 호출측 책임으로 반영.
# (3) 입력과 동일 통과 는 자연 통과 (changed=false 라도 OK), (4) timeout 은
# llm_assist.correct_endings() 가 담당.
_LENGTH_TOLERANCE: float = 0.5  # ±50% (ratio 기준)


def _token_set(text: str) -> set[str]:
    """공백 split 로 단어 집합 추출.

    구음장애 환자 발화 특성상 어절 단위 비교가 가장 보수적·간단.
    konlpy/mecab 미도입 (1차 MVP). 비교 전 strip + lower (한국어는 lower 영향 없음).
    """
    if not text:
        return set()
    return {tok for tok in text.strip().split() if tok}


def _recent_phrases(ctx: RankingContext, k: int = 10) -> list[str]:
    """ctx.freq_counts·last_at 에서 최신 발화 상위 ``k`` 건.

    newest-first 정렬. last_at 이 null 인 phrase 는 제외 (freq 만 있는 catalog seed 등).
    """
    pairs: list[tuple[str, Iterable]] = []
    for phrase in ctx.freq_counts:
        last = ctx.last_at.get(phrase)
        if last is not None:
            pairs.append((phrase, last))
    pairs.sort(key=lambda x: x[1], reverse=True)  # type: ignore[arg-type,return-value]
    return [p for p, _ in pairs[:k]]


def _guard_corrected(asr_text: str, corrected: str) -> tuple[bool, str | None]:
    """교정 결과 환각 가드 (설계 §5.1 (1)(2) 공백 split 기반 간이 구현).

    Returns:
        (ok, reason) — ok=False 면 reason 에 fallback 사유 (audit 용).
    """
    corr_stripped = corrected.strip()
    asr_stripped = asr_text.strip()
    if not corr_stripped:
        return False, "guard_empty_corrected"

    # (2) 길이 ±50% 체크
    asr_len = max(1, len(asr_stripped))
    corr_len = len(corr_stripped)
    ratio = corr_len / asr_len
    if ratio < (1.0 - _LENGTH_TOLERANCE) or ratio > (1.0 + _LENGTH_TOLERANCE):
        return False, "guard_length_out_of_range"

    # (1) 단어 집합 보존 — corrected 의 단어가 asr 단어 집합 안에 모두 포함되는지.
    #     종결어미만 바뀌는 경우 어절 자체는 유지되므로 "교정 어절 ⊆ ASR 어절" 체크.
    asr_tokens = _token_set(asr_stripped)
    corr_tokens = _token_set(corr_stripped)
    # ASR 어절이 1개(짧은 발화)면 공백 split 비교가 지나치게 엄격해지므로 skip.
    if len(asr_tokens) >= 2:
        extra = corr_tokens - asr_tokens
        if extra:
            # 신규 어절 추가 → 환각 가능성 — fallback.
            return False, "guard_new_tokens_added"

    return True, None


async def _correct_verbatim(
    final_transcript: str,
    ctx: RankingContext,
) -> tuple[str, bool, str | None, int | None]:
    """교정 LLM 호출 — 종결어미 보정 + hallucination guard (W2 wiring).

    설계 §5.1 환각 가드 책임 분담:
        * ``llm_assist.correct_endings()``       → (4) timeout + strict JSON schema
        * 본 함수 (``_guard_corrected``)         → (1) 단어 집합 비교, (2) 길이 ±50%
        * 입력과 동일 통과 (3)                     → changed=false 자연 통과

    환자 recent_phrases 는 ``ctx.freq_counts.last_at desc`` 로 10건 추출.

    Returns:
        (verbatim_text, llm_success, fallback_reason, llm_latency_ms)
        * llm_success=False 면 verbatim_text = ASR 원본 (final_transcript strip).
        * fallback_reason: None (성공) | "llm_returned_none" | "guard_*" | ...
    """
    asr_stripped = final_transcript.strip()
    if not asr_stripped:
        return final_transcript, False, "empty_final_transcript", None

    history = _recent_phrases(ctx, k=10)

    _t0 = time.perf_counter()
    try:
        result = await correct_endings(
            asr_stripped,
            history,
            timeout_s=LLM_CORRECTIVE_TIMEOUT_S,
        )
    except Exception as exc:  # 방어 — llm_assist 가 자체 처리하지만 안전망.
        err_type = type(exc).__name__
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log.warning("mid._correct_verbatim unexpected: %s (latency_ms=%d)", err_type, latency_ms)
        return asr_stripped, False, f"error:{err_type}", latency_ms

    latency_ms = int((time.perf_counter() - _t0) * 1000)

    if result is None:
        # timeout/429/circuit_open/parse_error — llm_assist 가 이미 로깅.
        return asr_stripped, False, "llm_returned_none", latency_ms

    # Hallucination guard — 설계 §5.1 (1)(2).
    corrected = result.corrected.strip()
    if not result.changed:
        # LLM 이 변경 없음 통지 — ASR 원본 재사용 (guard 생략).
        return asr_stripped, True, None, latency_ms

    ok, reason = _guard_corrected(asr_stripped, corrected)
    if not ok:
        log.warning(
            "mid.correct_verbatim guard rejected: reason=%s asr=%r corrected=%r",
            reason, asr_stripped, corrected,
        )
        return asr_stripped, False, reason, latency_ms

    return corrected, True, None, latency_ms


def _score_match(
    cand,
    ctx: RankingContext,
) -> tuple[FeatureScores, float]:
    """2위 매칭 후보 점수 계산 — F1+F2+F5+prefix 가중합 + preference bonus.

    설계 §4.2: recency·sequence 는 2위 매칭에서 제외.
    """
    scores = FeatureScores(
        prefix_match=f_prefix_match(cand, ctx),
        patient_frequency=f_patient_frequency(cand, ctx),
        time_context=f_time_context(cand, ctx),
        semantic_similarity=f_semantic_similarity(cand, ctx),
        preference_bonus=f_preference(cand, ctx),
        # 중신뢰 2위 매칭에서 제외되는 feature.
        sequence_context=0.0,
        recency=0.0,
        rejected_penalty=0.0,  # TODO(A3): rejected_penalty 적용 결정 (Q14).
    )
    base = (
        WEIGHTS_MID_MATCH["patient_frequency"] * scores.patient_frequency
        + WEIGHTS_MID_MATCH["time_context"] * scores.time_context
        + WEIGHTS_MID_MATCH["semantic_similarity"] * scores.semantic_similarity
        + WEIGHTS_MID_MATCH["prefix_match"] * scores.prefix_match
    )
    bonus = scores.preference_bonus
    total = min(1.15, max(0.0, base + bonus))
    return scores, total


async def strategy(
    input: AgentInput,
    ctx: RankingContext,
    session: AsyncSession,
) -> AgentOutput:
    """중신뢰 분기 — verbatim 1위 + phrase_log 매칭 2개 (총 3개).

    Args:
        input: AgentInput (0.90 ≤ asr_confidence < 0.98).
        ctx: RankingContext.
        session: AsyncSession.

    Returns:
        AgentOutput with ``branch="mid"``, 1~3개 suggestions
        (매칭 부족 fallback 로 보통 3개 보장), audit_payload 포함.
    """
    t0 = time.perf_counter()

    # ── 1위: verbatim (교정 LLM 결과) ────────────────────────────────────────
    verbatim_text = _compute_verbatim_text(input.final_transcript)
    llm_called = False
    llm_success = False
    accuracy_fallback_reason: str | None = None
    llm_latency_ms: int | None = None

    if verbatim_text is not None:
        llm_called = True
        verbatim_text, llm_success, accuracy_fallback_reason, llm_latency_ms = (
            await _correct_verbatim(verbatim_text, ctx)
        )

    # ── 2·3위: phrase_log 매칭 상위 2개 ─────────────────────────────────────
    pool = await gather_candidates(session, ctx)

    # patient_log 와 catalog 를 분리 — 매칭 부족 fallback 에서 catalog 활용.
    log_pool = [c for c in pool if c.source == "patient_log"]
    cat_pool = [c for c in pool if c.source == "catalog"]

    scored_log: list[CandidateScore] = []
    for cand in log_pool:
        # verbatim 과 동일 phrase 는 중복 제거 (Q22: 대체 규칙).
        if verbatim_text is not None and cand.phrase == verbatim_text:
            continue
        scores, total = _score_match(cand, ctx)
        if total < MID_MATCH_MIN_SCORE:
            continue
        scored_log.append(
            CandidateScore(
                phrase=cand.phrase,
                source=cand.source,
                scores=scores,
                base=total - scores.preference_bonus,
                bonus=scores.preference_bonus,
                total=total,
                reason_label=None,
            )
        )
    scored_log.sort(key=lambda s: s.total, reverse=True)
    match_top2 = scored_log[:2]

    # ── 매칭 부족 fallback (설계 §2.3b) ─────────────────────────────────────
    # phrase_log 2개 못 뽑으면 preset (catalog) semantic match 상위로 채워 3개 보장.
    fallback_filled: list[CandidateScore] = []
    if len(match_top2) < 2:
        seen_phrases = {c.phrase for c in match_top2}
        if verbatim_text is not None:
            seen_phrases.add(verbatim_text)
        scored_cat: list[CandidateScore] = []
        for cand in cat_pool:
            if cand.phrase in seen_phrases:
                continue
            scores, total = _score_match(cand, ctx)
            # catalog fallback 은 MIN_SCORE 완화 — "빈 slot 보강" 목적.
            scored_cat.append(
                CandidateScore(
                    phrase=cand.phrase,
                    source=cand.source,
                    scores=scores,
                    base=total - scores.preference_bonus,
                    bonus=scores.preference_bonus,
                    total=total,
                    reason_label=None,
                )
            )
        scored_cat.sort(key=lambda s: s.total, reverse=True)
        needed = 2 - len(match_top2)
        fallback_filled = scored_cat[:needed]

    matches = match_top2 + fallback_filled

    # ── 최종 suggestions 구성 ────────────────────────────────────────────────
    suggestions: list[dict] = []

    if verbatim_text is not None:
        verbatim_source = "llm_corrected" if llm_success else "asr_verbatim"
        suggestions.append(
            {
                "rank": 1,
                "text": verbatim_text,
                "reason_label": "verbatim",
                "reason_display": None,
                "score": None,  # verbatim 은 score 없음 (D1 설계)
                "source": verbatim_source,
            }
        )

    for i, m in enumerate(matches, start=len(suggestions) + 1):
        # label 할당 — verbatim 과 분리된 ladder (중신뢰 2위 매칭은 활성 feature 만).
        label = _pick_mid_match_label(m)
        suggestions.append(
            {
                "rank": i,
                "text": m.phrase,
                "reason_label": label,
                "reason_display": None,
                "score": round(m.total, 4),
                "source": m.source,
            }
        )
        if len(suggestions) >= 3:
            break

    # rank 재부여 (verbatim 미생성 시 1부터 시작).
    for idx, s in enumerate(suggestions, start=1):
        s["rank"] = idx

    latency_ms = int((time.perf_counter() - t0) * 1000)
    cold_start = ctx.freq_total == 0

    audit = {
        "branch": "mid",
        "weights": WEIGHTS_MID_MATCH,
        "mid_match_min_score": MID_MATCH_MIN_SCORE,
        "verbatim_text": verbatim_text,
        "verbatim_llm_success": llm_success,
        "llm_called": llm_called,
        "llm_timeout_s": LLM_CORRECTIVE_TIMEOUT_S,
        "accuracy_fallback_reason": accuracy_fallback_reason,
        "llm_latency_ms": llm_latency_ms,
        "cold_start": cold_start,
        "pool_size": len(pool),
        "log_pool_size": len(log_pool),
        "match_log_count": len(match_top2),
        "match_fallback_count": len(fallback_filled),
        "asr_confidence": input.asr_confidence,
        "final_transcript": input.final_transcript,
        "candidates": [
            {
                "phrase": m.phrase,
                "source": m.source,
                "scores": m.scores.model_dump(),
                "base": m.base,
                "bonus": m.bonus,
                "total": m.total,
            }
            for m in matches
        ],
    }

    return AgentOutput(
        suggestions=suggestions,
        latency_ms=latency_ms,
        cold_start=cold_start,
        accuracy_mode_used=llm_success,
        branch="mid",
        audit_payload=audit,
    )


def _pick_mid_match_label(c: CandidateScore) -> str | None:
    """중신뢰 2위 매칭용 label ladder (verbatim 제외, F1·F2·F5 중심).

    설계 §3.1 ladder 간소화: 중신뢰 2위 매칭에선 recency·sequence 미사용.
    """
    s = c.scores
    if c.bonus >= 0.12:
        return "registered_preference"
    if s.time_context >= 0.5:
        return "time_context"
    if s.patient_frequency >= 0.3:
        return "patient_frequency"
    if s.semantic_similarity >= 0.286:  # raw cosine >= 0.75
        return "semantic_similarity"
    return None
