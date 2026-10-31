"""Confidence 분기 라우터.

ASR confidence 값에 따라 4단계 분기 (retry / low / mid / high) 를 결정한다.
임계값은 ``app.config.settings`` 에서 환경변수 override 가능.

본 모듈은 분기 결정 전용이며, DB 접근·ranker 호출을 하지 않는다.
교정 LLM 호출 여부·기대 후보 수 등 "분기별 메타정보" 헬퍼도 함께 제공한다.
"""
from __future__ import annotations

from app.agent.contracts import ConfidenceBranch
from app.config import settings


def classify_branch(asr_confidence: float | None) -> ConfidenceBranch:
    """ASR confidence 값으로 분기를 결정한다.

    D3 fallback (사용자 결정, 2026-10-31):
        None / 음수 → 1.0 으로 간주 → ``high`` 반환.
        → ``asr_confidence`` 를 전달하지 않는 레거시 경로나
          ML 서버 응답 누락 시 안전하게 verbatim-only 로 동작.

    경계 처리 (ML 보고서 2026-11-01 반영):
        * ``conf < RETRY_CONFIDENCE (0.60)``                → ``"retry"``
        * ``RETRY_CONFIDENCE ≤ conf < LOW_UPPER (0.90)``    → ``"low"``
        * ``LOW_UPPER ≤ conf < MID_UPPER (0.98)``           → ``"mid"``
        * ``conf ≥ MID_UPPER (0.98)``                       → ``"high"``

    1차 MVP 는 **안정화 밴드 없음** (D4, 리스크 R2 수용).
    경계 flapping 은 환경변수 튜닝으로 대응한다.

    Args:
        asr_confidence: WS ``final.confidence`` 값. ``[0.0, 1.0]`` 범위 기대.
            ``None`` / 음수 입력 시 D3 fallback 으로 ``high`` 반환.

    Returns:
        ``ConfidenceBranch`` enum (``"retry" | "low" | "mid" | "high"``).
    """
    # D3 fallback: 미제공·음수 → high (1.0 간주)
    if asr_confidence is None or asr_confidence < 0.0:
        return "high"

    if asr_confidence < settings.retry_confidence:
        return "retry"
    if asr_confidence < settings.low_upper:
        return "low"
    if asr_confidence < settings.mid_upper:
        return "mid"
    return "high"


def is_llm_correction_enabled(branch: ConfidenceBranch) -> bool:
    """해당 분기에서 교정 LLM 을 호출해야 하는지 여부.

    * ``mid``  → ``True`` (종결어미 보정 특화)
    * ``low``  → ``True`` (저신뢰 범용 보정 — 호출 측에서 cold-start skip 결정)
    * ``high`` → ``False`` (verbatim 그대로, LLM skip)
    * ``retry`` → ``False`` (ranker 자체를 skip)

    저신뢰 분기의 cold-start LLM skip 규칙 (D25) 은 호출 측에서
    ``phrase_log`` 집계 상태를 보고 추가로 결정해야 한다.
    본 함수는 "분기 단계" 의 활성 여부만 돌려준다.

    Args:
        branch: ``classify_branch`` 반환값.

    Returns:
        교정 LLM 호출 여부 (bool).
    """
    return branch in ("low", "mid")


def expected_candidate_count(branch: ConfidenceBranch) -> int:
    """UI 가 기대하는 후보 수.

    실제 반환되는 후보 수는 매칭 상황·fallback 에 따라 가변 가능
    (예: 중신뢰에서 phrase_log 2건 미매칭 시 preset 보강).
    본 함수는 "분기별 기본 기대값" 을 돌려준다.

    * ``retry`` → 0 (agent skip, HTTP 422)
    * ``low``   → 3
    * ``mid``   → 3 (verbatim 1 + 매칭 2)
    * ``high``  → 1 (verbatim 단독)

    Args:
        branch: ``classify_branch`` 반환값.

    Returns:
        기대 후보 수 (int).
    """
    if branch == "retry":
        return 0
    if branch == "high":
        return 1
    # low, mid
    return 3
