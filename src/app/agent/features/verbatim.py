"""Verbatim strategy — build a `PoolCandidate` from ASR final transcript.

- 중/고신뢰 분기에서 호출되는 "교정 없이 그대로 노출" 또는 "교정 LLM 결과 반영" 경로용.
- 이 모듈 자체는 **I/O 금지**: LLM 호출은 `branches/mid.py` 가 수행하고, 그 결과(교정된 문자열)를
  `source="llm_corrected"` 로 넘겨주면 이 모듈이 `PoolCandidate` 로 포장만 한다.
- 고신뢰 분기에서는 ASR 결과를 그대로 `source="asr_verbatim"` 으로 포장.

생성 규칙:
    cnt_90d       = 0           # patient_frequency 입력 — verbatim 은 집계 비대상
    last_at       = now         # recency 가 계산되더라도 "방금 발화" 로 간주되도록
    catalog_id    = None        # catalog seed 가 아님
    embedding     = None        # ranker 가 skip (semantic_similarity 비활성)
    reason_label  = "verbatim"  # 응답 매핑은 branches/*.py 담당

"의미 있음" 기준 (사용자 지침):
    `strip()` 후 길이 ≥ 1 — 그렇지 않으면 `ValueError` (재질의 분기가 라우팅 단에서 걸러야 함).
"""
from __future__ import annotations

from datetime import UTC, datetime

from app.agent.contracts import CandidateSource, PoolCandidate

name = "verbatim"
# 가중합에 참여하지 않는 "구조적" feature — weight 는 상징적 1.0 (branches/*.py 가 직접 할당).
weight = 1.0

# 안전 상한: ASR 결과가 비정상적으로 길어도 schema (Candidate.text max_length) 와
# phrase_catalog.phrase (varchar 100) 와 정합을 맞춘다.
MAX_VERBATIM_LEN = 100


def is_meaningful(text: str | None) -> bool:
    """공백 제거 후 1자 이상이면 True (사용자 지침).

    - `None` / 빈 문자열 / 공백만 있는 문자열 → False.
    - 중신뢰·고신뢰 분기 진입 전 라우터가 1차 가드하지만, 방어적으로 재확인.
    """
    if text is None:
        return False
    return len(text.strip()) >= 1


def _truncate(text: str, limit: int = MAX_VERBATIM_LEN) -> str:
    """스키마 상한 초과 시 안전하게 자른다 (경계 공백 trim 재적용)."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip()


def build_verbatim_candidate(
    text: str,
    *,
    source: CandidateSource = "asr_verbatim",
    now: datetime | None = None,
) -> PoolCandidate:
    """ASR 결과(또는 교정 LLM 결과)를 `PoolCandidate` 로 포장.

    Parameters
    ----------
    text:
        verbatim 으로 노출할 문자열. 고신뢰 분기 → ASR final_transcript 원문,
        중신뢰 분기 → `llm_assist.corrective_rerank_with_llm()` 결과.
    source:
        `"asr_verbatim"` (고신뢰, 기본) 또는 `"llm_corrected"` (중신뢰).
        그 외 값이 들어오면 `ValueError`.
    now:
        현재 시각 (UTC naive 또는 aware). `None` 이면 `datetime.now(UTC)` 사용.

    Returns
    -------
    PoolCandidate
        `cnt_90d=0`, `last_at=now`, `catalog_id=None`, `embedding=None`.

    Raises
    ------
    ValueError
        - `is_meaningful(text) is False` 인 경우 (공백·None·빈 문자열).
        - `source` 가 `"asr_verbatim"` / `"llm_corrected"` 가 아닌 경우.
    """
    if not is_meaningful(text):
        raise ValueError("verbatim text must contain at least one non-whitespace character")
    if source not in ("asr_verbatim", "llm_corrected"):
        raise ValueError(
            f"verbatim source must be 'asr_verbatim' or 'llm_corrected', got {source!r}"
        )

    phrase = _truncate(text)
    stamp = now if now is not None else datetime.now(UTC)

    return PoolCandidate(
        phrase=phrase,
        source=source,
        cnt_90d=0,
        last_at=stamp,
        catalog_id=None,
        embedding=None,
    )


def compute(cand: object, ctx: object) -> float:  # type: ignore[override]
    """구조적 feature — 가중합 참여 X.

    `features/*.compute(cand, ctx)` 시그니처와 맞춰두되, branches/*.py 가 verbatim 후보에 대해
    `base = 1.0` 을 **직접 할당** 하는 설계이므로 이 함수는
    호출될 일이 없다. 호환성 유지를 위한 noop.
    """
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    source: str = cand.source  # type: ignore[attr-defined]
    if source in ("asr_verbatim", "llm_corrected") and is_meaningful(phrase):
        return 1.0
    return 0.0
