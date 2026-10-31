"""Agent I/O contracts (v4). Immutable per-request shapes."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ConfidenceBranch = Literal["retry", "low", "mid", "high"]
CandidateSource = Literal["patient_log", "catalog", "llm_gen", "asr_verbatim", "llm_corrected"]


class AgentInput(BaseModel):
    """One request into the ranker."""

    patient_id: UUID
    prefix: str = Field(min_length=0, max_length=200)
    event_type: Literal["interim", "final"] = "final"
    final_transcript: str | None = None
    asr_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="WS final.confidence 값. 분기 결정 입력. 미제공 시 1.0 → 고신뢰로 fallback (D3).",
    )
    previous_phrase: str | None = Field(
        default=None,
        description="직전 phrase (30분 이내). sequence_context 입력.",
    )
    now: datetime
    accuracy_mode: bool = False


class PoolCandidate(BaseModel):
    """후보 풀 1건. UNION 결과."""

    phrase: str
    source: CandidateSource
    cnt_90d: int = 0  # patient_frequency 입력
    last_at: datetime | None = None  # recency 입력
    catalog_id: int | None = None  # catalog 추적
    embedding: list[float] | None = None  # F5 입력, nullable


class FeatureScores(BaseModel):
    """7-feature (6 base + rejected_penalty) + bonus 분해 — audit_log 저장용."""

    prefix_match: float = 0.0
    patient_frequency: float = 0.0
    time_context: float = 0.0
    sequence_context: float = 0.0
    semantic_similarity: float = 0.0
    recency: float = 0.0
    preference_bonus: float = 0.0
    rejected_penalty: float = 0.0  # 음수 or 0. rejected feedback.


class CandidateScore(BaseModel):
    """ranker 내부 state. API 응답으로 바로 노출되진 않음."""

    phrase: str
    source: CandidateSource
    scores: FeatureScores
    base: float
    bonus: float
    total: float
    reason_label: str | None = None


class RankingContext(BaseModel):
    """단일 트랜잭션 선계산 결과. 모든 feature가 이걸 공유."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patient_id: UUID
    prefix: str
    prefix_vec: list[float] | None = None  # None if len<3
    now: datetime
    previous_phrase: str | None
    accuracy_mode: bool
    time_bucket_kst: Literal["morning", "afternoon", "evening", "night"]
    # 집계 선계산
    freq_counts: dict[str, int] = Field(default_factory=dict)  # phrase -> 90d count
    freq_total: int = 0
    time_bucket_counts: dict[str, dict[str, int]] = Field(default_factory=dict)  # phrase -> {bucket: cnt}
    time_bucket_total: dict[str, int] = Field(default_factory=dict)  # phrase -> total 60d
    bigram_counts: dict[str, int] = Field(default_factory=dict)  # (prev→X) next_phrase -> cnt
    bigram_total: int = 0
    last_at: dict[str, datetime] = Field(default_factory=dict)  # phrase -> MAX(created_at)
    prefs: list[dict] = Field(default_factory=list)  # patient_preference rows
    # 24h rejected 이벤트 집계 (features/rejected_penalty.compute 입력).
    # phrase -> [(rejected_at_utc, multiplicity), ...]. 비어있으면 penalty=0.
    # 저신뢰 분기에서만 build_ranking_context 가 채운다 (고신뢰/재질의는 ctx skip).
    rejected_counts: dict[str, list[tuple[datetime, int]]] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """ranker 반환 — API 응답 매핑 소스."""

    suggestions: list[dict]  # final 후보 (text, rank, reason_label, score, source)
    latency_ms: int
    cold_start: bool
    accuracy_mode_used: bool = False
    branch: ConfidenceBranch = "low"  # 분기 결정 결과 (audit·테스트용)
    audit_payload: dict | None = None  # per-candidate FeatureScores for audit_log
