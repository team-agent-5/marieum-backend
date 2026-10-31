"""SuggestResponseV2 — accuracy-mode agent 응답 shape (Mode C / D1).

v2는 /phrase/suggest의 accuracy-mode 전용 응답 형식.
Realtime-mode (/phrase/bypass) 및 preset (/phrase/preset)는 변경 없음.

설계 참조: docs/design/agent-mode-c-api-contract.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.schemas import _ZDatetime  # UTC-Z ISO 8601 직렬화 베이스 재사용

# ─────────── Shared enums ───────────

CandidateSourceV2 = Literal[
    "asr_verbatim",       # 고/중신뢰 rank-1: final_transcript 원문 그대로
    "llm_inference",      # 중신뢰 rank-2/3: LLM이 allowed_set 내에서 선택
    "llm_reconstruction", # 저신뢰 rank-1: LLM이 prefix+history로 문장 재구성
    "ranker_fallback",    # L1 결정론 fallback (분기 불문)
    "catalog_fallback",   # L2 preset catalog fallback
]
"""후보 생성 소스 (v2). audit log 및 프론트 디버그용 출처 식별자."""

BranchV2 = Literal["high", "mid", "low", "retry"]
"""confidence 분기. classify_branch() 반환값과 1:1 대응."""

ModeV2 = Literal["proactive", "candidates", "single", "retry"]
"""응답 shape 결정 discriminator. 프론트가 이 값으로 switch 분기한다."""


# ─────────── Sub-models ───────────

class VerbatimPhrase(BaseModel):
    """고신뢰(高) 분기 ASR 원문. TTS에 그대로 사용된다.

    LLM 제안 수락·거부와 무관하게 verbatim은 항상 즉시 재생된다.
    서버는 이 텍스트를 일체 수정하지 않는다.
    """

    text: str = Field(
        description="final_transcript.strip() 원문. 서버가 어떤 교정도 하지 않은 ASR 결과."
    )
    source: Literal["asr_verbatim"] = Field(
        default="asr_verbatim",
        description="항상 'asr_verbatim' 고정. 프론트 사이드 분기 검증용 리던던시.",
    )


class ProactiveSuggestion(BaseModel):
    """고신뢰(高) 분기 LLM 제안 문장. 환자가 모달에서 수락하면 TTS 재생.

    LLM 내부 출력은 _LLMProactiveSuggestion (Prompt Designer 소유).
    Orchestrator가 이 API 타입으로 변환해 반환한다.
    confidence < 0.5 이면 Orchestrator가 None 처리 → 모달 UI 비표시.
    """

    phrase: str = Field(
        min_length=1,
        max_length=30,
        description="LLM이 allowed_set 내에서 선택한 제안 문장. verbatim과 달라야 함. "
                    "환자 수락 시 flutter_tts.speak() 대상.",
    )
    reason_display: str = Field(
        min_length=1,
        max_length=60,
        description="모달 본문 앞 context 줄. 예: '저녁에 자주 요청하신 음료'. "
                    "이 문장이 왜 제안되었는지 한국어로 설명.",
    )
    modal_text: str = Field(
        min_length=1,
        max_length=80,
        description="모달 질문 전문. 닉네임 치환은 서버에서 완료. "
                    "예: '미지근한 물 주세요라고 말할까요?'",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM self-reported confidence. Orchestrator는 0.5 미만이면 이 제안을 null로 처리함.",
    )


class CandidateV2(BaseModel):
    """후보 1개 (v2). 중신뢰·저신뢰·fallback 응답에서 사용.

    v1 Candidate와의 차이: source 필드 추가 (LLM 경로 감사용).
    rank 상한은 기존과 동일 (le=3).
    """

    rank: int = Field(
        ge=1,
        le=3,
        description="후보 순위. 1이 가장 높은 우선순위. 1·2·3 중 하나.",
    )
    text: str = Field(
        description="추천 문장 원문. 환자 터치 시 flutter_tts.speak()에 전달될 텍스트."
    )
    reason_label: str | None = Field(
        default=None,
        description="추천 근거 enum 값. ReasonLabel 참조 "
                    "(verbatim·time_context·patient_frequency·recency 등). "
                    "fallback 경로는 기존 값 그대로. 근거 없으면 null.",
    )
    reason_display: str | None = Field(
        default=None,
        description="앱 UI용 한국어 라벨. 예: '이 시간대에 자주 선택'. "
                    "reason_label과 1:1 대응. null이면 UI에 라벨 비표시.",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="가중합 점수 (0~1). LLM 생성 후보는 null; "
                    "결정론 ranker 후보만 0~1 범위 가중합 결과를 채움.",
    )
    source: CandidateSourceV2 = Field(
        description="후보 출처. LLM 감사(§12.2 안전 보증) 및 fallback 레벨 파악용. "
                    "asr_verbatim·llm_inference·llm_reconstruction·ranker_fallback·catalog_fallback.",
    )


# ─────────── Base ───────────

class SuggestResponseV2Base(_ZDatetime):
    """모든 v2 응답 공통 필드.

    _ZDatetime 베이스를 상속해 datetime 필드가 'YYYY-MM-DDTHH:MM:SS.sssZ' 형식으로
    JSON 직렬화된다 (UTC, trailing Z).
    """

    trace_id: str = Field(
        description="요청 trace_id echo. 요청에 없었으면 서버가 UUID v4로 발급. "
                    "/phrase/transcribe 응답의 trace_id와 동일해야 같은 발화 세션으로 묶임."
    )
    mode: ModeV2 = Field(
        description="응답 분기 discriminator. 프론트는 이 값으로 switch(mode) 해야 함. "
                    "proactive·candidates·single·retry 4종.",
    )
    branch: BranchV2 = Field(
        description="confidence 분기 (debug/audit용). mode와 1:1이나 별도 노출. "
                    "high·mid·low·retry 4종.",
    )
    asr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="분기 결정에 사용된 ASR confidence echo. 요청에 없으면 1.0으로 처리됨. "
                    "retry 분기는 임계값 미만 신뢰도이므로 null 가능.",
    )
    cold_start: bool = Field(
        default=False,
        description="phrase_log 5건 미만 신규 환자 여부. "
                    "true면 프론트가 '아직 학습 중이에요' 안내 문구 표시 가능 (선택).",
    )
    generated_at: datetime = Field(
        description="후보 생성 완료 시각 (UTC). ISO 8601 trailing Z 형식으로 직렬화됨."
    )
    generation_latency_ms: int = Field(
        ge=0,
        description="서버 내부 처리 지연 ms. LLM 호출 시간 포함. "
                    "목표: 高 ≤500ms / 中 ≤1000ms / 低 ≤1500ms.",
    )
    accuracy_fallback_reason: str | None = Field(
        default=None,
        description="LLM 정상 경로 이탈 사유. null이면 LLM 경로 성공. "
                    "값 예시: '429_rate_limit' | 'llm_timeout' | 'hallucinated_phrase' | "
                    "'low_llm_confidence' | 'ranker_fallback' | 'preset_fallback'.",
    )


# ─────────── 高 분기 ───────────

class ProactiveModeResponse(SuggestResponseV2Base):
    """高 분기 (conf ≥ 0.98) — verbatim 즉시 TTS 재생 + 선택적 모달 제안.

    verbatim은 항상 존재하며 즉시 TTS 재생. suggestion은 LLM 실패 시 null →
    모달 UI 생략, 환자는 verbatim만 받는다.
    """

    mode: Literal["proactive"] = Field(
        default="proactive",
        description="高 분기 discriminator. 항상 'proactive'.",
    )
    branch: Literal["high"] = Field(
        default="high",
        description="confidence 分岐. 高 분기는 항상 'high'.",
    )
    verbatim: VerbatimPhrase = Field(
        description="환자가 말한 그대로 ASR 전사. suggestion과 무관하게 항상 즉시 TTS 재생."
    )
    suggestion: ProactiveSuggestion | None = Field(
        default=None,
        description="LLM 제안 문장. null이면 모달 UI를 렌더링하지 않음. "
                    "null 사유는 accuracy_fallback_reason 참조.",
    )


# ─────────── 中 분기 ───────────

class CandidatesModeResponse(SuggestResponseV2Base):
    """中 분기 (0.90 ≤ conf < 0.98) — 3후보 (verbatim 1 + AI 추론 2).

    candidates[0].source == asr_verbatim (rank 1, verbatim 그대로).
    candidates[1~2].source == llm_inference (rank 2·3), fallback 시 ranker_fallback.
    """

    mode: Literal["candidates"] = Field(
        default="candidates",
        description="中 분기 discriminator. 항상 'candidates'.",
    )
    branch: Literal["mid"] = Field(
        default="mid",
        description="confidence 分岐. 中 분기는 항상 'mid'.",
    )
    candidates: list[CandidateV2] = Field(
        min_length=1,
        max_length=3,
        description="신뢰도 기반 후보 1~3개. 정상 LLM 경로: len=3 "
                    "(verbatim 1 + LLM 추론 2). fallback 시 가변.",
    )


# ─────────── 低 분기 ───────────

class SingleModeResponse(SuggestResponseV2Base):
    """低 분기 (0.40 ≤ conf < 0.90) — AI 단일 후보.

    ML verbatim은 서버에서 audit만 하고 응답에 포함하지 않는다.
    프론트는 큰 글자 1개 UI로 렌더링 (터치 영역 넓게).
    """

    mode: Literal["single"] = Field(
        default="single",
        description="低 분기 discriminator. 항상 'single'.",
    )
    branch: Literal["low"] = Field(
        default="low",
        description="confidence 分岐. 低 분기는 항상 'low'.",
    )
    candidate: CandidateV2 = Field(
        description="LLM 재구성 단일 후보. fallback 시 ranker top-1. "
                    "배열이 아닌 단수 객체임에 주의 (candidates_mode와 다름).",
    )


# ─────────── retry 분기 ───────────

class RetryModeResponse(SuggestResponseV2Base):
    """retry 분기 (conf < 0.40) — 재질의 안내.

    기존에는 HTTP 422 ASR_LOW_CONFIDENCE를 반환했으나 accuracy-mode에서는
    200 + RetryModeResponse로 항상 응답을 보장한다.
    환자가 반드시 뭔가를 받아야 한다는 UX 원칙 유지.
    """

    mode: Literal["retry"] = Field(
        default="retry",
        description="retry 분기 discriminator. 항상 'retry'.",
    )
    branch: Literal["retry"] = Field(
        default="retry",
        description="confidence 分岐. retry 분기는 항상 'retry'.",
    )
    message: str = Field(
        default="다시 말씀해 주세요",
        description="앱 안내 문구 (한국어). 프론트가 그대로 토스트/안내 UI에 표시. "
                    "예: '잘 못 들었어요. 다시 말씀해 주세요.'",
    )


# ─────────── Discriminated Union ───────────

SuggestResponseV2 = Annotated[
    ProactiveModeResponse | CandidatesModeResponse | SingleModeResponse | RetryModeResponse,
    Field(discriminator="mode"),
]
"""POST /phrase/suggest v2 응답 타입 (discriminated union).

FastAPI는 response_model=SuggestResponseV2 어노테이션으로
자동으로 oneOf + discriminator OpenAPI 3.1 형태를 생성한다.

프론트엔드 사용법:
    switch (response.mode) {
        case "proactive":   // ProactiveModeResponse
        case "candidates":  // CandidatesModeResponse
        case "single":      // SingleModeResponse
        case "retry":       // RetryModeResponse
    }
"""


__all__ = [
    "BranchV2",
    "CandidateSourceV2",
    "CandidateV2",
    "CandidatesModeResponse",
    "ModeV2",
    "ProactiveModeResponse",
    "ProactiveSuggestion",
    "RetryModeResponse",
    "SingleModeResponse",
    "SuggestResponseV2",
    "SuggestResponseV2Base",
    "VerbatimPhrase",
]
