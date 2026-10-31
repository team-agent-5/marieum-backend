"""API 요청/응답 Pydantic 모델 (v4 — 환자 1인 앱).

api-spec.md v4.1 기준. v1 돌봄 스키마 전면 폐기.
모든 Field에 `description` 명시 — Swagger 문서 품질 유지.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

# ────────── 공통 Literal / Enum ──────────

PhraseSource = Literal["voice", "preset", "fallback", "voice_bypass", "rejected"]
"""발화 로그의 소스.

- `voice`         — 추천 선택 모드 (M4~M8). 후보 중 1개 선택.
- `preset`        — 빠른 문장 10개 중 선택 (M9).
- `fallback`      — 재질의 3회 실패 후 문장판 복귀 (S13).
- `voice_bypass`  — 실시간 대화 모드 ASR final 즉시 재생 (S16).
- `rejected`      — '다시 말하기' 선택 — negative feedback 로그 (M11).
"""

ReasonLabel = Literal[
    "verbatim",
    "registered_preference",
    "time_context",
    "recency",
    "patient_frequency",
    "sequence_context",
    "semantic_similarity",
    "cold_start",
]
"""후보 추천 근거(라벨). agent-personalization.md §4 + agent-algorithm.md §2.3·§3.9.

- `verbatim`              — "들린 그대로" (중/고신뢰 분기 ASR 원문·LLM 교정)
- `registered_preference` — "등록된 선호" (환자 등록 시 명시)
- `time_context`          — "이 시간대에 자주 선택"
- `recency`               — "최근 요청" (24h 이내)
- `patient_frequency`     — "자주 쓰는 말" (환자 본인 선택 횟수)
- `sequence_context`      — "다음에 자주 하는 말" (bigram)
- `semantic_similarity`   — "비슷한 말투" (벡터 코사인, prefix ≥3자)
- `cold_start`            — phrase_catalog seed (신규 환자 fallback / 저신뢰)
"""

CandidateSource = Literal[
    "voice",
    "preset",
    "fallback",
    "voice_bypass",
    "asr_verbatim",
    "llm_corrected",
    "catalog",
    "patient_log",
    "llm_gen",
    # D2 accuracy-mode agent sources
    "llm_reconstruction",   # 저신뢰 single — history 기반 복원
    "llm_inference",         # 중신뢰 candidates r2/r3 — LLM 추론
    "catalog_fallback",      # L2 time_based preset
    "ranker_fallback",       # L1 deterministic ranker
]
"""후보 생성 소스. LLM 자유 생성 사후 감사용(§12.2 안전 보증).

- `voice`         — 저신뢰 분기 phrase_log 매칭 (6-feature 가중합)
- `preset`        — 빠른 문장판 선택
- `fallback`      — 재질의 3회 실패 후 문장판 복귀
- `voice_bypass`  — 실시간 대화 모드 ASR final 즉시 재생
- `asr_verbatim`  — 중/고신뢰 분기 ASR 원문 그대로 (G8 verbatim 분기)
- `llm_corrected` — 중신뢰 분기 종결어미 LLM 교정본 (G8 verbatim 분기)
- `catalog`       — phrase_catalog seed (cold-start fallback)
- `patient_log`   — phrase_log 집계 후보
- `llm_gen`       — Groq LLM accuracy_mode 재생성
"""

RejectReason = Literal["not_this_phrase", "wrong_context", "all_bad"]
"""환자 거부 사유. `action=rejected` 전용.

- `not_this_phrase` — 후보에 하려던 말이 없음
- `wrong_context`   — 상황에 안 맞음
- `all_bad`         — 전부 틀림
"""

TtsRate = Literal["slow", "normal", "fast"]
"""TTS 속도. 클라가 flutter_tts에 전달 시 매핑: slow=0.3 / normal=0.45 / fast=0.6"""

TtsVolume = Literal["low", "normal", "high"]
"""TTS 음량. 매핑: low=0.5 / normal=1.0 / high=1.5"""

TTS_RATE_MAP: dict[str, float] = {"slow": 0.3, "normal": 0.45, "fast": 0.6}
TTS_VOLUME_MAP: dict[str, float] = {"low": 0.5, "normal": 1.0, "high": 1.5}

SelectionAction = Literal["selected", "rejected"]
"""환자 선택 액션. selected=후보 1개 터치(M7), rejected='다시 말하기' 터치(M11)."""

ErrorCode = Literal[
    "VALIDATION_ERROR",
    "UNAUTHORIZED",
    "FORBIDDEN",
    "NOT_FOUND",
    "CONFLICT",
    "UNPROCESSABLE",
    "AUDIO_TOO_LONG",
    "INVALID_FORMAT",
    "ASR_LOW_CONFIDENCE",
    "UPSTREAM_UNAVAILABLE",
    "TIMEOUT",
    "INTERNAL",
]
"""에러 코드. api-spec.md §1.4"""


# ────────── 공통 베이스 ──────────


class _ZDatetime(BaseModel):
    """datetime을 "...Z" ISO 8601 UTC로 serialize하는 베이스."""

    model_config = ConfigDict(populate_by_name=True)

    @field_serializer("*", when_used="json", check_fields=False)
    def _z(self, v: Any) -> Any:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=UTC)
            return v.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return v


# ────────── Health ──────────


class HealthzResponse(_ZDatetime):
    """`GET /healthz` 응답."""

    status: Literal["ok", "degraded"] = Field(default="ok", description="전체 상태")
    version: str = Field(description="앱 버전 (예: '0.4.1')")
    ts: datetime = Field(description="응답 시각 (UTC)")
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="상류 의존성 상태. db·ml_asr·groq_llm·anthropic 등. 자유 문자열 (예: 'ok','down','unknown','configured','configured_proxy','disabled').",
    )


# ────────── Auth ──────────


class SignupRequest(BaseModel):
    """회원가입 요청 (S0, M0)."""

    user_id: str = Field(
        min_length=4,
        max_length=20,
        pattern=r"^[A-Za-z0-9_]+$",
        description="로그인 ID. 영문/숫자/_만, 4~20자. 유니크.",
        examples=["kim_patient"],
    )
    password: str = Field(
        min_length=8,
        max_length=128,
        description="비밀번호. 8자 이상. 서버에서 scrypt 해시 저장.",
        examples=["P@ssw0rd!"],
    )
    password_confirm: str = Field(description="비밀번호 재확인. password와 완전 일치 필요.")
    nickname: str = Field(
        min_length=1,
        max_length=16,
        description="앱 상단 환자 호칭용 닉네임. 1~16자.",
        examples=["김환자"],
    )

    @field_validator("password_confirm")
    @classmethod
    def _match(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        if v != info.data.get("password"):
            raise ValueError("password_confirm_mismatch")
        return v


class LoginRequest(BaseModel):
    """로그인 요청 (M1)."""

    user_id: str = Field(description="회원가입 시 등록한 user_id.", examples=["kim_patient"])
    password: str = Field(description="비밀번호.", examples=["P@ssw0rd!"])


class PatientBrief(BaseModel):
    """로그인·가입 응답에 포함되는 환자 요약."""

    id: UUID = Field(description="환자 UUID (JWT sub).")
    nickname: str = Field(description="앱 상단 표시용.")
    onboarding_completed: bool = Field(description="첫 가입 시 false. S2 온보딩 1회 완료 후 true.")


class AuthResponse(BaseModel):
    """로그인·회원가입 공통 응답."""

    access_token: str = Field(description="JWT HS256. `Authorization: Bearer <token>`로 사용.")
    token_type: Literal["bearer"] = Field(default="bearer", description="항상 'bearer'.")
    expires_in: int = Field(
        ge=0,
        description="토큰 TTL 초 단위. 365d = 31,536,000s.",
        examples=[31536000],
    )
    patient: PatientBrief = Field(description="가입/로그인한 환자 요약.")


# ────────── Patient ──────────


class PatientPreferences(BaseModel):
    """환자 설정. 로컬이 primary, 서버는 다중 디바이스 복구용 스냅샷."""

    tts_rate: TtsRate = Field(
        default="normal",
        description="TTS 속도. slow(0.3)·normal(0.45)·fast(0.6).",
    )
    tts_volume: TtsVolume = Field(
        default="normal",
        description="TTS 음량. low(0.5)·normal(1.0)·high(1.5).",
    )
    large_text: bool = Field(default=False, description="화면 글자 크기 36sp→48sp.")
    realtime_mode: bool = Field(
        default=False,
        description="S16 실시간 대화 모드 토글. true면 /phrase/bypass 사용 (추천 생략).",
    )
    onboarding_completed: bool = Field(
        default=False,
        description="S2 온보딩 1회 완료. 한 번 true면 false로 되돌리지 않음.",
    )


class PatientMeOut(_ZDatetime):
    """`GET /patient/me` 응답 (M2)."""

    id: UUID = Field(description="환자 UUID.")
    nickname: str = Field(description="앱 상단 호칭.")
    display_name: str | None = Field(
        default=None, description="실명 표시용 (nullable). 백엔드 사후 등록 시만 노출."
    )
    room: str | None = Field(default=None, description="호실 번호 (nullable).")
    disease: str | None = Field(default=None, description="주요 질환 (nullable).")
    onboarding_completed: bool = Field(description="S2 온보딩 완료 플래그.")
    preferences: PatientPreferences = Field(description="TTS·화면·모드 설정 스냅샷.")
    created_at: datetime = Field(description="가입 시각 (UTC).")


class PreferencesPatch(BaseModel):
    """`PATCH /patient/preferences` 요청 — 모든 필드 optional, 보낸 것만 업데이트."""

    tts_rate: TtsRate | None = Field(default=None, description="S14 TTS 속도 변경.")
    tts_volume: TtsVolume | None = Field(default=None, description="S14 TTS 음량 변경.")
    large_text: bool | None = Field(default=None, description="큰 글자 모드 토글.")
    realtime_mode: bool | None = Field(default=None, description="S16 실시간 모드 on/off.")
    onboarding_completed: bool | None = Field(
        default=None, description="S15 온보딩 완료 표시. true로 세팅할 때만 반영됨 (false는 무시)."
    )


# ────────── Phrase — Suggest (M6) ──────────


class SuggestRequest(BaseModel):
    """`POST /phrase/suggest` 요청 — ASR interim/final transcript로 후보 3개 요청."""

    patient_prefix: str = Field(
        min_length=1,
        max_length=200,
        description="ASR 중간 전사 또는 ellipsis prefix. 예: '물' (환자가 아직 발화 중).",
        examples=["물"],
    )
    final_transcript: str | None = Field(
        default=None,
        max_length=400,
        description="ASR final transcript (있을 때). ranking의 prefix_match·semantic_similarity 입력.",
        examples=["물 좀 주세요"],
    )
    asr_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="WS final.confidence 값. 분기 결정 입력 (retry/low/mid/high). "
                    "미제공 시 1.0 → 고신뢰 fallback (D3 결정).",
    )
    source: Literal["voice"] = Field(
        default="voice",
        description="현재 'voice' 고정. preset·bypass는 각자 전용 엔드포인트 사용.",
    )
    trace_id: str | None = Field(
        default=None,
        description="같은 발화 세션 묶음용. /phrase/transcribe 응답의 trace_id 그대로. 미제공 시 서버 발급.",
    )
    previous_log_id: int | None = Field(
        default=None,
        description="직전 phrase_log.id. sequence_context feature 입력. "
        "null이면 서버가 30분 창 내 가장 최근 로그로 자동 유추.",
    )
    accuracy_mode: bool = Field(
        default=False,
        description="환자 '정확 모드' 토글. true면 결정론 top-5 → Groq LLM 재생성 3개. "
                    "false면 결정론 top-3만. final_transcript가 null이면 무시됨.",
    )
    client_ts: datetime | None = Field(
        default=None,
        description="클라이언트 시각 (선택). 서버 drift 보정·time_context 재계산용.",
    )
    # patient_id는 JWT에서 추출 — body에 없음


class Candidate(BaseModel):
    """후보 1개. `/phrase/suggest` 응답에 노출."""

    rank: int = Field(ge=1, le=3, description="순위. 1·2·3.")
    text: str = Field(description="추천 문장 원문. 환자가 터치 시 그대로 TTS 재생될 텍스트.")
    reason_label: ReasonLabel | None = Field(
        default=None,
        description="추천 근거 (로그·분석용 enum). 매칭 feature 없으면 null.",
    )
    reason_display: str | None = Field(
        default=None,
        description="앱 UI용 한국어 라벨. 예: '자주 쓰는 말'. reason_label과 1:1.",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="종합 점수 (0~1). 디버그/정렬 참고용. "
                    "verbatim 분기 (중·고신뢰) 는 ranking 가중합이 의미 없어 **null** 로 반환. "
                    "저신뢰 분기는 0~1 범위의 가중합 결과.",
    )


class SuggestResponse(_ZDatetime):
    """`POST /phrase/suggest` 응답."""

    trace_id: str = Field(description="요청 trace_id echo (없었으면 서버 발급).")
    candidates: list[Candidate] = Field(
        max_length=3,
        description="confidence 분기별 후보 1~3개 (api-spec §3.4.2). "
                    "저신뢰: 6-feature 가중합 top-3 (empty 시 catalog seed 1건). "
                    "중신뢰: verbatim 1위 + 매칭 2위·3위. "
                    "고신뢰: verbatim 1건만. "
                    "재질의 분기는 HTTP 422 ASR_LOW_CONFIDENCE로 반환되므로 이 응답에 도달하지 않음.",
    )
    generated_at: datetime = Field(description="후보 생성 완료 시각.")
    generation_latency_ms: int = Field(
        ge=0, description="서버 내부 ranking 지연 (ms). 500 이내 목표."
    )


# ────────── Phrase — Selection (M7·M11, 피드백 루프) ──────────


class CandidateSnapshot(BaseModel):
    """phrase_log.candidates_shown JSONB 저장용 스냅샷.

    `reason_display`는 로그에 불필요(앱 UI 전용)이라 제외.
    `source`는 LLM 생성 후보 사후 감사용(§12.2 안전 보증).
    """

    rank: int | None = Field(default=None, ge=1, le=3, description="노출 순위. D2 agent single 모드는 null.")
    text: str = Field(description="노출된 후보 문장.")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="ranking 점수. D2 agent는 confidence 또는 null.")
    reason_label: ReasonLabel | None = Field(default=None, description="추천 근거 enum.")
    source: CandidateSource = Field(
        description="후보 생성 소스. catalog·patient_log·llm_gen·llm_*."
    )


class SelectionRequest(BaseModel):
    """`POST /phrase/selection` — 환자 선택/거부 로그 (피드백 루프 ★)."""

    trace_id: str = Field(description="/phrase/suggest 응답의 trace_id.")
    action: SelectionAction = Field(
        description="selected(M7, 후보 터치) or rejected(M11, '다시 말하기' 터치)."
    )
    prefix: str = Field(
        min_length=1,
        max_length=200,
        description="ASR 중간 전사 snapshot (ranking input 재현용).",
    )
    final_transcript: str | None = Field(
        default=None, max_length=400, description="ASR final transcript (있을 때)."
    )
    candidates_shown: list[CandidateSnapshot] = Field(
        min_length=1,
        max_length=3,
        description="노출된 후보 3개 (1~3개 가능). 환자가 뭘 보고 결정했는지 재현.",
    )
    selected_phrase: str | None = Field(
        default=None,
        max_length=400,
        description="환자가 선택한 문장. action=selected 시 필수. rejected 시 null.",
    )
    selected_rank: Literal[1, 2, 3] | None = Field(
        default=None,
        description="선택한 후보의 순위. action=selected 시 1·2·3. rejected 또는 fallback 시 null.",
    )
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="후보 노출 → 터치까지 ms. 평가 지표 '선택 완료 시간' 산출용. 미측정 시 null.",
    )
    source: Literal["voice", "fallback"] = Field(
        default="voice",
        description="voice=후보 선택 모드. fallback=재질의 실패 후 문장판 선택.",
    )
    prev_log_id: int | None = Field(
        default=None,
        description="직전 phrase_log.id. sequence_context 체인 구성. 미제공 시 서버 30분창 자동 유추.",
    )
    reject_reason: RejectReason | None = Field(
        default=None,
        description="거부 사유 (action=rejected 전용). ranking engine negative signal 세분화.",
    )

    @model_validator(mode="after")
    def _enforce_action(self) -> Self:
        if self.action == "selected":
            if self.selected_phrase is None:
                raise ValueError("selected_phrase required when action=selected")
            if self.reject_reason is not None:
                raise ValueError("reject_reason must be null when action=selected")
        else:  # rejected
            if self.selected_phrase is not None or self.selected_rank is not None:
                raise ValueError("selected_phrase/rank must be null when action=rejected")
        return self


class SelectionResponse(_ZDatetime):
    """`POST /phrase/selection` 응답."""

    log_id: int = Field(description="phrase_log 테이블 PK (bigint). GET /phrase/history로 조회 가능.")
    final_text: str | None = Field(
        description="action=selected: selected_phrase. rejected: null. 앱이 받자마자 flutter_tts.speak()."
    )
    created_at: datetime = Field(description="로그 저장 시각.")


# ────────── Phrase — Preset (M9) ──────────


class PresetRequest(BaseModel):
    """`POST /phrase/preset` — 빠른 문장 10개 중 1개 즉시 선택.

    **10개 catalog** (하드코딩, 코드 변경 없이 심사장 변경 불가):
    1. 💧 물 주세요
    2. 🚽 화장실 가고 싶어요
    3. 👍 네
    4. 👎 아니요
    5. 🙏 고마워요
    6. 😔 미안해요
    7. 😣 아파요
    8. 🥶 춥거나 더워요
    9. 🆘 도와주세요
    10. 🙋 불러주세요
    """

    preset_index: int = Field(
        ge=1,
        le=10,
        description="빠른 문장 번호 1~10. 상세 매핑은 모델 docstring 참조.",
        examples=[1],
    )


class PresetResponse(_ZDatetime):
    """`POST /phrase/preset` 응답."""

    log_id: int = Field(description="phrase_log PK.")
    final_text: str = Field(
        description="재생할 문장. 앱이 받자마자 flutter_tts.speak().",
        examples=["물 주세요"],
    )
    preset_index: int = Field(description="요청한 preset_index echo.")
    created_at: datetime = Field(description="로그 저장 시각.")


# ────────── Phrase — Bypass (S16 실시간 대화 모드) ──────────


class BypassRequest(BaseModel):
    """`POST /phrase/bypass` — 실시간 대화 모드. ASR final transcript 즉시 재생.

    환자가 S3에서 ⚡ 실시간 모드를 선택한 상태에서만 호출. 추천(/suggest) 생략.
    """

    final_transcript: str = Field(
        min_length=1,
        max_length=400,
        description="/phrase/transcribe 응답의 final_transcript 그대로.",
    )
    asr_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="/phrase/transcribe 응답의 asr_confidence 그대로. **0.4 미만이면 서버가 422로 재질의 강제**.",
    )
    trace_id: str | None = Field(default=None, description="/phrase/transcribe 응답의 trace_id.")
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="녹음 종료 → bypass 호출까지 ms. 평가 지표용 (선택).",
    )


class BypassResponse(_ZDatetime):
    """`POST /phrase/bypass` 응답."""

    log_id: int = Field(description="phrase_log PK (source=voice_bypass).")
    final_text: str = Field(description="재생할 문장. 앱이 받자마자 flutter_tts.speak().")
    created_at: datetime = Field(description="로그 저장 시각.")


# ────────── Phrase — History (S12) ──────────


class PhraseHistoryItem(_ZDatetime):
    """히스토리 1개 아이템."""

    log_id: int = Field(description="phrase_log PK. '다시 재생'은 서버 호출 없이 로컬 처리.")
    final_text: str = Field(description="당시 재생된 문장.")
    source: PhraseSource = Field(description="voice·preset·fallback·voice_bypass.")
    selected_rank: int | None = Field(
        default=None,
        description="voice 모드일 때 선택한 후보 순위. preset·bypass는 null.",
    )
    created_at: datetime = Field(description="저장 시각.")


class PhraseHistoryResponse(BaseModel):
    """`GET /phrase/history` 응답."""

    items: list[PhraseHistoryItem] = Field(description="최신순 발화 로그.")
    next_cursor: str | None = Field(default=None, description="다음 페이지 커서. null이면 끝.")
    total_estimate: int = Field(ge=0, description="추정 전체 건수.")


# ────────── 공통 에러 ──────────


class ErrorDetail(BaseModel):
    """validation·domain 에러 세부 1건."""

    field: str | None = Field(default=None, description="문제 발생 필드 경로. 'body.password' 등.")
    issue: str = Field(description="문제 요약. 'missing' / 'too_short' / 'below_threshold' 등.")
    value: str | int | float | bool | None = Field(
        default=None, description="참고용 실제 값 (민감정보면 생략)."
    )


class ErrorResponse(BaseModel):
    """공통 에러 응답 (api-spec.md §1.4)."""

    error_code: ErrorCode = Field(description="12종 enum 중 하나.")
    message: str = Field(description="사람이 읽을 수 있는 요약 (앱 UI 토스트용).")
    trace_id: str = Field(description="요청 trace_id. 로그 상관관계 추적용.")
    details: list[ErrorDetail] = Field(
        default_factory=list, description="field 수준 세부 (validation·low confidence 등).",
    )


# ────────── 인증 Principal (내부 전용) ──────────


class PatientPrincipal(BaseModel):
    """JWT 디코딩 결과. 응답에 노출되지 않는 내부 전용 (`Depends(require_patient)` 반환)."""

    patient_id: UUID
    nickname: str


# ────────── Phrase — Transcribe (M4·M5, 멀티파트 오디오 ASR) ──────────


class TranscribeResponse(_ZDatetime):
    """`POST /phrase/transcribe` 응답 — 멀티파트 오디오 업로드 ASR 결과.

    v4(B안): WebSocket 제거 후 환자앱이 녹음 완료된 오디오 blob을 **1회 POST**로 전송.
    서버가 ASR → final transcript + confidence 반환. 이후 클라가 그 값을 들고
    `/phrase/suggest` 또는 `/phrase/bypass`를 호출.
    """

    trace_id: str = Field(description="세션 ID. 이후 /suggest·/bypass·/selection에 이어 사용.")
    final_transcript: str = Field(
        description="ASR 최종 전사. 저신뢰(asr_confidence < 0.4)이면 S13 재질의 안내 분기.",
    )
    asr_confidence: float = Field(
        ge=0.0, le=1.0, description="ASR 신뢰도. 0~1. 0.4 미만이면 UI에서 재녹음 유도."
    )
    duration_ms: int = Field(
        ge=0, description="녹음 길이 (ms). 10,000ms 상한. 서버 측 재계산 (파일 메타 기반)."
    )
    heard_text: str | None = Field(
        default=None,
        description="저신뢰 시 '들린 내용' 노출용 (S13). 신뢰도가 임계값 이상이면 null.",
    )


# ────────── 공통 에러 ──────────


# (ErrorDetail/ErrorResponse 이미 위에 정의됨 — 중복 방지)


# ────────── Streaming WebSocket 메시지 모델 (S16 실시간 대화 모드) ──────────
#
# **시니어 결정 §4 (M2) 기준으로 통일됨**:
# - 서버 → 앱 메시지는 `transcript{final: bool}` 1종 (interim/final 분리 모델 폐기)
# - WS 전용 ErrorCode 5종만: UNAUTHORIZED, AUDIO_TOO_LONG,
#   ML_UPSTREAM_UNAVAILABLE, ASR_LOW_CONFIDENCE, INTERNAL
# - WebSocket close code: 1000 정상 · 1001 환자 cancel · 4001 JWT 유효 X ·
#   4013 10초 초과 · 4503 ML 상류 장애
#
# **격리 조건 (M4)**:
# - 이 모델들은 `app/agent/streaming/schemas.py` 로 이동할 수도 있으나,
#   1차 MVP에서는 schema 중복을 피하기 위해 메인 schemas.py에 포함.
# - 다만 `agent/ranking/*` 에서는 이 모델들을 import하지 말 것 (상호 import 금지).
#
# **M3 (phrase_log INSERT 책임 분리)**:
# - WS 라우터(`agent/streaming/router.py`)는 phrase_log 저장 금지.
# - 환자앱이 `final: true` 수신 후 `tts.setCompletionHandler`에서
#   `POST /phrase/bypass` 를 호출해 저장 (유일한 저장 경로).


WSErrorCode = Literal[
    "UNAUTHORIZED",
    "AUDIO_TOO_LONG",
    "ML_UPSTREAM_UNAVAILABLE",
    "ASR_LOW_CONFIDENCE",
    "INTERNAL",
]
"""실시간 모드 전용 ErrorCode (M2 기준 5종).

메인 `ErrorCode`와는 교집합은 있지만 WS 세션 운영상 필요한 축소 집합.
그 외 모든 상류 예외는 `INTERNAL`로 수렴시킨다.
"""


# WebSocket close code 상수 (M2 §4):
#   1000 정상 종료 (final 수신 완료)
#   1001 환자 cancel (Going Away)
#   4001 JWT 유효하지 않음 (UNAUTHORIZED)
#   4013 녹음 10초 초과 (AUDIO_TOO_LONG)
#   4503 ML 상류 장애 (ML_UPSTREAM_UNAVAILABLE)
WSCloseCode = Literal[1000, 1001, 4001, 4013, 4503]


# ── Client → Server ──


class WSAudioIn(BaseModel):
    """클라→서버: 오디오 청크 (base64 JSON 대체 경로).

    기본 전송은 WebSocket binary frame (PCM16 raw) 이며 이 모델은
    base64 JSON 경로로만 사용. `app/agent/streaming/router.py` 측에서
    binary opcode 수신 시 이 모델을 거치지 않고 바로 ML proxy로 pass-through.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["audio"] = Field(description="고정.")
    chunk: str = Field(description="base64 encoded PCM16 또는 Opus bytes.")
    seq: int = Field(ge=0, description="서버 재조립용 시퀀스.")
    format: Literal["pcm16", "opus"] = Field(default="pcm16", description="오디오 포맷.")
    sample_rate: Literal[16000, 24000, 48000] = Field(
        default=16000, description="샘플링 레이트. 1차 MVP는 16000 고정 사용."
    )


class WSEndIn(BaseModel):
    """클라→서버: 오디오 입력 종료. 최종 인식을 요청하며 연결은 유지한다."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["end"] = Field(description="고정.")


class WSCancelIn(BaseModel):
    """클라→서버: 환자가 '다시 말하기' 터치. 서버는 phrase_log 저장 안 함."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["cancel"] = Field(description="고정.")


class WSCloseIn(BaseModel):
    """클라→서버: 대화 연결 종료. 백엔드와 ML 연결을 함께 정리한다."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["close"] = Field(description="고정.")


WSClientMessage = Annotated[
    WSAudioIn | WSEndIn | WSCancelIn | WSCloseIn,
    Field(discriminator="type"),
]
"""클라→서버 제어 메시지 discriminated Union.

바이너리 오디오 프레임은 이 Union에 포함되지 않는다
(WebSocket opcode 수준에서 분기).
"""


# ── Server → Client ──


class WSTranscriptOut(BaseModel):
    """서버→앱: 전사 (interim/final 통합).

    **시니어 결정 §4 (M2) 기준으로 interim/final 분리 모델 폐기**.
    `final: bool` 플래그 1종으로 통일. 흐릿/선명 투명도 구분 없음 —
    `text` 는 항상 환자가 바로 쓸 수 있는 형태.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["transcript"] = Field(description="고정.")
    text: str = Field(
        min_length=0,
        max_length=400,
        description="앱이 그대로 UI 표시 또는 TTS 재생할 문자열. "
        "현재 실시간 경로는 Haiku를 연결하지 않고 ML 전사에 문자 필터만 적용한다.",
    )
    final: bool = Field(
        description="false=interim (화면 갱신용), true=final (TTS 재생 신호).",
    )
    trace_id: str = Field(description="세션 UUID. 서버가 WS accept 직후 발급.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="final=true 일 때 ML ASR confidence를 그대로 전달. LLM 자체 confidence와 별개. interim은 null/생략.",
    )


class WSErrorOut(BaseModel):
    """서버→앱: 오류 안내. 인증 후 대화 연결은 프론트가 종료를 결정한다."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = Field(description="고정.")
    error_code: WSErrorCode = Field(description="M2 기준 5종 enum.")
    message: str = Field(description="한국어 토스트 문구. 민감 정보 포함 금지.")
    trace_id: str = Field(description="로그 상관관계 추적용.")


class WSEndNoticeOut(BaseModel):
    """서버→앱: 세션 강제 종료 알림 (close 전).

    1차 MVP는 `reason="max_duration"` 만 구현.
    client_cancel 은 close code 1001 로 커버됨.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["end_notice"] = Field(description="고정.")
    reason: Literal["max_duration"] = Field(
        description="1차 MVP: 10초 초과 'max_duration' 만 구현.",
    )


WSServerMessage = Annotated[
    WSTranscriptOut | WSErrorOut | WSEndNoticeOut,
    Field(discriminator="type"),
]
"""서버→앱 메시지 discriminated Union.

**시니어 결정 §4 (M2)**: interim/final 분리 모델(`WSInterimOut`/`WSFinalOut`) 폐기.
`WSTranscriptOut{final: bool}` 1종 + 에러 + end_notice 로 통일.
"""


__all__ = [
    "TTS_RATE_MAP",
    "TTS_VOLUME_MAP",
    "AuthResponse",
    "BypassRequest",
    "BypassResponse",
    "Candidate",
    "CandidateSnapshot",
    "CandidateSource",
    "ErrorCode",
    # Error
    "ErrorDetail",
    "ErrorResponse",
    # Health
    "HealthzResponse",
    "LoginRequest",
    "PatientBrief",
    "PatientMeOut",
    # Patient
    "PatientPreferences",
    "PatientPrincipal",
    "PhraseHistoryItem",
    "PhraseHistoryResponse",
    # Enums
    "PhraseSource",
    "PreferencesPatch",
    "PresetRequest",
    "PresetResponse",
    "ReasonLabel",
    "RejectReason",
    "SelectionAction",
    "SelectionRequest",
    "SelectionResponse",
    # Auth
    "SignupRequest",
    "SuggestRequest",
    "SuggestResponse",
    # Phrase
    "TranscribeResponse",
    "TtsRate",
    "TtsVolume",
    # Streaming WS (S16 실시간 모드)
    "WSAudioIn",
    "WSCancelIn",
    "WSClientMessage",
    "WSCloseCode",
    "WSCloseIn",
    "WSEndIn",
    "WSEndNoticeOut",
    "WSErrorCode",
    "WSErrorOut",
    "WSServerMessage",
    "WSTranscriptOut",
]
