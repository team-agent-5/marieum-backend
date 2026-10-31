"""Phrase 라우터 (v4) — HTTP 발화 흐름.

- transcribe: 멀티파트 오디오 → ASR 전사 (M4·M5, WS 대체)
- suggest: 추천 후보 3개 (agent 플레이스홀더)
- selection: 환자 선택·거부 로그
- preset: 빠른 문장 10개 즉시 로그
- bypass: 실시간 모드 ASR final 즉시 로그
- history: 발화 히스토리

**agent 실제 ranking 로직은 미구현** — mock 응답 반환. ranking engine 작성 시
`app/agent/ranker.py`로 분리 예정.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import audit_append
from app.db import get_session
from app.middleware import get_trace_id
from app.models import PhraseLog
from app.schemas import (
    BypassRequest,
    BypassResponse,
    PatientPrincipal,
    PhraseHistoryItem,
    PhraseHistoryResponse,
    PresetRequest,
    PresetResponse,
    SelectionRequest,
    SelectionResponse,
    SuggestRequest,
    TranscribeResponse,
)
from app.schemas_v2 import (
    CandidatesModeResponse,
    CandidateV2,
    ProactiveModeResponse,
    ProactiveSuggestion,
    RetryModeResponse,
    SingleModeResponse,
    SuggestResponseV2,
    VerbatimPhrase,
)
from app.security import require_patient
from app.services.embedding import embed_text

log = logging.getLogger(__name__)

# 오디오 업로드 상한: 10초 녹음 + 여유 = 2MiB
_MAX_AUDIO_BYTES = 2 * 1024 * 1024
_ASR_LOW_CONFIDENCE = 0.4

router = APIRouter(prefix="/phrase", tags=["phrase"])


# ────────── 빠른 문장 10개 (하드코딩) ──────────

PRESET_PHRASES: list[str] = [
    "물 주세요",
    "화장실 가고 싶어요",
    "네",
    "아니요",
    "고마워요",
    "미안해요",
    "아파요",
    "춥거나 더워요",
    "도와주세요",
    "불러주세요",
]


# ────────── POST /phrase/transcribe — 멀티파트 오디오 ASR ──────────


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="멀티파트 오디오 → ASR 전사 (M4·M5, WS 대체)",
    responses={
        413: {"description": "오디오 길이/크기 초과 (10초, 2MiB)"},
        415: {"description": "지원하지 않는 Content-Type"},
    },
)
async def transcribe(
    request: Request,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
    audio: Annotated[UploadFile, File(description="녹음 오디오 blob. PCM16 WAV 또는 Opus webm.")],
    duration_ms: Annotated[
        int | None,
        Form(ge=0, le=15_000, description="클라 측정 녹음 길이 (ms). 서버 추정이 어려우면 fallback."),
    ] = None,
) -> TranscribeResponse:
    # Content-Type 가드 (audio/* 만 허용)
    ctype = (audio.content_type or "").lower()
    if not ctype.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error_code": "INVALID_FORMAT", "message": f"audio/* 가 아닙니다: {ctype or 'unknown'}"},
        )

    # 사이즈 가드 (스트리밍 read + 상한 체크, 메모리 폭발 방지)
    data = await audio.read(_MAX_AUDIO_BYTES + 1)
    if len(data) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error_code": "AUDIO_TOO_LONG",
                "message": "오디오 크기가 2MiB를 초과했습니다.",
            },
        )

    trace_id = get_trace_id(request) or str(uuid.uuid4())

    # TODO(ML): 실제 ASR 서버 호출. 지금은 하드코딩 mock.
    final_text = "물 좀 주세요"
    confidence = 0.82
    effective_duration = duration_ms if duration_ms is not None else 1800

    await audit_append(
        session,
        trace_id=trace_id,
        patient_id=me.patient_id,
        event="transcript_final",
        payload={
            "duration_ms": effective_duration,
            "bytes": len(data),
            "content_type": ctype,
            "confidence": confidence,
        },
    )
    await session.commit()

    return TranscribeResponse(
        trace_id=trace_id,
        final_transcript=final_text,
        asr_confidence=confidence,
        duration_ms=effective_duration,
        heard_text=final_text if confidence < _ASR_LOW_CONFIDENCE else None,
    )


# ────────── POST /phrase/suggest — 추천 후보 (mock) ──────────


# B2.2 — CandidateSnapshot.source (API CandidateSource enum, 9종) → phrase_log.source
#       (DB CHECK constraint, voice|preset|fallback|voice_bypass|rejected) 매핑.
#
# **중요**: 응답의 ``CandidateSource`` enum 과 DB ``phrase_log.source`` 는 서로 다른 집합.
#   API 는 "후보 생성 소스" (asr_verbatim, llm_corrected, catalog, patient_log, llm_gen ...)
#   DB  는 "선택 결과 소스" (voice/preset/fallback/voice_bypass/rejected)
# selection 저장 시 candidates_shown JSONB 내부 source 값을 DB enum 으로 매핑해
# 운영 데이터 분석 쿼리 (SQL CHECK 안 걸리지만 자연스런 join 가능) 를 간소화.
_SOURCE_TO_DB: dict[str, str] = {
    # DB 와 동일 식별자는 그대로.
    "voice": "voice",
    "preset": "preset",
    "fallback": "fallback",
    "voice_bypass": "voice_bypass",
    # API 전용 (후보 생성 소스) → voice (추천 선택 모드로 흡수).
    "asr_verbatim": "voice",
    "llm_corrected": "voice",
    "patient_log": "voice",
    "llm_gen": "voice",
    # catalog seed 는 "fallback" 성격 (phrase_catalog 수동 fallback 과 동일).
    "catalog": "fallback",
}


def _map_candidate_source_to_db(source: str | None) -> str:
    """API ``CandidateSource`` → DB ``phrase_log.source`` 매핑 (B2.2).

    매핑 규칙은 ``_SOURCE_TO_DB`` 참고. 미지정·미등록 값은 ``"voice"`` 로 흡수.
    """
    if not source:
        return "voice"
    return _SOURCE_TO_DB.get(source, "voice")


@router.post(
    "/suggest",
    response_model=SuggestResponseV2,
    summary="후보 추천 (v2, mode별 shape) — D1 mock 응답, D2+에서 LLM 결선",
    description=(
        "**D1 accuracy-mode mock**: LLM/ranker 호출 없이 asr_confidence 기반으로 "
        "4분기(proactive/candidates/single/retry) 중 하나를 즉시 반환.\n\n"
        "- `mode=proactive` (conf ≥ 0.98): verbatim 즉시 TTS + 선택적 LLM 제안 모달\n"
        "- `mode=candidates` (0.90 ≤ conf < 0.98): 3후보 선택 UI\n"
        "- `mode=single` (0.40 ≤ conf < 0.90): 단일 후보 큰 글자 UI\n"
        "- `mode=retry` (conf < 0.40): 재질의 안내 (200 OK, HTTP 422 아님)\n\n"
        "프론트엔드는 `mode` 필드로 switch 분기. D2+에서 실제 LLM 로직 교체 예정."
    ),
)
async def suggest(
    body: SuggestRequest,
    request: Request,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SuggestResponseV2:
    """D1 mock — confidence 기반 4분기 응답. 실제 LLM/ranker 호출 없음.

    asr_confidence 미제공 시 1.0(high 분기)으로 처리 (D3 fallback).
    retry 분기도 HTTP 422 대신 200 + RetryModeResponse 반환 (accuracy-mode UX 원칙).
    """
    import time

    from app.agent.confidence_router import classify_branch

    t0 = time.perf_counter()
    trace_id = body.trace_id or get_trace_id(request)
    now = datetime.now(UTC)

    # D3 fallback: asr_confidence 미제공이면 1.0 (high 분기)으로 처리.
    conf: float = body.asr_confidence if body.asr_confidence is not None else 1.0
    branch = classify_branch(conf)

    base_kwargs: dict = dict(
        trace_id=trace_id,
        asr_confidence=conf,
        cold_start=False,  # D1 mock: 실제 phrase_log 집계 없음
        generated_at=now,
    )

    # ── retry 분기 ──────────────────────────────────────────────────────────
    if branch == "retry":
        response: SuggestResponseV2 = RetryModeResponse(
            mode="retry",
            branch="retry",
            message="잘 못 들었어요. 다시 말씀해 주세요.",
            **base_kwargs,
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── high 분기 (proactive) ────────────────────────────────────────────
    elif branch == "high":
        verbatim_text = (body.final_transcript or body.patient_prefix or "").strip()
        response = ProactiveModeResponse(
            mode="proactive",
            branch="high",
            verbatim=VerbatimPhrase(text=verbatim_text),
            suggestion=ProactiveSuggestion(
                phrase="미지근한 물 주세요",
                reason_display="저녁에 자주 요청하신 음료",
                modal_text="미지근한 물 주세요라고 말할까요?",
                confidence=0.86,
            ),
            **base_kwargs,
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── mid 분기 (candidates) ────────────────────────────────────────────
    elif branch == "mid":
        verbatim_text = (body.final_transcript or body.patient_prefix or "물 주세요").strip()
        response = CandidatesModeResponse(
            mode="candidates",
            branch="mid",
            candidates=[
                CandidateV2(
                    rank=1,
                    text=verbatim_text,
                    reason_label="verbatim",
                    reason_display="방금 들린 그대로",
                    score=None,
                    source="asr_verbatim",
                ),
                CandidateV2(
                    rank=2,
                    text="미지근한 물 주세요",
                    reason_label="time_context",
                    reason_display="저녁에 자주 선택하신 말",
                    score=None,
                    source="llm_inference",
                ),
                CandidateV2(
                    rank=3,
                    text="따뜻한 물 주세요",
                    reason_label="patient_frequency",
                    reason_display="자주 쓰시는 패턴",
                    score=None,
                    source="llm_inference",
                ),
            ],
            **base_kwargs,
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── low 분기 (single) ────────────────────────────────────────────────
    else:  # branch == "low"
        response = SingleModeResponse(
            mode="single",
            branch="low",
            candidate=CandidateV2(
                rank=1,
                text="물 주세요",
                reason_label="patient_frequency",
                reason_display="이 시간대에 자주 요청하신 문장",
                score=None,
                source="llm_reconstruction",
            ),
            **base_kwargs,
            generation_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── audit log ────────────────────────────────────────────────────────
    # D1 mock이라도 운영 로그 패턴 유지: event="candidates_generated"
    await audit_append(
        session,
        trace_id=trace_id,
        patient_id=me.patient_id,
        event="candidates_generated",
        payload={
            "patient_id": str(me.patient_id),
            "prefix": body.patient_prefix,
            "mode": branch,   # proactive/candidates/single/retry
            "branch": branch,
            "asr_confidence": conf,
            "cold_start": False,
            "mock": True,     # D1 식별자 — D2+에서 제거
            "generation_latency_ms": response.generation_latency_ms,  # type: ignore[union-attr]
        },
    )
    await session.commit()

    return response


# ────────── POST /phrase/selection — 환자 선택·거부 ──────────


@router.post(
    "/selection",
    response_model=SelectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="환자 선택/거부 로그 (M7·M11, 피드백 루프 ★)",
)
async def selection(
    body: SelectionRequest,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SelectionResponse:
    """환자 선택·거부 로그 저장.

    B2.2 매핑 (중요):
        응답 ``CandidateSource`` enum ≠ DB ``phrase_log.source`` enum.
        * API/후보 생성 소스: ``asr_verbatim`` / ``llm_corrected`` /
          ``patient_log`` / ``catalog`` / ``llm_gen`` / voice / preset / ...
        * DB  선택 결과 소스: ``voice`` / ``preset`` / ``fallback`` /
          ``voice_bypass`` / ``rejected``
        candidates_shown JSONB 안의 ``source`` 를 ``_map_candidate_source_to_db``
        로 정규화해 저장 — 분석 쿼리 join 간소화. ``body.source`` (voice|fallback)
        는 phrase_log.source 결정 입력 그대로 사용.
    """
    # candidate_shown에 selected_phrase가 있는지 (voice 모드) 검증
    if body.action == "selected" and body.source == "voice":
        candidate_texts = {c.text for c in body.candidates_shown}
        if body.selected_phrase not in candidate_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "VALIDATION_ERROR",
                    "message": "selected_phrase가 candidates_shown에 없습니다",
                },
            )

    embedding: list[float] | None = None
    if body.action == "selected" and body.selected_phrase:
        try:
            embedding = await embed_text(body.selected_phrase)
        except Exception:
            log.warning("embed_text failed for selection; falling back to embedding=None", exc_info=True)

    # B2.2 — candidates_shown JSONB 저장 전 각 snapshot 의 source 를 DB enum 으로 정규화.
    # 응답/CandidateSource (asr_verbatim, llm_corrected 등) ↔ DB enum 불일치 흡수.
    normalized_candidates: list[dict] = []
    for cand in body.candidates_shown:
        c_dict = cand.model_dump()
        c_dict["source"] = _map_candidate_source_to_db(c_dict.get("source"))
        normalized_candidates.append(c_dict)

    row = PhraseLog(
        patient_id=me.patient_id,
        trace_id=body.trace_id,
        partial_transcript=body.prefix,
        final_transcript=body.final_transcript,
        source=body.source if body.action == "selected" else "rejected",
        candidates_shown=normalized_candidates,
        selected_phrase=body.selected_phrase,
        selected_rank=body.selected_rank,
        reject_reason=body.reject_reason,
        selection_latency_ms=body.latency_ms,
        prev_phrase_log_id=body.prev_log_id,
        embedding=embedding,
    )
    session.add(row)
    await audit_append(
        session,
        trace_id=body.trace_id,
        patient_id=me.patient_id,
        event="phrase_selected" if body.action == "selected" else "phrase_rejected",
        payload={
            "action": body.action,
            "selected_phrase": body.selected_phrase,
            "selected_rank": body.selected_rank,
            "latency_ms": body.latency_ms,
        },
    )
    await session.commit()
    await session.refresh(row)

    return SelectionResponse(
        log_id=row.id,
        final_text=body.selected_phrase,  # action=rejected면 null
        created_at=row.created_at,
    )


# ────────── POST /phrase/preset — 빠른 문장 ──────────


@router.post(
    "/preset",
    response_model=PresetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="빠른 문장 10개 중 1개 즉시 선택 (M9)",
)
async def preset(
    body: PresetRequest,
    request: Request,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PresetResponse:
    phrase = PRESET_PHRASES[body.preset_index - 1]
    trace_id = get_trace_id(request)

    preset_embedding: list[float] | None = None
    try:
        preset_embedding = await embed_text(phrase)
    except Exception:
        log.warning("embed_text failed for preset; falling back to embedding=None", exc_info=True)

    row = PhraseLog(
        patient_id=me.patient_id,
        trace_id=trace_id,
        source="preset",
        selected_phrase=phrase,
        candidates_shown=None,
        selected_rank=None,
        embedding=preset_embedding,
    )
    session.add(row)
    await audit_append(
        session,
        trace_id=trace_id,
        patient_id=me.patient_id,
        event="preset_selected",
        payload={"preset_index": body.preset_index, "phrase": phrase},
    )
    await session.commit()
    await session.refresh(row)

    return PresetResponse(
        log_id=row.id,
        final_text=phrase,
        preset_index=body.preset_index,
        created_at=row.created_at,
    )


# ────────── POST /phrase/bypass — 실시간 모드 (S16) ──────────
#
# **M3 (시니어 결정 §4): phrase_log INSERT는 이 HTTP 라우트에서만 발생.**
#
# 책임 분리:
# - WebSocket 라우터 (`agent/streaming/router.py`) 는 phrase_log 저장 금지.
#   WS 는 ASR interim/final 중계만 수행하고 종료.
# - 환자앱(Flutter)이 `transcript{final:true}` 수신 후 `flutter_tts.speak()` 호출
#   → `tts.setCompletionHandler` 에서 **이 라우트를 호출**해 phrase_log 저장.
# - 즉 S16 실시간 대화 모드의 phrase_log 저장은 이 HTTP 라우트가 **유일한 책임자**.
#
# 사유 (06-senior-review §4 M3):
# 1. 학습 신호 혼선 방지. 실시간 모드는 ranking engine 학습 가중치 낮음(0.3).
#    WS 라우터가 저장하면 ranking agent 와의 결합이 발생.
# 2. TTS 재생 성공 시점에서야 "환자가 실제로 발화 완료" 로 간주되므로,
#    `setCompletionHandler` 시점에 저장하는 것이 데이터 품질상 정확.
# 3. phrase_log 저장 실패해도 TTS 재생에 영향 없음 (best-effort).
#
# ranking agent 측 조회 가드 (M3):
# - `agent/ranking/*` 가 phrase_log 조회 시 `WHERE source != 'voice_bypass'` SQL
#   가드 필수. 실시간 모드 bypass 로그가 ranking engine 을 오염시키지 않도록 격리.
# - WebSocket 라우터 (`agent/streaming/*`) 와 ranking agent (`agent/ranking/*`) 간
#   상호 import 금지 (ruff tidy-imports 린터로 강제 예정).


@router.post(
    "/bypass",
    response_model=BypassResponse,
    summary="실시간 대화 모드 — ASR final 즉시 로그 (S16)",
    description=(
        "**M3 (시니어 결정 §4): 실시간 대화 모드 phrase_log INSERT 의 유일한 책임자.**\n\n"
        "- WebSocket 라우터(`/ws/phrase/stream`)는 phrase_log 저장 금지.\n"
        "- 환자앱이 WS 로 `transcript{final:true}` 수신 → `flutter_tts.speak()` →\n"
        "  `tts.setCompletionHandler` 에서 이 라우트 호출해 저장.\n"
        "- `source='voice_bypass'` 로 기록. ranking agent 는 `WHERE source != 'voice_bypass'`\n"
        "  가드로 조회 (ranking engine 오염 방지)."
    ),
    responses={
        422: {"description": "ASR confidence < 0.4 → S13 재질의 강제"},
    },
)
async def bypass(
    body: BypassRequest,
    request: Request,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> BypassResponse:
    """실시간 모드 전용 phrase_log 저장 라우트.

    M3 (시니어 결정 §4) 핵심: WS 라우터는 저장 안 함. 이 HTTP 라우트가 유일.
    """
    # Confidence 재검증 — 저신뢰면 강제 재질의
    if body.asr_confidence < 0.4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "ASR_LOW_CONFIDENCE",
                "message": "잘 못 들었어요. 다시 말씀해 주세요.",
                "details": [{"field": "asr_confidence", "issue": "below_threshold", "value": body.asr_confidence}],
            },
        )

    trace_id = body.trace_id or get_trace_id(request)

    bypass_embedding: list[float] | None = None
    try:
        bypass_embedding = await embed_text(body.final_transcript)
    except Exception:
        log.warning("embed_text failed for bypass; falling back to embedding=None", exc_info=True)

    row = PhraseLog(
        patient_id=me.patient_id,
        trace_id=trace_id,
        final_transcript=body.final_transcript,
        source="voice_bypass",
        selected_phrase=body.final_transcript,
        selected_rank=None,
        candidates_shown=None,
        selection_latency_ms=body.latency_ms,
        embedding=bypass_embedding,
    )
    session.add(row)
    await audit_append(
        session,
        trace_id=trace_id,
        patient_id=me.patient_id,
        event="bypass_logged",
        payload={
            "final_transcript": body.final_transcript,
            "asr_confidence": body.asr_confidence,
        },
    )
    await session.commit()
    await session.refresh(row)

    return BypassResponse(
        log_id=row.id,
        final_text=body.final_transcript,
        created_at=row.created_at,
    )


# ────────── GET /phrase/history — 히스토리 ──────────


def _decode_history_cursor(cursor: str) -> tuple[datetime, int]:
    """cursor = base64url("<iso8601>|<log_id>")."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii") + b"==").decode("utf-8")
        ts_str, id_str = raw.split("|", 1)
        return (datetime.fromisoformat(ts_str), int(id_str))
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "VALIDATION_ERROR", "message": "invalid cursor"},
        ) from exc


def _encode_history_cursor(created_at: datetime, log_id: int) -> str:
    raw = f"{created_at.isoformat()}|{log_id}".encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _parse_since(since: str) -> datetime:
    now = datetime.now(UTC)
    if since == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if since == "7d":
        return now - timedelta(days=7)
    try:
        parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "VALIDATION_ERROR",
                "message": "since must be 'today' | '7d' | ISO 8601",
            },
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@router.get(
    "/history",
    response_model=PhraseHistoryResponse,
    summary="발화 히스토리 (S12)",
)
async def history(
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(20, ge=1, le=100),
    since: str | None = Query(
        None, description="today | 7d | ISO 8601. created_at >= since 필터."
    ),
    cursor: str | None = Query(
        None, description="이전 응답의 next_cursor. 과거 방향으로 페이지네이션."
    ),
) -> PhraseHistoryResponse:
    conditions = [
        PhraseLog.patient_id == me.patient_id,
        PhraseLog.deleted_at.is_(None),
    ]

    if since is not None:
        conditions.append(PhraseLog.created_at >= _parse_since(since))

    if cursor is not None:
        cur_ts, cur_id = _decode_history_cursor(cursor)
        # (created_at, id) 복합 커서 — 동일 ts 중복 안전.
        conditions.append(
            (PhraseLog.created_at < cur_ts)
            | and_(PhraseLog.created_at == cur_ts, PhraseLog.id < cur_id)
        )

    stmt = (
        select(PhraseLog)
        .where(*conditions)
        .order_by(desc(PhraseLog.created_at), desc(PhraseLog.id))
        .limit(limit + 1)
    )
    rows = list((await session.execute(stmt)).scalars().all())

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    items = [
        PhraseHistoryItem(
            log_id=r.id,
            final_text=r.selected_phrase or r.final_transcript or "",
            source=r.source,  # type: ignore[arg-type]
            selected_rank=r.selected_rank,
            created_at=r.created_at,
        )
        for r in page_rows
        if r.selected_phrase or r.final_transcript
    ]

    next_cursor = (
        _encode_history_cursor(page_rows[-1].created_at, page_rows[-1].id)
        if has_more and page_rows
        else None
    )

    return PhraseHistoryResponse(
        items=items,
        next_cursor=next_cursor,
        total_estimate=len(items) + (1 if has_more else 0),
    )
