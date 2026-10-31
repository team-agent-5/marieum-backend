"""Phrase 라우터 (v4) — HTTP 발화 흐름.

- transcribe: 멀티파트 오디오 → ASR 전사 (M4·M5, WS 대체)
- suggest: 추천 후보 (accuracy-mode orchestrator — D2)
- selection: 환자 선택·거부 로그
- preset: 빠른 문장 10개 즉시 로그
- bypass: 실시간 모드 ASR final 즉시 로그
- history: 발화 히스토리
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid

import httpx
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

from app.agent.orchestrator import suggest as orchestrator_suggest
from app.audit import audit_append
from app.config import settings
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
    SuggestResponseV2,
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

    # ML /asr forward — spec: POST {ml_asr_url}/asr, body=raw bytes, header=Content-Type+X-Trace-Id.
    ml_model_version = ""
    try:
        async with httpx.AsyncClient(timeout=settings.ml_asr_timeout_sec) as client:
            r = await client.post(
                f"{settings.ml_asr_url}/asr",
                content=data,
                headers={"Content-Type": ctype, "X-Trace-Id": trace_id},
            )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        log.warning("ML /asr unreachable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_code": "ML_UNAVAILABLE", "message": "ASR 서버에 연결할 수 없습니다."},
        ) from exc
    except httpx.ReadTimeout as exc:
        log.warning("ML /asr read timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error_code": "ASR_TIMEOUT", "message": "ASR 처리 시간이 초과됐습니다."},
        ) from exc

    if r.status_code != status.HTTP_200_OK:
        try:
            err = r.json()
            ml_code = str(err.get("code", "INTERNAL"))
            ml_msg = str(err.get("message", "ASR 오류"))
        except Exception:
            ml_code = "INTERNAL"
            ml_msg = r.text[:200] or "ASR 오류"
        log.warning("ML /asr error %d code=%s", r.status_code, ml_code)
        # 상태 매핑: 413 그대로, 400/415는 INVALID_FORMAT, 503은 MODEL_UNAVAILABLE, 그 외 502.
        if r.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"error_code": "AUDIO_TOO_LONG", "message": ml_msg},
            )
        if r.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={"error_code": "INVALID_FORMAT", "message": ml_msg},
            )
        if r.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error_code": "MODEL_UNAVAILABLE", "message": ml_msg},
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_code": ml_code, "message": ml_msg},
        )

    try:
        ml_resp = r.json()
    except Exception as exc:
        log.warning("ML /asr malformed response")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_code": "ASR_MALFORMED", "message": "ASR 응답 파싱 실패."},
        ) from exc

    final_text = str(ml_resp.get("text", ""))
    try:
        confidence = float(ml_resp.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    ml_duration = ml_resp.get("duration_ms")
    effective_duration = (
        int(ml_duration) if ml_duration is not None
        else (duration_ms if duration_ms is not None else 0)
    )
    ml_model_version = str(ml_resp.get("model_version", ""))

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
            "ml_model": ml_model_version,
        },
    )
    await session.commit()

    return TranscribeResponse(
        trace_id=trace_id,
        final_transcript=final_text,
        asr_confidence=confidence,
        duration_ms=effective_duration,
        heard_text=final_text if (final_text and confidence < _ASR_LOW_CONFIDENCE) else None,
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
    summary="후보 추천 (v2, mode별 shape) — accuracy-mode orchestrator (D2)",
    description=(
        "**D2 accuracy-mode**: RAG + LLM + 3-level fallback chain.\n\n"
        "- `mode=proactive` (conf ≥ 0.98): verbatim 즉시 TTS + 선택적 LLM 제안 모달\n"
        "- `mode=candidates` (0.90 ≤ conf < 0.98): 3후보 선택 UI\n"
        "- `mode=single` (0.40 ≤ conf < 0.90): 단일 후보 큰 글자 UI\n"
        "- `mode=retry` (conf < 0.40): 재질의 안내 (200 OK, HTTP 422 아님)\n\n"
        "프론트엔드는 `mode` 필드로 switch 분기."
    ),
)
async def suggest(
    body: SuggestRequest,
    request: Request,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SuggestResponseV2:
    """D2 orchestrator — RAG + LLM + fallback 체인.

    asr_confidence 미제공 시 1.0(high 분기)으로 처리 (D3 fallback).
    모든 분기에서 HTTP 200 보장 (retry 포함).
    """
    trace_id = body.trace_id or get_trace_id(request)

    response = await orchestrator_suggest(
        patient_id=me.patient_id,
        ml_text=body.final_transcript or body.patient_prefix or "",
        asr_confidence=body.asr_confidence,
        trace_id=trace_id,
        session=session,
        accuracy_mode=body.accuracy_mode if hasattr(body, "accuracy_mode") else True,
    )

    # audit_log 기록
    await audit_append(
        session,
        trace_id=trace_id,
        patient_id=me.patient_id,
        event="candidates_generated",
        payload={
            "mode": response.mode,
            "branch": response.branch,
            "asr_confidence": body.asr_confidence,
            "cold_start": response.cold_start,
            "fallback_reason": response.accuracy_fallback_reason,
            "latency_ms": response.generation_latency_ms,
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
