"""WebSocket 라우터 — ``/ws/phrase/stream``.

JWT는 쿼리 파라미터 ``?token=<jwt>`` 로 받는다 (§S2: Cloudflare/브라우저 호환).
trace_id는 accept 직후 서버가 UUIDv4로 발급한다 (클라이언트 입력 무시, §S1).

앱 → 서버 오디오: PCM16 little-endian / 16kHz / mono, 100ms당 3,200바이트.
WAV 헤더 없이 바이너리로 전송하고 마지막 잔여 청크 뒤 JSON end를 보낸다.
수신 바이너리는 크기 변환 없이 ML proxy에 전달한다.

흐름:
  1. JWT 쿼리 파라미터 검증 → PatientPrincipal
  2. WebSocket accept
  3. StreamingSession 생성 · 시작
  4. 메시지 루프: 바이너리 → ML forward, 텍스트 → 제어 메시지
  5. 세션 종료 정리 (idempotent)

격리 조건:
  - agent/ranking import 금지
  - agent/streaming/* 자기 패키지 import OK
"""

from __future__ import annotations

import logging

import jwt as pyjwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.agent.streaming.session import StreamingSession
from app.config import settings
from app.security import decode_token

log = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])

# ─── 엔드포인트 ───────────────────────────────────────────────────────────────


@router.websocket("/ws/phrase/stream")
async def ws_phrase_stream(
    ws: WebSocket,
    token: str = Query(..., description="JWT access token (쿼리 파라미터)"),
) -> None:
    """실시간 모드 WebSocket 엔드포인트.

    Args:
        ws:    FastAPI WebSocket 연결 객체.
        token: JWT access token (쿼리 파라미터 ``?token=<jwt>``).

    Close codes:
        1000 — 정상 종료
        1001 — 클라이언트 cancel
        4001 — JWT 인증 실패
    """
    # 1. JWT 검증 (accept 전에 수행 — 4001 close를 보내기 위해 먼저 accept 필요)
    try:
        principal = decode_token(token)
    except (pyjwt.PyJWTError, Exception):
        # decode_token 내부에서 HTTPException 발생 가능 → WS에서는 close로 처리
        await ws.accept()
        log.warning(
            "ws_phrase_stream.jwt_invalid",
            extra={"remote": ws.client},
        )
        await ws.close(code=4001, reason="JWT 인증 실패")
        return

    # 2. WebSocket accept
    await ws.accept()
    log.info(
        "ws_phrase_stream.connected",
        extra={
            "patient_id": str(principal.patient_id),
            "remote": ws.client,
        },
    )

    # 3. 운영 ML 주소. 미설정/장애는 오류로 알리고 연결 종료는 프론트가 결정한다.
    ml_ws_url = settings.ml_asr_ws_url

    # 4. 세션 생성
    session = StreamingSession(
        app_ws=ws,
        patient_principal=principal,
        ml_ws_url=ml_ws_url,
    )

    # 5. 세션 시작 (ML 연결 · pipeline 초기화 · 백그라운드 태스크)
    try:
        await session.start()
    except Exception:
        log.exception(
            "ws_phrase_stream.session_start_failed",
            extra={"patient_id": str(principal.patient_id)},
        )
        await session._push_error("INTERNAL", "실시간 연결 초기화에 실패했습니다")

    # 6. 메시지 루프
    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info(
                    "ws_phrase_stream.client_disconnect",
                    extra={
                        "trace_id": session.trace_id,
                        "patient_id": str(principal.patient_id),
                    },
                )
                break

            try:
                if "bytes" in msg and msg["bytes"] is not None:
                    if not await session.handle_client_message(msg["bytes"]):
                        break
                elif (
                    "text" in msg
                    and msg["text"] is not None
                    and not await session.handle_client_message(msg["text"])
                ):
                    break
            except WebSocketDisconnect:
                raise
            except Exception:
                log.exception(
                    "ws_phrase_stream.message_processing_error",
                    extra={"trace_id": session.trace_id},
                )
                await session._push_error("INTERNAL", "입력 처리 중 오류가 발생했습니다")

    except WebSocketDisconnect:
        log.info(
            "ws_phrase_stream.websocket_disconnect",
            extra={
                "trace_id": session.trace_id,
                "patient_id": str(principal.patient_id),
            },
        )
    except (OSError, RuntimeError):
        log.exception(
            "ws_phrase_stream.transport_receive_failed",
            extra={
                "trace_id": session.trace_id,
                "patient_id": str(principal.patient_id),
            },
        )
    finally:
        # 7. 세션 정리 (idempotent — 이미 닫혔으면 no-op)
        await session._close(code=1000)
        log.info(
            "ws_phrase_stream.session_ended",
            extra={
                "trace_id": session.trace_id,
                "patient_id": str(principal.patient_id),
            },
        )
