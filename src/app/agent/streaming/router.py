"""WebSocket 라우터 — ``/ws/phrase/stream``.

JWT는 **Sec-WebSocket-Protocol 서브프로토콜**로 받는다. 클라이언트 예시::

    new WebSocket(url, ["marieum.jwt.v1", "<access_token>"])

서버는 토큰 문자열을 꺼내 검증하고 ``marieum.jwt.v1``을 echo 해 accept 한다.
토큰이 액세스 로그·CDN 로그·브라우저 히스토리에 남지 않게 하기 위함.

**레거시**: ``?token=<jwt>`` 쿼리 파라미터도 당분간 허용(프론트 마이그레이션 완료 후 제거).
쿼리 사용 시 warning 로그가 찍힌다.

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

import asyncio
import logging
import time
from collections import defaultdict

import jwt as pyjwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.agent.streaming.session import StreamingSession
from app.config import settings
from app.security import decode_token

log = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])

# 환자당 동시 WS 접속 cap. 한 환자가 N개 소켓을 열어 reconnect/watchdog/ML 커넥션을
# N배 쓰는 것을 막는다. 초과 시 4003으로 거절.
MAX_CONN_PER_PATIENT = 1
_conn_counts: dict[str, int] = defaultdict(int)
_conn_lock = asyncio.Lock()


async def _try_acquire_slot(patient_id: str) -> bool:
    async with _conn_lock:
        if _conn_counts[patient_id] >= MAX_CONN_PER_PATIENT:
            return False
        _conn_counts[patient_id] += 1
        return True


async def _release_slot(patient_id: str) -> None:
    async with _conn_lock:
        if _conn_counts[patient_id] <= 1:
            _conn_counts.pop(patient_id, None)
        else:
            _conn_counts[patient_id] -= 1


# 서브프로토콜 식별자. 클라는 ["marieum.jwt.v1", "<token>"] 2개를 보낸다.
WS_SUBPROTOCOL = "marieum.jwt.v1"


def _extract_token_from_subprotocols(ws: WebSocket) -> str | None:
    """Sec-WebSocket-Protocol 헤더에서 토큰 추출.

    규약: 첫 번째 프로토콜은 식별자(WS_SUBPROTOCOL), 두 번째가 실제 JWT.
    둘 중 하나라도 없으면 None.
    """
    raw = ws.headers.get("sec-websocket-protocol")
    if not raw:
        return None
    protocols = [p.strip() for p in raw.split(",") if p.strip()]
    if WS_SUBPROTOCOL not in protocols:
        return None
    # 식별자를 제외한 나머지 중 첫 번째를 토큰으로 본다.
    for p in protocols:
        if p != WS_SUBPROTOCOL:
            return p
    return None

# ─── 엔드포인트 ───────────────────────────────────────────────────────────────


@router.websocket("/ws/phrase/stream")
async def ws_phrase_stream(
    ws: WebSocket,
    token: str | None = Query(None, description="(레거시) JWT access token. 서브프로토콜 사용 권장."),
) -> None:
    """실시간 모드 WebSocket 엔드포인트.

    Auth:
        우선순위 1 — ``Sec-WebSocket-Protocol: marieum.jwt.v1, <jwt>``
        우선순위 2 (레거시) — ``?token=<jwt>`` 쿼리

    Close codes:
        1000 — 정상 종료
        1001 — 클라이언트 cancel
        4001 — JWT 인증 실패
        4003 — 동시 접속 한도 초과
    """
    # 타이밍 측정 — 각 단계 지연 분해.
    t0 = time.perf_counter()

    # 1. 토큰 추출: 서브프로토콜 우선, 쿼리는 레거시 폴백
    subproto_token = _extract_token_from_subprotocols(ws)
    accept_subprotocol: str | None = None
    if subproto_token is not None:
        token_value = subproto_token
        accept_subprotocol = WS_SUBPROTOCOL
    elif token is not None:
        token_value = token
        log.warning(
            "ws_phrase_stream.legacy_query_token",
            extra={"remote": ws.client},
        )
    else:
        await ws.accept()
        log.warning("ws_phrase_stream.no_token", extra={"remote": ws.client})
        await ws.close(code=4001, reason="JWT 누락")
        return

    # 2. JWT 검증
    try:
        principal = decode_token(token_value)
    except (pyjwt.PyJWTError, Exception):
        # decode_token 내부에서 HTTPException 발생 가능 → WS에서는 close로 처리
        await ws.accept(subprotocol=accept_subprotocol)
        log.warning(
            "ws_phrase_stream.jwt_invalid",
            extra={"remote": ws.client},
        )
        await ws.close(code=4001, reason="JWT 인증 실패")
        return
    t_jwt = time.perf_counter()

    # 3. WebSocket accept (서브프로토콜 echo)
    await ws.accept(subprotocol=accept_subprotocol)
    t_accept = time.perf_counter()
    log.info(
        "ws_phrase_stream.connected",
        extra={
            "patient_id": str(principal.patient_id),
            "remote": ws.client,
            "jwt_ms": int((t_jwt - t0) * 1000),
            "accept_ms": int((t_accept - t_jwt) * 1000),
        },
    )

    # 2b. 환자당 동시 접속 cap. 초과 시 4003으로 거절.
    patient_id_str = str(principal.patient_id)
    if not await _try_acquire_slot(patient_id_str):
        log.warning(
            "ws_phrase_stream.conn_limit_exceeded",
            extra={"patient_id": patient_id_str, "limit": MAX_CONN_PER_PATIENT},
        )
        await ws.close(code=4003, reason="동시 접속 한도 초과")
        return
    t_cap = time.perf_counter()

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
        await session.push_error("INTERNAL", "실시간 연결 초기화에 실패했습니다")
    t_ready = time.perf_counter()
    log.info(
        "ws_phrase_stream.ready trace_id=%s total=%dms jwt=%dms accept=%dms cap=%dms session_start=%dms",
        session.trace_id,
        int((t_ready - t0) * 1000),
        int((t_jwt - t0) * 1000),
        int((t_accept - t_jwt) * 1000),
        int((t_cap - t_accept) * 1000),
        int((t_ready - t_cap) * 1000),
    )

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
                await session.push_error("INTERNAL", "입력 처리 중 오류가 발생했습니다")

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
        await session.close(code=1000)
        await _release_slot(patient_id_str)
        log.info(
            "ws_phrase_stream.session_ended",
            extra={
                "trace_id": session.trace_id,
                "patient_id": str(principal.patient_id),
            },
        )
