"""Listen HTTP + WebSocket 라우터.

4개 HTTP 엔드포인트 + 1개 WS 엔드포인트.

- `POST   /listen/sessions`          환자 JWT. 세션 생성/교체.
- `GET    /listen/sessions/current`  환자 JWT. 현재 세션 조회.
- `DELETE /listen/sessions/current`  환자 JWT. 세션 종료.
- `POST   /listen/join`              NO auth. IP당 rate limit.
- `WS     /ws/listen`                `Sec-WebSocket-Protocol: marieum.listen.v1, <listener_token>`

격리:
- `app.agent.*` import 금지.
- `app.listen.store`를 통해서만 세션 상태 조작.

Audit 이벤트 (fire-and-forget, 본문 없음):
- `listen_session_created`
- `listen_joined`
- `listen_left`
- `listen_played`
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from app.audit import audit_append
from app.db import AsyncSessionLocal
from app.listen import store
from app.schemas import (
    CreateListenSessionResponse,
    CurrentListenSessionResponse,
    JoinRequest,
    JoinResponse,
    PatientPrincipal,
    WSListenPingOut,
    WSListenPlayedIn,
)
from app.security import require_patient

log = logging.getLogger(__name__)

router = APIRouter(tags=["listen"])

# WS 서브프로토콜. 클라는 ["marieum.listen.v1", "<listener_token>"] 2개를 보낸다.
WS_LISTEN_SUBPROTOCOL = "marieum.listen.v1"

# Cloudflare 100s idle 방지. streaming과 동일.
WS_LISTEN_PING_INTERVAL_S = 25.0

# WebSocket close codes (listen 전용).
WS_CLOSE_NORMAL = 1000
WS_CLOSE_AUTH = 4001
WS_CLOSE_SESSION_ENDED = 4004
WS_CLOSE_ALREADY_JOINED = 4009


# ────────── 헬퍼 ──────────


def _client_ip(request: Request) -> str:
    """역방향 프록시(X-Forwarded-For) 우선, 없으면 request.client.host."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client is not None:
        return request.client.host
    return "unknown"


async def _audit(
    *,
    event: str,
    payload: dict[str, Any],
    trace_id: str,
    patient_id=None,
) -> None:
    """Audit 1건 기록. 실패해도 요청 흐름을 막지 않는다."""
    try:
        async with AsyncSessionLocal() as db:
            await audit_append(
                db,
                trace_id=trace_id,
                event=event,
                payload=payload,
                patient_id=patient_id,
            )
            await db.commit()
    except Exception:
        log.exception("listen.audit_failed", extra={"event": event, "trace_id": trace_id})


_audit_tasks: set[asyncio.Task] = set()


def _fire_audit(**kwargs: Any) -> None:
    """현재 루프에 audit task를 띄우고 결과는 신경쓰지 않는다."""
    with contextlib.suppress(RuntimeError):
        loop = asyncio.get_running_loop()
        task = loop.create_task(_audit(**kwargs))
        _audit_tasks.add(task)
        task.add_done_callback(_audit_tasks.discard)


def _trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "listen")


# ────────── HTTP 엔드포인트 ──────────


@router.post(
    "/listen/sessions",
    response_model=CreateListenSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_listen_session(
    request: Request,
    principal: Annotated[PatientPrincipal, Depends(require_patient)],
) -> CreateListenSessionResponse:
    """환자용 Listen 세션 생성/교체. 같은 환자의 기존 세션은 store가 revoke 처리."""
    session = await _maybe_await(store.create_session(principal.patient_id))
    response = CreateListenSessionResponse(
        listen_session_id=session.listen_session_id,
        join_code=session.join_code,
        join_expires_at=session.join_expires_at,
        expires_at=session.expires_at,
    )
    _fire_audit(
        event="listen_session_created",
        payload={
            "listen_session_id": session.listen_session_id,
            "expires_at": _iso(session.expires_at),
            "join_expires_at": _iso(session.join_expires_at),
        },
        trace_id=_trace_id(request),
        patient_id=principal.patient_id,
    )
    return response


@router.get(
    "/listen/sessions/current",
    response_model=CurrentListenSessionResponse,
)
async def get_current_listen_session(
    principal: Annotated[PatientPrincipal, Depends(require_patient)],
) -> CurrentListenSessionResponse:
    """환자의 현재 세션. 없으면 404 LISTEN_NO_SESSION."""
    session = await _maybe_await(store.get_by_patient(principal.patient_id))
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "LISTEN_NO_SESSION",
                "message": "활성 세션이 없습니다.",
            },
        )
    state = _derive_state(session)
    join_code = session.join_code if state == "waiting" and not session.join_code_consumed else None
    return CurrentListenSessionResponse(
        listen_session_id=session.listen_session_id,
        state=state,
        join_code=join_code,
        join_expires_at=session.join_expires_at if join_code else None,
        expires_at=session.expires_at,
    )


@router.delete(
    "/listen/sessions/current",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_current_listen_session(
    request: Request,
    principal: Annotated[PatientPrincipal, Depends(require_patient)],
) -> Response:
    """현재 세션 종료 (patient_left)."""
    session = await _maybe_await(store.get_by_patient(principal.patient_id))
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "LISTEN_NO_SESSION",
                "message": "활성 세션이 없습니다.",
            },
        )
    await _maybe_await(store.end_session(session, "patient_left"))
    _fire_audit(
        event="listen_left",
        payload={
            "listen_session_id": session.listen_session_id,
            "reason": "session_ended",
        },
        trace_id=_trace_id(request),
        patient_id=principal.patient_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/listen/join", response_model=JoinResponse)
async def join_listen_session(request: Request, body: JoinRequest) -> JoinResponse:
    """상대방 join. JWT 불필요, IP 분당 10회 rate limit."""
    ip = _client_ip(request)
    try:
        session, listener_token = await store.consume_join_code(body.join_code, ip)
    except store.RateLimitedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error_code": "RATE_LIMITED", "message": "잠시 후 다시 시도하세요."},
        ) from exc
    except store.ListenerAlreadyJoinedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "LISTENER_ALREADY_JOINED",
                "message": "이미 참여한 세션입니다.",
            },
        ) from exc
    except store.JoinCodeInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "LISTEN_CODE_INVALID",
                "message": "유효하지 않은 코드입니다.",
            },
        ) from exc

    _fire_audit(
        event="listen_joined",
        payload={
            "listen_session_id": session.listen_session_id,
            "listener_ip": ip,
        },
        trace_id=_trace_id(request),
        patient_id=session.patient_id,
    )
    return JoinResponse(
        listener_token=listener_token,
        expires_at=session.expires_at,
    )


# ────────── WebSocket ──────────


def _extract_listener_token(ws: WebSocket) -> str | None:
    """Sec-WebSocket-Protocol: 'marieum.listen.v1, <token>'에서 토큰 추출."""
    raw = ws.headers.get("sec-websocket-protocol")
    if not raw:
        return None
    protocols = [p.strip() for p in raw.split(",") if p.strip()]
    if WS_LISTEN_SUBPROTOCOL not in protocols:
        return None
    for p in protocols:
        if p != WS_LISTEN_SUBPROTOCOL:
            return p
    return None


@router.websocket("/ws/listen")
async def ws_listen(ws: WebSocket) -> None:
    """상대방 WebSocket.

    Auth:
        Sec-WebSocket-Protocol: marieum.listen.v1, <listener_token>

    Close codes:
        1000 — 정상
        4001 — 토큰 누락/무효
        4004 — 세션 종료/없음
        4009 — 이미 다른 상대방이 연결됨
    """
    token = _extract_listener_token(ws)
    if not token:
        await ws.accept()
        await ws.close(code=WS_CLOSE_AUTH, reason="listener_token 누락")
        return

    # attach_listener 전에 사전 조회 — 토큰 자체가 유효하지 않으면 바로 4001.
    session_preview = await _maybe_await(store.get_by_listener_token(token))
    if session_preview is None:
        await ws.accept(subprotocol=WS_LISTEN_SUBPROTOCOL)
        await ws.close(code=WS_CLOSE_AUTH, reason="유효하지 않은 토큰")
        return

    await ws.accept(subprotocol=WS_LISTEN_SUBPROTOCOL)

    try:
        session = await store.attach_listener(token, ws)
    except store.ListenerAlreadyJoinedError:
        await ws.close(code=WS_CLOSE_ALREADY_JOINED, reason="이미 다른 상대방이 연결됨")
        return
    except Exception as exc:
        # SessionEndedError 또는 유사한 "세션 종료" 예외
        name = type(exc).__name__
        if "SessionEnded" in name or "NotFound" in name:
            await ws.close(code=WS_CLOSE_SESSION_ENDED, reason="세션 종료됨")
            return
        log.exception("ws_listen.attach_failed")
        await ws.close(code=WS_CLOSE_SESSION_ENDED, reason="세션 초기화 실패")
        return

    log.info(
        "ws_listen.connected",
        extra={
            "listen_session_id": session.listen_session_id,
            "remote": ws.client,
        },
    )

    ping_task: asyncio.Task | None = asyncio.create_task(_ping_loop(ws))
    now_listener_left = False

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if text is None:
                # binary/unknown — 무시
                continue
            try:
                import json

                data = json.loads(text)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            msg_type = data.get("type")
            if msg_type == "played":
                try:
                    played = WSListenPlayedIn.model_validate(data)
                except Exception:
                    continue
                await _handle_played(session, played.utterance_id)
            # Unknown types silently ignored.

    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError):
        log.exception("ws_listen.transport_error")
    finally:
        if ping_task is not None:
            ping_task.cancel()
            with contextlib.suppress(BaseException):
                await ping_task
        try:
            await _maybe_await(store.detach_listener(session, now_listener_left=now_listener_left))
        except Exception:
            log.exception("ws_listen.detach_failed")
        _fire_audit(
            event="listen_left",
            payload={
                "listen_session_id": session.listen_session_id,
                "reason": "disconnect",
            },
            trace_id="listen_ws",
            patient_id=session.patient_id,
        )
        with contextlib.suppress(Exception):
            await ws.close(code=WS_CLOSE_NORMAL)


async def _ping_loop(ws: WebSocket) -> None:
    """25s 주기로 ping. 실패하면 조용히 종료."""
    try:
        payload = WSListenPingOut(type="ping").model_dump()
        while True:
            await asyncio.sleep(WS_LISTEN_PING_INTERVAL_S)
            try:
                await ws.send_json(payload)
            except (WebSocketDisconnect, OSError, RuntimeError):
                return
    except asyncio.CancelledError:
        return


async def _handle_played(session, utterance_id: str) -> None:
    """상대방이 '재생 시작' 알림을 보냈을 때 환자 측으로 중계 + audit."""
    push = getattr(store, "push_played_event", None)
    if push is not None:
        try:
            result = push(session, utterance_id)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            log.exception("ws_listen.push_played_failed")

    # 재생 이벤트 audit (text 없음).
    now_iso = datetime.now().astimezone().isoformat()
    _fire_audit(
        event="listen_played",
        payload={"utterance_id": utterance_id, "played_at": now_iso},
        trace_id="listen_ws",
        patient_id=session.patient_id,
    )


# ────────── 상태 파생 ──────────


def _derive_state(session) -> str:
    """store 상태에서 ListenSessionState 파생."""
    if getattr(session, "ended", False):
        return "ended"
    if session.listener_ws is not None or session.grace_until is not None:
        return "connected"
    if session.join_code_consumed:
        # 소비됐지만 아직 WS는 안 붙음 → connected 과도기로 간주
        return "connected"
    return "waiting"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.isoformat()


async def _maybe_await(value):
    """store API가 동기/비동기 혼재해 있어 유연하게 처리."""
    if asyncio.iscoroutine(value):
        return await value
    return value
