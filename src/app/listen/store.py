"""Listen 세션 메모리 store — 단일 uvicorn 프로세스 전제.

책임:
- 환자당 1개 ListenSession 유지
- join_code ↔ session, listener_token ↔ session 양방향 조회
- 상대방 WebSocket 핸들 보관 (중계용)
- utterance_id 멱등 (최근 50개 캐시)
- 30초 재접속 grace
- 만료 정리 (background task)
- join 시도 rate limit (IP당 분당 10회)

격리:
- `agent/`를 import하지 않는다.
- streaming/session.py가 이 모듈을 import하는 건 허용 (ranking이 아니므로 격리 룰 유지).

구현 Agent 참고:
- 모든 변경은 asyncio.Lock으로 보호 (단일 이벤트 루프라도 await 경계에서 재진입 가능).
- WebSocket 타입은 fastapi.WebSocket. send_json 호출은 상대방 쪽에서만.
- 만료/grace 정리는 10초 주기 background task로 처리.
"""
from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from app.schemas import WSListenSayOut, WSListenSessionEndOut

if TYPE_CHECKING:
    from fastapi import WebSocket

# ────────── 상수 ──────────

JOIN_CODE_LEN = 6
JOIN_CODE_CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 0/O, 1/I 제외 (Crockford-style).
JOIN_TTL = timedelta(minutes=10)      # join_code 유효기간
SESSION_TTL = timedelta(hours=2)      # 세션 전체 수명
RECONNECT_GRACE_S = 30                # 상대방 WS 끊긴 뒤 재접속 유예
UTTERANCE_DEDUP_WINDOW = 50           # 최근 utterance_id 캐시 크기
LISTENER_TOKEN_BYTES = 32             # secrets.token_urlsafe() 바이트 수

# join_code 충돌 체크 시 재시도 상한 (실질 충돌 확률은 32^6 / 환자 수 ≪ 1).
MAX_JOIN_CODE_RETRIES = 8

# IP당 /listen/join 분당 허용 횟수.
JOIN_RATE_LIMIT_PER_MIN = 10
RATE_LIMIT_WINDOW_S = 60

CLEANUP_INTERVAL_S = 10


# ────────── 데이터 ──────────


@dataclass
class ListenSession:
    """환자 1명당 1건."""

    listen_session_id: str                         # "ls_<ulid>"
    patient_id: UUID
    join_code: str                                 # 6자. 소비되면 다른 값으로 교체되지 않고 그대로 유지(1회용 정책).
    join_code_consumed: bool = False               # True면 신규 join 거절.
    listener_token: str | None = None              # join 성공 시 발급. 세션 1회만.
    join_expires_at: datetime = field(default_factory=lambda: datetime.now().astimezone() + JOIN_TTL)
    expires_at: datetime = field(default_factory=lambda: datetime.now().astimezone() + SESSION_TTL)
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    # 런타임 상태 (persist 안 됨)
    listener_ws: WebSocket | None = None           # 활성 상대방 소켓. None이면 미연결/grace 중.
    grace_until: datetime | None = None            # listener_ws None이면서 grace 중이면 설정됨.
    ended: bool = False
    end_reason: str | None = None                  # "patient_left" | "expired" | "revoked"

    # utterance_id 멱등 캐시. 최근 N개 (OrderedDict처럼 삽입순 유지).
    recent_utterances: dict[str, bool] = field(default_factory=dict)


# ────────── 예외 ──────────


class JoinCodeInvalidError(Exception):
    """코드가 없거나 만료/소비됨 → 404 LISTEN_CODE_INVALID."""


class ListenerAlreadyJoinedError(Exception):
    """해당 세션에 이미 다른 상대방이 연결/grace 중 → 409 LISTENER_ALREADY_JOINED."""


class RateLimitedError(Exception):
    """IP 분당 시도 한도 초과 → 429 RATE_LIMITED."""


class SessionEndedError(Exception):
    """세션이 존재하지 않거나 종료됨 → 라우터는 close(4004)로 처리."""


# ────────── 내부 상태 ──────────

_lock: asyncio.Lock = asyncio.Lock()

# 비동기 백그라운드 task(revoke 시 WS close 등) 참조 보관. GC 방지.
_background_tasks: set[asyncio.Task[None]] = set()

_sessions_by_patient: dict[UUID, ListenSession] = {}
_sessions_by_code: dict[str, ListenSession] = {}
_sessions_by_token: dict[str, ListenSession] = {}

_event_queues: dict[UUID, asyncio.Queue[str]] = {}
_played_queues: dict[UUID, asyncio.Queue[str]] = {}

# 환자 WS가 못 따라갈 때 상태/played 이벤트가 무제한 쌓이지 않도록 cap.
# 환자 1명당 상태 전환은 몇 건 수준이라 16이면 충분. 초과 시 drop-oldest.
EVENT_QUEUE_MAXSIZE = 16

# IP → sliding window deque[monotonic timestamp]
_rate_limit: dict[str, deque[float]] = {}


def _now() -> datetime:
    return datetime.now().astimezone()


def _gen_join_code() -> str:
    return "".join(secrets.choice(JOIN_CODE_CHARSET) for _ in range(JOIN_CODE_LEN))


def _gen_session_id() -> str:
    return "ls_" + secrets.token_urlsafe(10)


def _check_rate_limit(ip: str) -> None:
    """Sliding 1-min window. All attempts count (brute-force deterrent)."""
    loop = asyncio.get_event_loop()
    now_ts = loop.time()
    dq = _rate_limit.setdefault(ip, deque())
    cutoff = now_ts - RATE_LIMIT_WINDOW_S
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= JOIN_RATE_LIMIT_PER_MIN:
        dq.append(now_ts)
        raise RateLimitedError("join rate limit exceeded")
    dq.append(now_ts)


def _free_indexes(session: ListenSession) -> None:
    _sessions_by_patient.pop(session.patient_id, None)
    _sessions_by_code.pop(session.join_code, None)
    if session.listener_token:
        _sessions_by_token.pop(session.listener_token, None)


def _enqueue_drop_oldest(q: asyncio.Queue[str], item: str) -> None:
    """Bounded queue에 drop-oldest 전략으로 push."""
    while q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
    with contextlib.suppress(asyncio.QueueFull):
        q.put_nowait(item)


async def _emit_event(patient_id: UUID, state: str) -> None:
    q = _event_queues.get(patient_id)
    if q is None:
        return
    _enqueue_drop_oldest(q, state)


async def _send_end_and_close(ws: WebSocket, reason: str) -> None:
    try:
        msg = WSListenSessionEndOut(type="session_end", reason=reason)  # type: ignore[arg-type]
        await ws.send_json(msg.model_dump(mode="json"))
    except Exception:
        pass
    with contextlib.suppress(Exception):
        await ws.close(code=4004, reason=reason)


# ────────── 공개 API ──────────


def create_session(patient_id: UUID) -> ListenSession:
    """환자용 세션을 새로 만든다.

    - 같은 환자의 기존 세션이 있으면 `end_session(..., "revoked")`로 종료 후 교체.
    - 교체 시 기존 세션의 listener_ws에게 `WSListenSessionEndOut(reason="revoked")` 전송 후 close(4004).
    - 신규 join_code 발급 (충돌 시 재시도).
    """
    existing = _sessions_by_patient.get(patient_id)
    if existing is not None and not existing.ended:
        ws = existing.listener_ws
        existing.ended = True
        existing.end_reason = "revoked"
        existing.listener_ws = None
        _free_indexes(existing)
        if ws is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    bg = loop.create_task(_send_end_and_close(ws, "revoked"))
                    _background_tasks.add(bg)
                    bg.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                pass

    code = ""
    for _ in range(MAX_JOIN_CODE_RETRIES):
        candidate = _gen_join_code()
        if candidate not in _sessions_by_code:
            code = candidate
            break
    if not code:
        code = _gen_join_code()

    session = ListenSession(
        listen_session_id=_gen_session_id(),
        patient_id=patient_id,
        join_code=code,
    )
    _sessions_by_patient[patient_id] = session
    _sessions_by_code[code] = session
    return session


def get_by_patient(patient_id: UUID) -> ListenSession | None:
    s = _sessions_by_patient.get(patient_id)
    if s is None or s.ended:
        return None
    return s


def get_by_join_code(code: str) -> ListenSession | None:
    s = _sessions_by_code.get(code)
    if s is None or s.ended:
        return None
    return s


def get_by_listener_token(token: str) -> ListenSession | None:
    s = _sessions_by_token.get(token)
    if s is None or s.ended:
        return None
    return s


async def consume_join_code(code: str, ip: str) -> tuple[ListenSession, str]:
    async with _lock:
        _check_rate_limit(ip)

        session = _sessions_by_code.get(code)
        if session is None or session.ended or session.join_code_consumed:
            raise JoinCodeInvalidError("invalid or consumed join code")
        if _now() >= session.join_expires_at:
            raise JoinCodeInvalidError("join code expired")

        if session.listener_ws is not None:
            raise ListenerAlreadyJoinedError("listener already attached")
        if session.grace_until is not None and _now() < session.grace_until:
            raise ListenerAlreadyJoinedError("listener in grace reconnect window")

        token = secrets.token_urlsafe(LISTENER_TOKEN_BYTES)
        session.join_code_consumed = True
        session.listener_token = token
        _sessions_by_token[token] = session
        return session, token


async def attach_listener(token: str, ws: WebSocket) -> ListenSession:
    async with _lock:
        session = _sessions_by_token.get(token)
        if session is None or session.ended:
            raise SessionEndedError("session not found or ended")

        if session.listener_ws is not None:
            raise ListenerAlreadyJoinedError("another listener socket attached")

        # Grace reconnect: do NOT re-emit "joined"
        if session.grace_until is not None and _now() < session.grace_until:
            session.listener_ws = ws
            session.grace_until = None
            return session

        # Fresh join
        session.listener_ws = ws
        session.grace_until = None
        await _emit_event(session.patient_id, "joined")
        return session


async def detach_listener(session: ListenSession, now_listener_left: bool = False) -> None:
    async with _lock:
        session.listener_ws = None
        if now_listener_left:
            session.grace_until = None
            await _emit_event(session.patient_id, "left")
        else:
            session.grace_until = _now() + timedelta(seconds=RECONNECT_GRACE_S)


async def end_session(session: ListenSession, reason: str) -> None:
    if session.ended:
        return
    session.ended = True
    session.end_reason = reason
    ws = session.listener_ws
    session.listener_ws = None
    if ws is not None:
        await _send_end_and_close(ws, reason)
    _free_indexes(session)


def record_utterance(session: ListenSession, utterance_id: str, delivered: bool) -> None:
    if utterance_id in session.recent_utterances:
        return
    session.recent_utterances[utterance_id] = delivered
    while len(session.recent_utterances) > UTTERANCE_DEDUP_WINDOW:
        oldest = next(iter(session.recent_utterances))
        del session.recent_utterances[oldest]


def lookup_utterance(session: ListenSession, utterance_id: str) -> bool | None:
    return session.recent_utterances.get(utterance_id)


def is_listener_active(session: ListenSession) -> bool:
    return session.listener_ws is not None and not session.ended


# ────────── Background cleanup ──────────


async def _cleanup_once() -> None:
    now = _now()
    to_end: list[ListenSession] = []
    to_mark_left: list[ListenSession] = []
    async with _lock:
        for session in list(_sessions_by_patient.values()):
            if session.ended:
                continue
            if (
                session.listener_ws is None
                and session.grace_until is not None
                and now >= session.grace_until
            ):
                session.grace_until = None
                to_mark_left.append(session)
            if now >= session.expires_at:
                to_end.append(session)

    for session in to_mark_left:
        await _emit_event(session.patient_id, "left")

    for session in to_end:
        await end_session(session, "expired")


async def _cleanup_loop() -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_S)
            await _cleanup_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            # 루프는 죽이지 않는다.
            continue


async def start_cleanup_task() -> asyncio.Task:
    return asyncio.create_task(_cleanup_loop())


# ────────── 환자↔상대방 중계 지원 ──────────


async def relay_say_to_listener(
    session: ListenSession, utterance_id: str, text: str
) -> bool:
    ws = session.listener_ws
    if ws is None or session.ended:
        record_utterance(session, utterance_id, False)
        return False
    try:
        msg = WSListenSayOut(
            type="say",  # type: ignore[arg-type]
            utterance_id=utterance_id,
            text=text,
            server_ts=_now().isoformat(),
        )
        await ws.send_json(msg.model_dump(mode="json"))
        record_utterance(session, utterance_id, True)
        return True
    except Exception:
        record_utterance(session, utterance_id, False)
        return False


# ────────── Listener 상태 이벤트 (환자 WS로 전파용) ──────────


async def subscribe_listener_events(patient_id: UUID) -> asyncio.Queue[str]:
    q = _event_queues.get(patient_id)
    if q is None:
        q = asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
        _event_queues[patient_id] = q
    return q


async def unsubscribe_listener_events(patient_id: UUID) -> None:
    _event_queues.pop(patient_id, None)


# ────────── Played 이벤트 (상대방→환자) ──────────


async def subscribe_played_events(patient_id: UUID) -> asyncio.Queue[str]:
    """환자 WS가 상대방 played 알림을 받기 위한 큐. 큐 메시지는 utterance_id."""
    q = _played_queues.get(patient_id)
    if q is None:
        q = asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
        _played_queues[patient_id] = q
    return q


async def unsubscribe_played_events(patient_id: UUID) -> None:
    _played_queues.pop(patient_id, None)


def push_played_event(session: ListenSession, utterance_id: str) -> None:
    """상대방이 played를 보냈을 때 환자 측 큐에 enqueue.

    환자 WS가 구독 중이 아니면(queue 없음) 조용히 drop. drop-oldest 정책.
    """
    q = _played_queues.get(session.patient_id)
    if q is None:
        return
    _enqueue_drop_oldest(q, utterance_id)


# ────────── 테스트/lifespan 용 리셋 ──────────


def _reset_for_tests() -> None:
    """모든 모듈 전역 상태 초기화. 테스트 전용."""
    _sessions_by_patient.clear()
    _sessions_by_code.clear()
    _sessions_by_token.clear()
    _event_queues.clear()
    _played_queues.clear()
    _rate_limit.clear()
