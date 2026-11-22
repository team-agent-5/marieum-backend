"""WebSocket /ws/listen 테스트."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.listen import router as listen_router
from app.listen import store as listen_store


@dataclass
class _FakeSession:
    listen_session_id: str
    patient_id: uuid.UUID
    join_code: str = "ABCDEF"
    join_code_consumed: bool = True
    listener_token: str | None = None
    join_expires_at: datetime = field(
        default_factory=lambda: datetime.now().astimezone() + timedelta(minutes=10)
    )
    expires_at: datetime = field(
        default_factory=lambda: datetime.now().astimezone() + timedelta(hours=2)
    )
    listener_ws: Any = None
    grace_until: datetime | None = None
    ended: bool = False
    end_reason: str | None = None


class _State:
    def __init__(self) -> None:
        self.sessions: dict[str, _FakeSession] = {}
        self.played: list[tuple[str, str]] = []  # (session_id, utterance_id)
        self.attached: set[str] = set()
        self.detached: list[str] = []

    def add(self, token: str) -> _FakeSession:
        sess = _FakeSession(
            listen_session_id=f"ls_{token}",
            patient_id=uuid.uuid4(),
            listener_token=token,
        )
        self.sessions[token] = sess
        return sess


@pytest.fixture
def ws_state(monkeypatch: pytest.MonkeyPatch) -> _State:
    st = _State()

    def get_by_listener_token(token: str) -> _FakeSession | None:
        s = st.sessions.get(token)
        if s is None or s.ended:
            return None
        return s

    async def attach_listener(token: str, ws) -> _FakeSession:
        s = st.sessions.get(token)
        if s is None or s.ended:
            raise RuntimeError("SessionEndedError")
        if s.listener_token in st.attached:
            raise listen_store.ListenerAlreadyJoinedError
        st.attached.add(token)
        s.listener_ws = ws
        return s

    async def detach_listener(session, now_listener_left: bool = False) -> None:
        st.attached.discard(session.listener_token or "")
        session.listener_ws = None
        st.detached.append(session.listen_session_id)

    def push_played_event(session, utterance_id: str) -> None:
        st.played.append((session.listen_session_id, utterance_id))

    monkeypatch.setattr(listen_store, "get_by_listener_token", get_by_listener_token)
    monkeypatch.setattr(listen_store, "attach_listener", attach_listener)
    monkeypatch.setattr(listen_store, "detach_listener", detach_listener)
    monkeypatch.setattr(listen_store, "push_played_event", push_played_event, raising=False)
    monkeypatch.setattr(listen_router, "_fire_audit", lambda **kw: None)
    # ping 주기를 짧게
    monkeypatch.setattr(listen_router, "WS_LISTEN_PING_INTERVAL_S", 0.05)
    return st


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(listen_router.router)
    return app


def _connect(client: TestClient, token: str):
    return client.websocket_connect(
        "/ws/listen",
        subprotocols=["marieum.listen.v1", token],
    )


def test_missing_token_closes_4001(app, ws_state):
    with TestClient(app) as client:
        # 토큰 없이 subprotocol만.
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                "/ws/listen", subprotocols=["marieum.listen.v1"]
            ) as ws:
                ws.receive_json()
    assert exc.value.code == 4001


def test_invalid_token_closes_4001(app, ws_state):
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as exc:
        with _connect(client, "bogus") as ws:
            ws.receive_json()
    assert exc.value.code == 4001


def test_valid_token_connects_and_receives_ping(app, ws_state):
    sess = ws_state.add("tok_valid")
    with TestClient(app) as client, _connect(client, "tok_valid") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "ping"
    assert sess.listen_session_id in ws_state.detached


def test_already_joined_closes_4009(app, ws_state):
    ws_state.add("tok_dup")
    with TestClient(app) as client, _connect(client, "tok_dup") as ws1:
        ws1.receive_json()  # 첫 ping으로 연결 확인
        # 두 번째 접속 — attach_listener가 ListenerAlreadyJoinedError 발생.
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "tok_dup") as ws2:
            ws2.receive_json()
        assert exc.value.code == 4009


def test_played_message_recorded(app, ws_state):
    sess = ws_state.add("tok_play")
    with TestClient(app) as client, _connect(client, "tok_play") as ws:
        ws.send_json({"type": "played", "utterance_id": "u-1"})
        # ping 하나 받기 (짧은 주기이므로 거의 즉시)
        ws.receive_json()
    # played가 store에 기록됐는지 (eventually)
    # 짧게 대기해서 핸들러가 실행됐는지 확인
    for _ in range(20):
        if ws_state.played:
            break
    assert any(uid == "u-1" for _, uid in ws_state.played)


def test_session_ended_error_closes_4004(app, ws_state, monkeypatch):
    sess = ws_state.add("tok_ended")

    class SessionEndedError(Exception):
        pass

    async def attach(token: str, ws) -> _FakeSession:
        raise SessionEndedError

    monkeypatch.setattr(listen_store, "attach_listener", attach)
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as exc:
        with _connect(client, "tok_ended") as ws:
            ws.receive_json()
    assert exc.value.code == 4004
