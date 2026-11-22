"""HTTP 엔드포인트 통합 테스트 (/listen/*).

store.py 구현이 아직이라, store의 공개 함수를 monkeypatch한 in-process fake로 테스트.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient

from app.listen import router as listen_router
from app.listen import store as listen_store
from app.security import decode_token, issue_token


async def _dep_require_patient_override(request: Request):
    authorization = request.headers.get("authorization")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "Bearer 토큰이 필요합니다."},
        )
    return decode_token(authorization.split(" ", 1)[1])


# ────────── Fake store ──────────


@dataclass
class _FakeSession:
    listen_session_id: str
    patient_id: uuid.UUID
    join_code: str
    join_code_consumed: bool = False
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


class _FakeStore:
    def __init__(self) -> None:
        self.by_patient: dict[uuid.UUID, _FakeSession] = {}
        self.by_code: dict[str, _FakeSession] = {}
        self.by_token: dict[str, _FakeSession] = {}
        self.rate_hits: dict[str, int] = {}
        self.rate_limit = 10
        self.counter = 0

    def create_session(self, patient_id: uuid.UUID) -> _FakeSession:
        # revoke 기존
        existing = self.by_patient.get(patient_id)
        if existing:
            existing.ended = True
            existing.end_reason = "revoked"
            self.by_code.pop(existing.join_code, None)
        self.counter += 1
        code = f"ABC{self.counter:03d}"
        sess = _FakeSession(
            listen_session_id=f"ls_{self.counter:06d}",
            patient_id=patient_id,
            join_code=code,
        )
        self.by_patient[patient_id] = sess
        self.by_code[code] = sess
        return sess

    def get_by_patient(self, patient_id: uuid.UUID) -> _FakeSession | None:
        s = self.by_patient.get(patient_id)
        if s and s.ended:
            return None
        return s

    def get_by_join_code(self, code: str) -> _FakeSession | None:
        return self.by_code.get(code)

    def get_by_listener_token(self, token: str) -> _FakeSession | None:
        s = self.by_token.get(token)
        if s and s.ended:
            return None
        return s

    async def consume_join_code(self, code: str, ip: str) -> tuple[_FakeSession, str]:
        self.rate_hits[ip] = self.rate_hits.get(ip, 0) + 1
        if self.rate_hits[ip] > self.rate_limit:
            raise listen_store.RateLimitedError
        sess = self.by_code.get(code)
        if sess is None or sess.ended or sess.join_code_consumed:
            raise listen_store.JoinCodeInvalidError
        sess.join_code_consumed = True
        token = f"tok_{sess.listen_session_id}_{'x' * 24}"
        sess.listener_token = token
        self.by_token[token] = sess
        return sess, token

    async def end_session(self, session: _FakeSession, reason: str) -> None:
        session.ended = True
        session.end_reason = reason
        self.by_code.pop(session.join_code, None)


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> _FakeStore:
    fs = _FakeStore()
    monkeypatch.setattr(listen_store, "create_session", fs.create_session)
    monkeypatch.setattr(listen_store, "get_by_patient", fs.get_by_patient)
    monkeypatch.setattr(listen_store, "get_by_join_code", fs.get_by_join_code)
    monkeypatch.setattr(listen_store, "get_by_listener_token", fs.get_by_listener_token)
    monkeypatch.setattr(listen_store, "consume_join_code", fs.consume_join_code)
    monkeypatch.setattr(listen_store, "end_session", fs.end_session)
    return fs


@pytest.fixture
def app_with_listen(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    # audit은 DB를 쓰므로 fire-and-forget을 no-op으로.
    monkeypatch.setattr(listen_router, "_fire_audit", lambda **kw: None)

    from app.errors import install as install_error_handlers
    from app.security import require_patient

    app = FastAPI()
    install_error_handlers(app)
    app.include_router(listen_router.router)
    app.dependency_overrides[require_patient] = _dep_require_patient_override
    return app


@pytest.fixture
def auth_headers() -> tuple[uuid.UUID, dict[str, str]]:
    pid = uuid.uuid4()
    token, _ = issue_token(pid, "tester")
    return pid, {"Authorization": f"Bearer {token}"}


# ────────── Tests ──────────


def test_post_sessions_creates_and_returns_201(app_with_listen, fake_store, auth_headers):
    pid, headers = auth_headers
    with TestClient(app_with_listen) as client:
        r = client.post("/listen/sessions", headers=headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["listen_session_id"].startswith("ls_")
    assert len(body["join_code"]) == 6
    assert "expires_at" in body and "join_expires_at" in body
    assert pid in fake_store.by_patient


def test_get_current_waiting_includes_join_code(app_with_listen, fake_store, auth_headers):
    _, headers = auth_headers
    with TestClient(app_with_listen) as client:
        client.post("/listen/sessions", headers=headers)
        r = client.get("/listen/sessions/current", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "waiting"
    assert body["join_code"] is not None


def test_get_current_404_when_no_session(app_with_listen, fake_store, auth_headers):
    _, headers = auth_headers
    with TestClient(app_with_listen) as client:
        r = client.get("/listen/sessions/current", headers=headers)
    assert r.status_code == 404
    assert r.json()["error_code"] == "LISTEN_NO_SESSION"


def test_post_join_success(app_with_listen, fake_store, auth_headers):
    _, headers = auth_headers
    with TestClient(app_with_listen) as client:
        created = client.post("/listen/sessions", headers=headers).json()
        r = client.post("/listen/join", json={"join_code": created["join_code"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["listener_token"].startswith("tok_")
    assert "expires_at" in body


def test_post_join_invalid_code(app_with_listen, fake_store):
    with TestClient(app_with_listen) as client:
        r = client.post("/listen/join", json={"join_code": "ZZZZZZ"})
    assert r.status_code == 404
    assert r.json()["error_code"] == "LISTEN_CODE_INVALID"


def test_post_join_rate_limited(app_with_listen, fake_store):
    with TestClient(app_with_listen) as client:
        # 10회 잘못된 시도 — 모두 404. 11회차가 429.
        for _ in range(10):
            client.post("/listen/join", json={"join_code": "ZZZZZZ"})
        r = client.post("/listen/join", json={"join_code": "ZZZZZZ"})
    assert r.status_code == 429
    assert r.json()["error_code"] == "RATE_LIMITED"


def test_delete_current_204(app_with_listen, fake_store, auth_headers):
    _, headers = auth_headers
    with TestClient(app_with_listen) as client:
        client.post("/listen/sessions", headers=headers)
        r = client.delete("/listen/sessions/current", headers=headers)
    assert r.status_code == 204


def test_delete_current_404_when_none(app_with_listen, fake_store, auth_headers):
    _, headers = auth_headers
    with TestClient(app_with_listen) as client:
        r = client.delete("/listen/sessions/current", headers=headers)
    assert r.status_code == 404
    assert r.json()["error_code"] == "LISTEN_NO_SESSION"


def test_replacing_session_ends_previous(app_with_listen, fake_store, auth_headers):
    pid, headers = auth_headers
    with TestClient(app_with_listen) as client:
        first = client.post("/listen/sessions", headers=headers).json()
        second = client.post("/listen/sessions", headers=headers).json()
    assert first["listen_session_id"] != second["listen_session_id"]
    # 이전 세션이 store에서 ended 상태이거나 by_code에서 사라졌는지 확인.
    old = fake_store.by_code.get(first["join_code"])
    assert old is None or old.ended
