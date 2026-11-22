"""Listen relay — streaming session의 say/listener/played 중계 단위 테스트.

store를 전면 모킹해서 session.py 로직만 검증한다. 실제 store 동작은 별도 스위트가 담당.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.streaming.session import StreamingSession
from app.schemas import PatientPrincipal

PATIENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────


def make_session():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    session = StreamingSession(
        app_ws=ws,
        patient_principal=PatientPrincipal(patient_id=PATIENT_ID, nickname="환자"),
        ml_ws_url="ws://mock-ml/asr-stream",
    )
    return session, ws


async def blocking_events():
    await asyncio.Event().wait()
    yield  # never


def make_proxy():
    proxy = AsyncMock()
    proxy.recv_events = blocking_events
    return proxy


def build_store_mock(
    *,
    session_obj=None,
    utterance_lookup: dict[str, bool] | None = None,
    relay_result: bool = True,
    listener_queue: asyncio.Queue | None = None,
    played_queue: asyncio.Queue | None = None,
):
    store = MagicMock()
    store.get_by_patient = MagicMock(return_value=session_obj)
    store.lookup_utterance = MagicMock(
        side_effect=lambda s, uid: (utterance_lookup or {}).get(uid)
    )
    store.record_utterance = MagicMock()
    store.relay_say_to_listener = AsyncMock(return_value=relay_result)
    store.subscribe_listener_events = AsyncMock(
        return_value=listener_queue if listener_queue is not None else asyncio.Queue()
    )
    store.unsubscribe_listener_events = AsyncMock()
    if played_queue is not None:
        store.subscribe_played_events = AsyncMock(return_value=played_queue)
        store.unsubscribe_played_events = AsyncMock()
    else:
        # played event helper 없음(hasattr False) — 실제 Agent A 미구현 상황 재현.
        del store.subscribe_played_events
        del store.unsubscribe_played_events
    return store


def pushed(ws):
    return [c.args[0] for c in ws.send_json.call_args_list]


def by_type(ws, t):
    return [m for m in pushed(ws) if m.get("type") == t]


# ── 테스트 ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_without_listen_session_pushes_listener_none() -> None:
    proxy = make_proxy()
    store = build_store_mock(session_obj=None)
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            states = by_type(ws, "listener")
            assert len(states) == 1
            assert states[0]["state"] == "none"
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_connect_with_active_listen_session_pushes_listener_joined() -> None:
    proxy = make_proxy()
    store = build_store_mock(session_obj=SimpleNamespace(ended=False))
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            states = by_type(ws, "listener")
            assert states[0]["state"] == "joined"
        finally:
            await session._close()


def _say_msg(uid: str = "u1", text: str = "물 좀 주세요") -> str:
    return json.dumps(
        {
            "type": "say",
            "utterance_id": uid,
            "text": text,
            "trace_id": "ignored",
            "client_ts": "2026-11-21T10:00:00+09:00",
        }
    )


@pytest.mark.asyncio
async def test_say_without_listen_session_returns_delivered_false() -> None:
    proxy = make_proxy()
    store = build_store_mock(session_obj=None)
    audits: list[tuple[str, dict]] = []

    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()

        async def fake_audit(event, payload):
            audits.append((event, payload))

        session._audit = fake_audit  # type: ignore[assignment]
        await session.start()
        try:
            await session.handle_client_message(_say_msg("u1"))
            saids = by_type(ws, "said")
            assert saids == [{"type": "said", "utterance_id": "u1", "delivered": False}]
            store.relay_say_to_listener.assert_not_called()
            listen_audits = [a for a in audits if a[0] == "listen_say"]
            assert len(listen_audits) == 1
            payload = listen_audits[0][1]
            assert payload["delivered"] is False
            assert payload["utterance_id"] == "u1"
            assert payload["text_len"] == len("물 좀 주세요")
            # text 자체는 저장 안 됨.
            assert "text" not in payload
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_say_with_active_session_relays_and_delivers() -> None:
    proxy = make_proxy()
    sess_obj = SimpleNamespace(ended=False)
    store = build_store_mock(session_obj=sess_obj, relay_result=True)
    audits: list[tuple[str, dict]] = []

    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        session._audit = lambda e, p: audits.append((e, p)) or asyncio.sleep(0)  # type: ignore[assignment]
        await session.start()
        try:
            await session.handle_client_message(_say_msg("u2", "안녕"))
            store.relay_say_to_listener.assert_awaited_once_with(sess_obj, "u2", "안녕")
            store.record_utterance.assert_called_once_with(sess_obj, "u2", True)
            saids = by_type(ws, "said")
            assert saids[-1] == {"type": "said", "utterance_id": "u2", "delivered": True}
            assert any(a[0] == "listen_say" and a[1]["delivered"] is True for a in audits)
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_duplicate_say_uses_idempotency_cache() -> None:
    proxy = make_proxy()
    sess_obj = SimpleNamespace(ended=False)
    store = build_store_mock(
        session_obj=sess_obj,
        utterance_lookup={"u3": True},
        relay_result=True,
    )
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            await session.handle_client_message(_say_msg("u3"))
            # 캐시 히트 → relay 안 함, record 안 함.
            store.relay_say_to_listener.assert_not_called()
            store.record_utterance.assert_not_called()
            saids = by_type(ws, "said")
            assert saids[-1] == {"type": "said", "utterance_id": "u3", "delivered": True}
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_listener_event_from_queue_pushes_to_patient() -> None:
    proxy = make_proxy()
    q: asyncio.Queue[str] = asyncio.Queue()
    store = build_store_mock(session_obj=None, listener_queue=q)
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            await q.put("joined")
            # 큐 소비 대기.
            for _ in range(20):
                await asyncio.sleep(0.01)
                if any(
                    m.get("type") == "listener" and m.get("state") == "joined"
                    for m in pushed(ws)
                ):
                    break
            states = by_type(ws, "listener")
            # 최초 'none' + 이후 'joined'.
            assert states[0]["state"] == "none"
            assert any(s["state"] == "joined" for s in states)
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_played_event_from_queue_pushes_played_message() -> None:
    proxy = make_proxy()
    played_q: asyncio.Queue[str] = asyncio.Queue()
    store = build_store_mock(session_obj=None, played_queue=played_q)
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            await played_q.put("u9")
            for _ in range(20):
                await asyncio.sleep(0.01)
                if by_type(ws, "played"):
                    break
            plays = by_type(ws, "played")
            assert plays == [{"type": "played", "utterance_id": "u9"}]
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_malformed_say_pushes_internal_error() -> None:
    proxy = make_proxy()
    store = build_store_mock(session_obj=None)
    with (
        patch("app.agent.streaming.session.MLProxySession", return_value=proxy),
        patch("app.agent.streaming.session.listen_store", store),
    ):
        session, ws = make_session()
        await session.start()
        try:
            bad = json.dumps({"type": "say", "utterance_id": "u", "text": "x"})  # trace/client_ts 누락
            await session.handle_client_message(bad)
            errors = by_type(ws, "error")
            assert errors
            assert errors[-1]["error_code"] == "INTERNAL"
            store.relay_say_to_listener.assert_not_called()
        finally:
            await session._close()
