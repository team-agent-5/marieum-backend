"""Listen in-memory store 단위 테스트."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.listen import store
from app.listen.store import (
    JoinCodeInvalidError,
    ListenerAlreadyJoinedError,
    RateLimitedError,
    SessionEndedError,
)


@pytest.fixture(autouse=True)
def _reset():
    store._reset_for_tests()
    yield
    store._reset_for_tests()


# ────────── create / revoke ──────────


def test_create_session_generates_fields():
    pid = uuid4()
    s = store.create_session(pid)
    assert s.patient_id == pid
    assert s.listen_session_id.startswith("ls_")
    assert len(s.join_code) == store.JOIN_CODE_LEN
    assert all(c in store.JOIN_CODE_CHARSET for c in s.join_code)
    assert not s.join_code_consumed
    assert not s.ended
    assert store.get_by_patient(pid) is s
    assert store.get_by_join_code(s.join_code) is s


async def test_create_session_revokes_existing_and_closes_listener():
    pid = uuid4()
    old = store.create_session(pid)
    ws = AsyncMock()
    old.listener_ws = ws

    new = store.create_session(pid)
    assert new is not old
    assert old.ended is True
    assert old.end_reason == "revoked"
    # old join_code freed from index
    assert store.get_by_join_code(old.join_code) is None
    assert store.get_by_patient(pid) is new
    # New code is different (practically — allow retry collision; still uniqueness guaranteed)
    assert new.join_code != old.join_code or len(store._sessions_by_code) == 1

    # Allow the scheduled close task to run
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    ws.send_json.assert_awaited()
    ws.close.assert_awaited()


def test_create_session_join_codes_are_unique_across_patients():
    codes = set()
    for _ in range(50):
        s = store.create_session(uuid4())
        assert s.join_code not in codes
        codes.add(s.join_code)


# ────────── join_code lifecycle ──────────


async def test_consume_join_code_success_then_second_fails():
    pid = uuid4()
    s = store.create_session(pid)
    sess, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    assert sess is s
    assert isinstance(token, str) and len(token) > 20
    assert s.join_code_consumed is True
    assert s.listener_token == token
    assert store.get_by_listener_token(token) is s

    with pytest.raises(JoinCodeInvalidError):
        await store.consume_join_code(s.join_code, "1.1.1.1")


async def test_consume_join_code_unknown_code_raises_invalid():
    with pytest.raises(JoinCodeInvalidError):
        await store.consume_join_code("ZZZZZZ", "1.1.1.1")


async def test_consume_join_code_expired():
    pid = uuid4()
    s = store.create_session(pid)
    s.join_expires_at = store._now() - timedelta(seconds=1)
    with pytest.raises(JoinCodeInvalidError):
        await store.consume_join_code(s.join_code, "1.1.1.1")


# ────────── Rate limit ──────────


async def test_rate_limit_11th_attempt_raises():
    ip = "9.9.9.9"
    # 10 attempts all with invalid code — still count toward rate limit
    for _ in range(store.JOIN_RATE_LIMIT_PER_MIN):
        with pytest.raises(JoinCodeInvalidError):
            await store.consume_join_code("NOPE12", ip)
    with pytest.raises(RateLimitedError):
        await store.consume_join_code("NOPE12", ip)


async def test_rate_limit_per_ip_isolation():
    for _ in range(store.JOIN_RATE_LIMIT_PER_MIN):
        with pytest.raises(JoinCodeInvalidError):
            await store.consume_join_code("NOPE12", "a.a.a.a")
    # Different IP unaffected
    with pytest.raises(JoinCodeInvalidError):
        await store.consume_join_code("NOPE12", "b.b.b.b")


# ────────── attach / detach ──────────


async def test_attach_listener_fresh_emits_joined():
    pid = uuid4()
    s = store.create_session(pid)
    q = await store.subscribe_listener_events(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    ws = AsyncMock()
    attached = await store.attach_listener(token, ws)
    assert attached is s
    assert s.listener_ws is ws
    assert store.is_listener_active(s)
    assert q.get_nowait() == "joined"


async def test_attach_listener_twice_raises_already_joined():
    pid = uuid4()
    s = store.create_session(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    await store.attach_listener(token, AsyncMock())
    with pytest.raises(ListenerAlreadyJoinedError):
        await store.attach_listener(token, AsyncMock())


async def test_attach_listener_unknown_token_raises_session_ended():
    with pytest.raises(SessionEndedError):
        await store.attach_listener("bogus-token", AsyncMock())


async def test_attach_listener_after_end_raises():
    pid = uuid4()
    s = store.create_session(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    await store.end_session(s, "patient_left")
    with pytest.raises(SessionEndedError):
        await store.attach_listener(token, AsyncMock())


async def test_grace_reconnect_does_not_re_emit_joined():
    pid = uuid4()
    s = store.create_session(pid)
    q = await store.subscribe_listener_events(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    ws1 = AsyncMock()
    await store.attach_listener(token, ws1)
    assert q.get_nowait() == "joined"

    # Disconnect → grace
    await store.detach_listener(s, now_listener_left=False)
    assert s.listener_ws is None
    assert s.grace_until is not None
    assert store.is_listener_active(s) is False
    assert q.empty()  # no "left" yet

    # Reconnect
    ws2 = AsyncMock()
    await store.attach_listener(token, ws2)
    assert s.listener_ws is ws2
    assert s.grace_until is None
    # No new emit
    assert q.empty()


async def test_detach_grace_expiry_emits_left_via_cleanup():
    pid = uuid4()
    s = store.create_session(pid)
    q = await store.subscribe_listener_events(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    await store.attach_listener(token, AsyncMock())
    assert q.get_nowait() == "joined"

    await store.detach_listener(s, now_listener_left=False)
    # Simulate grace expiry
    s.grace_until = store._now() - timedelta(seconds=1)
    await store._cleanup_once()
    assert q.get_nowait() == "left"
    assert s.grace_until is None


async def test_detach_with_now_listener_left_emits_immediately():
    pid = uuid4()
    s = store.create_session(pid)
    q = await store.subscribe_listener_events(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    await store.attach_listener(token, AsyncMock())
    q.get_nowait()  # consume "joined"

    await store.detach_listener(s, now_listener_left=True)
    assert q.get_nowait() == "left"
    assert s.grace_until is None


# ────────── end_session ──────────


async def test_end_session_revoked_sends_end_and_closes_ws():
    pid = uuid4()
    s = store.create_session(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    ws = AsyncMock()
    await store.attach_listener(token, ws)

    await store.end_session(s, "revoked")
    assert s.ended is True
    assert s.end_reason == "revoked"
    ws.send_json.assert_awaited_once()
    sent = ws.send_json.await_args.args[0]
    assert sent["type"] == "session_end"
    assert sent["reason"] == "revoked"
    ws.close.assert_awaited_once()
    # Indexes cleaned up
    assert store.get_by_patient(pid) is None
    assert store.get_by_join_code(s.join_code) is None
    assert store.get_by_listener_token(token) is None


async def test_end_session_is_idempotent():
    pid = uuid4()
    s = store.create_session(pid)
    await store.end_session(s, "expired")
    await store.end_session(s, "revoked")  # should no-op
    assert s.end_reason == "expired"


async def test_cleanup_ends_expired_session():
    pid = uuid4()
    s = store.create_session(pid)
    s.expires_at = store._now() - timedelta(seconds=1)
    await store._cleanup_once()
    assert s.ended is True
    assert s.end_reason == "expired"


# ────────── utterance dedup ──────────


def test_record_and_lookup_utterance():
    pid = uuid4()
    s = store.create_session(pid)
    store.record_utterance(s, "u1", True)
    assert store.lookup_utterance(s, "u1") is True
    # Second record keeps first value
    store.record_utterance(s, "u1", False)
    assert store.lookup_utterance(s, "u1") is True
    assert store.lookup_utterance(s, "unknown") is None


def test_utterance_cache_evicts_oldest_after_51():
    pid = uuid4()
    s = store.create_session(pid)
    for i in range(store.UTTERANCE_DEDUP_WINDOW):
        store.record_utterance(s, f"u{i}", True)
    assert len(s.recent_utterances) == store.UTTERANCE_DEDUP_WINDOW
    # Add one more
    store.record_utterance(s, "u_new", True)
    assert len(s.recent_utterances) == store.UTTERANCE_DEDUP_WINDOW
    assert store.lookup_utterance(s, "u0") is None  # oldest evicted
    assert store.lookup_utterance(s, "u_new") is True


# ────────── relay_say ──────────


async def test_relay_say_success_records_delivered_true():
    pid = uuid4()
    s = store.create_session(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    ws = AsyncMock()
    await store.attach_listener(token, ws)
    ok = await store.relay_say_to_listener(s, "u1", "hello")
    assert ok is True
    ws.send_json.assert_awaited_once()
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "say"
    assert payload["utterance_id"] == "u1"
    assert payload["text"] == "hello"
    assert "server_ts" in payload
    assert store.lookup_utterance(s, "u1") is True


async def test_relay_say_no_listener_returns_false():
    pid = uuid4()
    s = store.create_session(pid)
    ok = await store.relay_say_to_listener(s, "u1", "hi")
    assert ok is False
    assert store.lookup_utterance(s, "u1") is False


async def test_relay_say_send_failure_returns_false():
    pid = uuid4()
    s = store.create_session(pid)
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    ws = AsyncMock()
    ws.send_json.side_effect = RuntimeError("boom")
    await store.attach_listener(token, ws)
    ok = await store.relay_say_to_listener(s, "u1", "hi")
    assert ok is False
    assert store.lookup_utterance(s, "u1") is False


# ────────── subscribe / unsubscribe ──────────


async def test_subscribe_returns_same_queue():
    pid = uuid4()
    q1 = await store.subscribe_listener_events(pid)
    q2 = await store.subscribe_listener_events(pid)
    assert q1 is q2


async def test_unsubscribe_removes_queue_and_stops_emit():
    pid = uuid4()
    q = await store.subscribe_listener_events(pid)
    await store.unsubscribe_listener_events(pid)
    # Re-subscribe → brand new queue
    q2 = await store.subscribe_listener_events(pid)
    assert q is not q2


# ────────── is_listener_active ──────────


async def test_is_listener_active_states():
    pid = uuid4()
    s = store.create_session(pid)
    assert store.is_listener_active(s) is False
    _, token = await store.consume_join_code(s.join_code, "1.1.1.1")
    await store.attach_listener(token, AsyncMock())
    assert store.is_listener_active(s) is True
    await store.detach_listener(s, now_listener_left=False)
    # Grace in progress — not active
    assert store.is_listener_active(s) is False


# ────────── cleanup task lifecycle ──────────


async def test_start_cleanup_task_returns_cancellable_task():
    task = await store.start_cleanup_task()
    assert isinstance(task, asyncio.Task)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
