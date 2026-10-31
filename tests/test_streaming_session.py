"""실시간 세션: DB 문맥 없이 ASR/교정 연결과 자원 종료 검증."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.streaming.ml_proxy import MLProxyConnectionError, MLProxyEvent, MLProxyEventType
from app.agent.streaming.session import StreamingSession
from app.schemas import PatientPrincipal

PATIENT_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def make_session(proxy, *, correct=None):
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    session = StreamingSession(
        app_ws=ws,
        patient_principal=PatientPrincipal(patient_id=PATIENT_ID, nickname="테스트"),
        ml_ws_url="ws://mock-ml/asr-stream",
        llm_correct=correct,
    )
    return session, ws


async def blocking_events():
    await asyncio.Event().wait()
    yield  # async generator


def make_proxy():
    proxy = AsyncMock()
    proxy.recv_events = blocking_events
    return proxy


@pytest.mark.asyncio
async def test_start_requires_no_database_and_close_joins_tasks() -> None:
    proxy = make_proxy()
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy) as factory:
        session, ws = make_session(proxy)
        await session.start()
        assert uuid.UUID(session.trace_id).version == 4
        assert factory.call_args.kwargs["patient_id"] == str(PATIENT_ID)
        proxy.open.assert_awaited_once()
        await session._close()
        assert session._ml_event_task.done()
        proxy.close.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1000, reason="")
        await session._close()
        ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_corrected_without_closing_connections() -> None:
    proxy = make_proxy()

    async def events():
        yield MLProxyEvent(type=MLProxyEventType.TRANSCRIPT_FINAL, text="믈", confidence=0.39)

    proxy.recv_events = events
    correct = AsyncMock(return_value=("물", 0.99))
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy, correct=correct)
        await session.start()
        await asyncio.wait_for(session._ml_event_task, 1)
        correct.assert_awaited_once_with("믈")
        msg = ws.send_json.call_args.args[0]
        assert msg["text"] == "물"
        assert msg["confidence"] == 0.39
        assert msg["final"] is True
        proxy.close.assert_not_awaited()
        ws.close.assert_not_awaited()
        await session._close()


@pytest.mark.asyncio
async def test_client_end_delivers_corrected_final_without_closing() -> None:
    proxy = make_proxy()
    ended = asyncio.Event()
    proxy.send_end.side_effect = ended.set

    async def events():
        await ended.wait()
        await asyncio.sleep(0.02)
        yield MLProxyEvent(
            type=MLProxyEventType.TRANSCRIPT_FINAL, text="믈", confidence=0.82, is_last=True
        )

    async def correct(raw):
        await asyncio.sleep(0.02)
        return "물", 0.99

    proxy.recv_events = events
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy, correct=correct)
        await session.start()
        await session.handle_client_message('{"type":"end"}')
        await asyncio.wait_for(session._ml_event_task, 1)
        messages = [call.args[0] for call in ws.send_json.call_args_list]
        assert messages[-1]["text"] == "물"
        assert messages[-1]["final"] is True
        assert messages[-1]["confidence"] == 0.82
        proxy.send_end.assert_awaited_once()
        proxy.close.assert_not_awaited()
        ws.close.assert_not_awaited()
        assert not await session.handle_client_message('{"type":"close"}')
        proxy.close.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1000, reason="")


@pytest.mark.asyncio
async def test_end_cancel_and_invalid_control_messages() -> None:
    proxy = make_proxy()
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy)
        await session.start()
        for message in ["broken", "[]", "null", "42"]:
            assert await session.handle_client_message(message)
        assert await session.handle_client_message(b"pcm")
        proxy.send_audio.assert_awaited_once_with(b"pcm")
        assert await session.handle_client_message('{"type":"end"}')
        assert await session.handle_client_message('{"type":"end"}')
        assert await session.handle_client_message(b"late audio")
        proxy.send_end.assert_awaited_once()
        proxy.send_audio.assert_awaited_once_with(b"pcm")
        assert not await session.handle_client_message('{"type":"cancel"}')
        ws.close.assert_awaited_once_with(code=1001, reason="")
        assert session._ml_event_task.done()


@pytest.mark.asyncio
async def test_end_without_ml_final_keeps_connections_open_past_old_timeout() -> None:
    proxy = make_proxy()
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy)
        await session.start()
        await session.handle_client_message('{"type":"end"}')
        await asyncio.sleep(2.1)
        assert not session._ml_event_task.done()
        proxy.close.assert_not_awaited()
        ws.close.assert_not_awaited()
        await session.handle_client_message('{"type":"close"}')
        assert session._ml_event_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "no_speech"), [("", 1.0), ("  ", 0.0), ("없는 말", 1.0)])
async def test_silent_final_asks_again_without_llm(text, no_speech) -> None:
    proxy = make_proxy()
    correct = AsyncMock()

    async def events():
        yield MLProxyEvent(
            type=MLProxyEventType.TRANSCRIPT_FINAL,
            text=text,
            confidence=0.73,
            no_speech_prob=no_speech,
        )

    proxy.recv_events = events
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy, correct=correct)
        await session.start()
        await asyncio.wait_for(session._ml_event_task, 1)
        correct.assert_not_awaited()
        assert ws.send_json.call_args.args[0]["error_code"] == "ASR_LOW_CONFIDENCE"
        proxy.close.assert_not_awaited()
        ws.close.assert_not_awaited()
        await session._close()


@pytest.mark.asyncio
async def test_ml_audio_limit_error_still_delivers_final() -> None:
    proxy = make_proxy()

    async def events():
        yield MLProxyEvent(type=MLProxyEventType.ML_ERROR, error_code="AUDIO_TOO_LONG")
        yield MLProxyEvent(type=MLProxyEventType.TRANSCRIPT_FINAL, text="믈", confidence=0.82)

    proxy.recv_events = events
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy, correct=AsyncMock(return_value=("물", 0.99)))
        await session.start()
        await asyncio.wait_for(session._ml_event_task, 1)
        messages = [call.args[0] for call in ws.send_json.call_args_list]
        assert messages[0]["error_code"] == "AUDIO_TOO_LONG"
        assert messages[-1]["text"] == "물"
        assert messages[-1]["final"] is True
        proxy.send_end.assert_not_awaited()
        proxy.close.assert_not_awaited()
        ws.close.assert_not_awaited()
        await session._close()


@pytest.mark.asyncio
async def test_ml_open_failure_leaves_closing_to_frontend() -> None:
    proxy = make_proxy()
    proxy.open.side_effect = MLProxyConnectionError("unavailable")
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy)
        await session.start()
        assert ws.send_json.call_args.args[0]["error_code"] == "ML_UPSTREAM_UNAVAILABLE"
        ws.close.assert_not_awaited()
        await session.handle_client_message('{"type":"close"}')


@pytest.mark.asyncio
async def test_ml_disconnect_does_not_close_frontend():
    proxy = make_proxy()

    async def events():
        yield MLProxyEvent(type=MLProxyEventType.CONNECTION_LOST)

    proxy.recv_events = events
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy)
        await session.start()
        await asyncio.wait_for(session._ml_event_task, 1)
        assert ws.send_json.call_args.args[0]["error_code"] == "ML_UPSTREAM_UNAVAILABLE"
        ws.close.assert_not_awaited()
        assert session._reconnect_task is not None
        await session.handle_client_message('{"type":"close"}')
        ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_normal_ml_completion_is_not_reported_as_an_error():
    proxy = make_proxy()

    async def events():
        yield MLProxyEvent(type=MLProxyEventType.CONNECTION_CLOSED)

    proxy.recv_events = events
    with patch("app.agent.streaming.session.MLProxySession", return_value=proxy):
        session, ws = make_session(proxy)
        await session.start()
        await asyncio.wait_for(session._ml_event_task, 1)
        ws.send_json.assert_not_awaited()
        ws.close.assert_not_awaited()
        await session.handle_client_message('{"type":"close"}')
