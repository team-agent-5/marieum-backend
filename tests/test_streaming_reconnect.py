"""ML 응답 마감/재연결은 프론트 WebSocket 수명과 분리된다."""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

from app.agent.streaming import session as module
from app.agent.streaming.session import StreamingSession
from app.schemas import PatientPrincipal


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def frontend_session(url):
    ws = MagicMock(send_json=AsyncMock(), close=AsyncMock())
    session = StreamingSession(
        ws, PatientPrincipal(patient_id=uuid.uuid4(), nickname="synthetic"), url
    )
    return session, ws


@pytest.mark.asyncio
async def test_unanswered_audio_times_out_and_reconnects_without_closing_frontend(monkeypatch):
    assert module.ML_RESPONSE_TIMEOUT_S == 30
    monkeypatch.setattr(module, "ML_RESPONSE_TIMEOUT_S", 0.12)
    monkeypatch.setattr(module, "ML_RETRY_INITIAL_S", 0.01)
    connections = []
    first_closed = asyncio.Event()

    async def ml(ws):
        connections.append(ws)
        start = json.loads(await ws.recv())
        assert start["mode"] == "conversation"
        connection = len(connections)
        try:
            async for _audio in ws:
                if connection == 1:
                    continue  # 연결은 살아 있지만 ASR 응답이 없다
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript_final",
                            "text": "물 주세요",
                            "segment_id": 0,
                            "confidence": 0.9,
                            "is_last": False,
                        }
                    )
                )
        finally:
            if connection == 1:
                first_closed.set()

    async with websockets.serve(ml, "127.0.0.1", 0) as server:
        session, frontend = frontend_session(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        try:
            await session.start()
            # 오디오를 보내지 않은 idle에는 마감 시간이 없다.
            await asyncio.sleep(0.15)
            assert len(connections) == 1
            # 계속 보내도 미응답 마감 시간을 연장하지 않는다.
            for _ in range(10):
                await session.handle_client_message(b"\0\0" * 1600)
                await asyncio.sleep(0.02)
            await until(lambda: len(connections) == 2 and session._ml_available)
            assert first_closed.is_set()
            await session.handle_client_message(b"\0\0" * 1600)
            await until(
                lambda: any(c.args[0].get("final") for c in frontend.send_json.call_args_list)
            )
            await until(lambda: session._response_pending_since is None)
            frontend.close.assert_not_awaited()
        finally:
            await session._close()
        assert session._watchdog_task.done()
        assert session._reconnect_task.done()


@pytest.mark.asyncio
async def test_dropped_ml_reconnects_and_forwards_buffered_audio_and_end(monkeypatch):
    monkeypatch.setattr(module, "ML_RETRY_INITIAL_S", 0.04)
    connections = []
    received = []

    async def ml(ws):
        connections.append(ws)
        await ws.recv()
        if len(connections) == 1:
            await ws.close(code=1011)
            return
        async for message in ws:
            received.append(message)
            if isinstance(message, str):
                await ws.send(
                    json.dumps({"type": "transcript_final", "text": "끝", "is_last": True})
                )

    async with websockets.serve(ml, "127.0.0.1", 0) as server:
        session, frontend = frontend_session(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        try:
            await session.start()
            await until(lambda: not session._ml_available)
            await session.handle_client_message(b"\0\0" * 1600)
            await session.handle_client_message('{"type":"end"}')
            await until(lambda: len(received) == 2)
            assert received[0] == b"\0\0" * 1600
            assert json.loads(received[1])["type"] == "end"
            await until(lambda: not session._end_pending)
            frontend.close.assert_not_awaited()
            assert len(connections) == 2
        finally:
            await session._close()


@pytest.mark.asyncio
async def test_frontend_close_cancels_ml_retry(monkeypatch):
    monkeypatch.setattr(module, "ML_RETRY_INITIAL_S", 0.05)
    proxy = AsyncMock()
    proxy.open.side_effect = module.MLProxyConnectionError("unavailable")
    monkeypatch.setattr(module, "MLProxySession", MagicMock(return_value=proxy))
    session, frontend = frontend_session("ws://unavailable")
    await session.start()
    await session.handle_client_message('{"type":"close"}')
    await asyncio.sleep(0.08)
    proxy.open.assert_awaited_once()
    assert session._reconnect_task.done()
    assert session._watchdog_task.done()
    frontend.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_interim_refreshes_deadline_and_final_clears_it(monkeypatch):
    monkeypatch.setattr(module, "ML_RESPONSE_TIMEOUT_S", 0.12)
    connections = []

    async def ml(ws):
        connections.append(ws)
        await ws.recv()
        await ws.recv()
        for _ in range(5):
            await asyncio.sleep(0.04)
            await ws.send(json.dumps({"type": "transcript_interim", "text": "물", "segment_id": 0}))
        await ws.send(json.dumps({"type": "transcript_final", "text": "물", "is_last": False}))
        await ws.wait_closed()

    async with websockets.serve(ml, "127.0.0.1", 0) as server:
        session, frontend = frontend_session(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        try:
            await session.start()
            await session.handle_client_message(b"\0\0")
            await until(
                lambda: any(c.args[0].get("final") for c in frontend.send_json.call_args_list)
            )
            await asyncio.sleep(0.15)
            assert len(connections) == 1
            assert session._response_pending_since is None
            frontend.close.assert_not_awaited()
        finally:
            await session._close()
