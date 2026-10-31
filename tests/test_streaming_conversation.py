"""대화의 종료 주체는 프론트이며 구간 final은 연결을 종료하지 않는다."""

import asyncio
import json
import socket
import uuid

import pytest
import uvicorn
import websockets
from fastapi import FastAPI
from websockets.protocol import State

from app.agent.streaming.ml_proxy import MLProxyEventType, MLProxySession
from app.agent.streaming.router import router
from app.config import settings
from app.security import issue_token


@pytest.mark.asyncio
async def test_ml_conversation_receives_multiple_finals_until_client_end():
    upstream_closed = asyncio.Event()

    async def ml(ws):
        start = json.loads(await ws.recv())
        assert start["mode"] == "conversation"
        for segment, text in enumerate(["물 주세요", "문 열어 주세요"]):
            assert await ws.recv() == b"\0\0" * 1600
            await ws.send(
                json.dumps(
                    {
                        "type": "transcript_final",
                        "text": text,
                        "segment_id": segment,
                        "is_last": False,
                        "confidence": 0.9,
                    }
                )
            )
        assert json.loads(await ws.recv())["type"] == "end"
        await ws.send(
            json.dumps({"type": "transcript_final", "text": "", "segment_id": 2, "is_last": True})
        )
        await ws.wait_closed()
        upstream_closed.set()

    async with websockets.serve(ml, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with MLProxySession("conversation", "synthetic", f"ws://127.0.0.1:{port}") as proxy:
            await proxy.open()
            events = proxy.recv_events()
            for text in ["물 주세요", "문 열어 주세요"]:
                await proxy.send_audio(b"\0\0" * 1600)
                event = await asyncio.wait_for(anext(events), 1)
                assert event.text == text
                assert event.is_final is True
                assert event.is_last is False
                assert not upstream_closed.is_set()
            await proxy.send_end()
            last = await asyncio.wait_for(anext(events), 1)
            assert last.is_last is True
            assert not upstream_closed.is_set()
        await asyncio.wait_for(upstream_closed.wait(), 1)


@pytest.mark.asyncio
async def test_conversation_audio_is_not_cut_off_at_ten_seconds():
    received = bytearray()

    async def ml(ws):
        await ws.recv()  # start
        async for message in ws:
            if isinstance(message, bytes):
                received.extend(message)
            else:
                assert json.loads(message)["type"] == "end"
                await ws.send(json.dumps({"type": "transcript_final", "text": "echo"}))

    async with websockets.serve(ml, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with MLProxySession(
            "long-conversation", "synthetic", f"ws://127.0.0.1:{port}"
        ) as proxy:
            await proxy.open()
            audio = b"\0" * 352_000
            await proxy.send_audio(audio)
            assert not proxy._input_ended
            await proxy.send_end()
            event = await asyncio.wait_for(proxy.recv_event(), 1)
            assert event.text == "echo"
            assert received == audio


@pytest.mark.asyncio
async def test_ml_normal_close_after_last_final_is_not_an_upstream_failure():
    async def ml(ws):
        await ws.recv()  # start
        assert json.loads(await ws.recv())["type"] == "end"
        await ws.send(json.dumps({"type": "transcript_final", "text": "끝", "is_last": True}))

    async with websockets.serve(ml, "127.0.0.1", 0) as upstream:
        port = upstream.sockets[0].getsockname()[1]
        async with MLProxySession("normal-end", "synthetic", f"ws://127.0.0.1:{port}") as proxy:
            await proxy.open()
            await proxy.send_end()
            async with asyncio.timeout(2):
                events = [event async for event in proxy.recv_events()]
            assert [event.type for event in events] == [
                MLProxyEventType.TRANSCRIPT_FINAL,
                MLProxyEventType.CONNECTION_CLOSED,
            ]


@pytest.mark.asyncio
async def test_real_websocket_stays_open_through_idle_multiple_finals_and_end(monkeypatch):
    upstream_ready = asyncio.Event()
    upstream_closed = asyncio.Event()
    corrected_inputs = []

    async def correct(raw):
        corrected_inputs.append(raw)
        return raw, 0.9

    # 실제 HTTP 제공자 대신 교정 함수만 주입하고 양쪽 WebSocket은 실제 소켓으로 검증한다.
    from app.agent.streaming.session import StreamingSession

    class TestSession(StreamingSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, llm_correct=correct)

    monkeypatch.setattr("app.agent.streaming.router.StreamingSession", TestSession)

    async def ml(ws):
        await ws.recv()  # start
        upstream_ready.set()
        segment = 0
        async for message in ws:
            if isinstance(message, bytes):
                text = ["물 주세요", "문 열어 주세요"][segment]
                await ws.send(
                    json.dumps({"type": "transcript_interim", "text": text, "segment_id": segment})
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript_final",
                            "text": text,
                            "segment_id": segment,
                            "confidence": 0.9,
                            "is_last": False,
                        }
                    )
                )
                segment += 1
            else:
                assert json.loads(message)["type"] == "end"
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript_final",
                            "text": "마지막 문장",
                            "segment_id": segment,
                            "confidence": 0.8,
                            "is_last": True,
                        }
                    )
                )
        upstream_closed.set()

    app = FastAPI()
    app.include_router(router)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = None
    try:
        async with websockets.serve(ml, "127.0.0.1", 0) as upstream:
            ml_port = upstream.sockets[0].getsockname()[1]
            monkeypatch.setattr(settings, "ml_asr_ws_url", f"ws://127.0.0.1:{ml_port}")
            task = asyncio.create_task(server.serve(sockets=[listener]))
            async with asyncio.timeout(3):
                while not server.started:
                    if task.done():
                        task.result()
                    await asyncio.sleep(0.01)
            token, _ = issue_token(uuid.uuid4(), "synthetic-conversation")
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/ws/phrase/stream?token={token}"
            ) as ws:
                await asyncio.wait_for(upstream_ready.wait(), 2)
                # 실제 시간으로 기존 10초 타이머와 ML 무응답 30초 제한을 모두 넘긴다.
                await asyncio.sleep(31)
                assert ws.state == State.OPEN
                assert not upstream_closed.is_set()
                for text in ["물 주세요", "문 열어 주세요"]:
                    await ws.send(b"\0\0" * 1600)
                    interim = json.loads(await asyncio.wait_for(ws.recv(), 2))
                    final = json.loads(await asyncio.wait_for(ws.recv(), 2))
                    assert interim["text"] == final["text"] == text
                    assert interim["final"] is False
                    assert final["final"] is True
                    assert final["confidence"] == 0.9
                    assert ws.state == State.OPEN
                    assert not upstream_closed.is_set()
                await ws.send(json.dumps({"type": "end"}))
                final = json.loads(await asyncio.wait_for(ws.recv(), 2))
                assert final["text"] == "마지막 문장"
                assert final["final"] is True
                await asyncio.sleep(2.1)
                assert ws.state == State.OPEN
                assert not upstream_closed.is_set()
                await ws.close()
                await asyncio.wait_for(upstream_closed.wait(), 2)
            assert corrected_inputs == ["물 주세요", "문 열어 주세요", "마지막 문장"]
    finally:
        server.should_exit = True
        if task is not None:
            await asyncio.wait_for(task, 3)
        listener.close()
