"""실제 앱/ML WebSocket에서 Haiku 없이 누적 전사와 final을 전달한다."""

from __future__ import annotations

import asyncio
import json
import socket
import uuid
from unittest.mock import MagicMock

import pytest
import uvicorn
import websockets
from fastapi import FastAPI

from app.agent.streaming.router import router
from app.config import settings
from app.security import issue_token


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raws",
    [
        ["믈", "믈 좀", "믈 좀 주세여"],
        ["물을", "불을", "불을 꺼 주세여"],
    ],
)
async def test_cumulative_ml_transcripts_reach_app_without_haiku(monkeypatch, raws) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "streaming_llm_enabled", True)
    ml_inputs = []
    haiku = MagicMock(side_effect=AssertionError("Haiku must not be connected"))
    monkeypatch.setattr("app.agent.streaming.llm_corrector.HaikuCorrector", haiku)

    async def ml_handler(ws) -> None:
        start = json.loads(await ws.recv())
        ml_inputs.append(start)
        for raw in raws[:2]:
            audio = await ws.recv()
            assert isinstance(audio, bytes)
            assert audio == b"\x00\x00" * 1600  # 프론트의 100ms PCM16 3,200B를 보존한다.
            ml_inputs.append(audio)
            await ws.send(
                json.dumps(
                    {
                        "type": "transcript_interim",
                        "text": raw,
                        "stable_text": "",
                        "unstable_text": raw,
                        "trace_id": start["trace_id"],
                    }
                )
            )
        end = json.loads(await ws.recv())
        assert end["type"] == "end"
        await ws.send(
            json.dumps(
                {
                    "type": "transcript_final",
                    "text": raws[-1],
                    "confidence": 0.82,
                    "is_last": True,
                    "trace_id": start["trace_id"],
                }
            )
        )
        await ws.wait_closed()

    app = FastAPI()
    app.include_router(router)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    server_task = None
    outputs = []
    try:
        async with websockets.serve(ml_handler, "127.0.0.1", 0) as ml_server:
            ml_port = ml_server.sockets[0].getsockname()[1]
            monkeypatch.setattr(settings, "ml_asr_ws_url", f"ws://127.0.0.1:{ml_port}/asr-stream")
            server_task = asyncio.create_task(server.serve(sockets=[listener]))
            async with asyncio.timeout(3):
                while not server.started:
                    if server_task.done():
                        server_task.result()
                    await asyncio.sleep(0.01)
            token, _ = issue_token(uuid.uuid4(), "test")
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/ws/phrase/stream?token={token}"
            ) as ws:
                for raw in raws[:2]:
                    await ws.send(b"\x00\x00" * 1600)
                    outputs.append(json.loads(await asyncio.wait_for(ws.recv(), 3)))
                    assert outputs[-1]["text"] == raw
                await ws.send(json.dumps({"type": "end"}))
                outputs.append(json.loads(await asyncio.wait_for(ws.recv(), 3)))
                await ws.send(json.dumps({"type": "close"}))
                await asyncio.wait_for(ws.wait_closed(), 3)
                assert ws.close_code == 1000
    finally:
        server.should_exit = True
        if server_task is not None:
            await asyncio.wait_for(server_task, 3)
        listener.close()

    haiku.assert_not_called()
    assert [o["text"] for o in outputs] == raws
    assert [o["final"] for o in outputs] == [False] * (len(outputs) - 1) + [True]
    assert [o["confidence"] for o in outputs] == [None] * (len(outputs) - 1) + [0.82]
    assert all(o["trace_id"] == ml_inputs[0]["trace_id"] for o in outputs)
