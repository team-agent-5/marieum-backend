"""pytest unit tests for agent/streaming/ml_proxy.py — B1-3 명세 2건."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
import websockets
import websockets.server

from app.agent.streaming.ml_proxy import (
    MLProxyEventType,
    MLProxySession,
)

# ─── Mock ASR Server fixture ──────────────────────────────────────────────────


@pytest_asyncio.fixture
async def mock_asr_server():
    """인라인 Mock WS ASR 서버. start 수신 → final 응답."""
    received: dict[str, object] = {"start": None, "bytes": 0, "end": False}

    async def handler(ws: websockets.ServerConnection) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                received["bytes"] = int(received["bytes"]) + len(msg)
            else:
                data = json.loads(msg)
                if data["type"] == "start":
                    received["start"] = data
                elif data["type"] == "end":
                    received["end"] = True
                    await ws.send(
                        json.dumps(
                            {
                                "type": "transcript_final",
                                "trace_id": data["trace_id"],
                                "text": "echo",
                                "confidence": 0.9,
                                "duration_ms": 1000,
                                "model_version": "mock",
                                "no_speech_prob": 0.0,
                                "segment_id": 0,
                                "is_last": True,
                            }
                        )
                    )
                    break

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield received, f"ws://127.0.0.1:{port}/asr-stream"
    server.close()
    await server.wait_closed()


# ─── Test 1: open → start 메시지 전송 확인 ────────────────────────────────────


@pytest.mark.asyncio
async def test_open_success_sends_start(mock_asr_server):
    """open() 호출 시 ML 서버에 start 메시지가 전송되어야 한다."""
    received, url = mock_asr_server

    session = MLProxySession(
        trace_id="test-trace-001",
        patient_id="p-TEST-001",
        ml_ws_url=url,
    )
    await session.open()

    # send_end → final 수신 → close
    await session.send_end()

    # final 이벤트 수신 확인
    evt = await session.recv_event()
    assert evt is not None
    assert evt.type == MLProxyEventType.TRANSCRIPT_FINAL
    assert evt.text == "echo"
    assert evt.is_final is True

    await session.close()

    # start 메시지가 Mock 서버에 전달됐는지 확인
    start_data = received["start"]
    assert start_data is not None, "start 메시지가 ML 서버에 도달하지 않았습니다"
    assert isinstance(start_data, dict)
    assert start_data["type"] == "start"
    assert start_data["trace_id"] == "test-trace-001"
    assert start_data["sample_rate"] == 16000
    assert start_data["encoding"] == "pcm16"
    assert start_data["patient_id"] == "p-TEST-001"
    assert start_data["language"] == "ko"
    assert start_data["mode"] == "conversation"


@pytest.mark.asyncio
async def test_ml_audio_limit_error_is_followed_by_final():
    async def handler(ws):
        start = json.loads(await ws.recv())
        await ws.recv()  # end
        await ws.send(json.dumps({"type": "error", "code": "AUDIO_TOO_LONG"}))
        await ws.send(
            json.dumps(
                {
                    "type": "transcript_final",
                    "trace_id": start["trace_id"],
                    "text": "마지막 문장",
                    "confidence": 0.82,
                    "is_last": True,
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with MLProxySession(
            trace_id="limit-final", patient_id="synthetic", ml_ws_url=f"ws://127.0.0.1:{port}"
        ) as session:
            await session.open()
            await session.send_end()
            events = []
            async for event in session.recv_events():
                events.append(event)
                if event.type == MLProxyEventType.TRANSCRIPT_FINAL:
                    break
    assert [event.type for event in events] == [
        MLProxyEventType.ML_ERROR,
        MLProxyEventType.TRANSCRIPT_FINAL,
    ]
    assert events[-1].text == "마지막 문장"


# ─── Test 2: 10초 초과 → ML_ERROR(AUDIO_TOO_LONG) 이벤트 emit ────────────────


@pytest.mark.asyncio
async def test_audio_too_long_triggers_error(mock_asr_server):
    """320,001 바이트 전송 시 AUDIO_TOO_LONG MLProxyEvent가 emit되어야 한다."""
    received, url = mock_asr_server

    session = MLProxySession(
        trace_id="test-trace-002",
        patient_id="p-TEST-002",
        ml_ws_url=url,
        mode="utterance",
    )
    await session.open()

    # 320,001 바이트 → 10초 상한 초과
    oversized_audio = b"\x00" * 320_001
    await session.send_audio(oversized_audio)

    # AUDIO_TOO_LONG 이벤트가 큐에 들어가야 함
    evt = await session.recv_event()
    assert evt is not None, "AUDIO_TOO_LONG 이벤트가 emit되지 않았습니다"
    assert evt.type == MLProxyEventType.ML_ERROR
    assert evt.error_code == "AUDIO_TOO_LONG"

    # 상한까지의 오디오는 보존하고 end 후 final까지 받아야 한다.
    assert session._closed is False
    final = await session.recv_event()
    assert final.type == MLProxyEventType.TRANSCRIPT_FINAL
    assert received["bytes"] == 320_000
    assert received["end"] is True
    await session.close()
