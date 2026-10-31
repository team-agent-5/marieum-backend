"""Mock ML ASR WebSocket 서버 — 데모 fallback (B1-1).

URL: ws://127.0.0.1:19001/asr-stream  (env MARIEUM_MOCK_ASR_PORT)
MARIEUM_MOCK_SCENARIO: water(기본) | silent | slow | error_audio_too_long | error_model_unavailable
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import websockets
from websockets import serve
from websockets.asyncio.server import Server, ServerConnection

# ─── 환경변수 ──────────────────────────────────────────────────────────────

HOST = os.getenv("MARIEUM_MOCK_ASR_HOST", "127.0.0.1")
PORT = int(os.getenv("MARIEUM_MOCK_ASR_PORT", "19001"))
SCENARIO = os.getenv("MARIEUM_MOCK_SCENARIO", "water")

# 10 s 상한: 16000 Hz × 2 bytes × 10 s = 320,000 bytes
MAX_SESSION_BYTES = 320_000

# ─── 시나리오 정의 ────────────────────────────────────────────────────────

# (sleep_before_ms, text, stable_text, unstable_text, confidence, audio_end_ms)
_WATER_INTERIMS = [
    (300,  "물",        "물",    "",     0.65, 600),
    (700,  "물 좀",     "물",    "좀",   0.78, 1300),
    (500,  "물 좀 주세", "물 좀", "주세", 0.85, 1800),
]

_WATER_FINAL = dict(
    text="물 좀 주세요",
    confidence=0.98,
    duration_ms=2200,
    model_version="mock-whisper-v1",
    no_speech_prob=0.0,
    segment_id=0,
    is_last=True,
)

_SILENT_FINAL = dict(
    text="",
    confidence=0.0,
    duration_ms=2200,
    model_version="mock-whisper-v1",
    no_speech_prob=1.0,
    segment_id=0,
    is_last=True,
)


def _log(msg: str) -> None:
    print(f"[mock-asr] {msg}", flush=True)


# ─── 핵심 핸들러 ──────────────────────────────────────────────────────────

async def mock_asr_handler(ws: ServerConnection) -> None:
    """발화 1건 = 1 WebSocket 연결 처리."""
    remote = ws.remote_address
    _log(f"client connected: {remote}")

    trace_id: str | None = None
    total_bytes = 0
    seq = 0
    start_time = time.monotonic()
    audio_too_long_sent = False

    try:
        async for raw in ws:
            # ── 바이너리 (오디오 chunk) ──
            if isinstance(raw, bytes):
                chunk_len = len(raw)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                total_bytes += chunk_len
                _log(f"bytes received: {chunk_len} (seq={seq}, elapsed={elapsed_ms}ms, total={total_bytes})")
                seq += 1

                # 10초 초과 가드
                if total_bytes > MAX_SESSION_BYTES and not audio_too_long_sent:
                    audio_too_long_sent = True
                    audio_end_ms = total_bytes * 1000 // 32000
                    if SCENARIO == "error_audio_too_long":
                        await ws.send(json.dumps({
                            "type": "error",
                            "code": "AUDIO_TOO_LONG",
                            "message": (
                                f"오디오 길이 초과: "
                                f"{total_bytes} bytes > {MAX_SESSION_BYTES}"
                            ),
                        }))
                        _log("error sent: AUDIO_TOO_LONG")
                    # ML 스펙: AUDIO_TOO_LONG 뒤 final 자동 연쇄
                    await asyncio.sleep(0.2)
                    await _send_final(
                        ws, trace_id or "", _SILENT_FINAL,
                        extra={"audio_end_ms": audio_end_ms},
                    )
                    _log('sending final: "" (audio_too_long fallback)')
                    await ws.close(1000)
                    _log("closed cleanly code=1000")
                    return

                continue

            # ── 텍스트 (JSON 제어 메시지) ──
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                _log(f"invalid JSON, ignored: {raw[:80]}")
                continue

            kind = msg.get("type")

            # ── start ──
            if kind == "start":
                tid = msg.get("trace_id")
                if not tid:
                    await _send_error(ws, "INVALID_FORMAT", "missing trace_id")
                    await ws.close(1000)
                    return

                trace_id = tid
                sr = msg.get("sample_rate", "?")
                enc = msg.get("encoding", "?")
                mode = msg.get("mode", "?")
                _log(f"client connected: {remote} trace_id={trace_id}")
                _log(f"start: sample_rate={sr}, encoding={enc}, mode={mode}")

                # error_model_unavailable: start 받고 바로 에러
                if SCENARIO == "error_model_unavailable":
                    await _send_error(
                        ws, "MODEL_UNAVAILABLE",
                        "모델 로딩 중입니다. 잠시 후 재시도해 주세요.",
                        trace_id=trace_id,
                    )
                    _log("error sent: MODEL_UNAVAILABLE")
                    await ws.close(1013)
                    _log("closed cleanly code=1013")
                    return

                # 나머지 시나리오는 end 받을 때까지 대기
                continue

            # ── end ──
            elif kind == "end":
                end_tid = msg.get("trace_id")
                if not end_tid:
                    await _send_error(
                        ws, "INVALID_FORMAT", "missing trace_id in end",
                        trace_id=trace_id,
                    )
                    await ws.close(1000)
                    return
                if trace_id and end_tid != trace_id:
                    await _send_error(
                        ws, "INVALID_FORMAT",
                        f"trace_id mismatch: expected {trace_id} got {end_tid}",
                        trace_id=trace_id,
                    )
                    await ws.close(1000)
                    return

                _log("end received")
                await _run_scenario(ws, trace_id or end_tid, total_bytes)
                await ws.close(1000)
                _log("closed cleanly code=1000")
                return

            # ── 알 수 없는 타입 ── (무시, 연결 유지)
            else:
                _log(f"unknown message type={kind!r}, ignored")

    except websockets.ConnectionClosed as exc:
        _log(f"connection closed unexpectedly: code={exc.code}")


async def _run_scenario(
    ws: ServerConnection,
    trace_id: str,
    total_bytes: int,
) -> None:
    """시나리오에 따라 interim + final 송신."""
    multiplier = 2.0 if SCENARIO == "slow" else 1.0

    # water / slow: interim 3회 전송
    if SCENARIO in ("water", "slow"):
        for sleep_ms, text, stable, unstable, conf, _ in _WATER_INTERIMS:
            await asyncio.sleep(sleep_ms / 1000 * multiplier)
            audio_end_ms = total_bytes * 1000 // 32000
            await ws.send(json.dumps({
                "type": "transcript_interim",
                "trace_id": trace_id,
                "text": text,
                "stable_text": stable,
                "unstable_text": unstable,
                "confidence": conf,
                "is_final": False,
                "segment_id": 0,
                "audio_end_ms": audio_end_ms,
            }))
            _log(f"sent interim: {text!r} confidence={conf} audio_end_ms={audio_end_ms}")

    # Whisper 처리 지연 모방
    await asyncio.sleep(0.2)

    # final 선택
    if SCENARIO == "silent":
        final_data = _SILENT_FINAL
    else:
        final_data = _WATER_FINAL

    await _send_final(ws, trace_id, final_data)
    _log(f"sending final: {final_data['text']!r} confidence={final_data['confidence']}")


async def _send_final(
    ws: ServerConnection,
    trace_id: str,
    data: dict,
    extra: dict | None = None,
) -> None:
    payload = {
        "type": "transcript_final",
        "trace_id": trace_id,
        **data,
    }
    if extra:
        payload.update(extra)
    await ws.send(json.dumps(payload))


async def _send_error(
    ws: ServerConnection,
    code: str,
    message: str,
    trace_id: str | None = None,
) -> None:
    payload: dict = {"type": "error", "code": code, "message": message}
    if trace_id:
        payload["trace_id"] = trace_id
    await ws.send(json.dumps(payload))


# ─── 공개 API (pytest fixture) ────────────────────────────────────────────

_server_instance: Server | None = None


async def start_mock_server(
    host: str = HOST,
    port: int = PORT,
) -> Server:
    """pytest-asyncio fixture에서 호출 가능.

    Example::

        @pytest_asyncio.fixture
        async def asr_server():
            server = await start_mock_server(port=19001)
            yield server
            await stop_mock_server(server)
    """
    global _server_instance
    _server_instance = await serve(mock_asr_handler, host, port)
    return _server_instance


async def stop_mock_server(
    server: Server | None = None,
) -> None:
    """서버 종료."""
    target = server or _server_instance
    if target is not None:
        target.close()
        await target.wait_closed()


# ─── standalone 진입점 ────────────────────────────────────────────────────

async def main() -> None:
    _log(f"listening on ws://{HOST}:{PORT}/asr-stream (scenario={SCENARIO})")
    server = await serve(mock_asr_handler, HOST, PORT)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
