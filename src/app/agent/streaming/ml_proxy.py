"""ML ASR WebSocket proxy — 실시간 모드 전용.

ML팀 공유 스펙(ws://10.1.2.11:8001/asr-stream, feat/asr-stream)에 정확히 맞춤.
대화 연결에서 여러 구간 final을 받으며 프론트 종료 요청까지 연결을 유지한다.

격리: agent/ranking/ 어디서도 이 모듈을 import하지 않는다.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import websockets
import websockets.exceptions
from websockets.protocol import State

log = logging.getLogger(__name__)

# ─── 상수 ────────────────────────────────────────────────────────────────────

ML_WS_URL = "ws://10.1.2.11:8001/asr-stream"
MAX_SESSION_BYTES = 320_000  # 10초 상한 (PCM16 16kHz mono)


# ─── 예외 ─────────────────────────────────────────────────────────────────────


class MLProxyError(Exception):
    """ml_proxy 모듈 공통."""


class MLProxyConnectionError(MLProxyError):
    """연결 실패."""


# ─── 이벤트 타입 ──────────────────────────────────────────────────────────────


class MLProxyEventType(StrEnum):
    TRANSCRIPT_INTERIM = "transcript_interim"
    TRANSCRIPT_FINAL = "transcript_final"
    ML_ERROR = "ml_error"  # ML이 반환한 에러
    CONNECTION_LOST = "connection_lost"  # 네트워크 끊김
    CONNECTION_CLOSED = "connection_closed"  # 마지막 final 후 ML이 정상 종료


# ─── 이벤트 dataclass ─────────────────────────────────────────────────────────


@dataclass
class MLProxyEvent:
    type: MLProxyEventType
    text: str | None = None
    confidence: float | None = None
    is_final: bool = False
    is_last: bool = False  # 구간 final과 대화 종료를 구분한다.
    stable_text: str | None = None  # transcript_interim 전용
    unstable_text: str | None = None  # transcript_interim 전용
    audio_end_ms: int | None = None  # transcript_interim 전용
    duration_ms: int | None = None  # transcript_final 전용
    model_version: str | None = None  # transcript_final 전용
    no_speech_prob: float | None = None  # transcript_final 전용
    segment_id: int = 0
    error_code: str | None = None  # ml_error / connection_lost
    error_message: str | None = None
    trace_id: str | None = None


# ─── 에러 매핑 ────────────────────────────────────────────────────────────────


def _map_ml_error(ml_code: str) -> str:
    """ML 서버 에러 코드 → 백엔드 WSErrorCode (schemas에 정의된 5종)."""
    mapping: dict[str, str] = {
        "AUDIO_TOO_LONG": "AUDIO_TOO_LONG",
        "INVALID_FORMAT": "ML_UPSTREAM_UNAVAILABLE",
        "MODEL_UNAVAILABLE": "ML_UPSTREAM_UNAVAILABLE",
    }
    return mapping.get(ml_code, "INTERNAL")


# ─── 세션 ─────────────────────────────────────────────────────────────────────


class MLProxySession:
    """ML ASR WebSocket 1건 세션.

    사용 패턴::

        session = MLProxySession(trace_id=..., patient_id=..., ml_ws_url=...)
        await session.open()
        await session.send_audio(chunk)
        await session.send_end()
        async for evt in session.recv_events():
            ...
        await session.close()

    컨텍스트 매니저::

        async with session:
            ...
    """

    def __init__(
        self,
        trace_id: str,
        patient_id: str,
        ml_ws_url: str = ML_WS_URL,
        mode: Literal["conversation", "utterance"] = "conversation",
    ) -> None:
        self.trace_id = trace_id
        self.patient_id = patient_id
        self.ml_ws_url = ml_ws_url
        self.mode = mode
        self._ws: websockets.ClientConnection | None = None
        self._bytes_sent: int = 0
        self._event_queue: asyncio.Queue[MLProxyEvent] = asyncio.Queue(maxsize=100)
        self._recv_task: asyncio.Task[None] | None = None
        self._closed: bool = False
        self._input_ended: bool = False
        self._last_final_received = False

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> MLProxySession:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── open ──────────────────────────────────────────────────────────────────

    async def open(self) -> None:
        """ML WebSocket 연결 열고 start 메시지 전송 + 수신 태스크 시작."""
        try:
            self._ws = await websockets.connect(
                self.ml_ws_url,
                open_timeout=5.0,
                close_timeout=5.0,
                max_size=1_000_000,
                compression=None,
            )
        except (OSError, websockets.exceptions.WebSocketException) as e:
            raise MLProxyConnectionError(f"ML connection failed: {e}") from e

        start_msg: dict[str, object] = {
            "type": "start",
            "trace_id": self.trace_id,
            "sample_rate": 16000,
            "encoding": "pcm16",
            "patient_id": self.patient_id,
            "language": "ko",
            "mode": self.mode,
        }
        await self._ws.send(json.dumps(start_msg))
        log.info("ml_proxy.start.sent", extra={"trace_id": self.trace_id})

        self._recv_task = asyncio.create_task(self._recv_loop())
        log.info(
            "ml_proxy.connected",
            extra={"trace_id": self.trace_id, "url": self.ml_ws_url},
        )

    # ── 송신 ─────────────────────────────────────────────────────────────────

    async def send_audio(self, chunk: bytes) -> None:
        """오디오 forward. 발화 모드만 10초 상한, conversation은 ML이 구간을 나눈다."""
        if self._closed or self._ws is None:
            raise MLProxyError("Session not open")
        if self._input_ended:
            return

        if self.mode == "conversation":
            await self._ws.send(chunk)
            self._bytes_sent += len(chunk)
            return

        remaining = MAX_SESSION_BYTES - self._bytes_sent
        allowed_chunk = chunk[:remaining]
        if allowed_chunk:
            await self._ws.send(allowed_chunk)
            self._bytes_sent += len(allowed_chunk)
        if len(chunk) > remaining:
            log.warning(
                "ml_proxy.audio_cap_reached",
                extra={"trace_id": self.trace_id, "bytes_sent": self._bytes_sent},
            )
            await self._event_queue.put(
                MLProxyEvent(
                    type=MLProxyEventType.ML_ERROR,
                    error_code="AUDIO_TOO_LONG",
                    error_message="녹음 10초 초과",
                    trace_id=self.trace_id,
                )
            )
            # 유효한 10초 오디오의 final을 잃지 않도록 end 후 수신을 유지한다.
            await self.send_end()

    async def send_end(self) -> None:
        """end 메시지 전송."""
        if self._closed or self._ws is None or self._input_ended:
            return
        self._input_ended = True
        end_msg = {"type": "end", "trace_id": self.trace_id}
        await self._ws.send(json.dumps(end_msg))
        log.info("ml_proxy.end.sent", extra={"trace_id": self.trace_id})

    # ── 수신 ─────────────────────────────────────────────────────────────────

    async def recv_event(self) -> MLProxyEvent | None:
        """다음 ML 이벤트 수신. None이면 세션 종료."""
        if self._closed and self._event_queue.empty():
            return None
        # 무음 구간에는 전사가 오지 않는다. 수신 간격으로 대화를 종료하지 않는다.
        return await self._event_queue.get()

    async def recv_events(self):  # type: ignore[return]
        """ML 이벤트 async generator. 세션 종료 시 자동 중단."""
        while True:
            evt = await self.recv_event()
            if evt is None:
                return
            yield evt
            if evt.type in (MLProxyEventType.CONNECTION_LOST, MLProxyEventType.CONNECTION_CLOSED):
                return

    # ── 내부 수신 루프 ────────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        """ML 메시지 수신 → 이벤트 queue에 push."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    # ML은 바이너리 응답 안 보냄
                    continue

                try:
                    data: dict[str, object] = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                evt_type = data.get("type")

                if evt_type == "transcript_interim":
                    await self._event_queue.put(
                        MLProxyEvent(
                            type=MLProxyEventType.TRANSCRIPT_INTERIM,
                            text=str(data.get("text", "")),
                            stable_text=str(data["stable_text"])
                            if data.get("stable_text") is not None
                            else None,
                            unstable_text=str(data["unstable_text"])
                            if data.get("unstable_text") is not None
                            else None,
                            confidence=float(data["confidence"])
                            if data.get("confidence") is not None
                            else None,  # type: ignore[arg-type]
                            is_final=False,
                            audio_end_ms=int(data["audio_end_ms"])
                            if data.get("audio_end_ms") is not None
                            else None,  # type: ignore[arg-type]
                            segment_id=int(data.get("segment_id", 0)),  # type: ignore[arg-type]
                            trace_id=str(data.get("trace_id", self.trace_id)),
                        )
                    )

                elif evt_type == "transcript_final":
                    self._last_final_received = bool(data.get("is_last", True))
                    await self._event_queue.put(
                        MLProxyEvent(
                            type=MLProxyEventType.TRANSCRIPT_FINAL,
                            text=str(data.get("text", "")),
                            confidence=float(data["confidence"])
                            if data.get("confidence") is not None
                            else None,  # type: ignore[arg-type]
                            is_final=True,
                            is_last=self._last_final_received,
                            duration_ms=int(data["duration_ms"])
                            if data.get("duration_ms") is not None
                            else None,  # type: ignore[arg-type]
                            model_version=str(data["model_version"])
                            if data.get("model_version") is not None
                            else None,
                            no_speech_prob=float(data["no_speech_prob"])
                            if data.get("no_speech_prob") is not None
                            else None,  # type: ignore[arg-type]
                            segment_id=int(data.get("segment_id", 0)),  # type: ignore[arg-type]
                            trace_id=str(data.get("trace_id", self.trace_id)),
                        )
                    )
                    # 구간 final 후 다음 구간을 계속 받는다. 종료는 앱 요청이 결정한다.

                elif evt_type == "error":
                    code = str(data.get("code", "INTERNAL"))
                    mapped = _map_ml_error(code)
                    log.warning(
                        "ml_proxy.recv.error",
                        extra={
                            "trace_id": self.trace_id,
                            "code": code,
                            "mapped": mapped,
                        },
                    )
                    await self._event_queue.put(
                        MLProxyEvent(
                            type=MLProxyEventType.ML_ERROR,
                            error_code=mapped,
                            error_message=str(data.get("message", "")),
                            trace_id=self.trace_id,
                        )
                    )
                    if code == "AUDIO_TOO_LONG":
                        # ML은 이 오류 다음에 자동 final을 보낸다 (HANDOFF §6).
                        self._input_ended = True
                        continue
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # 정상 close도 포함해 실제 ML 연결 종료를 알리되, 앱 WS는 닫지 않는다.
            if not self._closed:
                if self._last_final_received and self._ws.close_code == 1000:
                    await self._event_queue.put(
                        MLProxyEvent(
                            type=MLProxyEventType.CONNECTION_CLOSED,
                            trace_id=self.trace_id,
                        )
                    )
                else:
                    log.warning(
                        "ml_proxy.connection_lost",
                        extra={"trace_id": self.trace_id},
                    )
                    await self._event_queue.put(
                        MLProxyEvent(
                            type=MLProxyEventType.CONNECTION_LOST,
                            error_code="ML_UPSTREAM_UNAVAILABLE",
                            error_message="ML 연결이 끊어졌습니다",
                            trace_id=self.trace_id,
                        )
                    )

    # ── close ─────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """정리. 재사용 불가."""
        if self._closed:
            return
        self._closed = True

        if self._ws is not None and self._ws.state != State.CLOSED:
            with contextlib.suppress(Exception):
                await self._ws.close(code=1000)

        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task

        log.info(
            "ml_proxy.closed",
            extra={
                "trace_id": self.trace_id,
                "bytes_sent": self._bytes_sent,
            },
        )
