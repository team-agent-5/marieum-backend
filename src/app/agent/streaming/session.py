"""세션 매니저 — 실시간 대화 모드 (S16) WebSocket 1건 생명주기.

결선 레이어: ML proxy + StreamingPipeline + 앱 WebSocket 사이 조율.

책임:
- trace_id 발급 (UUIDv4, 서버 생성 — 클라 입력 무시)
- MLProxySession 열기·닫기
- StreamingPipeline 생성·주입
- 앱 WS 바이너리 chunk → ML proxy forward
- ML interim/final → pipeline 전달
- 구간 final과 입력 종료(end)를 연결 종료와 분리
- 프론트 close/cancel/연결 해제에 맞춰 양쪽 연결 정리
- 세션 종료 정리 (idempotent)

격리 조건:
- agent/ranking import 금지
- phrase_log INSERT 금지 (M3 — bypass 라우트 전용)
- version_id는 pipeline에서만 사용, 앱에 미노출
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.streaming.ml_proxy import (
    MLProxyConnectionError,
    MLProxyEventType,
    MLProxySession,
)
from app.agent.streaming.pipeline import (
    StreamingPipeline,
    StreamingSessionState,
)
from app.config import settings
from app.schemas import (
    PatientPrincipal,
    WSErrorOut,
    WSTranscriptOut,
)

log = logging.getLogger(__name__)

ML_RESPONSE_TIMEOUT_S = 30.0
ML_RETRY_INITIAL_S = 1.0
ML_RETRY_MAX_S = 8.0
MAX_RECONNECT_BUFFER_BYTES = 320_000


# ─── 세션 ────────────────────────────────────────────────────────────────────


class StreamingSession:
    """실시간 모드 세션 매니저. WS 1건당 1개 instance.

    책임:
    - trace_id 발급 (서버가 UUIDv4 생성, 쿼리 입력은 무시)
    - MLProxySession 열기
    - StreamingPipeline 생성·주입
    - 앱 WS 바이너리 chunk → ML proxy forward
    - ML interim/final → pipeline 전달
    - 구간 final/end 후 연결 유지
    - 프론트 close/cancel/연결 해제 시 자원 정리
    - 세션 종료 정리
    """

    def __init__(
        self,
        app_ws: WebSocket,
        patient_principal: PatientPrincipal,
        ml_ws_url: str,
        llm_correct: Callable[[str], Awaitable[tuple[str, float | None]]] | None = None,
    ) -> None:
        self.app_ws = app_ws
        self.patient_principal = patient_principal
        self.ml_ws_url = ml_ws_url
        # Haiku 구현/프롬프트는 보존한다. 운영 실시간 경로에는 교정기를 연결하지 않는다.
        self.llm_correct = llm_correct

        # 서버가 발급 — 클라 입력 무시
        self.trace_id: str = str(uuid.uuid4())

        self._session_state: StreamingSessionState | None = None
        self._ml_proxy: MLProxySession | None = None
        self._pipeline: StreamingPipeline | None = None
        self._ml_event_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._ml_lock = asyncio.Lock()
        self._response_pending_since: float | None = None
        self._pending_audio = bytearray()
        self._end_pending = False
        self._input_ended = False
        self._ml_available = False
        self._segment_id: int | None = None
        self._closed: bool = False

    # ── 세션 시작 ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """세션 초기화: ML proxy open → 문자 필터 pipeline → 백그라운드 수신."""
        if not self.ml_ws_url.strip():
            await self._push_error(
                "ML_UPSTREAM_UNAVAILABLE", "ML ASR WebSocket 주소가 설정되지 않았습니다"
            )
            return
        log.info(
            "streaming_session.start",
            extra={
                "trace_id": self.trace_id,
                "patient_id": str(self.patient_principal.patient_id),
            },
        )

        self._watchdog_task = asyncio.create_task(self._response_watchdog())

        # 3. ML proxy 열기
        self._ml_proxy = MLProxySession(
            trace_id=self.trace_id,
            patient_id=str(self.patient_principal.patient_id),
            ml_ws_url=self.ml_ws_url,
        )
        try:
            await self._ml_proxy.open()
        except MLProxyConnectionError as e:
            log.error(
                "streaming_session.ml_connect_failed",
                extra={"trace_id": self.trace_id, "error": str(e)},
            )
            await self._push_error("ML_UPSTREAM_UNAVAILABLE", str(e))
            self._schedule_reconnect()
            return

        self._ml_available = True
        await self._select_segment(0)
        self._ml_event_task = asyncio.create_task(self._ml_event_loop())

        log.info(
            "streaming_session.started",
            extra={"trace_id": self.trace_id},
        )

    async def _select_segment(self, segment_id: int) -> None:
        """구간별 final 가드/교정 상태를 분리해 다음 문장의 결과도 처리한다."""
        if self._segment_id == segment_id and self._pipeline is not None:
            return
        if self._pipeline is not None:
            await self._pipeline.close()
        self._segment_id = segment_id
        self._session_state = StreamingSessionState(trace_id=self.trace_id)
        self._pipeline = StreamingPipeline(
            state=self._session_state,
            app_ws_push=self._push_message,
            llm_correct=self.llm_correct,
            llm_timeout_s=settings.anthropic_timeout_sec,
        )

    # ── 클라 메시지 처리 ─────────────────────────────────────────────────────

    async def handle_client_message(self, raw: bytes | str) -> bool:
        """앱에서 받은 메시지 처리. False 반환 시 세션 종료."""
        if self._closed:
            return False

        # 바이너리 → ML forward
        if isinstance(raw, bytes):
            if self._input_ended:
                return True
            async with self._ml_lock:
                if self._ml_proxy is None or not self._ml_available:
                    await self._buffer_audio(raw)
                    self._schedule_reconnect()
                    return True
                try:
                    self._mark_request_pending()
                    await self._ml_proxy.send_audio(raw)
                except Exception:
                    await self._buffer_audio(raw)
                    await self._push_error("ML_UPSTREAM_UNAVAILABLE", "ML 전송 실패. 재연결합니다")
                    self._schedule_reconnect()
            return True

        # JSON 제어 메시지
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return True  # 무시

        if not isinstance(data, dict):
            return True
        mtype = data.get("type")
        if mtype == "end":
            await self._end_input()
            return True  # final을 기다리되 앱/ML 연결은 닫지 않는다.
        elif mtype in ("close", "cancel"):
            await self._close(code=1001 if mtype == "cancel" else 1000)
            return False
        return True

    # ── ML 이벤트 루프 ───────────────────────────────────────────────────────

    async def _ml_event_loop(self) -> None:
        """ML proxy에서 이벤트 꺼내 pipeline에 전달."""
        if self._ml_proxy is None or self._pipeline is None:
            return
        try:
            async for event in self._ml_proxy.recv_events():
                if self._response_pending_since is not None:
                    self._response_pending_since = asyncio.get_running_loop().time()
                if event.type == MLProxyEventType.TRANSCRIPT_INTERIM:
                    await self._select_segment(event.segment_id)
                    await self._pipeline.handle_interim(event.text or "")
                elif event.type == MLProxyEventType.TRANSCRIPT_FINAL:
                    if not self._end_pending or event.is_last:
                        self._response_pending_since = None
                    if event.is_last:
                        self._end_pending = False
                    await self._select_segment(event.segment_id)
                    # end 후 빈 마지막 final은 잔여 음성이 없다는 정상 응답이다.
                    if self._input_ended and event.is_last and not (event.text or "").strip():
                        continue
                    if not (event.text or "").strip() or event.no_speech_prob == 1.0:
                        await self._pipeline.close()
                        await self._push_error(
                            "ASR_LOW_CONFIDENCE", "말소리를 인식하지 못했어요. 다시 말해 주세요."
                        )
                        continue
                    await self._pipeline.handle_final(
                        event.text or "",
                        event.confidence or 0.0,
                    )
                    # 구간 final / is_last 모두 프론트의 연결 종료 요청이 아니다.
                elif event.type == MLProxyEventType.ML_ERROR:
                    await self._push_error(
                        event.error_code or "INTERNAL",
                        event.error_message or "ML 오류",
                    )
                elif event.type == MLProxyEventType.CONNECTION_LOST:
                    self._ml_available = False
                    await self._push_error(
                        "ML_UPSTREAM_UNAVAILABLE",
                        "ML 연결이 끊어졌습니다",
                    )
                    self._schedule_reconnect()
                    break
                elif event.type == MLProxyEventType.CONNECTION_CLOSED:
                    self._ml_available = False
                    self._schedule_reconnect()
                    break  # ML이 정상 종료해도 프론트 연결은 닫지 않는다.
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "streaming_session.ml_event_loop_crashed",
                extra={"trace_id": self.trace_id},
            )
            await self._push_error("INTERNAL", "전사 처리 중 오류가 발생했습니다")
            self._schedule_reconnect()

    def _mark_request_pending(self) -> None:
        # 계속 들어오는 오디오는 첫 미응답 요청의 마감 시간을 연장하지 않는다.
        if self._response_pending_since is None:
            self._response_pending_since = asyncio.get_running_loop().time()

    async def _buffer_audio(self, raw: bytes) -> None:
        if len(self._pending_audio) + len(raw) > MAX_RECONNECT_BUFFER_BYTES:
            await self._push_error(
                "AUDIO_TOO_LONG", "ML 재연결 대기 오디오가 10초를 초과했습니다. 초과분은 버립니다"
            )
            return
        self._pending_audio.extend(raw)

    def _schedule_reconnect(self) -> None:
        if self._closed or not self.ml_ws_url.strip():
            return
        self._ml_available = False
        self._response_pending_since = None
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_ml())

    async def _reconnect_ml(self) -> None:
        """프론트를 유지한 채 ML만 정리하고 1·2·4·8초 간격으로 재시도한다."""
        delay = ML_RETRY_INITIAL_S
        while not self._closed:
            async with self._ml_lock:
                if self._ml_event_task is not None and not self._ml_event_task.done():
                    self._ml_event_task.cancel()
                    await asyncio.gather(self._ml_event_task, return_exceptions=True)
                if self._ml_proxy is not None:
                    await self._ml_proxy.close()
            await asyncio.sleep(delay)
            async with self._ml_lock:
                if self._closed:
                    return
                self._ml_proxy = MLProxySession(
                    trace_id=self.trace_id,
                    patient_id=str(self.patient_principal.patient_id),
                    ml_ws_url=self.ml_ws_url,
                )
                try:
                    await self._ml_proxy.open()
                    # ML 재연결은 segment_id가 다시 0부터 시작한다.
                    self._segment_id = None
                    await self._select_segment(0)
                    if self._pending_audio:
                        self._mark_request_pending()
                        await self._ml_proxy.send_audio(bytes(self._pending_audio))
                        self._pending_audio.clear()
                    if self._end_pending:
                        self._mark_request_pending()
                        await self._ml_proxy.send_end()
                except Exception:
                    log.warning(
                        "streaming_session.ml_retry_failed", extra={"trace_id": self.trace_id}
                    )
                    delay = min(delay * 2, ML_RETRY_MAX_S)
                    continue
                self._ml_available = True
                self._ml_event_task = asyncio.create_task(self._ml_event_loop())
                log.info("streaming_session.ml_reconnected", extra={"trace_id": self.trace_id})
                return

    async def _response_watchdog(self) -> None:
        """오디오/end 전송 후 30초 미응답일 때 ML만 교체한다. 단순 idle은 제외."""
        while not self._closed:
            await asyncio.sleep(min(0.25, ML_RESPONSE_TIMEOUT_S / 4))
            pending = self._response_pending_since
            if (
                self._ml_available
                and pending is not None
                and asyncio.get_running_loop().time() - pending >= ML_RESPONSE_TIMEOUT_S
            ):
                self._schedule_reconnect()
                await self._push_error(
                    "ML_UPSTREAM_UNAVAILABLE", "ML 응답 대기 30초 초과. ML에 다시 연결합니다"
                )

    async def _end_input(self) -> None:
        """end는 입력 종료이며 프론트 연결을 닫지 않는다."""
        if self._input_ended or self._closed:
            return
        self._input_ended = True
        self._end_pending = True
        async with self._ml_lock:
            if self._ml_proxy is not None and self._ml_available:
                try:
                    self._mark_request_pending()
                    await self._ml_proxy.send_end()
                except Exception:
                    await self._push_error("ML_UPSTREAM_UNAVAILABLE", "ML 입력 종료 전달 실패")
                    self._schedule_reconnect()
            else:
                self._schedule_reconnect()

    # ── 메시지 push 유틸 ─────────────────────────────────────────────────────

    async def _push_message(self, msg: WSTranscriptOut | WSErrorOut) -> None:
        """앱 WS로 JSON 전송. version_id는 pipeline에서 사용하므로 여기서 노출 안 됨."""
        if self._closed:
            return
        try:
            await self.app_ws.send_json(msg.model_dump(mode="json"))
        except (WebSocketDisconnect, OSError):
            log.info("streaming_session.client_disconnected", extra={"trace_id": self.trace_id})
        except Exception:
            log.exception(
                "streaming_session.app_ws_send_failed",
                extra={"trace_id": self.trace_id},
            )

    async def _push_error(self, error_code: str, message: str) -> None:
        """에러 메시지 push."""
        msg = WSErrorOut(
            type="error",
            error_code=error_code,  # type: ignore[arg-type]
            message=message,
            trace_id=self.trace_id,
        )
        await self._push_message(msg)

    # ── 세션 종료 ────────────────────────────────────────────────────────────

    async def _close(self, code: int = 1000, reason: str = "") -> None:
        """세션 전체 정리. idempotent."""
        if self._closed:
            return
        self._closed = True

        log.info(
            "streaming_session.closing",
            extra={"trace_id": self.trace_id, "code": code, "reason": reason},
        )

        # 종료를 실행하는 자기 자신은 취소하지 않는다. 형제 task 종료를 기다린다.
        tasks = [
            task
            for task in (self._ml_event_task, self._reconnect_task, self._watchdog_task)
            if task is not None and task is not asyncio.current_task()
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        self._pending_audio.clear()
        self._response_pending_since = None

        # ML proxy close
        if self._ml_proxy is not None:
            await self._ml_proxy.close()

        # Pipeline close
        if self._pipeline is not None:
            await self._pipeline.close()

        # 앱 WS close
        with contextlib.suppress(Exception):
            await self.app_ws.close(code=code, reason=reason)

        log.info(
            "streaming_session.closed",
            extra={"trace_id": self.trace_id, "code": code},
        )
