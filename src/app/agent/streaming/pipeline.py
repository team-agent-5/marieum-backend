"""실시간 전사 파이프라인 — 실시간 대화 모드 (S16) 전용.

운영 기본 경로는 ML interim/final → 문자 필터 → 앱 WebSocket push다.
교정기를 명시적으로 주입할 때의 debounce/stale/fallback 구현은 보존한다.

격리 조건: agent/ranking/ import 금지 (ruff ban 대상).
환자 문맥 조회·DB 읽기/쓰기 금지. 현재 발화만 Haiku에 전달한다.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.schemas import WSErrorOut, WSTranscriptOut

log = logging.getLogger(__name__)

_TRANSCRIPT_PUNCTUATION = frozenset(".,?!:;'\"()[]/-+%=<>~°℃℉…·\u2013—“”\u2018\u2019≤≥\u00d7÷&")


def sanitize_transcript_text(text: str) -> str:
    """발화 문자·숫자·단위·문장부호를 보존하고 장식/비가시 문자를 제거한다."""
    cleaned = []
    for char in unicodedata.normalize("NFC", text):
        if char.isspace():
            cleaned.append(" ")
            continue
        codepoint = ord(char)
        if (
            0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
            or char in "\u20e3\u115f\u1160\u3164\uffa0"
        ):
            continue
        category = unicodedata.category(char)
        if category[0] in "LNM" or category == "Sc" or char in _TRANSCRIPT_PUNCTUATION:
            cleaned.append(char)
    result = " ".join("".join(cleaned).split())
    return result if any(char.isalnum() for char in result) else ""


# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class StreamingSessionState:
    """실시간 대화 모드 WebSocket 1 세션의 메모리 상태.

    주의:
    - ranking agent에서 import하지 않는다.
    - 세션 종료 시 debounce_task를 cancel해 메모리 누수 방지.
    - Redis·DB에 저장하지 않는다 (§2.2).
    """

    trace_id: str
    last_corrected: str = ""
    """WS로 마지막으로 push한 문자열 (ASR 미리보기 또는 교정본)."""
    last_raw_text: str = ""
    """가장 최근 ASR cumulative text (debounce 발사 시 전송할 raw)."""
    pending_version: int = 0
    """다음 LLM 호출에 부여할 version (단조 증가)."""
    last_completed_version: int = 0
    """가장 최근 완료 교정의 version (stale drop 기준)."""
    debounce_task: asyncio.Task | None = None
    """지연 호출 핸들 (debounce). None이면 예약 없음."""


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class StreamingPipeline:
    """ML interim 수신 → debounce → LLM 교정 → 앱 push.

    세션 단위 instance (WebSocket 1건당 1개).

    ``llm_correct`` 는 외부에서 주입받는다 (dependency injection).
    실제 Anthropic Haiku 구현은 ``agent/streaming/llm_corrector.py``.
    """

    def __init__(
        self,
        state: StreamingSessionState,
        app_ws_push: Callable[[WSTranscriptOut | WSErrorOut], Awaitable[None]],
        llm_correct: Callable[[str], Awaitable[tuple[str, float | None]]] | None = None,
        *,
        debounce_ms: int = 150,
        llm_timeout_s: float = 1.5,
    ) -> None:
        """
        Args:
            state: 세션 상태 (per-WebSocket).
            app_ws_push: WS push 콜백. ``WSTranscriptOut`` 또는 ``WSErrorOut`` 를 받는다.
            llm_correct: ``(raw_text)``
                → ``(corrected_text, confidence)`` 코루틴. 실패·타임아웃 처리는 pipeline 책임.
            debounce_ms: interim debounce 창 (ms). 기본 150ms.
            llm_timeout_s: LLM 호출 최대 허용 시간 (s). 기본 1.5s.
        """
        self.state = state
        self.app_ws_push = app_ws_push
        self.llm_correct = llm_correct
        self.debounce_ms = debounce_ms
        self.llm_timeout_s = llm_timeout_s
        self._closed = False
        self._final_received = False

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    async def handle_interim(self, text: str) -> None:
        """ML ``transcript_interim`` 수신 시 호출.

        text는 앞 단어를 포함한 현재 발화 전체(cumulative)다. ASR 수정도 반영하도록
        이전 text에 이어 붙이지 않고 최신 전체 text로 교체한다.
        150ms debounce 창으로 뭉쳐서 LLM 교정을 호출한다.
        새 interim이 들어오면 기존 debounce 태스크를 cancel하고 재예약.
        """
        if self._closed or self._final_received:
            return
        text = sanitize_transcript_text(text)
        if not text:
            return
        self.state.last_raw_text = text
        self.state.pending_version += 1
        my_version = self.state.pending_version
        # ML 간격보다 LLM 응답이 느려도 발화 중 화면은 계속 갱신한다.
        self.state.last_corrected = text
        await self.app_ws_push(
            WSTranscriptOut(type="transcript", text=text, final=False, trace_id=self.state.trace_id)
        )

        log.debug(
            "handle_interim: version=%d text=%.30r",
            my_version,
            text,
        )

        # 기존 debounce 태스크 cancel
        if self.state.debounce_task is not None and not self.state.debounce_task.done():
            self.state.debounce_task.cancel()

        if self.llm_correct is None:
            return  # 교정기 미연결: ASR 미리보기만 전달하고 LLM 작업을 만들지 않는다.

        async def _delayed_correct() -> None:
            try:
                await asyncio.sleep(self.debounce_ms / 1000)
            except asyncio.CancelledError:
                return
            await self._correct_and_push(my_version, is_final=False)

        self.state.debounce_task = asyncio.create_task(_delayed_correct())

    async def handle_final(self, text: str, confidence: float) -> None:
        """ML ``transcript_final`` 수신 시 호출.

        debounce 생략 — 즉시 LLM 교정 호출 (§8.5).
        """
        if self._closed or self._final_received:
            return
        text = sanitize_transcript_text(text)
        self.state.last_raw_text = text
        self.state.pending_version += 1
        my_version = self.state.pending_version

        log.debug(
            "handle_final: version=%d text=%.30r confidence=%.2f",
            my_version,
            text,
            confidence,
        )

        # 기존 debounce cancel
        if self.state.debounce_task is not None and not self.state.debounce_task.done():
            self.state.debounce_task.cancel()

        self._final_received = True
        if not text:
            await self.app_ws_push(
                WSErrorOut(
                    type="error",
                    error_code="ASR_LOW_CONFIDENCE",
                    message="말소리를 인식하지 못했어요. 다시 말해 주세요.",
                    trace_id=self.state.trace_id,
                )
            )
            return
        await self._correct_and_push(my_version, is_final=True, raw_confidence=confidence)

    async def close(self) -> None:
        """정리. debounce 태스크 cancel.

        WS close 핸들러에서 반드시 호출해야 메모리 누수 방지.
        """
        self._closed = True
        task = self.state.debounce_task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        log.debug("StreamingPipeline.close: trace_id=%s", self.state.trace_id)

    # ── 내부 ──────────────────────────────────────────────────────────────────

    async def _correct_and_push(
        self,
        my_version: int,
        is_final: bool,
        raw_confidence: float | None = None,
    ) -> None:
        """LLM 교정 호출 + stale 체크 + 앱 push.

        두 지점에서 stale 체크:
        1. LLM 호출 전 (빠른 포기)
        2. LLM 응답 수신 후 (느린 응답 drop)
        """
        # 빠른 stale 포기 (호출 전)
        if (
            self._closed
            or my_version != self.state.pending_version
            or my_version <= self.state.last_completed_version
        ):
            log.debug(
                "stale_pre_llm: version=%d last_completed=%d (skipped LLM call)",
                my_version,
                self.state.last_completed_version,
            )
            return

        raw = self.state.last_raw_text  # race 방지 로컬 스냅샷

        log.debug(
            "_correct_and_push: version=%d is_final=%s raw=%.30r",
            my_version,
            is_final,
            raw,
        )

        # LLM 교정 호출 (실패 시 raw fallback — §7)
        corrected = raw
        if self.llm_correct is not None:
            try:
                corrected, llm_confidence = await asyncio.wait_for(
                    self.llm_correct(raw),
                    timeout=self.llm_timeout_s,
                )
                log.debug(
                    "streaming_correction_completed: version=%d llm_confidence=%s",
                    my_version,
                    llm_confidence,
                )
            except Exception as exc:
                # 교정기를 명시적으로 연결한 경로에서도 실패 시 실제 ASR로 복구한다.
                log.warning(
                    "llm_correction_failed: version=%d trace_id=%s error_type=%s",
                    my_version,
                    self.state.trace_id,
                    type(exc).__name__,
                )

        # LLM 응답과 원문 fallback 모두 같은 문자 정책으로 앱에 전달한다.
        corrected = sanitize_transcript_text(corrected) or raw

        # stale 체크 — 느린 LLM 응답이 더 최신 결과를 덮지 않도록 (§8.2)
        if (
            self._closed
            or my_version != self.state.pending_version
            or my_version <= self.state.last_completed_version
        ):
            log.debug(
                "stale_version_dropped: version=%d last_completed=%d",
                my_version,
                self.state.last_completed_version,
            )
            return

        # state 갱신
        previous_text = self.state.last_corrected
        self.state.last_completed_version = my_version
        self.state.last_corrected = corrected
        if not is_final and corrected == previous_text:
            return  # 원문 미리보기와 동일한 중간 결과를 중복 전송하지 않는다.

        # 앱 push
        msg = WSTranscriptOut(
            type="transcript",
            text=corrected,
            final=is_final,
            trace_id=self.state.trace_id,
            confidence=raw_confidence if is_final else None,
        )
        await self.app_ws_push(msg)

        log.debug(
            "push_done: version=%d is_final=%s corrected=%.30r",
            my_version,
            is_final,
            corrected,
        )
