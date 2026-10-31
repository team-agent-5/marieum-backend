"""pytest unit tests — StreamingPipeline (Phase B2-1).

3건:
  1. test_debounce_merges_rapid_interims  — 짧은 간격 interim 여러 번 → LLM 1번만 호출
  2. test_stale_response_dropped          — 느린 LLM 응답은 version 체크로 drop
  3. test_llm_failure_fallback_to_raw     — LLM 예외 → raw text fallback push
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from app.agent.streaming.pipeline import (
    StreamingPipeline,
    StreamingSessionState,
)
from app.schemas import WSTranscriptOut

# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_state(trace_id: str = "test-trace") -> StreamingSessionState:
    return StreamingSessionState(trace_id=trace_id)


def _make_pipeline(
    state: StreamingSessionState,
    push_cb: Any,
    llm_cb: Any,
    *,
    debounce_ms: int = 150,
    llm_timeout_s: float = 2.0,
) -> StreamingPipeline:
    return StreamingPipeline(
        state=state,
        app_ws_push=push_cb,
        llm_correct=llm_cb,
        debounce_ms=debounce_ms,
        llm_timeout_s=llm_timeout_s,
    )


# ─── Test 1: debounce ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_debounce_merges_rapid_interims() -> None:
    """100ms 간격 interim 3회 → debounce 150ms 후 LLM 1번만 호출, push 1건."""
    state = _make_state()
    pushed: list[WSTranscriptOut] = []

    async def push_cb(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    llm_call_count = 0

    async def llm_cb(raw: str) -> tuple[str, float]:
        nonlocal llm_call_count
        llm_call_count += 1
        return f"교정: {raw}", 0.9

    pipeline = _make_pipeline(state, push_cb, llm_cb, debounce_ms=150)

    # 100ms 간격으로 interim 3회 (마지막 interim 이후 debounce 150ms 기다림)
    await pipeline.handle_interim("믈")
    await asyncio.sleep(0.1)
    await pipeline.handle_interim("믈ㅈ")
    await asyncio.sleep(0.1)
    await pipeline.handle_interim("믈ㅈㅅ")

    # debounce 완료 대기 (마지막 interim + 150ms + 여유 50ms)
    await asyncio.sleep(0.25)

    await pipeline.close()

    assert llm_call_count == 1, f"LLM 호출 횟수 기대값 1, 실제 {llm_call_count}"
    assert [m.text for m in pushed] == ["믈", "믈ㅈ", "믈ㅈㅅ", "교정: 믈ㅈㅅ"]
    assert all(m.final is False for m in pushed)
    assert pushed[-1].trace_id == "test-trace"


# ─── Test 2: stale drop ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stale_response_dropped() -> None:
    """느린 LLM 응답(0.4s)이 돌아오기 전에 handle_final이 더 최신 version 완료 →
    느린 응답은 stale drop, 최종 push는 1건 (final).
    """
    state = _make_state()
    pushed: list[WSTranscriptOut] = []

    async def push_cb(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    call_order: list[int] = []

    async def llm_cb(raw: str) -> tuple[str, float]:
        # 호출 시점의 pending_version을 기록
        call_order.append(state.pending_version)
        if "느린" in raw:
            # interim v=1 호출 — 의도적으로 느리게
            await asyncio.sleep(0.4)
            return "느린 교정", 0.5
        # final v=2 호출 — 빠르게
        return "빠른 최종", 0.95

    pipeline = _make_pipeline(state, push_cb, llm_cb, debounce_ms=10, llm_timeout_s=2.0)

    # interim: debounce 10ms 후 LLM 호출 시작 (느린 호출, 0.4s 걸림)
    await pipeline.handle_interim("느린")
    await asyncio.sleep(0.05)  # debounce 발사 대기

    # final: 즉시 LLM 호출 (빠른 호출, 거의 즉시) — v=2가 v=1보다 먼저 완료
    await pipeline.handle_final("빠른", confidence=0.95)

    # 느린 LLM 응답 완료 대기
    await asyncio.sleep(0.5)

    await pipeline.close()

    # 빠른 final(v=2)이 push됨
    assert [m.text for m in pushed] == ["느린", "빠른 최종"]
    assert pushed[-1].final is True

    # stale drop: last_completed_version은 최소 v=2
    assert state.last_completed_version >= 2


# ─── Test 3: LLM failure fallback ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_failure_fallback_to_raw(caplog: pytest.LogCaptureFixture) -> None:
    """LLM 예외 발생 → raw text 그대로 push, warning 로그 발생."""
    state = _make_state(trace_id="fail-trace")
    pushed: list[WSTranscriptOut] = []

    async def push_cb(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    async def llm_cb(raw: str) -> tuple[str, float]:
        raise RuntimeError("Groq API 500")

    pipeline = _make_pipeline(state, push_cb, llm_cb, debounce_ms=10, llm_timeout_s=1.5)

    with caplog.at_level(logging.WARNING, logger="app.agent.streaming.pipeline"):
        await pipeline.handle_final("아파여", confidence=0.7)

    await pipeline.close()

    # raw text 그대로 push
    assert len(pushed) == 1, f"push 건수 기대값 1, 실제 {len(pushed)}"
    assert pushed[0].text == "아파여", f"raw fallback 기대값 '아파여', 실제 {pushed[0].text!r}"
    assert pushed[0].final is True
    assert pushed[0].trace_id == "fail-trace"

    # warning 로그 발생 확인
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("llm_correction_failed" in m for m in warning_msgs), (
        f"'llm_correction_failed' 로그 없음. 기록된 경고: {warning_msgs}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [0.0, 0.39, 0.40, 0.82, 0.98, 1.0])
async def test_corrected_final_preserves_asr_confidence(confidence: float) -> None:
    pushed = []

    async def correct(raw: str) -> tuple[str, float]:
        assert raw == "믈 조세여"
        return "물 주세요", 0.99

    async def push(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    pipeline = _make_pipeline(_make_state(), push, correct)
    await pipeline.handle_final("믈 조세여", confidence)
    await pipeline.handle_interim("늦은 interim")
    await pipeline.close()
    assert len(pushed) == 1
    assert pushed[0].text == "물 주세요"
    assert pushed[0].confidence == confidence


@pytest.mark.asyncio
async def test_timeout_uses_raw_text_and_asr_confidence() -> None:
    pushed = []
    cancelled = asyncio.Event()

    async def correct(raw: str) -> tuple[str, float]:
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return "사용 안 됨", 1.0

    async def push(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    pipeline = _make_pipeline(_make_state(), push, correct, llm_timeout_s=0.01)
    await pipeline.handle_final("믈", 0.39)
    await pipeline.close()
    assert cancelled.is_set()
    assert pushed[0].text == "믈"
    assert pushed[0].confidence == 0.39


@pytest.mark.asyncio
async def test_close_cancels_pending_correction_without_push() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    pushed = []

    async def correct(raw: str) -> tuple[str, float]:
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return raw, 1.0

    async def push(msg: WSTranscriptOut) -> None:
        pushed.append(msg)

    state = _make_state()
    pipeline = _make_pipeline(state, push, correct, debounce_ms=0)
    await pipeline.handle_interim("취소할 발화")
    await asyncio.wait_for(started.wait(), 1)
    await pipeline.close()
    assert cancelled.is_set()
    assert state.debounce_task.done()
    assert [m.text for m in pushed] == ["취소할 발화"]


@pytest.mark.asyncio
async def test_interim_preview_does_not_wait_for_slow_haiku():
    started = asyncio.Event()
    pushed = []

    async def slow(raw):
        started.set()
        await asyncio.sleep(60)
        return raw, 0.9

    async def push(msg):
        pushed.append(msg)

    pipeline = _make_pipeline(_make_state(), push, slow, debounce_ms=0)
    await pipeline.handle_interim("믈")
    await asyncio.wait_for(started.wait(), 1)
    await pipeline.handle_interim("믈 좀")
    # 첫 LLM 응답이 없더라도 최신 누적 전사는 이미 앱에 도달한다.
    assert [m.text for m in pushed] == ["믈", "믈 좀"]
    assert all(not m.final for m in pushed)
    await pipeline.close()
