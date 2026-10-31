"""문자 필터가 의미 있는 숫자/단위를 보존하고 모든 WS 전사 경로를 보호한다."""

import pytest

from app.agent.streaming.pipeline import (
    StreamingPipeline,
    StreamingSessionState,
    sanitize_transcript_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("물 좀 주세요.", "물 좀 주세요."),
        ("**물** __좀__ `주세요` 🙂✨", "물 좀 주세요"),
        ("# 물\n좀\t주세요", "물 좀 주세요"),
        ("물\u200b 좀\u202e 주세요\x00", "물 좀 주세요"),
        ("체온 37.5℃, -5도, 10~20mL, 50%", "체온 37.5℃, -5도, 10~20mL, 50%"),
        ("3.5mg / 2회, ₩1,000", "3.5mg / 2회, ₩1,000"),
        ("안 먹을래요! 정말인가요?", "안 먹을래요! 정말인가요?"),
        ("Café, 김 씨와 A씨", "Café, 김 씨와 A씨"),
        ("1️⃣개", "1개"),
        ("👨‍👩‍👧‍👦🧑🏽‍⚕️", ""),
        ("\u3164\u115f\uffa0", ""),
        ("**...?!**", ""),
        ("   \n\t", ""),
        ("", ""),
    ],
)
def test_transcript_character_policy(text, expected):
    assert sanitize_transcript_text(text) == expected


@pytest.mark.asyncio
async def test_interim_and_corrected_final_are_filtered():
    pushed = []
    received = []

    async def push(message):
        pushed.append(message)

    async def correct(raw):
        received.append(raw)
        return "**물 좀 주세요.**🙂\u200b", 0.99

    pipeline = StreamingPipeline(StreamingSessionState("filtered"), push, correct)
    await pipeline.handle_interim("**물**🙂")
    await pipeline.handle_final("믈 좀 주세요.✨", 0.82)
    await pipeline.close()
    assert received == ["믈 좀 주세요."]
    assert [message.text for message in pushed] == ["물", "물 좀 주세요."]
    assert pushed[-1].confidence == 0.82
    assert pushed[-1].final is True


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", ["symbols", "failure"])
async def test_unusable_correction_uses_filtered_raw(correction):
    pushed = []

    async def push(message):
        pushed.append(message)

    async def correct(raw):
        if correction == "failure":
            raise RuntimeError("provider failed")
        return "✨🙂**\u200b", 0.9

    pipeline = StreamingPipeline(StreamingSessionState("fallback"), push, correct)
    await pipeline.handle_final("**안 먹을래요!**🙂", 0.7)
    await pipeline.close()
    assert len(pushed) == 1
    assert pushed[0].text == "안 먹을래요!"
    assert pushed[0].confidence == 0.7


@pytest.mark.asyncio
async def test_symbols_only_do_not_become_empty_transcripts_or_llm_requests():
    pushed = []

    async def push(message):
        pushed.append(message)

    async def correct(raw):
        pytest.fail("No readable ASR text should reach the LLM")

    pipeline = StreamingPipeline(StreamingSessionState("empty"), push, correct)
    await pipeline.handle_interim("🙂✨\u200b")
    assert pushed == []
    await pipeline.handle_final("🙂✨\u200b", 0.8)
    await pipeline.close()
    assert len(pushed) == 1
    assert pushed[0].type == "error"
    assert pushed[0].error_code == "ASR_LOW_CONFIDENCE"
