"""Anthropic 요청은 MockTransport로 검증하며 실제 API 키나 환자 데이터는 사용하지 않는다."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.agent.streaming.llm_corrector import HaikuCorrector
from app.config import settings


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "streaming_llm_enabled", True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
        ("https://copa.codyssey.kr", "https://copa.codyssey.kr/v1/messages"),
        ("https://copa.codyssey.kr/", "https://copa.codyssey.kr/v1/messages"),
        ("https://copa.codyssey.kr/v1", "https://copa.codyssey.kr/v1/messages"),
    ],
)
async def test_haiku_receives_only_current_utterance(
    configured, monkeypatch, base_url, endpoint
) -> None:
    monkeypatch.setattr(settings, "anthropic_base_url", base_url)

    def handler(request):
        assert request.url == endpoint
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["model"] == settings.anthropic_model
        assert len(body["messages"]) == 1
        assert json.loads(body["messages"][0]["content"]) == {"asr_transcript": "믈 조세여"}
        assert "patient_id" not in body
        assert "metadata" not in body
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [
                    {"type": "text", "text": '{"corrected_text":"물 주세요","confidence":0.9}'}
                ],
            },
        )

    corrector = HaikuCorrector()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    corrector._client = client
    assert await corrector("믈 조세여") == ("물 주세요", 0.9)
    await corrector.aclose()
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"stop_reason": "max_tokens", "content": []},
        {"stop_reason": "refusal", "content": []},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "not json"}]},
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '설명\n```json\n{"corrected_text":"물"}\n```'}],
        },
        {
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": '```json\n{"corrected_text":"물"}\n```\n추가 설명'}
            ],
        },
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"corrected_text":""}'}]},
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"corrected_text":"물","confidence":2}'}],
        },
    ],
)
async def test_invalid_response_rejected_for_raw_fallback(configured, payload) -> None:
    corrector = HaikuCorrector()
    corrector._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    try:
        with pytest.raises((ValueError, ValidationError)):
            await corrector("믈")
    finally:
        await corrector.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["```json", "```"])
async def test_haiku_accepts_complete_json_code_block(configured, fence) -> None:
    # Copa 실제 응답은 올바른 교정 JSON을 코드블록으로 감싸 반환한다.
    payload = {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": fence + '\n{"corrected_text":"물 좀 주세요","confidence":0.95}\n```',
            }
        ],
    }
    corrector = HaikuCorrector()
    corrector._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    try:
        assert await corrector("믈 좀 주세여") == ("물 좀 주세요", 0.95)
    finally:
        await corrector.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_provider_errors_reach_pipeline_fallback(configured, status) -> None:
    corrector = HaikuCorrector()
    corrector._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"error": "test"})
        )
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await corrector("믈")
    finally:
        await corrector.aclose()


@pytest.mark.asyncio
async def test_missing_key_or_disabled_does_not_open_connection(monkeypatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "streaming_llm_enabled", True)
    corrector = HaikuCorrector()
    assert await corrector("믈") == ("믈", None)
    assert corrector._client is None
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "streaming_llm_enabled", False)
    assert await corrector("믈") == ("믈", None)
    assert corrector._client is None
