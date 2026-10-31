"""
Groq LLM 클라이언트. OpenAI SDK로 Groq OpenAI-compatible endpoint 사용.
구조화 출력(JSON mode) + 타임아웃 가드레일.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings

log = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_llm() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


async def classify_intent_llm(text: str, candidate_intents: list[str]) -> tuple[str, float]:
    """
    L3 분류. 반드시 {intent: str, confidence: float} JSON 반환.
    실패하면 ("other", 0.0).
    """
    client = get_llm()
    prompt = (
        "너는 돌봄 요청 분류기다. 환자의 발화를 아래 의도 중 하나로 분류하라.\n"
        f"후보: {', '.join(candidate_intents)}\n"
        f'발화: "{text}"\n'
        '응답은 JSON으로: {"intent": "...", "confidence": 0.0~1.0}'
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.groq_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=80,
            timeout=settings.groq_timeout_sec,
            messages=[
                {"role": "system", "content": "You are a Korean dysarthria intent classifier."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        obj: dict[str, Any] = json.loads(raw)
        intent = str(obj.get("intent", "other"))
        if intent not in candidate_intents:
            intent = "other"
        conf = float(obj.get("confidence", 0.0))
        return intent, max(0.0, min(1.0, conf))
    except Exception as exc:
        log.warning("Groq intent 분류 실패: %s", exc)
        return "other", 0.0


async def generate_explanation(
    *,
    intent: str,
    priority: int,
    high_stakes: list[str],
    assignee_name: str,
    assignee_role: str,
    patient_name: str,
    patient_disease: str,
    history_summary: str,
    patient_notes: list[str],
    clinical_refs: list[str],
) -> str:
    """
    추천 카드의 '왜 이 추천인가' 1~2 문장 생성.
    의료 조언 금지 — 설명 뿐.
    """
    client = get_llm()
    notes_block = "\n- ".join(patient_notes) if patient_notes else "(없음)"
    refs_block = "\n- ".join(clinical_refs) if clinical_refs else "(없음)"
    hs_block = ", ".join(high_stakes) if high_stakes else "(없음)"

    system = (
        "너는 요양 돌봄 보조 AI다. 돌봄인력에게 '왜 이 추천인가'를 한국어 1~2문장으로 설명한다. "
        "진단·치료·투약 결정·중증도 평가를 절대 하지 말고, 참조만 요약하라."
    )
    user = (
        f"환자: {patient_name} ({patient_disease})\n"
        f"의도: {intent}, 긴급도: {priority}, High-stakes: {hs_block}\n"
        f"추천 담당자: {assignee_name} ({assignee_role})\n"
        f"환자 24h 요약: {history_summary}\n"
        f"[환자 특이사항 메모]\n- {notes_block}\n"
        f"[참조 임상 가이드]\n- {refs_block}\n"
        "위 정보를 바탕으로 설명 1~2문장만 작성."
    )
    resp = await client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.2,
        max_tokens=200,
        timeout=settings.groq_timeout_sec,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
