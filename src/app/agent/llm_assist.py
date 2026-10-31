"""Accuracy-mode LLM rerank via PydanticAI + Groq.

Design: rag-strategy.md §6, agent-personalization.md §3.5.
- Trigger: accuracy_mode=True AND final_transcript is not None (gated upstream in ranker.py)
- Input to LLM: top-5 deterministic candidates + patient history (text only) + time_bucket + prefs meta
- Output: Pydantic-enforced top-3 refined candidates
- Timeout: 1.5s via agent.run(timeout=...) native kwarg (pydantic-ai 2.64)
- Retries: {'output': 0} (don't burn budget on auto-retries)
- Fallback: on timeout/exception/429/validation → return deterministic top-3
- 429 circuit breaker: 3 consecutive 429 → disable LLM 60s
- Prompt injection defense: patient_prefix NEVER passed to LLM
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelRateLimitError,
    RunTimedOut,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from app.config import settings

from .contracts import CandidateScore, RankingContext

log = logging.getLogger(__name__)


# ────────── Output schema (strict shape, Groq-safe) ──────────


class RerankItem(BaseModel):
    phrase: str = Field(min_length=1, max_length=50, description="후보 문장 원문")
    rank: int = Field(ge=1, le=3)


class RerankResult(BaseModel):
    top3: list[RerankItem] = Field(min_length=3, max_length=3)


# ────────── Dependencies for @agent.instructions ──────────


@dataclass
class RerankDeps:
    candidates: list[str]            # 5 phrase texts, enumerated
    patient_history: list[str]       # last 10 phrases, newest-first
    time_bucket_kst: str              # morning|afternoon|evening|night
    prefs_meta: list[dict]            # [{content, preference_type}] — no embeddings


# ────────── Module-level singletons ──────────

_http_client: httpx.AsyncClient | None = None
_agent: Agent[RerankDeps, RerankResult] | None = None
_agent_lock = asyncio.Lock()  # protects _get_agent idempotency under concurrent first-callers

# 429 circuit breaker state (module globals — Python GIL protects int/float writes;
# acceptable concurrency contract for hackathon scope)
_consecutive_429 = 0
_disabled_until: float = 0.0


async def _get_agent() -> Agent[RerankDeps, RerankResult] | None:
    """Lazy agent construction. Returns None if GROQ_API_KEY missing (gracefully disables accuracy mode).

    Double-checked locking: fast path returns cached agent; slow path serializes construction
    to prevent orphaned httpx clients under concurrent first-request bursts.
    """
    global _http_client, _agent
    if _agent is not None:
        return _agent
    if not settings.groq_api_key:
        log.warning("GROQ_API_KEY not set — accuracy mode disabled")
        return None
    async with _agent_lock:
        if _agent is not None:
            return _agent
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        provider = GroqProvider(api_key=settings.groq_api_key, http_client=_http_client)
        model = GroqModel(settings.groq_model, provider=provider)
        _agent = Agent(
            model,
            deps_type=RerankDeps,
            output_type=RerankResult,
            retries={"output": 0, "tools": 0},  # 0 output retries protects 1.5s budget
            instructions=(
                "너는 한국 구음장애 환자용 AAC(보완대체의사소통) 보조다. "
                "환자가 방금 ASR로 전사된 문장을 말했고, 결정론적 ranker가 5개 후보를 뽑아줬다. "
                "환자의 과거 발화 패턴·시간대·등록 선호를 반영해 그 중에서 상위 3개를 다시 랭킹하라. "
                "원문 그대로 반환 — 변형·요약·생성 금지. 반드시 입력 5개 중에서만 선택. "
                "응답은 JSON 스키마 그대로."
            ),
        )

        @_agent.instructions
        def _inject_context(ctx) -> str:
            deps: RerankDeps = ctx.deps
            cand_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(deps.candidates))
            hist_block = "\n".join(f"- {h}" for h in deps.patient_history[:10]) or "(없음)"
            prefs_block = "\n".join(
                f"- [{p['preference_type']}] {p['content']}" for p in deps.prefs_meta
            ) or "(없음)"
            return (
                f"[후보 5개 — 반드시 이 중에서만 선택]\n{cand_block}\n\n"
                f"[환자 과거 발화 최근 10건]\n{hist_block}\n\n"
                f"[환자 선호]\n{prefs_block}\n\n"
                f"[현재 시간대] {deps.time_bucket_kst}"
            )

    return _agent


# ────────── 429 circuit breaker ──────────


def _circuit_open() -> bool:
    return time.monotonic() < _disabled_until


def _record_failure_429() -> None:
    global _consecutive_429, _disabled_until
    _consecutive_429 += 1
    if _consecutive_429 >= 3:
        _disabled_until = time.monotonic() + 60.0
        log.warning("LLM 429 circuit opened for 60s (consecutive=%d)", _consecutive_429)


def _record_success() -> None:
    global _consecutive_429, _disabled_until
    _consecutive_429 = 0
    _disabled_until = 0.0


# ────────── Public API ──────────


async def rerank_with_llm(
    ctx: RankingContext,
    top5: list[CandidateScore],
    timeout_s: float = 1.5,
) -> tuple[list[CandidateScore] | None, str | None, int]:
    """Rerank top-5 using LLM. Returns (reranked_top3, None, latency_ms) on success,
    (None, reason, latency_ms) on fallback (None → deterministic path).

    latency_ms is 0 on paths that never call the agent (empty top5, circuit open,
    groq not configured). On paths that call agent.run, it measures wall time.

    Prompt injection defense: ctx.prefix is NEVER passed into the LLM.
    """
    if len(top5) == 0:
        return None, "empty_top5", 0
    if _circuit_open():
        return None, "circuit_open_429", 0

    agent = await _get_agent()
    if agent is None:
        return None, "groq_not_configured", 0

    # Build patient_history: last 10 phrases from ctx.freq_counts keys sorted by last_at desc
    history = sorted(
        [(p, ctx.last_at.get(p)) for p in ctx.freq_counts if ctx.last_at.get(p)],
        key=lambda x: x[1],  # type: ignore[arg-type,return-value]
        reverse=True,
    )
    patient_history = [p for p, _ in history[:10]]

    # Prepare deps — scrub embeddings from prefs
    prefs_meta = [
        {"content": p.get("content") or "", "preference_type": p.get("preference_type") or ""}
        for p in ctx.prefs
    ]

    deps = RerankDeps(
        candidates=[c.phrase for c in top5[:5]],
        patient_history=patient_history,
        time_bucket_kst=ctx.time_bucket_kst,
        prefs_meta=prefs_meta,
    )

    # Prompt-injection defense: ctx.prefix is intentionally excluded from deps.
    # Candidates may share a prefix substring — that is expected and safe.
    # We only forbid raw prefix text as a standalone injected instruction.

    _t0 = time.perf_counter()
    try:
        result = await agent.run(
            "입력된 5개 후보 중에서 환자 맥락에 가장 맞는 3개를 골라 rank 1~3으로 반환하라.",
            deps=deps,
            model_settings={"temperature": 0.1, "max_tokens": 512},
            timeout=timeout_s,
        )
        _record_success()
    except RunTimedOut:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        log.warning("llm_rerank timed out after %.1fs", timeout_s)
        return None, "timeout", _latency_ms
    except ModelRateLimitError:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        # Scrubbed — rate-limit error bodies may include provider internals.
        _record_failure_429()
        log.warning("llm_rerank rate-limited (429)")
        return None, "429_rate_limit", _latency_ms
    except UnexpectedModelBehavior as exc:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        # Debug-level: UnexpectedModelBehavior.__str__ can echo raw model output.
        log.debug("llm_rerank malformed output: %s", exc)
        log.warning("llm_rerank malformed output (type=UnexpectedModelBehavior)")
        return None, "malformed_output", _latency_ms
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        # Scrub exception content — may include API key fragments or sensitive payload.
        err_type = type(exc).__name__
        log.warning("llm_rerank unexpected: %s", err_type)
        return None, f"error:{err_type}", _latency_ms

    _latency_ms = int((time.perf_counter() - _t0) * 1000)

    # Map LLM output back to CandidateScore objects (preserve scoring metadata)
    by_phrase: dict[str, CandidateScore] = {c.phrase: c for c in top5}
    picked: list[CandidateScore] = []
    for item in sorted(result.output.top3, key=lambda x: x.rank):
        orig = by_phrase.get(item.phrase)
        if orig is None:
            # LLM hallucinated a phrase not in top-5. Fallback.
            log.warning("llm_rerank hallucinated phrase: %r", item.phrase)
            return None, "hallucinated_phrase", _latency_ms
        picked.append(orig)

    # Dedup (LLM could return duplicates)
    seen: set[str] = set()
    unique_picked: list[CandidateScore] = []
    for c in picked:
        if c.phrase in seen:
            continue
        seen.add(c.phrase)
        unique_picked.append(c)

    if len(unique_picked) < 3:
        return None, "fewer_than_3_unique", _latency_ms

    return unique_picked[:3], None, _latency_ms


# ────────── A5: 종결어미 보정 특화 함수 (중신뢰 분기 전용) ──────────
#
# 설계 참조: agent-algorithm.md §5.1, agent-rework-plan.md §2 Phase A5.
# 호출 지점: branches/mid.py (0.90 ≤ asr_confidence < 0.98).
# 목적: ASR 종결어미 왜곡 역보정 ("왔었어요" → "왔어요?").
#       어휘·명사·동사 보존, 환각 억제.
# ML 보고서 §49·§50 실측 재현: "전화 왔어요?" / "전화 왔어요." 가
# 전사 "전화 왔었어요" (conf 0.9100 / 0.9477) 로 왜곡된 2건.


class CorrectedEnding(BaseModel):
    """교정 LLM 응답 — strict JSON schema (환각 억제)."""

    corrected: str = Field(min_length=1, max_length=100, description="교정된 phrase")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM 자체 확신도")
    changed: bool = Field(description="종결어미 바뀌었는지 여부 (false면 원본과 동일)")


# ────────── Corrective agent (별도 agent — 다른 output_type) ──────────


@dataclass
class CorrectDeps:
    """종결어미 교정 전용 deps (ctx 전체가 아니라 최소 필드만 주입)."""

    asr_text: str
    patient_recent_phrases: list[str]  # newest-first, 최대 10건


_correct_agent: Agent[CorrectDeps, CorrectedEnding] | None = None
_correct_agent_lock = asyncio.Lock()


async def _get_correct_agent() -> Agent[CorrectDeps, CorrectedEnding] | None:
    """Lazy 생성. GROQ_API_KEY 미설정 시 None (중신뢰 분기는 ASR 원본 fallback)."""
    global _http_client, _correct_agent
    if _correct_agent is not None:
        return _correct_agent
    if not settings.groq_api_key:
        log.warning("GROQ_API_KEY not set — correct_endings disabled")
        return None
    async with _correct_agent_lock:
        if _correct_agent is not None:
            return _correct_agent
        # _http_client 는 rerank agent 와 공유. 아직 미생성이면 생성.
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        provider = GroqProvider(api_key=settings.groq_api_key, http_client=_http_client)
        model = GroqModel(settings.groq_model, provider=provider)
        _correct_agent = Agent(
            model,
            deps_type=CorrectDeps,
            output_type=CorrectedEnding,
            retries={"output": 0, "tools": 0},  # 0.5s 예산 보호
            instructions=(
                "너는 한국 구음장애 환자용 AAC(보완대체의사소통) 서비스의 ASR 보정기다. "
                "환자가 방금 발화한 ASR 전사 결과에서 종결어미만 자연스럽게 보정하라.\n\n"
                "[엄격 규칙]\n"
                "1. 종결어미 (예: ~어요, ~었어요, ~었었어요, ~어요?, ~해줘, ~줘, ~죠, ~네) 만 보정.\n"
                "2. 어휘·명사·동사·조사·구조는 입력에 있는 것만 사용. 추가 생성 금지.\n"
                "3. 환자 최근 발화 패턴과 어울리는 종결어미 선택 (예: 평상시 '~어요?' 사용 → 의문형 유지).\n"
                "4. 보정 결과가 입력과 동일할 수도 있음 (명확 발화는 그대로, changed=false).\n"
                "5. 환각 금지: 입력에 없는 명사·동사 추가 절대 금지. 의미 바꾸기 금지.\n"
                "6. 응답은 JSON 스키마 그대로. 다른 설명·마크다운 금지.\n\n"
                "[Few-shot 예시]\n"
                "예시 1 (ML 보고서 §50 재현 — 의문형 왜곡 역보정):\n"
                '  입력 ASR: "전화 왔었어요"\n'
                '  환자 최근 발화: ["전화 왔어요?", "엄마 전화 왔어요?", "집에 왔어요?"]\n'
                '  응답: {"corrected": "전화 왔어요?", "confidence": 0.92, "changed": true}\n\n'
                "예시 2 (명확 발화 — 보정 불필요):\n"
                '  입력 ASR: "여기까지 걸어 왔어요"\n'
                '  환자 최근 발화: ["걸어 왔어요", "지금 와요"]\n'
                '  응답: {"corrected": "여기까지 걸어 왔어요", "confidence": 0.95, "changed": false}\n'
            ),
        )

        @_correct_agent.instructions
        def _inject_correct_context(ctx) -> str:
            deps: CorrectDeps = ctx.deps
            hist_block = (
                "\n".join(f"- {h}" for h in deps.patient_recent_phrases[:10])
                or "(없음)"
            )
            return (
                f'[ASR 전사]\n"{deps.asr_text}"\n\n'
                f"[환자 최근 발화 10건 (newest first)]\n{hist_block}\n\n"
                f"[지시]\n"
                f"위 ASR 결과의 종결어미만 자연스럽게 보정하라. 어휘·구조는 보존. "
                f"JSON 으로 응답."
            )

    return _correct_agent


async def correct_endings(
    asr_text: str,
    patient_recent_phrases: list[str],
    *,
    timeout_s: float = 0.5,
) -> CorrectedEnding | None:
    """ASR 결과의 종결어미만 자연스럽게 보정 (중신뢰 분기 전용).

    설계 참조: agent-algorithm.md §5.1 (프롬프트 템플릿·few-shot).
    호출 지점: branches/mid.py — ctx 에서 patient_recent_phrases 를 뽑아 호출.

    환각 방지 가드 (4단계):
      1. 프롬프트 레벨 — "입력에 없는 명사·동사 추가 절대 금지" 명시 (System).
      2. 어휘·구조 보존 — 종결어미만 수정 지시 + few-shot 2개로 재강조.
      3. strict JSON schema — Pydantic `CorrectedEnding` 검증
         (corrected/confidence/changed; UnexpectedModelBehavior 로 거부).
      4. timeout 분리 — rerank 1.5s 와 독립적으로 0.5s (D5 결정).
         timeout·parse_error·429·circuit_open 모두 None 반환 →
         branches/mid.py 가 ASR 원본 fallback 처리.

    주의: 본 함수는 **형태소 레벨 후처리 가드** (단어 집합 비교·길이 ±50%) 는
          호출측 (branches/mid.py) 에서 수행한다. 설계 §5.1 환각 가드 4종
          중 (1)(2)(3) 는 호출측 책임. 본 함수는 (4) timeout + schema 만 담당.

    Args:
        asr_text: ASR final transcript (strip 완료 가정, 비어있지 않음).
        patient_recent_phrases: phrase_log 최근 10건 (newest-first).
        timeout_s: LLM 호출 timeout (초). D5 결정 — 중신뢰 분기 전용 0.5s.

    Returns:
        CorrectedEnding: 성공 시 strict-schema 응답.
        None: timeout / parse_error / 429 / circuit_open / groq_not_configured.
              호출측은 ASR 원본으로 fallback.
    """
    text_len = len(asr_text)
    log.info(
        "correct_endings called, text_len=%d, hist_len=%d, timeout_s=%.2f",
        text_len, len(patient_recent_phrases), timeout_s,
    )

    # 입력 가드 (호출측에서 strip 가정이지만 방어)
    if not asr_text or not asr_text.strip():
        log.warning("correct_endings empty asr_text — skip")
        return None

    # 429 circuit breaker 공유 (rerank 와 동일 Groq provider)
    if _circuit_open():
        log.warning("correct_endings skipped — circuit_open_429")
        return None

    agent = await _get_correct_agent()
    if agent is None:
        return None

    deps = CorrectDeps(
        asr_text=asr_text,
        patient_recent_phrases=list(patient_recent_phrases[:10]),
    )

    _t0 = time.perf_counter()
    try:
        result = await agent.run(
            "ASR 결과의 종결어미만 보정해 JSON 으로 응답.",
            deps=deps,
            model_settings={
                "temperature": 0.0,  # 환각 최소
                "max_tokens": 256,
            },
            timeout=timeout_s,
        )
        _record_success()
    except RunTimedOut:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        log.warning(
            "correct_endings timeout after %.2fs (latency_ms=%d)",
            timeout_s, _latency_ms,
        )
        return None
    except ModelRateLimitError:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        _record_failure_429()
        log.warning("correct_endings rate-limited 429 (latency_ms=%d)", _latency_ms)
        return None
    except UnexpectedModelBehavior as exc:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        log.debug("correct_endings malformed output: %s", exc)
        log.warning(
            "correct_endings parse_error (type=UnexpectedModelBehavior, "
            "latency_ms=%d)", _latency_ms,
        )
        return None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        err_type = type(exc).__name__
        log.warning(
            "correct_endings unexpected: %s (latency_ms=%d)",
            err_type, _latency_ms,
        )
        return None

    _latency_ms = int((time.perf_counter() - _t0) * 1000)
    log.info(
        "correct_endings ok, corrected_len=%d, confidence=%.3f, changed=%s, "
        "latency_ms=%d",
        len(result.output.corrected),
        result.output.confidence,
        result.output.changed,
        _latency_ms,
    )
    return result.output


# ─── D1 스캐폴드 — D2+에서 결선 ───────────────────────────────────────────────

try:
    from pydantic_ai.models.anthropic import AnthropicModel  # noqa: F401
    from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: F401
    _anthropic_available_import = True
except ImportError:
    _anthropic_available_import = False

_sonnet_http_client: httpx.AsyncClient | None = None
_sonnet_agent_proactive: Agent | None = None
_sonnet_agent_mid: Agent | None = None
_sonnet_agent_low: Agent | None = None
_sonnet_agent_lock = asyncio.Lock()


def _sonnet_available() -> bool:
    """Anthropic 키·base_url 세팅됐는지."""
    return bool(settings.anthropic_api_key) and bool(settings.accuracy_anthropic_base_url)


async def _get_sonnet_agent(variant: Literal["proactive", "mid", "low"]) -> Agent | None:
    """Branch-specific Agent lazy construction. Returns None if not configured.

    D2+ 에서 각 variant 의 instructions / output_type / @agent.instructions 를 채운다.
    """
    # TODO(D2): construct branch-specific agent per Prompt Designer spec
    #   - instructions= (static system prompt)
    #   - output_type=_LLMProactiveSuggestion / _LLMMidInferenceCandidates / _LLMReconstructedCandidate
    #   - @agent.instructions dynamic context injector
    #   - retries 세팅 — proactive/mid 분기는 {"output": 1, "tools": 0},
    #     low 분기는 {"output": 0, "tools": 0} (latency floor 1.5s tight).
    #     Reason: Sonnet 4.5 is chattier than Llama 3.3 — may emit prose before JSON on
    #     first try. 1 retry at ~400ms still fits 1.5s SLA for proactive/mid.
    #   - model_settings={"temperature": 0.2 (0.1 for low), "max_tokens": 512}
    raise NotImplementedError(f"D2 placeholder: variant={variant}")


# ─── D1 스캐폴드 끝 ───────────────────────────────────────────────────────────
