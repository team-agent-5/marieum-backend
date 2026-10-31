"""D2 Accuracy Agent test suite.

Group A — Unit tests (no DB, deterministic logic only)
Group B — Integration E2E (live Postgres + seed data, @pytest.mark.integration)
Group C — Fallback chain tests (mock)
Group D — ruff + import checks (preconditions)
"""
from __future__ import annotations

import subprocess
import unicodedata
from datetime import UTC, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Group A — Unit tests (no DB, deterministic logic only)
# ══════════════════════════════════════════════════════════════════════════════


class TestClassifyBranchBoundaries:
    """A1: classify_branch boundary values."""

    def _branch(self, val):
        from app.agent.confidence_router import classify_branch
        return classify_branch(val)

    def test_zero_is_retry(self):
        assert self._branch(0.0) == "retry"

    def test_0_39_is_retry(self):
        assert self._branch(0.39) == "retry"

    def test_0_40_is_low(self):
        assert self._branch(0.40) == "low"

    def test_0_89_is_low(self):
        assert self._branch(0.89) == "low"

    def test_0_90_is_mid(self):
        assert self._branch(0.90) == "mid"

    def test_0_97_is_mid(self):
        assert self._branch(0.97) == "mid"

    def test_0_98_is_high(self):
        assert self._branch(0.98) == "high"

    def test_1_0_is_high(self):
        assert self._branch(1.0) == "high"

    def test_none_is_high_d3_fallback(self):
        """None → high (D3 fallback design decision)."""
        assert self._branch(None) == "high"


class TestTimeBucketComputation:
    """A2: _compute_time_bucket KST hour → bucket string."""

    def _bucket(self, kst_hour: int) -> str:
        from app.agent.orchestrator import _compute_time_bucket

        # Build a UTC datetime whose KST conversion gives the desired hour
        # KST = UTC + 9h; so UTC_hour = kst_hour - 9 (mod 24)
        utc_hour = (kst_hour - 9) % 24
        # Use a fixed date; set tzinfo=UTC so astimezone works correctly
        dt = datetime(2026, 6, 15, utc_hour, 0, 0, tzinfo=UTC)
        return _compute_time_bucket(dt)

    def test_06_morning(self):
        assert self._bucket(6) == "morning"

    def test_12_afternoon(self):
        assert self._bucket(12) == "afternoon"

    def test_18_evening(self):
        assert self._bucket(18) == "evening"

    def test_23_night(self):
        assert self._bucket(23) == "night"

    def test_0559_night(self):
        """05:59 KST → night (before morning starts at 06:00)."""
        from app.agent.orchestrator import _compute_time_bucket

        # 05:59 KST = 20:59 UTC previous day
        dt = datetime(2026, 6, 15, 20, 59, 0, tzinfo=UTC)
        assert _compute_time_bucket(dt) == "night"

    def test_1159_morning(self):
        """11:59 KST → morning (morning ends at 12:00)."""
        from app.agent.orchestrator import _compute_time_bucket

        # 11:59 KST = 02:59 UTC
        dt = datetime(2026, 6, 15, 2, 59, 0, tzinfo=UTC)
        assert _compute_time_bucket(dt) == "morning"


class TestTimeBasedFallbackShapes:
    """A3: fallback_candidates() returns correctly-sized lists."""

    def test_morning_limit_3(self):
        from app.services.time_based_fallback import fallback_candidates
        result = fallback_candidates("morning", limit=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_evening_limit_1(self):
        from app.services.time_based_fallback import fallback_candidates
        result = fallback_candidates("evening", limit=1)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_all_four_buckets_non_empty(self):
        from app.services.time_based_fallback import fallback_candidates
        for bucket in ("morning", "afternoon", "evening", "night"):
            result = fallback_candidates(bucket, limit=3)  # type: ignore[arg-type]
            assert len(result) >= 1, f"bucket={bucket!r} returned empty list"

    def test_morning_items_are_strings(self):
        from app.services.time_based_fallback import fallback_candidates
        result = fallback_candidates("morning", limit=3)
        for item in result:
            assert isinstance(item, str) and item, f"Expected non-empty str, got {item!r}"


class TestNFCNormalizeDedup:
    """A4: NFC normalization and dedup in rag_retriever._nfc."""

    def _nfc(self, s: str) -> str:
        from app.agent.rag_retriever import _nfc
        return _nfc(s)

    def test_nfd_vs_nfc_same_after_normalize(self):
        """NFD-decomposed Korean vs NFC-composed → same after _nfc."""
        # 가 = U+AC00 (NFC precomposed)
        # 가 = ㄱ + ㅏ (NFD decomposed)
        nfc_str = "가"  # NFC '가'
        nfd_str = unicodedata.normalize("NFD", nfc_str)
        assert nfd_str != nfc_str, "Test precondition: NFD should differ from NFC"
        assert self._nfc(nfc_str) == self._nfc(nfd_str)

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace removed by _nfc."""
        assert self._nfc("  물 주세요  ") == "물 주세요"

    def test_same_string_after_double_normalize(self):
        """Idempotent: normalizing twice gives same result."""
        s = "미지근한 물 주세요"
        assert self._nfc(self._nfc(s)) == self._nfc(s)


class TestRAGRetrieverModelsImport:
    """A5: RetrievedCandidate and RetrievalSource can be imported."""

    def test_import_retrieved_candidate(self):
        from app.agent.rag_retriever import RetrievedCandidate  # noqa: F401
        assert RetrievedCandidate is not None

    def test_import_retrieval_source(self):
        from app.agent.rag_retriever import RetrievalSource  # noqa: F401
        assert RetrievalSource is not None

    def test_retrieval_source_catalog_cold_start(self):
        from app.agent.rag_retriever import RetrievalSource
        assert hasattr(RetrievalSource, "CATALOG_COLD_START")

    def test_retrieved_candidate_instantiation(self):
        from app.agent.rag_retriever import RetrievedCandidate, RetrievalSource
        cand = RetrievedCandidate(phrase="물 주세요", sources=[RetrievalSource.CATALOG_COLD_START])
        assert cand.phrase == "물 주세요"
        assert cand.cnt_90d == 0


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures for integration tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def app_client():
    """Spin up the full FastAPI app with lifespan via Starlette TestClient."""
    from starlette.testclient import TestClient

    from app.main import app as fastapi_app

    with TestClient(fastapi_app, raise_server_exceptions=True) as client:
        yield client


def _login(client, user_id: str, password: str = "demo1234!") -> str:
    """Login and return Bearer token."""
    resp = client.post("/auth/login", json={"user_id": user_id, "password": password})
    assert resp.status_code == 200, f"Login failed for {user_id}: {resp.text}"
    return resp.json()["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ══════════════════════════════════════════════════════════════════════════════
# Group B — Integration tests (live DB, demo patients)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_e2e_pa_high_branch(app_client):
    """B6: demo_pa conf=0.99 → mode=proactive, branch=high, verbatim.text exists."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            "asr_confidence": 0.99,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "proactive", f"Expected mode=proactive, got {body['mode']!r}"
    assert body["branch"] == "high", f"Expected branch=high, got {body['branch']!r}"
    assert "verbatim" in body, "verbatim field must exist in proactive response"
    assert body["verbatim"]["text"], "verbatim.text must be non-empty"
    # suggestion may be null (ANTHROPIC not configured) or a dict — both are valid
    suggestion = body.get("suggestion")
    assert suggestion is None or isinstance(suggestion, dict), (
        f"suggestion must be null or dict, got {type(suggestion)}"
    )
    # fallback path must be fast (< 100ms) when ANTHROPIC is not configured
    if body.get("accuracy_fallback_reason") is not None:
        assert body["generation_latency_ms"] < 100, (
            f"Fallback path took {body['generation_latency_ms']}ms, expected < 100ms"
        )


@pytest.mark.integration
def test_e2e_pa_mid_branch(app_client):
    """B7: demo_pa conf=0.93 → mode=candidates, branch=mid, candidates list len>=1."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            "asr_confidence": 0.93,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "candidates", f"Expected mode=candidates, got {body['mode']!r}"
    assert body["branch"] == "mid", f"Expected branch=mid, got {body['branch']!r}"
    candidates = body.get("candidates", [])
    assert len(candidates) >= 1, "Expected at least 1 candidate in mid branch"
    # rank=1 candidate should be verbatim (asr_verbatim source) OR fallback
    rank1 = next((c for c in candidates if c.get("rank") == 1), candidates[0])
    assert rank1.get("text"), "rank=1 candidate text must be non-empty"


@pytest.mark.integration
def test_e2e_pa_low_branch(app_client):
    """B8: demo_pa conf=0.65 → mode=single, branch=low, single candidate object."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물",
            "asr_confidence": 0.65,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "single", f"Expected mode=single, got {body['mode']!r}"
    assert body["branch"] == "low", f"Expected branch=low, got {body['branch']!r}"
    # 'candidate' is a single object, NOT a list
    assert "candidate" in body, "single branch must have 'candidate' key (not 'candidates')"
    assert not isinstance(body["candidate"], list), (
        "'candidate' must be a single object, not a list"
    )
    assert body["candidate"]["text"], "candidate.text must be non-empty"


@pytest.mark.integration
def test_e2e_retry_branch(app_client):
    """B9: conf=0.30 → mode=retry, HTTP 200 (NOT 422), message exists."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "asr_confidence": 0.30,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, (
        f"retry branch must return HTTP 200, not {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["mode"] == "retry", f"Expected mode=retry, got {body['mode']!r}"
    assert "message" in body, "retry response must include 'message' field"
    assert body["message"], "retry message must be non-empty"


@pytest.mark.integration
def test_e2e_pb_cold_start(app_client):
    """B10: demo_pb (zero phrase_log) conf=0.65 → mode=single, cold_start=true."""
    token = _login(app_client, "demo_pb")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물",
            "asr_confidence": 0.65,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "single", f"Expected mode=single, got {body['mode']!r}"
    assert body.get("cold_start") is True, (
        f"Expected cold_start=true for demo_pb (zero history), got {body.get('cold_start')!r}"
    )
    # candidate should come from catalog_fallback or time-based preset
    candidate = body.get("candidate", {})
    assert candidate.get("text"), "cold_start candidate.text must be non-empty"
    source = candidate.get("source", "")
    assert source in ("catalog_fallback", "ranker_fallback"), (
        f"cold_start candidate source should be catalog/ranker fallback, got {source!r}"
    )


@pytest.mark.integration
def test_e2e_pb_cold_start_high(app_client):
    """B11: demo_pb conf=0.99 → mode=proactive, cold_start=true, verbatim reproduced."""
    token = _login(app_client, "demo_pb")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            "asr_confidence": 0.99,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "proactive", f"Expected mode=proactive, got {body['mode']!r}"
    assert body.get("cold_start") is True, (
        f"Expected cold_start=true for demo_pb, got {body.get('cold_start')!r}"
    )
    assert body["verbatim"]["text"] == "물 주세요", (
        f"verbatim.text should echo final_transcript, got {body['verbatim']['text']!r}"
    )


@pytest.mark.integration
def test_e2e_asr_confidence_omitted(app_client):
    """B12: asr_confidence field absent → default 1.0 (high branch), HTTP 200, mode=proactive."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            # asr_confidence intentionally omitted
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "proactive", (
        f"Omitted asr_confidence should default to 1.0 (high branch), got mode={body['mode']!r}"
    )


@pytest.mark.integration
def test_e2e_audit_log_recorded(app_client):
    """B13: request → audit_log table has event='candidates_generated' with required fields."""
    import json

    import psycopg2

    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            "asr_confidence": 0.99,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    trace_id = resp.json().get("trace_id")
    assert trace_id, "Response must include trace_id"

    conn = psycopg2.connect(dbname="aac", user="aac", password="aac", host="127.0.0.1", port=5432)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM audit_log WHERE trace_id = %s AND event = 'candidates_generated' "
                "ORDER BY id DESC LIMIT 1",
                (trace_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row is not None, f"No audit_log row found for trace_id={trace_id}"
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])

    # Required fields per phrase.py audit_append call
    for field in ("mode", "branch", "cold_start", "fallback_reason"):
        assert field in payload, f"audit_log payload missing field '{field}': {payload}"


@pytest.mark.integration
def test_e2e_fallback_reason_set(app_client):
    """B14: demo_pa conf=0.99, ANTHROPIC not set → accuracy_fallback_reason='sonnet_not_configured' (or similar)."""
    token = _login(app_client, "demo_pa")
    resp = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "final_transcript": "물 주세요",
            "asr_confidence": 0.99,
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    # Basic shape checks regardless of fallback path
    assert body["mode"] == "proactive"
    assert body["verbatim"]["text"] == "물 주세요"
    # If ANTHROPIC_API_KEY not set, fallback_reason should be set (not null)
    fallback_reason = body.get("accuracy_fallback_reason")
    # Accept either None (if somehow LLM succeeded) or a non-empty string
    if fallback_reason is not None:
        assert isinstance(fallback_reason, str) and fallback_reason, (
            f"accuracy_fallback_reason should be non-empty string or null, got {fallback_reason!r}"
        )
        # The reason should be a known fallback kind
        assert any(
            keyword in fallback_reason
            for keyword in ("sonnet_not_configured", "not_configured", "timeout", "error", "hallucinated")
        ), f"Unexpected fallback reason: {fallback_reason!r}"


@pytest.mark.integration
def test_isolation_streaming_ranker_no_crosslink():
    """B15: run test_isolation.py tests and verify they all pass."""
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_isolation.py", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd="/Users/jeenee_1/python/hackathon/backend",
    )
    assert result.returncode == 0, (
        f"test_isolation.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group C — Fallback chain tests (mock)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_fallback_l1_when_llm_fails(app_client):
    """C16: Mock rerank_proactive to return LLMResult(output=None) → L1 fallback shape valid."""
    from app.agent.llm_assist import LLMResult

    mock_result = LLMResult(output=None, latency_ms=0, fallback_reason="timeout")

    with patch(
        "app.agent.orchestrator.rerank_proactive",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        token = _login(app_client, "demo_pa")
        resp = app_client.post(
            "/phrase/suggest",
            json={
                "patient_prefix": "물",
                "final_transcript": "물 주세요",
                "asr_confidence": 0.99,
            },
            headers=_auth_headers(token),
        )

    assert resp.status_code == 200, f"Expected 200 on LLM fallback, got {resp.status_code}: {resp.text}"
    body = resp.json()
    # On LLM failure for high branch → ProactiveModeResponse with suggestion=None
    assert body["mode"] == "proactive", f"Expected mode=proactive, got {body['mode']!r}"
    assert body.get("suggestion") is None, (
        "When LLM fails, suggestion must be None (L1 fallback to verbatim only)"
    )
    assert body.get("accuracy_fallback_reason") == "timeout", (
        f"fallback_reason should be 'timeout', got {body.get('accuracy_fallback_reason')!r}"
    )
    # verbatim still present
    assert body["verbatim"]["text"], "verbatim.text must be present even on LLM fallback"


@pytest.mark.integration
def test_fallback_l2_when_rag_empty(app_client):
    """C17: Mock retrieve_for_low to return [] → time_based_fallback (catalog_fallback source)."""
    with patch(
        "app.agent.orchestrator.retrieve_for_low",
        new_callable=AsyncMock,
        return_value=[],
    ):
        token = _login(app_client, "demo_pa")
        resp = app_client.post(
            "/phrase/suggest",
            json={
                "patient_prefix": "물",
                "final_transcript": "물",
                "asr_confidence": 0.65,
            },
            headers=_auth_headers(token),
        )

    assert resp.status_code == 200, f"Expected 200 on RAG empty, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["mode"] == "single", f"Expected mode=single, got {body['mode']!r}"
    candidate = body.get("candidate", {})
    assert candidate.get("text"), "candidate.text must be non-empty even on RAG empty"
    # When RAG returns empty AND LLM also skips (cold_start_skip or not_configured),
    # source should be catalog_fallback from time_based_fallback
    source = candidate.get("source", "")
    # Accept catalog_fallback or ranker_fallback (LLM may still pick from retrieved=[])
    assert source in ("catalog_fallback", "ranker_fallback", "llm_reconstruction"), (
        f"Unexpected source on RAG-empty path: {source!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group D — ruff + import checks (preconditions)
# ══════════════════════════════════════════════════════════════════════════════


def test_ruff_clean_agent():
    """D18: src/app/agent/ passes ruff check with returncode==0."""
    result = subprocess.run(
        ["ruff", "check", "src/app/agent/"],
        capture_output=True,
        text=True,
        cwd="/Users/jeenee_1/python/hackathon/backend",
    )
    assert result.returncode == 0, (
        f"ruff found issues in src/app/agent/:\n{result.stdout}\n{result.stderr}"
    )


def test_import_graph_no_cycle():
    """D19: from app.main import app succeeds without circular import error."""
    result = subprocess.run(
        ["python", "-c", "from app.main import app; print('ok')"],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PYTHONPATH": "/Users/jeenee_1/python/hackathon/backend/src",
        },
        cwd="/Users/jeenee_1/python/hackathon/backend",
    )
    assert result.returncode == 0, (
        f"Import cycle or startup error detected:\n{result.stdout}\n{result.stderr}"
    )
    assert "ok" in result.stdout, f"Expected 'ok' in output, got: {result.stdout!r}"
