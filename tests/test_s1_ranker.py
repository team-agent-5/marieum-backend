"""S1 Ranker test suite.

Group A — Pure feature unit tests (no DB, no model load)
Group B — pgvector parse_pgvector helper
Group C — DB-integrated ranker tests (require live Postgres + seed data)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _cand(phrase: str) -> SimpleNamespace:
    """Minimal PoolCandidate-like object for feature compute() calls."""
    return SimpleNamespace(phrase=phrase)


def _ctx(**kwargs: Any) -> SimpleNamespace:
    """Build a minimal RankingContext-like namespace for unit tests."""
    defaults: dict[str, Any] = {
        "prefix": "",
        "freq_counts": {},
        "time_bucket_counts": {},
        "time_bucket_total": {},
        "time_bucket_kst": "morning",
        "last_at": {},
        "now": datetime.now(UTC),
        "prefs": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Group A — Feature unit tests
# ══════════════════════════════════════════════════════════════════════════════


# ── F1 prefix_match ───────────────────────────────────────────────────────────

class TestF1PrefixMatch:
    """F1 prefix_match feature — pure Python, no DB."""

    def _compute(self, prefix: str, phrase: str) -> float:
        from app.agent.features.prefix_match import compute
        return compute(_cand(phrase), _ctx(prefix=prefix))

    def test_f1_prefix_short_startswith(self):
        """prefix '물' (len 1) on '물 주세요' → ~0.300 (startswith only)."""
        score = self._compute("물", "물 주세요")
        assert abs(score - 0.300) < 1e-9, f"Expected ~0.300, got {score}"

    def test_f1_prefix_long_trigram(self):
        """prefix '미지근한' (len 4, >=3) on '미지근한 물 주세요' → >= 0.6."""
        score = self._compute("미지근한", "미지근한 물 주세요")
        assert score >= 0.6, f"Expected >=0.6, got {score}"

    def test_f1_prefix_no_match(self):
        """prefix '약' on '물 주세요' → 0.0."""
        score = self._compute("약", "물 주세요")
        assert score == 0.0

    def test_f1_prefix_empty(self):
        """Empty prefix → 0.0."""
        score = self._compute("", "물 주세요")
        assert score == 0.0


# ── F2 patient_frequency ──────────────────────────────────────────────────────

class TestF2PatientFrequency:
    """F2 patient_frequency feature — pure Python, no DB."""

    def _compute(self, phrase: str, cnt: int) -> float:
        from app.agent.features.patient_frequency import compute
        return compute(_cand(phrase), _ctx(freq_counts={phrase: cnt}))

    def test_f2_zero(self):
        """cnt=0 → 0.0."""
        assert self._compute("물 주세요", 0) == 0.0

    def test_f2_saturation(self):
        """cnt=30 → ≈ 1.0 (log1p(30)/log1p(30))."""
        score = self._compute("물 주세요", 30)
        assert abs(score - 1.0) < 1e-9

    def test_f2_known(self):
        """cnt=18 → math.log1p(18)/math.log1p(30) ≈ 0.857."""
        expected = math.log1p(18) / math.log1p(30)
        score = self._compute("물 주세요", 18)
        assert abs(score - expected) < 1e-9


# ── F3 time_context ───────────────────────────────────────────────────────────

class TestF3TimeContext:
    """F3 time_context feature — pure Python, no DB."""

    def _compute(self, phrase: str, bucket: str, same: int, total: int) -> float:
        from app.agent.features.time_context import compute
        bucket_counts = {phrase: {bucket: same}} if same > 0 else {}
        return compute(
            _cand(phrase),
            _ctx(
                time_bucket_counts=bucket_counts,
                time_bucket_total={phrase: total},
                time_bucket_kst=bucket,
            ),
        )

    def test_f3_below_noise(self):
        """total=2 → 0.0 (gated: needs >=3 observations)."""
        score = self._compute("물 주세요", "morning", 2, 2)
        assert score == 0.0

    def test_f3_clipped(self):
        """same=12, total=15, bucket matches → min(1.0, 12/15*1.5) = 1.0."""
        score = self._compute("물 주세요", "morning", 12, 15)
        assert score == 1.0

    def test_f3_bucket_mismatch(self):
        """Candidate has no counts in current bucket → 0.0."""
        phrase = "물 주세요"
        from app.agent.features.time_context import compute
        # buckets are all "evening", but current bucket is "morning"
        score = compute(
            _cand(phrase),
            _ctx(
                time_bucket_counts={phrase: {"evening": 10}},
                time_bucket_total={phrase: 10},
                time_bucket_kst="morning",
            ),
        )
        assert score == 0.0


# ── F6 recency ────────────────────────────────────────────────────────────────

class TestF6Recency:
    """F6 recency feature — pure Python, no DB."""

    def _compute(self, phrase: str, last: datetime | None, now: datetime) -> float:
        from app.agent.features.recency import compute
        last_map = {phrase: last} if last is not None else {}
        return compute(_cand(phrase), _ctx(last_at=last_map, now=now))

    def test_f6_now(self):
        """last_at = now → 1.0."""
        now = datetime.now(UTC)
        score = self._compute("물 주세요", now, now)
        assert abs(score - 1.0) < 1e-9

    def test_f6_24h(self):
        """last_at = 24h ago → 0.5 (half-life = 24h)."""
        now = datetime.now(UTC)
        last = now - timedelta(hours=24)
        score = self._compute("물 주세요", last, now)
        assert abs(score - 0.5) < 1e-9

    def test_f6_none(self):
        """last_at absent → 0.0."""
        now = datetime.now(UTC)
        score = self._compute("물 주세요", None, now)
        assert score == 0.0


# ── F7 preference_bonus ───────────────────────────────────────────────────────

class TestF7PreferenceBonus:
    """F7 preference_bonus feature — pure Python, no DB."""

    def _compute(self, phrase: str, prefs: list[dict]) -> float:
        from app.agent.features.preference import compute
        return compute(_cand(phrase), _ctx(prefs=prefs))

    def test_f7_exact_content_match(self):
        """pref content == candidate.phrase → 0.15."""
        score = self._compute(
            "물 주세요",
            [{"preference_type": "preferred_phrase", "content": "물 주세요", "boost_phrases": []}],
        )
        assert abs(score - 0.15) < 1e-9

    def test_f7_boost_phrase_match(self):
        """candidate.phrase in boost_phrases → 0.15."""
        score = self._compute(
            "약 주세요",
            [{"preference_type": "preferred_phrase", "content": "물 주세요", "boost_phrases": ["약 주세요"]}],
        )
        assert abs(score - 0.15) < 1e-9

    def test_f7_regex_topic(self):
        """preference_type='topic_preference', regex match → 0.12."""
        score = self._compute(
            "물 주세요",
            [{"preference_type": "topic_preference", "content": "물.*주세요", "boost_phrases": []}],
        )
        assert abs(score - 0.12) < 1e-9

    def test_f7_no_match(self):
        """Nothing matches → 0.0."""
        score = self._compute(
            "약 주세요",
            [{"preference_type": "preferred_phrase", "content": "물 주세요", "boost_phrases": ["담요"]}],
        )
        assert score == 0.0

    def test_f7_bad_regex(self):
        """Malformed regex in topic_preference content must not crash; returns 0.0."""
        score = self._compute(
            "물 주세요",
            [{"preference_type": "topic_preference", "content": "[invalid(", "boost_phrases": []}],
        )
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Group B — pgvector parse_pgvector helper
# ══════════════════════════════════════════════════════════════════════════════


class TestParsePgvector:
    """parse_pgvector — normalises pgvector output to list[float] or None."""

    def _parse(self, v: Any):
        from app.services.embedding import parse_pgvector
        return parse_pgvector(v)

    def test_parse_pgvector_none(self):
        assert self._parse(None) is None

    def test_parse_pgvector_list(self):
        result = self._parse([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_parse_pgvector_string(self):
        result = self._parse("[0.1,0.2,0.3]")
        assert result is not None
        assert len(result) == 3
        assert abs(result[0] - 0.1) < 1e-9

    def test_parse_pgvector_empty_string(self):
        assert self._parse("[]") is None

    def test_parse_pgvector_malformed(self):
        """'garbage' → raises ValueError or returns None (graceful)."""
        from app.services.embedding import parse_pgvector
        try:
            result = parse_pgvector("garbage")
            # graceful return: must be None
            assert result is None
        except ValueError:
            pass  # also acceptable


# ══════════════════════════════════════════════════════════════════════════════
# Group C — DB-integrated ranker tests
# Require live Postgres + seed applied.
# ══════════════════════════════════════════════════════════════════════════════

pytestmark_integration = pytest.mark.integration


@pytest.fixture(scope="module")
def app_client():
    """Spin up the full FastAPI app with lifespan via Starlette TestClient.

    Uses a synchronous TestClient so the event loop is managed by the client,
    which makes it safe to call from sync test functions.
    """
    from starlette.testclient import TestClient

    from app.main import app as fastapi_app

    with TestClient(fastapi_app, raise_server_exceptions=True) as client:
        yield client


def _login(client, user_id: str, password: str) -> str:
    """Login and return Bearer token."""
    resp = client.post("/auth/login", json={"user_id": user_id, "password": password})
    assert resp.status_code == 200, f"Login failed for {user_id}: {resp.text}"
    return resp.json()["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_e2e_pa_heavy_history(app_client):
    """demo_pa (heavy history) suggests '물' → HTTP 200, candidates with 물, fast, no fallback."""
    token = _login(app_client, "demo_pa", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — asr_confidence=0.5 → 저신뢰 분기 (가중합 ranking 활성화).
        # 미지정 시 default 1.0 → 고신뢰 verbatim 분기로 라우팅돼 ranking 미실행.
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()

    assert "candidates" in body
    candidates = body["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) <= 3

    # Each candidate has rank, text, score
    for c in candidates:
        assert "rank" in c
        assert "text" in c
        assert "score" in c

    # At least 1 candidate contains '물' (prefix match should rank water-request first)
    texts = [c["text"] for c in candidates]
    assert any("물" in t for t in texts), f"No '물' in candidates: {texts}"

    # Latency
    assert body["generation_latency_ms"] < 500, f"Too slow: {body['generation_latency_ms']}ms"

    # Heavy history → no fallback_hint
    assert body.get("fallback_hint") is None, f"Expected no fallback for heavy-history patient, got: {body.get('fallback_hint')}"


@pytest.mark.integration
def test_e2e_pb_cold_start(app_client):
    """demo_pb (cold-start, 0 phrase_logs) suggests '물' → fallback_hint == 'preset'."""
    token = _login(app_client, "demo_pb", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 라우팅 (ranker 실행 검증 위해).
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()

    # Cold-start: all scores should be low, expect fallback_hint == 'preset'
    fallback = body.get("fallback_hint")
    assert fallback == "preset", (
        f"Expected fallback_hint='preset' for cold-start patient, got {fallback!r}. "
        f"Candidates: {body.get('candidates')}"
    )


@pytest.mark.integration
def test_e2e_pc_short_prefix(app_client):
    """demo_pc suggests '네' → rank-1 text == '네' (exact prefix match)."""
    token = _login(app_client, "demo_pc", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 라우팅 (prefix_match 가중합 검증).
        json={"patient_prefix": "네", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    candidates = body.get("candidates", [])

    assert len(candidates) >= 1, "Expected at least 1 candidate"
    rank1 = next((c for c in candidates if c["rank"] == 1), candidates[0])
    assert rank1["text"] == "네", f"Expected rank-1='네', got '{rank1['text']}'"


@pytest.mark.integration
def test_e2e_accuracy_mode_flag_accepted(app_client):
    """demo_pa with accuracy_mode=true → HTTP 200 (field accepted, S3 not yet impl)."""
    token = _login(app_client, "demo_pa", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기에서만 accuracy_mode flag 가 의미 있음 (고신뢰는 verbatim-only).
        json={"patient_prefix": "물", "accuracy_mode": True, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, (
        f"accuracy_mode=true was rejected with {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "candidates" in body


@pytest.mark.integration
def test_e2e_selection_writes_embedding(app_client):
    """demo_pa → suggest → pick top-1 → POST /phrase/selection → DB row has non-null embedding."""
    import psycopg2  # sync psycopg2 for direct DB query without event-loop conflicts

    token = _login(app_client, "demo_pa", "demo1234!")
    # 1. Suggest
    suggest_resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 라우팅.
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert suggest_resp.status_code == 200
    candidates = suggest_resp.json()["candidates"]
    assert len(candidates) >= 1
    top1 = candidates[0]
    trace_id = suggest_resp.json().get("trace_id", "test-trace-sel")

    # 2. Selection
    selection_body = {
        "trace_id": trace_id,
        "action": "selected",
        "prefix": "물",
        "final_transcript": None,
        "candidates_shown": [
            {
                "rank": c["rank"],
                "text": c["text"],
                "score": c["score"],
                "reason_label": c.get("reason_label"),
                "source": "patient_log",
            }
            for c in candidates
        ],
        "selected_phrase": top1["text"],
        "selected_rank": top1["rank"],
        "latency_ms": 123,
        "source": "voice",
        "prev_log_id": None,
        "reject_reason": None,
    }
    sel_resp = app_client.post(
        "/phrase/selection",
        json=selection_body,
        headers=_auth_headers(token),
    )
    assert sel_resp.status_code == 201, f"Selection failed: {sel_resp.status_code} — {sel_resp.text}"
    log_id = sel_resp.json()["log_id"]

    # 3. Query DB for non-null embedding via sync psycopg3 (avoids event-loop conflict)
    with psycopg2.connect("postgresql://aac:aac@localhost:5432/aac") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM phrase_log WHERE id = %s", (log_id,))
            row = cur.fetchone()

    assert row is not None, f"No phrase_log row found for id={log_id}"
    embedding_val = row[0]
    assert embedding_val is not None, (
        f"embedding is NULL for phrase_log id={log_id} — write-time RAG failed"
    )


@pytest.mark.integration
def test_e2e_preset_writes_embedding(app_client):
    """demo_pa → POST /phrase/preset → DB row with source='preset' has non-null embedding."""
    import psycopg2  # sync psycopg2

    token = _login(app_client, "demo_pa", "demo1234!")
    preset_resp = app_client.post(
        "/phrase/preset",
        json={"preset_index": 1},
        headers=_auth_headers(token),
    )
    assert preset_resp.status_code == 201, f"Preset failed: {preset_resp.status_code} — {preset_resp.text}"
    log_id = preset_resp.json()["log_id"]

    # Query DB via sync psycopg3
    with psycopg2.connect("postgresql://aac:aac@localhost:5432/aac") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding, source FROM phrase_log WHERE id = %s", (log_id,))
            row = cur.fetchone()

    assert row is not None
    embedding_val, source_val = row[0], row[1]
    assert source_val == "preset", f"Expected source='preset', got {source_val!r}"
    assert embedding_val is not None, (
        f"embedding is NULL for preset phrase_log id={log_id} — write-time RAG failed"
    )
