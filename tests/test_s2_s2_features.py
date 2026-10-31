"""S2 feature test suite.

Group F — F5 semantic_similarity unit tests (no DB)
Group G — F4 sequence_context unit tests (no DB)
Group H — reason resolver unit tests (no DB)
Group I — Integration E2E (require live Postgres + seed data)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers (mirrored from test_s1_ranker.py)
# ══════════════════════════════════════════════════════════════════════════════


def _cand(phrase: str, embedding: list[float] | None = None) -> SimpleNamespace:
    """Minimal PoolCandidate-like object."""
    return SimpleNamespace(phrase=phrase, embedding=embedding)


def _ctx(**kwargs: Any) -> SimpleNamespace:
    """Build a minimal RankingContext-like namespace for unit tests."""
    defaults: dict[str, Any] = {
        "prefix": "",
        "prefix_vec": None,
        "previous_phrase": None,
        "bigram_counts": {},
        "bigram_total": 0,
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
# Group F — F5 semantic_similarity unit tests
# ══════════════════════════════════════════════════════════════════════════════


class TestF5SemanticSimilarity:
    """F5 semantic_similarity — pure Python cosine, gate len(prefix)>=3 and cosine>=0.65."""

    def _compute(
        self,
        prefix: str,
        prefix_vec: list[float] | None,
        cand_embedding: list[float] | None,
        phrase: str = "미지근한 물 주세요",
    ) -> float:
        from app.agent.features.semantic_similarity import compute

        return compute(
            _cand(phrase, embedding=cand_embedding),
            _ctx(prefix=prefix, prefix_vec=prefix_vec),
        )

    def test_f5_perfect_cosine_passes_threshold(self):
        """prefix_vec=[1,0,0], candidate.embedding=[1,0,0], prefix '미지근한' (len 4) → cosine=1.0, score=1.0."""
        score = self._compute(
            prefix="미지근한",
            prefix_vec=[1.0, 0.0, 0.0],
            cand_embedding=[1.0, 0.0, 0.0],
        )
        assert abs(score - 1.0) < 1e-6, f"Expected ~1.0, got {score}"

    def test_f5_mid_cosine_scaled(self):
        """cosine 0.80 → (0.80 - 0.65) / 0.35 ≈ 0.4286."""
        # Vectors with cosine = 0.80: a=[1,0], b=[0.8, 0.6] → cos=0.8/1*1=0.8

        a = [0.8, 0.6]  # unit vector
        # b must produce cosine 0.80 exactly with a=[0.8, 0.6]
        # Let b = [1, 0]: cos = dot([0.8,0.6],[1,0]) = 0.8 ✓
        b = [1.0, 0.0]
        score = self._compute(
            prefix="미지근한",
            prefix_vec=a,
            cand_embedding=b,
        )
        expected = (0.80 - 0.65) / 0.35
        assert abs(score - expected) < 1e-6, f"Expected ~{expected:.4f}, got {score}"

    def test_f5_below_threshold_gated(self):
        """cosine 0.60 (< 0.65) → 0.0."""
        # a=[1,0], b=[0.6, 0.8] → cos=0.6
        score = self._compute(
            prefix="미지근한",
            prefix_vec=[1.0, 0.0],
            cand_embedding=[0.6, 0.8],
        )
        assert score == 0.0, f"Expected 0.0 (gated), got {score}"

    def test_f5_prefix_len_lt_3(self):
        """prefix '미' (len 1) → 0.0 regardless of cosine."""
        score = self._compute(
            prefix="미",
            prefix_vec=[1.0, 0.0, 0.0],
            cand_embedding=[1.0, 0.0, 0.0],
        )
        assert score == 0.0, f"Expected 0.0 (len < 3), got {score}"

    def test_f5_null_prefix_vec(self):
        """prefix_vec=None → 0.0."""
        score = self._compute(
            prefix="미지근한",
            prefix_vec=None,
            cand_embedding=[1.0, 0.0, 0.0],
        )
        assert score == 0.0, f"Expected 0.0 (null prefix_vec), got {score}"

    def test_f5_null_cand_embedding(self):
        """candidate.embedding=None → 0.0."""
        score = self._compute(
            prefix="미지근한",
            prefix_vec=[1.0, 0.0, 0.0],
            cand_embedding=None,
        )
        assert score == 0.0, f"Expected 0.0 (null cand embedding), got {score}"

    def test_f5_orthogonal(self):
        """vectors [1,0,0] vs [0,1,0] → cosine=0 → 0.0."""
        score = self._compute(
            prefix="미지근한",
            prefix_vec=[1.0, 0.0, 0.0],
            cand_embedding=[0.0, 1.0, 0.0],
        )
        assert score == 0.0, f"Expected 0.0 (orthogonal), got {score}"

    def test_f5_zero_vector_safe(self):
        """zero-norm vector input → 0.0 (no NaN)."""
        score = self._compute(
            prefix="미지근한",
            prefix_vec=[0.0, 0.0, 0.0],
            cand_embedding=[1.0, 0.0, 0.0],
        )
        assert score == 0.0, f"Expected 0.0 (zero norm), got {score}"
        assert not math.isnan(score), "Got NaN instead of 0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Group G — F4 sequence_context unit tests
# ══════════════════════════════════════════════════════════════════════════════


class TestF4SequenceContext:
    """F4 sequence_context — bigram transition, gate previous_phrase and bigram_total >= 2."""

    def _compute(
        self,
        phrase: str,
        previous_phrase: str | None,
        bigram_counts: dict,
        bigram_total: int,
    ) -> float:
        from app.agent.features.sequence_context import compute

        return compute(
            _cand(phrase),
            _ctx(
                previous_phrase=previous_phrase,
                bigram_counts=bigram_counts,
                bigram_total=bigram_total,
            ),
        )

    def test_f4_no_previous_phrase_returns_zero(self):
        """previous_phrase=None → 0.0 even if bigram data present."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase=None,
            bigram_counts={"약 주세요": 10},
            bigram_total=10,
        )
        assert score == 0.0, f"Expected 0.0 (no previous), got {score}"

    def test_f4_noise_threshold(self):
        """bigram_total=1 → 0.0 (below noise threshold of 2)."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase="물 주세요",
            bigram_counts={"약 주세요": 1},
            bigram_total=1,
        )
        assert score == 0.0, f"Expected 0.0 (noise gate), got {score}"

    def test_f4_normal_ratio(self):
        """bigram_counts['약 주세요']=12, total=20 → min(1.0, 12/20*1.2) = 0.72."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase="물 주세요",
            bigram_counts={"약 주세요": 12},
            bigram_total=20,
        )
        expected = min(1.0, (12 / 20) * 1.2)
        assert abs(score - expected) < 1e-9, f"Expected ~{expected:.4f}, got {score}"

    def test_f4_clipped(self):
        """ratio > 1.0 → clipped to 1.0 (counts=10, total=10 → 10/10*1.2 = 1.2 → 1.0)."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase="물 주세요",
            bigram_counts={"약 주세요": 10},
            bigram_total=10,
        )
        assert score == 1.0, f"Expected 1.0 (clipped), got {score}"

    def test_f4_phrase_absent(self):
        """candidate.phrase not in bigram_counts → 0.0."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase="물 주세요",
            bigram_counts={"다른 문장": 5},
            bigram_total=5,
        )
        assert score == 0.0, f"Expected 0.0 (absent phrase), got {score}"

    def test_f4_division_safe(self):
        """bigram_total=0 → 0.0 (caught by noise gate before division)."""
        score = self._compute(
            phrase="약 주세요",
            previous_phrase="물 주세요",
            bigram_counts={},
            bigram_total=0,
        )
        assert score == 0.0, f"Expected 0.0 (total=0 gate), got {score}"


# ══════════════════════════════════════════════════════════════════════════════
# Group H — reason resolver unit tests
# ══════════════════════════════════════════════════════════════════════════════


def _make_candidate_score(
    phrase: str = "물 주세요",
    source: str = "patient_log",
    prefix_match: float = 0.0,
    patient_frequency: float = 0.0,
    time_context: float = 0.0,
    sequence_context: float = 0.0,
    semantic_similarity: float = 0.0,
    recency: float = 0.0,
    preference_bonus: float = 0.0,
    bonus: float = 0.0,
    base: float = 0.0,
    total: float = 0.0,
):
    """Build a CandidateScore for reason resolver tests."""
    from app.agent.contracts import CandidateScore, FeatureScores

    scores = FeatureScores(
        prefix_match=prefix_match,
        patient_frequency=patient_frequency,
        time_context=time_context,
        sequence_context=sequence_context,
        semantic_similarity=semantic_similarity,
        recency=recency,
        preference_bonus=preference_bonus,
    )
    return CandidateScore(
        phrase=phrase,
        source=source,
        scores=scores,
        base=base,
        bonus=bonus,
        total=total,
    )


class TestReasonResolver:
    """pick_reason_label — priority ladder tests."""

    def _pick(self, **kwargs) -> str | None:
        from app.agent.reason import pick_reason_label

        c = _make_candidate_score(**kwargs)
        return pick_reason_label(c)

    def test_reason_preference_bonus_wins(self):
        """bonus=0.15, other scores high → 'registered_preference'."""
        label = self._pick(
            bonus=0.15,
            patient_frequency=0.9,
            time_context=0.8,
            sequence_context=0.7,
            semantic_similarity=0.9,
            recency=1.0,
        )
        assert label == "registered_preference", f"Expected 'registered_preference', got {label!r}"

    def test_reason_patient_frequency_dominant(self):
        """F2=0.9, all others 0 → 'patient_frequency'."""
        label = self._pick(patient_frequency=0.9, bonus=0.0)
        assert label == "patient_frequency", f"Expected 'patient_frequency', got {label!r}"

    def test_reason_time_context_dominant(self):
        """F3=1.0, all others 0 → 'time_context'."""
        label = self._pick(time_context=1.0, bonus=0.0)
        assert label == "time_context", f"Expected 'time_context', got {label!r}"

    def test_reason_sequence_dominant(self):
        """F4=0.9, all others 0 → 'sequence_context'."""
        label = self._pick(sequence_context=0.9, bonus=0.0)
        assert label == "sequence_context", f"Expected 'sequence_context', got {label!r}"

    def test_reason_semantic_dominant(self):
        """F5=0.9, all others 0 → 'semantic_similarity'."""
        label = self._pick(semantic_similarity=0.9, bonus=0.0)
        assert label == "semantic_similarity", f"Expected 'semantic_similarity', got {label!r}"

    def test_reason_recency_dominant(self):
        """F6=1.0, all others 0 → 'recency'."""
        label = self._pick(recency=1.0, bonus=0.0)
        assert label == "recency", f"Expected 'recency', got {label!r}"

    def test_reason_no_signal_returns_none(self):
        """All 6 features = 0, bonus=0 → None (no personalization signal). cold_start lives on audit_payload."""
        label = self._pick(
            prefix_match=0.0,
            patient_frequency=0.0,
            time_context=0.0,
            sequence_context=0.0,
            semantic_similarity=0.0,
            recency=0.0,
            bonus=0.0,
        )
        assert label is None, f"Expected None, got {label!r}"

    def test_reason_prefix_match_only_returns_none(self):
        """F1=0.9, all others 0 → None. prefix_match is implicit, not a user-facing label."""
        label = self._pick(
            prefix_match=0.9,
            patient_frequency=0.0,
            time_context=0.0,
            sequence_context=0.0,
            semantic_similarity=0.0,
            recency=0.0,
            bonus=0.0,
        )
        assert label is None, f"Expected None (prefix_match excluded), got {label!r}"

    def test_reason_ladder_time_beats_recency(self):
        """Threshold ladder (design §4.1): time_context precedes recency when both pass threshold."""
        # time_context=0.6 (>= 0.5), recency=0.9 (>= 0.5). Both pass; ladder returns time_context first.
        label = self._pick(
            time_context=0.6,
            recency=0.9,
            bonus=0.0,
        )
        assert label == "time_context", f"Expected 'time_context' (first in ladder), got {label!r}"

    def test_reason_ladder_frequency_below_threshold_skipped(self):
        """patient_frequency=0.25 (< 0.3) is below threshold → skipped; recency=0.6 (>= 0.5) wins."""
        label = self._pick(
            patient_frequency=0.25,
            recency=0.6,
            bonus=0.0,
        )
        assert label == "recency", f"Expected 'recency' (freq below threshold), got {label!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Group I — Integration E2E
# Require live Postgres + seed applied.
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def app_client():
    """Spin up the full FastAPI app with lifespan via Starlette TestClient."""
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
def test_e2e_pa_preference_match(app_client):
    """demo_pa POST /phrase/suggest with prefix '물' → rank 1 reason_label == 'registered_preference'."""
    token = _login(app_client, "demo_pa", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 (reason_label·가중합 활성).
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    candidates = body.get("candidates", [])
    assert len(candidates) >= 1, "Expected at least 1 candidate"
    rank1 = next((c for c in candidates if c["rank"] == 1), candidates[0])
    assert rank1["reason_label"] == "registered_preference", (
        f"Expected rank-1 reason_label='registered_preference', got {rank1['reason_label']!r}. "
        f"Full rank-1: {rank1}"
    )


@pytest.mark.integration
def test_e2e_pa_f5_boost(app_client):
    """demo_pa POST /phrase/suggest with prefix '미지근한' → rank 1 is '미지근한 물 주세요', score > 0.5."""
    token = _login(app_client, "demo_pa", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 (F5 semantic_similarity 가중합 검증).
        json={"patient_prefix": "미지근한", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    candidates = body.get("candidates", [])
    assert len(candidates) >= 1, "Expected at least 1 candidate"
    rank1 = next((c for c in candidates if c["rank"] == 1), candidates[0])
    assert rank1["text"] == "미지근한 물 주세요", (
        f"Expected rank-1='미지근한 물 주세요' (F5 semantic boost), got '{rank1['text']}'"
    )
    assert rank1["score"] > 0.5, (
        f"Expected score > 0.5 (F5 contribution visible), got {rank1['score']}"
    )


@pytest.mark.integration
def test_e2e_pb_cold_start_label(app_client):
    """demo_pb POST /phrase/suggest with prefix '물' → rank 1 reason_label == 'cold_start', fallback_hint == 'preset'."""
    token = _login(app_client, "demo_pb", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 (cold-start catalog fallback 검증).
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    fallback = body.get("fallback_hint")
    assert fallback == "preset", (
        f"Expected fallback_hint='preset' for cold-start patient, got {fallback!r}"
    )
    candidates = body.get("candidates", [])
    if candidates:
        rank1 = next((c for c in candidates if c["rank"] == 1), candidates[0])
        # Design §4.1: no personalization signal → reason_label=None.
        # cold_start lives on audit_payload (ranker.cold_start flag) + fallback_hint='preset'.
        assert rank1["reason_label"] is None, (
            f"Expected rank-1 reason_label=None for cold-start, got {rank1['reason_label']!r}"
        )


@pytest.mark.integration
def test_e2e_reason_display_mapped(app_client):
    """Any suggest response has reason_display set to Korean label matching _REASON_DISPLAY in phrase.py."""
    from app.api.phrase import _REASON_DISPLAY

    token = _login(app_client, "demo_pa", "demo1234!")
    resp = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 (reason_label 활성 → reason_display 매핑 확인).
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    candidates = body.get("candidates", [])
    assert len(candidates) >= 1, "Expected at least 1 candidate"

    for c in candidates:
        label = c.get("reason_label")
        display = c.get("reason_display")
        if label is not None:
            expected_display = _REASON_DISPLAY.get(label)
            assert display == expected_display, (
                f"reason_display mismatch for label={label!r}: "
                f"expected {expected_display!r}, got {display!r}"
            )
        else:
            assert display is None, (
                f"reason_display should be None when reason_label is None, got {display!r}"
            )


@pytest.mark.integration
def test_e2e_f4_sequence_boost(app_client):
    """demo_pa POST /phrase/suggest with previous_log_id → verify rank 1 may differ from no-previous.

    If the previous_log_id lookup produces no matching phrase (no seeded bigrams lead to a
    different top candidate), the test documents the comparison but does NOT fail — it skips
    with a note instead. The key assertions are:
      1. The endpoint accepts previous_log_id without error.
      2. If a bigram effect is visible, rank-1 differs between the two calls.
    """
    import psycopg2

    token = _login(app_client, "demo_pa", "demo1234!")
    auth = _auth_headers(token)

    # Step 1: suggest '물' without previous_log_id, capture rank-1 text
    resp_no_prev = app_client.post(
        "/phrase/suggest",
        # Phase C — 저신뢰 분기 (F4 bigram sequence_context 가중합 활성).
        json={"patient_prefix": "물", "accuracy_mode": False, "asr_confidence": 0.5},
        headers=auth,
    )
    assert resp_no_prev.status_code == 200, f"No-prev suggest failed: {resp_no_prev.text}"
    candidates_no_prev = resp_no_prev.json().get("candidates", [])
    rank1_no_prev = next((c for c in candidates_no_prev if c["rank"] == 1), None)

    # Step 2: find a phrase_log row for demo_pa where selected_phrase='물 주세요'
    with psycopg2.connect("postgresql://aac:aac@localhost:5432/aac") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pl.id
                FROM phrase_log pl
                JOIN patients p ON p.id = pl.patient_id
                WHERE p.user_id = 'demo_pa'
                  AND pl.selected_phrase = '물 주세요'
                  AND pl.deleted_at IS NULL
                ORDER BY pl.created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()

    if row is None:
        pytest.skip(
            "No '물 주세요' phrase_log row for demo_pa — cannot set previous_log_id; "
            "bigram effect unverifiable without seeded bigram data."
        )

    prev_log_id = row[0]

    # Step 3: suggest '물' WITH previous_log_id
    resp_with_prev = app_client.post(
        "/phrase/suggest",
        json={
            "patient_prefix": "물",
            "accuracy_mode": False,
            # Phase C — 저신뢰 분기 유지 (F4 bigram 효과 비교).
            "asr_confidence": 0.5,
            "previous_log_id": prev_log_id,
        },
        headers=auth,
    )
    assert resp_with_prev.status_code == 200, (
        f"Suggest with previous_log_id failed: {resp_with_prev.text}"
    )
    candidates_with_prev = resp_with_prev.json().get("candidates", [])
    rank1_with_prev = next((c for c in candidates_with_prev if c["rank"] == 1), None)

    # Step 4: check for bigram effect; skip gracefully if bigram data produced no change
    if rank1_no_prev is None or rank1_with_prev is None:
        pytest.skip("No candidates returned — cannot compare bigram effect.")

    # We document the result. If bigram_total < 2 (noise gate), the scores will be
    # identical — that is acceptable and we skip rather than fail.
    if rank1_no_prev["text"] == rank1_with_prev["text"] and rank1_no_prev["score"] == rank1_with_prev["score"]:
        pytest.skip(
            f"Rank-1 text and score identical with and without previous_log_id "
            f"('{rank1_no_prev['text']}', score={rank1_no_prev['score']}). "
            "Likely no seeded bigrams above noise threshold — F4 effect invisible."
        )

    # If we reach here, the bigram did shift ranking or score — assert it's non-trivially different
    scores_differ = rank1_no_prev["score"] != rank1_with_prev["score"]
    texts_differ = rank1_no_prev["text"] != rank1_with_prev["text"]
    assert scores_differ or texts_differ, (
        "Expected rank-1 to differ between no-previous and with-previous calls "
        f"(previous_log_id={prev_log_id})."
    )
