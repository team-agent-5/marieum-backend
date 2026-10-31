"""S0 Prerequisites verification tests.

Group A: contracts.py pure-pydantic validation (no DB, no model load)
Group B: SuggestRequest.accuracy_mode field
Group C: Dead code removal (static file/path assertions)
Group D: Migration 004 structural (no DB)
Group E: Lifespan + /health/warmup — SLOW, marks @pytest.mark.slow
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent  # /hackathon/backend/
SRC = ROOT / "src"
MIGRATION_PATH = ROOT / "alembic" / "versions" / "004_agent_prep.py"

# ═══════════════════════════════════════════════════════════════════════════
# Group A — contracts.py validation (pure pydantic)
# ═══════════════════════════════════════════════════════════════════════════


def test_agent_input_defaults():
    from app.agent.contracts import AgentInput

    inp = AgentInput(patient_id=uuid4(), prefix="", now=datetime.now(UTC))
    assert inp.event_type == "final"
    assert inp.accuracy_mode is False


def test_agent_input_rejects_bad_event_type():
    from app.agent.contracts import AgentInput

    with pytest.raises(ValidationError):
        AgentInput(patient_id=uuid4(), prefix="", now=datetime.now(UTC), event_type="foobar")


def test_pool_candidate_source_enum():
    from app.agent.contracts import PoolCandidate

    # Valid values should construct fine
    for src in ("patient_log", "catalog", "llm_gen"):
        pc = PoolCandidate(phrase="테스트", source=src)
        assert pc.source == src

    # Invalid should raise
    with pytest.raises(ValidationError):
        PoolCandidate(phrase="테스트", source="unknown_source")


def test_feature_scores_defaults_zero():
    from app.agent.contracts import FeatureScores

    fs = FeatureScores()
    assert fs.prefix_match == 0.0
    assert fs.patient_frequency == 0.0
    assert fs.time_context == 0.0
    assert fs.sequence_context == 0.0
    assert fs.semantic_similarity == 0.0
    assert fs.recency == 0.0
    assert fs.preference_bonus == 0.0


def test_candidate_score_computes_bounds():
    from app.agent.contracts import CandidateScore, FeatureScores

    cs = CandidateScore(
        phrase="물 주세요",
        source="patient_log",
        scores=FeatureScores(prefix_match=1.0),
        base=0.35,
        bonus=0.0,
        total=0.35,
    )
    assert cs.base == 0.35
    assert cs.bonus == 0.0
    assert cs.total == 0.35
    assert cs.scores.prefix_match == 1.0


def test_ranking_context_default_collections():
    from app.agent.contracts import RankingContext

    ctx = RankingContext(
        patient_id=uuid4(),
        prefix="물",
        now=datetime.now(UTC),
        previous_phrase=None,
        accuracy_mode=False,
        time_bucket_kst="morning",
    )
    assert ctx.freq_counts == {}
    assert ctx.bigram_counts == {}
    assert ctx.prefs == []


def test_agent_output_minimal():
    from app.agent.contracts import AgentOutput

    out = AgentOutput(suggestions=[], latency_ms=10, cold_start=False)
    assert out.suggestions == []
    assert out.latency_ms == 10
    assert out.cold_start is False
    assert out.accuracy_mode_used is False


# ═══════════════════════════════════════════════════════════════════════════
# Group B — SuggestRequest.accuracy_mode
# ═══════════════════════════════════════════════════════════════════════════


def test_suggest_request_accuracy_default_false():
    from app.schemas import SuggestRequest

    req = SuggestRequest.model_validate({"patient_prefix": "물", "trace_id": None})
    assert req.accuracy_mode is False


def test_suggest_request_accuracy_true():
    from app.schemas import SuggestRequest

    req = SuggestRequest.model_validate(
        {"patient_prefix": "물", "trace_id": None, "accuracy_mode": True}
    )
    assert req.accuracy_mode is True


def test_suggest_request_accuracy_rejects_non_bool():
    from app.schemas import SuggestRequest

    with pytest.raises(ValidationError):
        SuggestRequest.model_validate(
            {"patient_prefix": "물", "trace_id": None, "accuracy_mode": "yes"},
            strict=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Group C — Dead code removal (static)
# ═══════════════════════════════════════════════════════════════════════════


def test_dead_dirs_removed():
    orchestrator_dir = SRC / "app" / "orchestrator"
    mock_file = SRC / "app" / "services" / "orchestrator_mock.py"
    sse_file = SRC / "app" / "services" / "sse.py"
    asr_file = SRC / "app" / "services" / "asr.py"

    assert not orchestrator_dir.exists(), f"{orchestrator_dir} should not exist (dead code)"
    assert not mock_file.exists(), f"{mock_file} should not exist (dead code)"
    assert not sse_file.exists(), f"{sse_file} should not exist (dead code)"
    assert not asr_file.exists(), f"{asr_file} should not exist (dead code)"


def test_no_residual_imports():
    """No src/app .py file should import from the deleted modules."""
    app_src = SRC / "app"
    bad_patterns = [
        re.compile(r"from app\.orchestrator"),
        re.compile(r"from app\.services\.(orchestrator_mock|sse|asr)"),
    ]
    violations: list[str] = []
    for py_file in app_src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in bad_patterns:
            if pat.search(text):
                violations.append(f"{py_file}: matched '{pat.pattern}'")

    assert not violations, "Residual imports found:\n" + "\n".join(violations)


# ═══════════════════════════════════════════════════════════════════════════
# Group D — Migration 004 structural (no DB)
# ═══════════════════════════════════════════════════════════════════════════


def _load_migration():
    """Dynamically load the migration module without alembic env."""
    spec = importlib.util.spec_from_file_location("migration_004", MIGRATION_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Stub out alembic.op so the top-level import doesn't fail without a DB
    import types

    fake_alembic = types.ModuleType("alembic")
    fake_op = types.ModuleType("alembic.op")
    fake_alembic.op = fake_op
    sys.modules.setdefault("alembic", fake_alembic)
    sys.modules.setdefault("alembic.op", fake_op)
    spec.loader.exec_module(mod)
    return mod


def test_migration_004_chain():
    mod = _load_migration()
    assert mod.revision == "004_agent_prep"
    assert mod.down_revision == "003_patient_app_scope"


def test_migration_004_has_time_bucket_sql():
    text = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "AT TIME ZONE 'Asia/Seoul'" in text, "Missing Asia/Seoul timezone cast"
    assert "GENERATED ALWAYS AS" in text, "Missing GENERATED ALWAYS AS clause"
    for label in ("'morning'", "'afternoon'", "'evening'", "'night'"):
        assert label in text, f"Missing time bucket label {label}"


def test_migration_004_has_source_check():
    text = MIGRATION_PATH.read_text(encoding="utf-8")
    # All 7 enum values must appear inside the CHECK clause context
    for val in ("voice", "preset", "fallback", "voice_bypass", "rejected", "llm_assist", "manual"):
        assert f"'{val}'" in text, f"Missing source enum value '{val}' in migration"
    assert "CHECK" in text, "Missing CHECK constraint keyword"


def test_migration_004_backfill_before_check():
    text = MIGRATION_PATH.read_text(encoding="utf-8")
    # UPDATE (backfill) must appear before ADD CONSTRAINT ... CHECK
    update_pos = text.find("UPDATE phrase_log")
    check_pos = text.find("ADD CONSTRAINT")
    assert update_pos != -1, "Missing UPDATE backfill statement"
    assert check_pos != -1, "Missing ADD CONSTRAINT statement"
    assert update_pos < check_pos, (
        f"Backfill UPDATE (pos {update_pos}) must appear before ADD CONSTRAINT (pos {check_pos})"
    )


def test_migration_004_fk_safe_drop():
    text = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "DROP CONSTRAINT IF EXISTS" in text, (
        "Missing 'DROP CONSTRAINT IF EXISTS' pattern for safe FK drop"
    )


def test_migration_004_downgrade_blocked():
    mod = _load_migration()
    with pytest.raises(RuntimeError):
        mod.downgrade()


# ═══════════════════════════════════════════════════════════════════════════
# Group E — Lifespan + /health/warmup  (SLOW — requires HF model cache)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
@pytest.mark.asyncio
async def test_warmup_endpoint():
    """Full app boot: lifespan runs, /health/warmup returns expected shape."""
    import httpx

    try:
        from asgi_lifespan import LifespanManager
    except ImportError:
        LifespanManager = None  # type: ignore[assignment]

    from app.main import app

    if LifespanManager is not None:
        async with LifespanManager(app, startup_timeout=120, shutdown_timeout=30):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/warmup", timeout=120)
    else:
        # Fallback: use TestClient which triggers lifespan synchronously
        from starlette.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get("/health/warmup", timeout=120)

    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedder_dim"] > 0
    assert body["embed_latency_ms"] >= 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_lifespan_catalog_cache_shape():
    """After lifespan startup, app.state.catalog_embeddings is a dict.

    We do NOT boot the app a second time here — we read the state set by the
    warmup test (same process, same app object). If the catalog was loaded
    during lifespan it will be present; if the event loop already completed
    (TestClient path) we also check the already-set state attribute.
    """
    from app.main import app

    cache = getattr(app.state, "catalog_embeddings", None)
    assert isinstance(cache, dict), (
        f"app.state.catalog_embeddings should be dict, got {type(cache)}. "
        "Ensure test_warmup_endpoint ran first (same slow session)."
    )
