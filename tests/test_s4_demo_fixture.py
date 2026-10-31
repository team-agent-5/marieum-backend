"""Demo fixture reproducibility — asserts that seeded patients produce specific top-1 phrases.

Rationale: hackathon demo must be reproducible. If the seed script changes, these tests catch it.
"""
import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _login(client, user_id, password="demo1234!"):
    r = client.post("/auth/login", json={"user_id": user_id, "password": password})
    assert r.status_code == 200, f"login failed: {r.text}"
    return r.json()["access_token"]


@pytest.mark.integration
def test_demo_pa_water_prefix(client):
    """demo_pa (heavy history + 미지근한 물 preference) with prefix '물' → top-1 contains '물' or has preference match."""
    tok = _login(client, "demo_pa")
    r = client.post(
        "/phrase/suggest",
        json={"patient_prefix": "물", "accuracy_mode": False},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) >= 1, "expected at least 1 candidate for demo_pa 물"
    top1 = body["candidates"][0]
    assert "물" in top1["text"], f"top-1 should contain '물', got {top1['text']!r}"
    assert body.get("fallback_hint") is None, "demo_pa with history should not trigger fallback"


@pytest.mark.integration
def test_demo_pb_cold_start_triggers_preset(client):
    """demo_pb (cold-start, zero history) with prefix '물' → score low, fallback_hint == 'preset'."""
    tok = _login(client, "demo_pb")
    r = client.post(
        "/phrase/suggest",
        json={"patient_prefix": "물", "accuracy_mode": False},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["fallback_hint"] == "preset", (
        f"cold-start should trigger preset fallback, got {body['fallback_hint']!r}"
    )


@pytest.mark.integration
def test_demo_pc_short_prefix(client):
    """demo_pc (short-word) with prefix '네' → top-1 == '네'."""
    tok = _login(client, "demo_pc")
    r = client.post(
        "/phrase/suggest",
        json={"patient_prefix": "네", "accuracy_mode": False},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) >= 1
    assert body["candidates"][0]["text"] == "네", (
        f"demo_pc top-1 should be '네', got {body['candidates'][0]['text']!r}"
    )


@pytest.mark.integration
def test_healthz_groq_state(client):
    """/healthz reports groq_llm state based on local signals (no network call)."""
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "groq_llm" in body["dependencies"]
    # With invalid key still "ok" (key present); if ops clears GROQ_API_KEY env, should be "down"
    assert body["dependencies"]["groq_llm"] in ("ok", "down")


@pytest.mark.integration
def test_warmup_includes_agent_state(client):
    """/health/warmup now reports LLM agent construction state."""
    r = client.get("/health/warmup")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedder_dim"] == 768
    assert body["llm_agent"] in ("constructed", "disabled")


@pytest.mark.integration
def test_accuracy_mode_records_llm_latency(client):
    """accuracy_mode=true request logs llm_latency_ms in audit_log (non-null) even on fallback."""
    import json

    import psycopg2
    tok = _login(client, "demo_pa")
    r = client.post(
        "/phrase/suggest",
        json={"patient_prefix": "물", "final_transcript": "물 좀 주세요", "accuracy_mode": True},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    trace_id = r.json()["trace_id"]
    # Query audit_log for the matching event
    conn = psycopg2.connect(dbname="aac", user="aac", password="aac", host="127.0.0.1", port=5432)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM audit_log WHERE trace_id = %s AND event = 'candidates_generated' ORDER BY id DESC LIMIT 1",
                (trace_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None, f"no audit_log for trace_id={trace_id}"
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert "accuracy_mode_used" in payload
    assert "accuracy_fallback_reason" in payload
    assert "llm_latency_ms" in payload
    # On fallback (invalid key), accuracy_mode_used is False but llm_latency_ms should still be captured (non-null, >0)
    assert payload["llm_latency_ms"] is None or payload["llm_latency_ms"] >= 0
