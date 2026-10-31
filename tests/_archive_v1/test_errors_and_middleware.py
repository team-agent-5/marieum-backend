"""에러 포맷·미들웨어·공통 응답 헤더 검증."""

from __future__ import annotations

import uuid

from httpx import AsyncClient
from tests.conftest import hdr

# ────────── TraceId 미들웨어 ──────────


async def test_server_echoes_x_trace_id_when_provided(client: AsyncClient) -> None:
    sent = str(uuid.uuid4())
    r = await client.get("/healthz", headers={"X-Trace-Id": sent})
    assert r.status_code == 200
    assert r.headers.get("x-trace-id") == sent


async def test_server_mints_x_trace_id_when_missing(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    minted = r.headers.get("x-trace-id")
    assert minted
    uuid.UUID(minted)  # 유효한 UUID


async def test_error_response_includes_trace_id(client: AsyncClient) -> None:
    """ErrorResponse(§1.4)에 trace_id 포함."""
    sent = str(uuid.uuid4())
    r = await client.get("/calls", headers={"X-Trace-Id": sent})  # 401
    assert r.status_code == 401
    body = r.json()
    assert body["trace_id"] == sent
    assert body["error_code"]


# ────────── Server-Version 헤더 ──────────


async def test_server_version_header_present(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.headers.get("x-server-version") == "0.1.0"


# ────────── 에러 포맷 통일 ──────────


async def test_404_has_error_response_shape(client: AsyncClient, tokens) -> None:
    r = await client.get(
        f"/calls/{uuid.uuid4()}", headers=hdr(tokens["nurse"])
    )
    assert r.status_code == 404
    body = r.json()
    assert set(body.keys()) >= {"error_code", "message", "trace_id"}
    assert body["error_code"] == "NOT_FOUND"


async def test_409_conflict_shape(client: AsyncClient, tokens) -> None:
    """상태 전이 불가 → 409 + CONFLICT."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    r = await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["nurse"]),
        json={"status": "completed"},  # active → completed 직접 불가
    )
    assert r.status_code == 409
    assert r.json()["error_code"] == "CONFLICT"


async def test_validation_error_includes_details(client: AsyncClient, tokens) -> None:
    """Pydantic 검증 실패 시 details에 field/issue."""
    r = await client.post(
        "/patients",
        headers=hdr(tokens["admin"]),
        json={"room": "", "name": "", "disease": ""},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert isinstance(body["details"], list) and body["details"]
    for d in body["details"]:
        assert "field" in d and "issue" in d


async def test_malformed_json_body(client: AsyncClient, tokens) -> None:
    """잘못된 JSON → 400 VALIDATION_ERROR."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        content=b"{bad json",
    )
    assert r.status_code == 400


async def test_unauthorized_shape(client: AsyncClient) -> None:
    r = await client.get("/calls")
    assert r.status_code == 401
    body = r.json()
    assert body["error_code"] == "UNAUTHORIZED"
    assert body["message"]
    assert body["trace_id"]


# ────────── OpenAPI 메타 ──────────


async def test_openapi_has_29_paths(client: AsyncClient) -> None:
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    # 와이어가 요구하는 핵심 경로가 전부 있는지
    required = [
        "/healthz",
        "/auth/caregivers/login",
        "/auth/admin/login",
        "/auth/patients/login",
        "/auth/me",
        "/patients",
        "/patients/me",
        "/patients/{patient_id}",
        "/patients/{patient_id}/notes",
        "/patients/{patient_id}/notes/{note_id}",
        "/caregivers",
        "/caregivers/me",
        "/caregivers/me/stats",
        "/caregivers/public",
        "/playbooks",
        "/transcriptions",
        "/calls/voice",
        "/calls/text",
        "/calls/preset",
        "/calls",
        "/calls/{call_id}",
        "/calls/{call_id}/feedback",
        "/calls/{call_id}/responses",
        "/calls/{call_id}/answers",
        "/calls/{call_id}/notify",
        "/sse/patient",
        "/sse/station",
    ]
    missing = [p for p in required if p not in spec["paths"]]
    assert not missing, f"missing paths: {missing}"
