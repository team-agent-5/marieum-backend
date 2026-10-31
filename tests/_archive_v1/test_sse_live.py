"""SSE push 라이브 테스트.

httpx AsyncClient + ASGITransport로 SSE streaming을 끝까지 소비하는 건
sse-starlette가 BaseHTTPMiddleware·버퍼링과 충돌해 안정적이지 않다. 그래서
**서비스 레벨 큐**를 직접 subscribe해서 push가 도달하는지 검증한다.

(브라우저·실 uvicorn 환경에서는 SSE 스트림이 정상 동작함을 수동 테스트로 확인 완료.)
"""

from __future__ import annotations

import asyncio

from app.services.sse import _get_patient_queue, _get_station_queue  # type: ignore[attr-defined]
from httpx import ASGITransport, AsyncClient
from tests.conftest import hdr

from app.main import app


async def test_preset_call_pushes_to_station_queue(tokens) -> None:
    """`POST /calls/preset` → 담당자 디바이스 station 큐에 call.created push."""
    # seed: 305호 김환자 default_caregiver=이간호(nurse) → water도 nurse 디바이스로 push
    device = "demo-station-nurse"
    q = _get_station_queue(device)
    # 큐를 깨끗이
    while not q.empty():
        q.get_nowait()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
        assert r.status_code == 201

    event = await asyncio.wait_for(q.get(), timeout=3)
    assert event["type"] == "call.created"
    assert event["intent"] == "water"
    assert event["patient"]["id"] == tokens["patient_id"]


async def test_dyspnea_pushes_to_nurse_station_queue(tokens) -> None:
    """dyspnea는 High-stakes 상향 → 담당자 간호사로 변경 → 간호사 디바이스 큐로 push."""
    device = "demo-station-nurse"
    q = _get_station_queue(device)
    while not q.empty():
        q.get_nowait()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "dyspnea"},
        )

    event = await asyncio.wait_for(q.get(), timeout=3)
    assert event["type"] == "call.created"
    assert event["intent"] == "dyspnea"
    assert event["high_stakes"] is True
    assert event["priority"] == 5
    assert event["assignee"]["role"] == "nurse"


async def test_patch_acknowledged_pushes_to_patient_queue(tokens) -> None:
    """`PATCH status=acknowledged` → 환자 큐에 call.acknowledged push."""
    q = _get_patient_queue(tokens["patient_id"])
    while not q.empty():
        q.get_nowait()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "pain"},
        )
        call_id = r.json()["id"]
        # station push 소비 (환자 큐에만 관심)
        _ = _get_station_queue("demo-station-helper")  # exists

        await c.patch(
            f"/calls/{call_id}",
            headers=hdr(tokens["nurse"]),
            json={"status": "acknowledged"},
        )

    event = await asyncio.wait_for(q.get(), timeout=3)
    assert event["type"] == "call.acknowledged"
    assert event["call_id"] == call_id
    assert event["assignee"]["name"]


async def test_response_pushes_to_patient_queue(tokens) -> None:
    """`POST /responses` → 환자 큐에 patient.response push (mp3 자동 재생용)."""
    q = _get_patient_queue(tokens["patient_id"])
    while not q.empty():
        q.get_nowait()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
        call_id = r.json()["id"]

        await c.post(
            f"/calls/{call_id}/responses",
            headers=hdr(tokens["nurse"]),
            json={"mp3_key": "water_coming"},
        )

    event = await asyncio.wait_for(q.get(), timeout=3)
    assert event["type"] == "patient.response"
    assert event["mp3_key"] == "water_coming"
    assert event["call_id"] == call_id


async def test_notify_pushes_renotified_to_station(tokens) -> None:
    """`POST /notify` → station 큐에 call.renotified push, attempt 증가."""
    device = "demo-station-nurse"  # 김환자 water → 이간호
    q = _get_station_queue(device)
    while not q.empty():
        q.get_nowait()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
        call_id = r.json()["id"]
        # call.created 소비
        await asyncio.wait_for(q.get(), timeout=3)

        await c.post(f"/calls/{call_id}/notify", headers=hdr(tokens["patient"]))

    event = await asyncio.wait_for(q.get(), timeout=3)
    assert event["type"] == "call.renotified"
    assert event["call_id"] == call_id
    assert event["attempt"] == 2


# ────────── SSE 엔드포인트 권한 체크 ──────────


async def test_sse_station_rejects_patient_token(tokens) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get(f"/sse/station?token={tokens['patient']}")
        assert r.status_code == 403


async def test_sse_patient_rejects_nurse_token(tokens) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get(f"/sse/patient?token={tokens['nurse']}")
        assert r.status_code == 403
