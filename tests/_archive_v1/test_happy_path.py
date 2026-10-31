"""Happy-path 통합 테스트 — 와이어프레임 핵심 플로우 1건씩.

전제: `python scripts/seed.py` 사전 실행으로 간호사 1·요양보호사 1·환자 3명이 있음.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

PATIENT_PIN = "4321"
CAREGIVER_PIN = "1234"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "marieum2026"


async def _login_patient(client: AsyncClient) -> str:
    r = await client.post(
        "/auth/patients/login",
        json={"patient_code": "ignored-by-happy-path", "pin": PATIENT_PIN},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


async def _login_admin(client: AsyncClient) -> str:
    r = await client.post(
        "/auth/admin/login",
        json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


async def _login_first_nurse(client: AsyncClient) -> tuple[str, str]:
    r = await client.get("/caregivers/public")
    assert r.status_code == 200
    nurse_id = next(c["id"] for c in r.json() if c["role"] == "nurse")
    r = await client.post(
        "/auth/caregivers/login",
        json={"caregiver_id": nurse_id, "pin": CAREGIVER_PIN},
    )
    assert r.status_code == 200, r.text
    return nurse_id, r.json()["access_token"]


# ────────── System ──────────


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert r.headers.get("x-trace-id")  # TraceIdMiddleware 반드시 echo


# ────────── Auth ──────────


@pytest.mark.asyncio
async def test_admin_login_then_me(client: AsyncClient) -> None:
    token = await _login_admin(client)
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["type"] == "admin"


@pytest.mark.asyncio
async def test_patient_login_wrong_pin_returns_401(client: AsyncClient) -> None:
    r = await client.post(
        "/auth/patients/login",
        json={"patient_code": "302-KIM-ABCD", "pin": "0000"},
    )
    assert r.status_code == 401
    assert r.json()["error_code"] == "UNAUTHORIZED"


# ────────── Caregivers / Playbooks ──────────


@pytest.mark.asyncio
async def test_caregivers_public_lists_seeded(client: AsyncClient) -> None:
    r = await client.get("/caregivers/public")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    roles = {c["role"] for c in data}
    assert "nurse" in roles and "caregiver" in roles


@pytest.mark.asyncio
async def test_playbooks_10_intents(client: AsyncClient) -> None:
    nurse_id, token = await _login_first_nurse(client)
    r = await client.get("/playbooks", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 10
    intents = {pb["intent"] for pb in data}
    for required in ("water", "dyspnea", "fall", "pain"):
        assert required in intents


# ────────── Core: 환자 preset 호출 → High-stakes 룰 상향 ──────────


@pytest.mark.asyncio
async def test_dyspnea_preset_elevates_to_high_stakes(client: AsyncClient) -> None:
    token = await _login_patient(client)
    r = await client.post(
        "/calls/preset",
        headers={"Authorization": f"Bearer {token}"},
        json={"preset_intent": "dyspnea", "metadata": {}},
    )
    assert r.status_code == 201, r.text
    card = r.json()
    assert card["intent"] == "dyspnea"
    assert card["high_stakes"] is True
    assert card["priority"] == 5
    assert card["assignee"]["role"] == "nurse"  # §10.4 간호사 직통
    assert card["playbook"]["playbook_id"] == "pb_dyspnea_v1"
    assert len(card["playbook"]["steps"]) >= 3
    # 환자 메모 + 임상 가이드 RAG 결과 포함
    assert len(card["patient_notes"]) >= 1
    assert any(ref["source"] == "clinical_guide" for ref in card["clinical_refs"])


# ────────── Core: 상태 전이 + HITL + 재알림 ──────────


@pytest.mark.asyncio
async def test_state_machine_feedback_notify_flow(client: AsyncClient) -> None:
    patient_token = await _login_patient(client)
    nurse_id, nurse_token = await _login_first_nurse(client)

    # 1. Call 생성
    r = await client.post(
        "/calls/preset",
        headers={"Authorization": f"Bearer {patient_token}"},
        json={"preset_intent": "water", "metadata": {}},
    )
    assert r.status_code == 201
    call_id = r.json()["id"]

    # 2. 돌봄이 수락 (active → acknowledged)
    r = await client.patch(
        f"/calls/{call_id}",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"status": "acknowledged"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"

    # 3. 상태 전이 불가 → 409
    r = await client.patch(
        f"/calls/{call_id}",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"status": "rejected"},
    )
    assert r.status_code == 409
    assert r.json()["error_code"] == "CONFLICT"

    # 4. HITL accept
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"action": "accept"},
    )
    assert r.status_code == 201

    # 5. 응답 전송 (환자단말로 안심 멘트)
    r = await client.post(
        f"/calls/{call_id}/responses",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"mp3_key": "water_coming"},
    )
    assert r.status_code == 202
    assert r.json()["delivered"] is True

    # 6. 환자 "다시 부르기"
    r = await client.post(
        f"/calls/{call_id}/notify",
        headers={"Authorization": f"Bearer {patient_token}"},
    )
    assert r.status_code == 202
    assert r.json()["attempt"] >= 2

    # 7. 완료 처리 (acknowledged → completed)
    r = await client.patch(
        f"/calls/{call_id}",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"status": "completed"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"

    # 8. 완료 이후 전이 불가 → 409
    r = await client.patch(
        f"/calls/{call_id}",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"status": "acknowledged"},
    )
    assert r.status_code == 409


# ────────── 환자 등록 (A1) ──────────


@pytest.mark.asyncio
async def test_admin_can_create_patient_with_notes(client: AsyncClient) -> None:
    admin_token = await _login_admin(client)
    r = await client.post(
        "/patients",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "room": "999",
            "name": "테스트",
            "disease": "뇌졸중 후유증",
            "swallow_level": "normal",
            "notes": ["테스트 메모 1", "테스트 메모 2"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["room"] == "999"
    assert body["name"] == "테스트"
    assert body["patient_code"]  # 1회만 노출
    assert body["initial_pin"]
    assert body["notes_count"] == 2

    # 재조회 시 code/pin 미노출
    patient_id = body["id"]
    r2 = await client.get(
        f"/patients/{patient_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r2.status_code == 200
    assert r2.json().get("patient_code") is None
    assert r2.json().get("initial_pin") is None
    assert r2.json()["notes_count"] == 2
