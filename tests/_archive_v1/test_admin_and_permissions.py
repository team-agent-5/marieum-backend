"""관리자 A1 플로우 + 역할별 권한 매트릭스."""

from __future__ import annotations

from httpx import AsyncClient
from tests.conftest import hdr

# ────────── A1 관리자 ──────────


async def test_a1_create_patient_full_fields(client: AsyncClient, tokens) -> None:
    """A1 환자 등록 — 전필드 + notes 자동 RAG 적재."""
    r = await client.post(
        "/patients",
        headers=hdr(tokens["admin"]),
        json={
            "room": "T301",
            "name": "통합테스트A",
            "birthdate": "1945-03-01",
            "disease": "뇌졸중 후유증",
            "swallow_level": "reduced",
            "default_caregiver_id": tokens["nurse_id"],
            "notes": ["통합테스트 메모 1", "통합테스트 메모 2"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["patient_code"]
    assert body["initial_pin"]
    assert body["notes_count"] == 2
    assert body["default_caregiver"]["id"] == tokens["nurse_id"]
    assert body["default_caregiver"]["role"] == "nurse"


async def test_a1_notes_crud(client: AsyncClient, tokens) -> None:
    """환자 메모 추가·조회·삭제."""
    # 환자 생성
    r = await client.post(
        "/patients",
        headers=hdr(tokens["admin"]),
        json={
            "room": "T302",
            "name": "통합테스트B",
            "disease": "ALS",
            "notes": ["초기 메모"],
        },
    )
    patient_id = r.json()["id"]

    # 메모 추가
    r = await client.post(
        f"/patients/{patient_id}/notes",
        headers=hdr(tokens["admin"]),
        json={"content": "추가 메모 — 밤 호흡 보조 필수"},
    )
    assert r.status_code == 201
    note_id = r.json()["id"]
    assert r.json()["content"] == "추가 메모 — 밤 호흡 보조 필수"

    # 메모 목록
    r = await client.get(
        f"/patients/{patient_id}/notes", headers=hdr(tokens["admin"])
    )
    assert r.status_code == 200
    assert len(r.json()) >= 2

    # 메모 삭제 (admin만)
    r = await client.delete(
        f"/patients/{patient_id}/notes/{note_id}",
        headers=hdr(tokens["admin"]),
    )
    assert r.status_code == 204

    # 삭제 후 목록에서 사라짐 (soft delete)
    r = await client.get(
        f"/patients/{patient_id}/notes", headers=hdr(tokens["admin"])
    )
    remaining = [n["id"] for n in r.json()]
    assert note_id not in remaining


async def test_a1_patient_validation_min_length(client: AsyncClient, tokens) -> None:
    """빈 room/name은 Pydantic validation으로 400."""
    r = await client.post(
        "/patients",
        headers=hdr(tokens["admin"]),
        json={"room": "", "name": "", "disease": ""},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["details"]


# ────────── 권한 매트릭스 ──────────


async def test_no_token_returns_401(client: AsyncClient) -> None:
    """토큰 없이 보호 엔드포인트 호출."""
    r = await client.get("/calls")
    assert r.status_code == 401
    assert r.json()["error_code"] == "UNAUTHORIZED"


async def test_patient_cannot_list_calls(client: AsyncClient, tokens) -> None:
    """환자는 /calls 목록 접근 불가 (require_staff)."""
    r = await client.get("/calls", headers=hdr(tokens["patient"]))
    assert r.status_code == 403


async def test_patient_cannot_create_patient(client: AsyncClient, tokens) -> None:
    """환자는 POST /patients 불가 (require_admin)."""
    r = await client.post(
        "/patients",
        headers=hdr(tokens["patient"]),
        json={"room": "X", "name": "x", "disease": "x"},
    )
    assert r.status_code == 403


async def test_caregiver_cannot_create_patient(client: AsyncClient, tokens) -> None:
    """돌봄도 POST /patients 불가."""
    r = await client.post(
        "/patients",
        headers=hdr(tokens["nurse"]),
        json={"room": "X", "name": "x", "disease": "x"},
    )
    assert r.status_code == 403


async def test_caregiver_cannot_delete_note(client: AsyncClient, tokens) -> None:
    """돌봄은 메모 추가는 가능하지만 삭제는 admin only."""
    # admin으로 환자 + 메모 생성
    r = await client.post(
        "/patients",
        headers=hdr(tokens["admin"]),
        json={"room": "T303", "name": "통합테스트C", "disease": "x", "notes": ["m"]},
    )
    patient_id = r.json()["id"]
    r = await client.post(
        f"/patients/{patient_id}/notes",
        headers=hdr(tokens["admin"]),
        json={"content": "x"},
    )
    note_id = r.json()["id"]

    # 돌봄이 DELETE → 403
    r = await client.delete(
        f"/patients/{patient_id}/notes/{note_id}",
        headers=hdr(tokens["nurse"]),
    )
    assert r.status_code == 403


async def test_caregiver_cannot_create_call(client: AsyncClient, tokens) -> None:
    """돌봄이 환자 대신 호출 생성 불가 (require_patient)."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["nurse"]),
        json={"preset_intent": "water"},
    )
    assert r.status_code == 403


async def test_patient_cannot_submit_feedback(client: AsyncClient, tokens) -> None:
    """환자는 feedback 제출 불가 (require_caregiver)."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers=hdr(tokens["patient"]),
        json={"action": "accept"},
    )
    assert r.status_code == 403


async def test_admin_can_list_calls(client: AsyncClient, tokens) -> None:
    """관리자는 돌봄과 동일하게 /calls 접근 가능 (require_staff)."""
    r = await client.get("/calls", headers=hdr(tokens["admin"]))
    assert r.status_code == 200


# ────────── Caregivers 공개 리스트 ──────────


async def test_caregivers_public_without_auth(client: AsyncClient) -> None:
    """로그인 전 화면 — 인증 없이 조회 가능."""
    r = await client.get("/caregivers/public")
    assert r.status_code == 200
    data = r.json()
    # device_id 미노출
    assert all("device_id" not in c for c in data)
