"""돌봄 앱 와이어 C1~C3, C6 전수 테스트."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from tests.conftest import hdr

# ────────── C1 알림함 ──────────


async def test_c1_queue_lists_active_calls_sorted_by_priority(
    client: AsyncClient, tokens
) -> None:
    """C1 '호출 큐' — active만, 상단에 긴급."""
    # 2건 생성: 일반 + High-stakes
    await client.post(
        "/calls/preset", headers=hdr(tokens["patient"]), json={"preset_intent": "water"}
    )
    await client.post(
        "/calls/preset", headers=hdr(tokens["patient"]), json={"preset_intent": "dyspnea"}
    )

    r = await client.get("/calls?status=active&limit=50", headers=hdr(tokens["nurse"]))
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) >= 2
    priorities = {c["priority"] for c in items}
    assert 5 in priorities  # dyspnea
    assert 2 in priorities  # water


async def test_c1_me_stats_returns_counts(client: AsyncClient, tokens) -> None:
    """C1 상단 '2층·주간·담당 N명' → `GET /caregivers/me/stats`."""
    r = await client.get("/caregivers/me/stats", headers=hdr(tokens["nurse"]))
    assert r.status_code == 200
    body = r.json()
    assert body["assigned_patient_count"] >= 0
    assert body["active_call_count"] >= 0


# ────────── C2 추천 카드 상세 ──────────


async def test_c2_get_call_detail_has_all_blocks(client: AsyncClient, tokens) -> None:
    """C2 추천 카드 — 환자/의도/담당자/플레이북/환자메모/임상가이드/근거 7블록 모두."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": "water"},
    )
    call_id = r.json()["id"]

    r = await client.get(f"/calls/{call_id}", headers=hdr(tokens["nurse"]))
    assert r.status_code == 200
    card = r.json()
    for key in [
        "id",
        "trace_id",
        "patient",
        "raw_text",
        "corrected_text",
        "intent",
        "intent_confidence",
        "intent_source",
        "high_stakes",
        "priority",
        "assignee",
        "alternates",
        "playbook",
        "patient_notes",
        "clinical_refs",
        "reason",
        "status",
        "created_at",
    ]:
        assert key in card, f"missing field: {key}"
    assert card["patient"]["room"]
    assert len(card["patient_notes"]) >= 1  # 시드에 환자별 메모 2건 있음
    assert any(ref["source"] == "clinical_guide" for ref in card["clinical_refs"])


async def test_c2_feedback_accept(client: AsyncClient, tokens) -> None:
    """C2 '조치 완료 기록' → `POST /calls/{id}/feedback action=accept`."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers=hdr(tokens["nurse"]),
        json={"action": "accept"},
    )
    assert r.status_code == 201
    assert r.json()["id"]
    assert r.json()["created_at"]


async def test_c2_feedback_modify_with_fields(client: AsyncClient, tokens) -> None:
    """C2 '다른 조치' → action=modify + modified_fields."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers=hdr(tokens["nurse"]),
        json={
            "action": "modify",
            "modified_fields": {"playbook_steps_executed": [1, 2]},
            "note": "삼킴 저하 재확인",
        },
    )
    assert r.status_code == 201


async def test_c2_feedback_reject_requires_reason(
    client: AsyncClient, tokens
) -> None:
    """C2 '거부' — reject_reason 없으면 400."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    # 사유 없음 → 400
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers=hdr(tokens["nurse"]),
        json={"action": "reject"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "VALIDATION_ERROR"

    # 사유 있음 → 201
    r = await client.post(
        f"/calls/{call_id}/feedback",
        headers=hdr(tokens["nurse"]),
        json={"action": "reject", "reject_reason": "환자 상태 재평가 필요"},
    )
    assert r.status_code == 201


async def test_c2_reassign_to_other_caregiver(client: AsyncClient, tokens) -> None:
    """C2 '간호사에게' → `PATCH /calls/{id} assignee_id=nurse_id`."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},  # 기본 담당 = 보호사
        )
    ).json()["id"]
    r = await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["nurse"]),
        json={"assignee_id": tokens["nurse_id"], "note": "간호사 이관"},
    )
    assert r.status_code == 200
    assert r.json()["assignee"]["id"] == tokens["nurse_id"]


async def test_c2_send_response_to_patient_with_yesno(
    client: AsyncClient, tokens
) -> None:
    """C2 '환자에게 묻기' → `POST /responses` with expect_answer=yes_no."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "pain"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/responses",
        headers=hdr(tokens["nurse"]),
        json={
            "mp3_key": "is_it_water",
            "expect_answer": "yes_no",
        },
    )
    assert r.status_code == 202
    assert r.json()["delivered"] is True


async def test_c2_response_requires_mp3_or_text(client: AsyncClient, tokens) -> None:
    """`POST /responses`에 mp3_key, custom_text 둘 다 없으면 400."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "water"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/responses",
        headers=hdr(tokens["nurse"]),
        json={"expect_answer": "yes_no"},
    )
    assert r.status_code == 400


async def test_patient_answer_scale_1_5(client: AsyncClient, tokens) -> None:
    """P1 '네/아니요' 또는 '1~5' → `POST /calls/{id}/answers`."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "pain"},
        )
    ).json()["id"]
    r = await client.post(
        f"/calls/{call_id}/answers",
        headers=hdr(tokens["patient"]),
        json={"answer_type": "scale_1_5", "value": 4},
    )
    assert r.status_code == 201
    assert r.json()["delivered_to_station"] is True


# ────────── C3 긴급 ──────────


async def test_c3_emergency_acknowledge_me(client: AsyncClient, tokens) -> None:
    """C3-a '내가 갈게요' → `PATCH assignee_id=me, status=acknowledged`."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "dyspnea"},
        )
    ).json()["id"]
    r = await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["helper"]),
        json={"assignee_id": tokens["helper_id"], "status": "acknowledged"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"
    assert r.json()["assignee"]["id"] == tokens["helper_id"]


async def test_c3_arrive_complete(client: AsyncClient, tokens) -> None:
    """C3-b '도착·상황 해제' → `PATCH status=completed`."""
    call_id = (
        await client.post(
            "/calls/preset",
            headers=hdr(tokens["patient"]),
            json={"preset_intent": "fall"},
        )
    ).json()["id"]
    await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["nurse"]),
        json={"status": "acknowledged"},
    )
    r = await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["nurse"]),
        json={"status": "completed"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


# ────────── 상태 전이 가드 ──────────


@pytest.mark.parametrize(
    "from_status, to_status",
    [
        ("active", "completed"),  # active → completed 직접 전이 불가
        ("active", "active"),  # self-transition도 전이 아님 (no-op) — 하지만 PATCH는 통과
    ],
)
async def test_state_transition_rules(
    client: AsyncClient, tokens, from_status: str, to_status: str
) -> None:
    """허용되지 않은 전이는 409."""
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
        json={"status": to_status},
    )
    if from_status == to_status:
        # 같은 상태 PATCH는 통과 (no-op)
        assert r.status_code == 200
    else:
        assert r.status_code == 409
        assert r.json()["error_code"] == "CONFLICT"
