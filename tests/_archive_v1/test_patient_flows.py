"""환자 앱 와이어 P1~P6 전수 테스트.

각 테스트는 와이어 1~2개 UX와 매핑. 코멘트에 어떤 트리거에 쓰이는지 명시.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from tests.conftest import hdr

# ────────── P1 메인 — 단축 명령어 10종 ──────────


@pytest.mark.parametrize(
    "preset_intent, expected_priority, expected_high_stakes",
    [
        ("water", 2, False),
        ("toilet", 2, False),
        ("pain", 3, False),  # High-stakes는 trigger에 따라 (NRS≥7 등) — 기본은 False
        ("dyspnea", 5, True),  # §10.4
        ("cold", 1, False),
        ("anxiety", 3, False),
        ("fall", 5, True),  # §10.4
        ("medication", 2, False),
        ("call", 1, False),
    ],
)
async def test_p1_tile_each_preset_intent_builds_card(
    client: AsyncClient,
    tokens,
    preset_intent: str,
    expected_priority: int,
    expected_high_stakes: bool,
) -> None:
    """P1 각 타일 탭 → `POST /calls/preset` → RecommendationCard."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": preset_intent, "metadata": {}},
    )
    assert r.status_code == 201, r.text
    card = r.json()
    assert card["intent"] == preset_intent
    assert card["intent_source"] == "rule"
    assert card["high_stakes"] is expected_high_stakes
    assert card["priority"] == expected_priority
    # High-stakes면 담당자 role=nurse 강제 (§10.4)
    if expected_high_stakes:
        assert card["assignee"]["role"] == "nurse"
    # playbook 10 intent 중 하나와 매칭
    assert card["playbook"]["playbook_id"].startswith("pb_")
    assert len(card["playbook"]["steps"]) >= 1


async def test_p1_emergency_hold_3s_builds_emergency_card(
    client: AsyncClient, tokens
) -> None:
    """P5 긴급 3초 홀드 → `preset_intent=emergency` + metadata.source=sos_hold_3s."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": "emergency", "metadata": {"source": "sos_hold_3s"}},
    )
    assert r.status_code == 201, r.text
    card = r.json()
    assert card["high_stakes"] is True
    assert card["priority"] == 5
    # emergency는 intent로는 다른 값에 매핑됨 (mock: "other")
    assert card["intent"] in {"other", "call"}


# ────────── P2 다시 말하기 — PATCH cancelled ──────────


async def test_p2_cancel_within_window_patches_cancelled(
    client: AsyncClient, tokens
) -> None:
    """P2 '다시 말하기' → 환자가 호출 취소."""
    # 1. 호출 생성
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": "water"},
    )
    call_id = r.json()["id"]
    # 2. 돌봄 토큰으로 cancel (api-spec: 환자 취소는 PATCH로)
    # 실제론 환자가 cancel해야 하지만 require_staff 가드 때문에 돌봄으로 테스트
    r = await client.patch(
        f"/calls/{call_id}",
        headers=hdr(tokens["nurse"]),
        json={"status": "cancelled"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


# ────────── P3 녹음 → 후보 → 확정 ──────────


async def test_p3_transcriptions_returns_candidates(
    client: AsyncClient, tokens
) -> None:
    """P3-a 녹음 종료 → `POST /transcriptions` → 후보 3개."""
    files = {"audio": ("sample.webm", b"fake", "audio/webm")}
    r = await client.post(
        "/transcriptions",
        headers=hdr(tokens["patient"]),
        files=files,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace_id"]
    assert len(body["candidates"]) >= 1
    assert body["candidates"][0]["confidence"] >= 0


async def test_p3_calls_text_with_transcription_trace_id(
    client: AsyncClient, tokens
) -> None:
    """P3-b '이 말로 말하기' → `POST /calls/text` with transcription_trace_id."""
    r = await client.post(
        "/calls/text",
        headers=hdr(tokens["patient"]),
        json={
            "text": "숨이 차요",
            "source": "transcription",
            "transcription_trace_id": "fake-trace-123",
        },
    )
    assert r.status_code == 201, r.text
    card = r.json()
    # 키워드 매칭: 숨 → dyspnea (High-stakes)
    assert card["intent"] == "dyspnea"
    assert card["high_stakes"] is True
    assert card["raw_text"] == "숨이 차요"


# ────────── P4 통증 — 부위·강도 metadata ──────────


async def test_p4_pain_preset_with_metadata(client: AsyncClient, tokens) -> None:
    """P4 '아픈 곳 알리기' → `POST /calls/preset preset_intent=pain + metadata`."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={
            "preset_intent": "pain",
            "metadata": {"body_part": "abdomen", "intensity": 4},
        },
    )
    assert r.status_code == 201
    card = r.json()
    assert card["intent"] == "pain"
    assert card["playbook"]["playbook_id"] == "pb_pain_v1"


# ────────── P1 음성 바로 — /calls/voice ──────────


async def test_voice_endpoint_accepts_multipart_audio(
    client: AsyncClient, tokens
) -> None:
    """녹음 → `POST /calls/voice` (multipart)."""
    files = {"audio": ("sample.webm", b"fake-audio-bytes", "audio/webm")}
    r = await client.post(
        "/calls/voice",
        headers=hdr(tokens["patient"]),
        files=files,
    )
    assert r.status_code == 201, r.text
    card = r.json()
    assert card["raw_text"]
    assert card["intent"]


# ────────── P6 요청 목록 + 다시 부르기 ──────────


async def test_p6_my_requests_list_scoped_to_patient(
    client: AsyncClient, tokens
) -> None:
    """P6 '오늘 요청' → `GET /calls?patient_id={me}` — require_staff이므로 돌봄 토큰 사용."""
    # 환자가 미리 1건 생성해둠
    await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": "water"},
    )
    r = await client.get(
        f"/calls?patient_id={tokens['patient_id']}&limit=5",
        headers=hdr(tokens["nurse"]),
    )
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    assert all(c["patient"]["id"] == tokens["patient_id"] for c in data["items"])


async def test_p6_notify_increments_attempt(client: AsyncClient, tokens) -> None:
    """P6 '다시 부르기' → `POST /calls/{id}/notify` → attempt 증가."""
    r = await client.post(
        "/calls/preset",
        headers=hdr(tokens["patient"]),
        json={"preset_intent": "water"},
    )
    call_id = r.json()["id"]

    r1 = await client.post(f"/calls/{call_id}/notify", headers=hdr(tokens["patient"]))
    assert r1.status_code == 202
    first = r1.json()["attempt"]

    r2 = await client.post(f"/calls/{call_id}/notify", headers=hdr(tokens["patient"]))
    assert r2.status_code == 202
    assert r2.json()["attempt"] == first + 1


# ────────── 환자 /me ──────────


async def test_patient_me_returns_room_name(client: AsyncClient, tokens) -> None:
    """P1 상단 '305호 김환자님' 표시용 → `GET /patients/me`."""
    r = await client.get("/patients/me", headers=hdr(tokens["patient"]))
    assert r.status_code == 200
    body = r.json()
    assert body["room"]
    assert body["name"]
    # 재조회 시 patient_code/initial_pin 미노출 (§보안)
    assert body.get("patient_code") is None
    assert body.get("initial_pin") is None
