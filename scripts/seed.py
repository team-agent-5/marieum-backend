"""v5 idempotent demo seed — matches agent build spec.

Patients:
  demo_pa / Kim heavy (305)   : 30 phrase_log rows across varied times + 2 preferences
  demo_pb / Park cold-start (306): 0 phrase_log rows
  demo_pc / Lee short-word (307): 10 phrase_log rows with short phrases + 2 preferences

phrase_catalog: 25 Korean dysarthria AAC phrases with real embeddings.

Usage:
    JWT_SECRET=<secret> PYTHONPATH=src python scripts/seed.py

Idempotent: re-running is safe. Checks existing rows before inserting.
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.models import Patient, PatientPreference, PhraseCatalog, PhraseLog
from app.security import hash_password
from app.services.embedding import embed_batch, embed_text

# ────────── Catalog (25 phrases with time_tags) ──────────
# time_tags: 각 phrase가 자연스럽게 요청되는 시간대 (의료·돌봄 context)
# bucket 정의: morning=06-11, afternoon=12-17, evening=18-22, night=23-05 (KST)

CATALOG_PHRASES: list[tuple[str, list[str]]] = [
    # 음료
    ("물 주세요",             ["morning", "afternoon", "evening", "night"]),   # 하루종일
    ("화장실 가고 싶어요",    ["morning", "afternoon", "evening", "night"]),   # 하루종일
    ("배가 아파요",           ["morning", "afternoon", "evening", "night"]),   # 하루종일
    ("머리가 아파요",         ["morning", "afternoon", "evening", "night"]),   # 하루종일
    ("약 주세요",             ["morning", "night"]),                           # 복약 시간
    ("미지근한 물 주세요",    ["evening", "night"]),                           # 저녁·야간 선호
    ("뜨거운 물 주세요",      ["morning", "night"]),                           # 아침·야간
    ("조금만 주세요",         ["morning", "afternoon", "evening"]),            # 식사 시간대
    ("많이 주세요",           ["morning", "afternoon", "evening"]),            # 식사 시간대
    # 식사
    ("밥 먹을래요",           ["morning", "afternoon", "evening"]),            # 3끼
    ("죽 먹을래요",           ["morning", "evening"]),                         # 아침·저녁 죽
    ("국 더 주세요",          ["morning", "afternoon", "evening"]),            # 식사 중
    # 환경·조명
    ("불 꺼주세요",           ["night"]),                                      # 취침 전
    ("불 켜주세요",           ["morning"]),                                    # 기상 후
    ("창문 열어주세요",       ["morning", "afternoon"]),                       # 낮 환기
    ("창문 닫아주세요",       ["evening", "night"]),                           # 저녁·야간
    ("텔레비전 켜주세요",     ["afternoon", "evening"]),                       # 여가 시간
    ("텔레비전 꺼주세요",     ["night"]),                                      # 취침 전
    # 소통·감정
    ("조용히 해주세요",       ["afternoon", "night"]),                         # 낮잠·취침
    ("말 걸어 주세요",        ["morning", "afternoon"]),                       # 낮 시간 소통
    ("손 좀 잡아주세요",      ["evening", "night"]),                           # 저녁·야간 불안
    # 자세·이동
    ("일으켜 주세요",         ["morning"]),                                    # 기상 후 기립
    ("앉혀 주세요",           ["morning", "afternoon"]),                       # 낮 활동
    ("담요 덮어 주세요",      ["evening", "night"]),                           # 취침 준비
    ("선풍기 틀어 주세요",    ["afternoon"]),                                  # 오후 더위
]

# ────────── Demo patients ──────────
# NOTE: 아래 user_id/password 는 데모·테스트 전용 더미 계정이다.
#   운영 환자 계정은 /auth/signup 으로 자율 가입하며 비번 해시만 DB에 저장된다.
#   이 하드코딩 값이 공개돼도 운영 환자 보안에 영향 없다.

DEMO_PATIENTS = [
    {
        "user_id": "demo_pa",
        "password": "demo1234!",
        "nickname": "김중증",
        "room": "305",
        "role": "A_heavy",
    },
    {
        "user_id": "demo_pb",
        "password": "demo1234!",
        "nickname": "박신규",
        "room": "306",
        "role": "B_cold",
    },
    {
        "user_id": "demo_pc",
        "password": "demo1234!",
        "nickname": "이짧은",
        "room": "307",
        "role": "C_short",
    },
]


# ────────── Seed functions ──────────


async def seed_catalog(session) -> None:
    # Build lookup: text -> time_tags from CATALOG_PHRASES
    tags_by_text: dict[str, list[str]] = {text: tags for text, tags in CATALOG_PHRASES}

    existing_rows: list[PhraseCatalog] = (
        await session.execute(select(PhraseCatalog))
    ).scalars().all()
    existing_texts = {r.text for r in existing_rows}

    new_phrases = [text for text, _tags in CATALOG_PHRASES if text not in existing_texts]

    # ── Insert new phrases ────────────────────────────────────────────────────
    if new_phrases:
        vecs = await embed_batch(new_phrases)
        for phrase, vec in zip(new_phrases, vecs, strict=True):
            session.add(
                PhraseCatalog(
                    text=phrase,
                    category="general",
                    embedding=vec,
                    source="manual",
                    time_tags=tags_by_text[phrase],
                )
            )
        await session.commit()
        print(f"catalog seeded: {len(new_phrases)} new rows")
    else:
        print(f"catalog already seeded ({len(existing_texts)} rows)")

    # ── Backfill time_tags on existing rows that still have empty array ───────
    backfilled = 0
    for row in existing_rows:
        if not row.time_tags and row.text in tags_by_text:
            row.time_tags = tags_by_text[row.text]
            backfilled += 1
    if backfilled:
        await session.commit()
        print(f"catalog time_tags backfilled: {backfilled} rows updated")
    else:
        print("catalog time_tags: all existing rows already tagged")


async def seed_patients(session) -> dict[str, Patient]:
    patient_rows: dict[str, Patient] = {}
    for p in DEMO_PATIENTS:
        existing = (
            await session.execute(select(Patient).where(Patient.user_id == p["user_id"]))
        ).scalar_one_or_none()
        if existing is not None:
            print(f"patient {p['user_id']} already exists — skipping insert")
            patient_rows[p["role"]] = existing
            continue
        row = Patient(
            user_id=p["user_id"],
            password_hash=hash_password(p["password"]),
            nickname=p["nickname"],
            room=p["room"],
            tts_rate="normal",
            tts_volume="normal",
            onboarding_completed=True,
        )
        session.add(row)
        await session.flush()
        patient_rows[p["role"]] = row
        print(f"patient created: {p['user_id']} ({p['nickname']}, room {p['room']})")
    await session.commit()
    return patient_rows


async def seed_phrase_logs(
    session,
    patient_a: Patient,
    patient_c: Patient,
) -> None:
    # Check if already seeded (idempotent)
    a_count = (
        await session.execute(
            select(PhraseLog).where(PhraseLog.patient_id == patient_a.id)
        )
    ).scalars().all()
    c_count = (
        await session.execute(
            select(PhraseLog).where(PhraseLog.patient_id == patient_c.id)
        )
    ).scalars().all()

    if a_count:
        print(f"phrase_log for {patient_a.user_id} already has {len(a_count)} rows — skipping")
    else:
        await _seed_heavy_logs(session, patient_a)

    if c_count:
        print(f"phrase_log for {patient_c.user_id} already has {len(c_count)} rows — skipping")
    else:
        await _seed_short_logs(session, patient_c)

    await session.commit()


async def _seed_heavy_logs(session, patient: Patient) -> None:
    """P-A heavy: ~30 rows across varied times/phrases to populate freq/time/recency."""
    now = datetime.now(UTC)
    entries: list[tuple[str, str, int | None]] = []

    # Heaviest: 미지근한 물 주세요 ~12 times at evening hours
    for _ in range(12):
        entries.append(("미지근한 물 주세요", "voice", 1))
    # 물 주세요 ~6 times
    for _ in range(6):
        entries.append(("물 주세요", "voice", 1))
    # 약 주세요 ~4 times
    for _ in range(4):
        entries.append(("약 주세요", "voice", 2))
    # 담요 덮어 주세요 ~3 times
    for _ in range(3):
        entries.append(("담요 덮어 주세요", "preset", None))
    # misc 5 times
    misc_phrases = ["조용히 해주세요", "불 꺼주세요", "창문 닫아주세요", "말 걸어 주세요", "손 좀 잡아주세요"]
    for ph in misc_phrases:
        entries.append((ph, "voice", 3))

    # Precompute embeddings in batch
    phrases_only = [e[0] for e in entries]
    vecs = await embed_batch(phrases_only)

    rng = random.Random(42)
    for i, ((phrase, source, rank), vec) in enumerate(zip(entries, vecs, strict=True)):
        # Distribute times: last 72 hours, evening-biased for 미지근한 물 주세요
        if phrase == "미지근한 물 주세요":
            # Evening: 18:00~22:00 KST = 09:00~13:00 UTC
            base_hours = rng.choice([1, 25, 49, 73])
            minutes = rng.randint(0, 59)
            created_utc = now - timedelta(hours=base_hours, minutes=minutes)
            # Shift to ~20:00 KST offset within that day
            created_utc = created_utc.replace(
                hour=11,  # 11 UTC = 20 KST
                minute=rng.randint(0, 59),
            )
        else:
            hours_ago = rng.randint(1, 72)
            created_utc = now - timedelta(hours=hours_ago, minutes=rng.randint(0, 59))

        session.add(
            PhraseLog(
                patient_id=patient.id,
                trace_id=f"seed-a-{i:04d}",
                source=source,
                selected_phrase=phrase,
                selected_rank=rank,
                partial_transcript=phrase.split()[0] if source == "voice" else None,
                final_transcript=phrase if source != "preset" else None,
                created_at=created_utc.replace(tzinfo=None),
                embedding=vec,
            )
        )
    print(f"phrase_log seeded for {patient.user_id}: {len(entries)} rows")


async def _seed_short_logs(session, patient: Patient) -> None:
    """P-C short-word: ~10 rows with short phrases."""
    now = datetime.now(UTC)
    short_entries: list[tuple[str, str]] = [
        ("네", "preset"),
        ("아니요", "preset"),
        ("물 주세요", "voice"),
        ("네", "preset"),
        ("아니요", "preset"),
        ("물 주세요", "preset"),
        ("네", "voice_bypass"),
        ("아니요", "voice_bypass"),
        ("물 주세요", "voice"),
        ("네", "preset"),
    ]

    phrases_only = [e[0] for e in short_entries]
    vecs = await embed_batch(phrases_only)

    rng = random.Random(99)
    for i, ((phrase, source), vec) in enumerate(zip(short_entries, vecs, strict=True)):
        hours_ago = rng.randint(1, 48)
        created_utc = now - timedelta(hours=hours_ago, minutes=rng.randint(0, 59))
        session.add(
            PhraseLog(
                patient_id=patient.id,
                trace_id=f"seed-c-{i:04d}",
                source=source,
                selected_phrase=phrase,
                selected_rank=None,
                final_transcript=phrase if source == "voice_bypass" else None,
                created_at=created_utc.replace(tzinfo=None),
                embedding=vec,
            )
        )
    print(f"phrase_log seeded for {patient.user_id}: {len(short_entries)} rows")


async def seed_preferences(
    session,
    patient_a: Patient,
    patient_c: Patient,
) -> None:
    # Check idempotency
    a_prefs = (
        await session.execute(
            select(PatientPreference).where(PatientPreference.patient_id == patient_a.id)
        )
    ).scalars().all()
    c_prefs = (
        await session.execute(
            select(PatientPreference).where(PatientPreference.patient_id == patient_c.id)
        )
    ).scalars().all()

    added = 0

    if not a_prefs:
        pref_a1_text = "미지근한 물 주세요"
        pref_a2_text = "저녁 시간대에는 미지근한 물을 주로 선호합니다"
        a_vecs = await embed_batch([pref_a1_text, pref_a2_text])
        session.add(
            PatientPreference(
                patient_id=patient_a.id,
                preference_type="preferred_phrase",
                content=pref_a1_text,
                boost_phrases=["미지근한 물 주세요"],
                embedding=a_vecs[0],
            )
        )
        session.add(
            PatientPreference(
                patient_id=patient_a.id,
                preference_type="time_preference",
                content=pref_a2_text,
                boost_phrases=["미지근한 물 주세요", "물 주세요"],
                embedding=a_vecs[1],
            )
        )
        added += 2
        print(f"preferences seeded for {patient_a.user_id}: 2 rows")
    else:
        print(f"preferences for {patient_a.user_id} already exist ({len(a_prefs)} rows) — skipping")

    if not c_prefs:
        pref_c1_text = "네"
        pref_c2_text = "짧은 응답 선호: 네, 예, 아니요"
        c_vecs = await embed_batch([pref_c1_text, pref_c2_text])
        session.add(
            PatientPreference(
                patient_id=patient_c.id,
                preference_type="preferred_phrase",
                content=pref_c1_text,
                boost_phrases=["네", "예"],
                embedding=c_vecs[0],
            )
        )
        session.add(
            PatientPreference(
                patient_id=patient_c.id,
                preference_type="topic_preference",
                content=pref_c2_text,
                boost_phrases=["네", "예", "아니요"],
                embedding=c_vecs[1],
            )
        )
        added += 2
        print(f"preferences seeded for {patient_c.user_id}: 2 rows")
    else:
        print(f"preferences for {patient_c.user_id} already exist ({len(c_prefs)} rows) — skipping")

    await session.commit()
    if added:
        print(f"patient_preference total new rows: {added}")


# ────────── main ──────────


async def main() -> None:
    async with AsyncSessionLocal() as session:
        print("=== Seeding phrase_catalog ===")
        await seed_catalog(session)

        print("\n=== Seeding patients ===")
        patient_rows = await seed_patients(session)

        patient_a = patient_rows["A_heavy"]
        patient_c = patient_rows["C_short"]

        print("\n=== Seeding phrase_log ===")
        await seed_phrase_logs(session, patient_a, patient_c)

        print("\n=== Seeding patient_preference ===")
        await seed_preferences(session, patient_a, patient_c)

        print("\n=== Seed complete ===")
        print(f"  phrase_catalog : {len(CATALOG_PHRASES)} phrases")
        print(f"  patients       : {len(DEMO_PATIENTS)} (demo_pa heavy / demo_pb cold-start / demo_pc short-word)")
        print(f"  demo_pa id     : {patient_a.id}")
        print(f"  demo_pc id     : {patient_c.id}")


if __name__ == "__main__":
    asyncio.run(main())
