"""Demo seed — AAC 환자 persona 데이터 생성.

phrase_catalog + 3명 데모 환자 (phrase_log + preferences) 생성.
Weighted random sampling으로 환자 프로필에 맞는 자연스러운 분포 생성.

Usage:
    JWT_SECRET=<secret> PYTHONPATH=src python scripts/seed.py           # idempotent
    JWT_SECRET=<secret> PYTHONPATH=src python scripts/seed.py --reset   # phrase_log + prefs wipe
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select, text as sql_text

from app.db import AsyncSessionLocal
from app.models import Patient, PatientPreference, PhraseCatalog, PhraseLog
from app.security import hash_password
from app.services.embedding import embed_batch


# ────────── Catalog ──────────
# bucket: morning=06-11, afternoon=12-17, evening=18-22, night=23-05 (KST)

CATALOG_PHRASES: list[tuple[str, list[str]]] = [
    ("물 주세요",             ["morning", "afternoon", "evening", "night"]),
    ("미지근한 물 주세요",    ["evening", "night"]),
    ("뜨거운 물 주세요",      ["morning", "night"]),
    ("약 주세요",             ["morning", "evening", "night"]),
    ("진통제 주세요",         ["evening", "night"]),
    ("조금만 주세요",         ["morning", "afternoon", "evening"]),
    ("많이 주세요",           ["morning", "afternoon", "evening"]),
    ("밥 먹을래요",           ["morning", "afternoon", "evening"]),
    ("죽 먹을래요",           ["morning", "evening"]),
    ("국 더 주세요",          ["morning", "afternoon", "evening"]),
    ("화장실 가고 싶어요",    ["morning", "afternoon", "evening", "night"]),
    ("배가 아파요",           ["morning", "afternoon", "evening", "night"]),
    ("머리가 아파요",         ["morning", "afternoon", "evening", "night"]),
    ("불 꺼주세요",           ["night"]),
    ("불 켜주세요",           ["morning"]),
    ("창문 열어주세요",       ["morning", "afternoon"]),
    ("창문 닫아주세요",       ["evening", "night"]),
    ("텔레비전 켜주세요",     ["afternoon", "evening"]),
    ("텔레비전 꺼주세요",     ["night"]),
    ("조용히 해주세요",       ["afternoon", "night"]),
    ("말 걸어 주세요",        ["morning", "afternoon"]),
    ("손 좀 잡아주세요",      ["evening", "night"]),
    ("일으켜 주세요",         ["morning"]),
    ("앉혀 주세요",           ["morning", "afternoon"]),
    ("담요 덮어 주세요",      ["evening", "night"]),
    ("선풍기 틀어 주세요",    ["afternoon"]),
]


# phrase를 topic으로 묶어 persona weight 적용 시 사용
PHRASE_TOPICS: dict[str, list[str]] = {
    "hydration":     ["물 주세요", "미지근한 물 주세요", "뜨거운 물 주세요"],
    "medication":    ["약 주세요", "진통제 주세요"],
    "food":          ["밥 먹을래요", "죽 먹을래요", "국 더 주세요"],
    "quantity":      ["조금만 주세요", "많이 주세요"],
    "pain":          ["머리가 아파요", "배가 아파요"],
    "comfort":       ["담요 덮어 주세요", "손 좀 잡아주세요"],
    "lighting":      ["불 꺼주세요", "불 켜주세요"],
    "ventilation":   ["창문 열어주세요", "창문 닫아주세요"],
    "entertainment": ["텔레비전 켜주세요", "텔레비전 꺼주세요"],
    "communication": ["말 걸어 주세요", "조용히 해주세요"],
    "mobility":      ["일으켜 주세요", "앉혀 주세요"],
    "lavatory":      ["화장실 가고 싶어요"],
}


# ────────── Demo patients ──────────
# NOTE: 데모·테스트 전용 더미 계정. 운영은 /auth/signup.

DEMO_PATIENTS = [
    {"user_id": "demo_pa", "password": "demo1234!", "nickname": "김중증", "room": "305", "profile": "evening_active_senior"},
    {"user_id": "demo_pb", "password": "demo1234!", "nickname": "박신규", "room": "306", "profile": "newly_registered"},
    {"user_id": "demo_pc", "password": "demo1234!", "nickname": "이짧은", "room": "307", "profile": "minimal_speech"},
]


# ────────── Patient profiles — bucket별 topic 가중치 ──────────
# generate_logs()가 bucket 분포에 따라 각 slot마다 topic weighted-pick → phrase random-pick.
# 재현성은 random seed 고정으로 보장.

PATIENT_PROFILES: dict[str, dict] = {
    "evening_active_senior": {
        "total_logs": 60,
        "bucket_distribution": {
            "morning":   0.20,
            "afternoon": 0.17,
            "evening":   0.33,
            "night":     0.30,
        },
        "topic_weights_by_bucket": {
            "morning":   {"medication": 4, "hydration": 3, "food": 3, "mobility": 2, "ventilation": 1, "communication": 1, "lighting": 1},
            "afternoon": {"entertainment": 3, "hydration": 3, "communication": 3, "ventilation": 2, "mobility": 1, "quantity": 1, "food": 1},
            "evening":   {"hydration": 5, "food": 5, "comfort": 3, "quantity": 3, "ventilation": 2, "entertainment": 1},
            "night":     {"hydration": 4, "medication": 5, "pain": 4, "comfort": 3, "lighting": 2},
        },
        "preferences": [
            {"type": "preferred_phrase",  "content": "미지근한 물 주세요",
             "boost": ["미지근한 물 주세요"]},
            {"type": "time_preference",   "content": "저녁과 야간에 수분·안락 요청이 잦음",
             "boost": ["미지근한 물 주세요", "담요 덮어 주세요"]},
            {"type": "time_preference",   "content": "복약 시 물을 함께 요청하는 경향",
             "boost": ["약 주세요", "뜨거운 물 주세요", "물 주세요"]},
        ],
        "random_seed": 42,
    },
    "newly_registered": {
        "total_logs": 0,
        "preferences": [],
    },
    "minimal_speech": {
        "total_logs": 10,
        "bucket_distribution": {"evening": 0.4, "night": 0.6},
        "fixed_phrases": {   # 짧은 응답은 topic 추상화보다 직접 분포가 자연스러움
            "아니요": 4,
            "네":     4,
            "물 주세요": 2,
        },
        "preferences": [
            {"type": "preferred_phrase", "content": "네",
             "boost": ["네", "예"]},
            {"type": "topic_preference", "content": "짧은 응답 선호: 네, 예, 아니요",
             "boost": ["네", "예", "아니요"]},
        ],
        "random_seed": 99,
    },
}


# bucket → UTC hour (phrase_log.time_bucket_kst generated column이 UTC → KST 변환해 bucket 분류)
# KST = UTC+9. 각 bucket 중간 시각으로 세팅 (분·초는 random).
BUCKET_UTC_HOUR: dict[str, int] = {
    "morning":   1,   # KST 10
    "afternoon": 5,   # KST 14
    "evening":  11,   # KST 20
    "night":   17,   # KST 02 (당일 UTC)
}


def _generate_log_entries(profile: dict) -> list[tuple[str, str]]:
    """profile → [(phrase, bucket), ...]. source는 호출측에서 결정."""
    total = profile["total_logs"]
    if total == 0:
        return []

    rng = random.Random(profile["random_seed"])

    # 짧은 응답 환자처럼 phrase가 작은 집합일 때는 fixed 분포 사용
    if "fixed_phrases" in profile:
        entries: list[tuple[str, str]] = []
        buckets = list(profile["bucket_distribution"].keys())
        bucket_weights = list(profile["bucket_distribution"].values())
        for phrase, cnt in profile["fixed_phrases"].items():
            for _ in range(cnt):
                bucket = rng.choices(buckets, weights=bucket_weights)[0]
                entries.append((phrase, bucket))
        rng.shuffle(entries)
        return entries

    # 일반 환자: bucket 분포 → topic 가중치 → phrase 랜덤 선택
    entries: list[tuple[str, str]] = []
    for bucket, fraction in profile["bucket_distribution"].items():
        bucket_count = max(1, round(total * fraction))
        topic_weights = profile["topic_weights_by_bucket"][bucket]
        topics = list(topic_weights.keys())
        weights = list(topic_weights.values())
        for _ in range(bucket_count):
            topic = rng.choices(topics, weights=weights)[0]
            phrase = rng.choice(PHRASE_TOPICS[topic])
            entries.append((phrase, bucket))
    rng.shuffle(entries)
    return entries


# ────────── Seed functions ──────────


async def seed_catalog(session) -> None:
    tags_by_text: dict[str, list[str]] = {t: tags for t, tags in CATALOG_PHRASES}
    existing_rows: list[PhraseCatalog] = (
        await session.execute(select(PhraseCatalog))
    ).scalars().all()
    existing_texts = {r.text for r in existing_rows}

    new_phrases = [t for t, _ in CATALOG_PHRASES if t not in existing_texts]
    if new_phrases:
        vecs = await embed_batch(new_phrases)
        for phrase, vec in zip(new_phrases, vecs, strict=True):
            session.add(PhraseCatalog(
                text=phrase, category="general",
                embedding=vec, source="manual",
                time_tags=tags_by_text[phrase],
            ))
        await session.commit()
        print(f"catalog seeded: {len(new_phrases)} new rows")
    else:
        print(f"catalog already seeded ({len(existing_texts)} rows)")

    backfilled = 0
    for row in existing_rows:
        if row.text in tags_by_text and row.time_tags != tags_by_text[row.text]:
            row.time_tags = tags_by_text[row.text]
            backfilled += 1
    if backfilled:
        await session.commit()
        print(f"catalog time_tags updated: {backfilled} rows")


async def seed_patients(session) -> dict[str, Patient]:
    rows: dict[str, Patient] = {}
    for p in DEMO_PATIENTS:
        existing = (await session.execute(
            select(Patient).where(Patient.user_id == p["user_id"])
        )).scalar_one_or_none()
        if existing is not None:
            rows[p["user_id"]] = existing
            print(f"patient {p['user_id']} exists — skip")
            continue
        row = Patient(
            user_id=p["user_id"], password_hash=hash_password(p["password"]),
            nickname=p["nickname"], room=p["room"],
            tts_rate="normal", tts_volume="normal",
            onboarding_completed=True,
        )
        session.add(row)
        await session.flush()
        rows[p["user_id"]] = row
        print(f"patient created: {p['user_id']} ({p['nickname']}, room {p['room']})")
    await session.commit()
    return rows


async def seed_phrase_logs_for_patient(session, patient: Patient, profile_key: str) -> None:
    profile = PATIENT_PROFILES[profile_key]
    existing = (await session.execute(
        select(PhraseLog).where(PhraseLog.patient_id == patient.id)
    )).scalars().all()
    if existing:
        print(f"phrase_log for {patient.user_id} already has {len(existing)} — skip")
        return

    entries = _generate_log_entries(profile)
    if not entries:
        print(f"phrase_log for {patient.user_id}: {profile_key} 는 로그 없음")
        return

    phrases = [e[0] for e in entries]
    vecs = await embed_batch(phrases)
    rng = random.Random(profile["random_seed"] + 1)
    now = datetime.now(UTC)

    for i, ((phrase, bucket), vec) in enumerate(zip(entries, vecs, strict=True)):
        source = rng.choice(["voice", "voice", "preset"])  # voice 2:1 preset
        days_ago = rng.randint(1, 14)
        base_hour = BUCKET_UTC_HOUR[bucket]
        created = (now - timedelta(days=days_ago)).replace(
            hour=base_hour,
            minute=rng.randint(0, 59),
            second=rng.randint(0, 59),
            microsecond=0, tzinfo=None,
        )
        session.add(PhraseLog(
            patient_id=patient.id,
            trace_id=f"seed-{patient.user_id}-{i:04d}",
            source=source,
            selected_phrase=phrase,
            selected_rank=1 if source == "voice" else None,
            partial_transcript=phrase.split()[0] if source == "voice" else None,
            final_transcript=phrase if source != "preset" else None,
            created_at=created,
            embedding=vec,
        ))
    await session.commit()
    print(f"phrase_log seeded for {patient.user_id}: {len(entries)} rows ({profile_key})")


async def seed_preferences_for_patient(session, patient: Patient, profile_key: str) -> None:
    profile = PATIENT_PROFILES[profile_key]
    prefs = profile.get("preferences", [])
    if not prefs:
        return

    existing = (await session.execute(
        select(PatientPreference).where(PatientPreference.patient_id == patient.id)
    )).scalars().all()
    if existing:
        print(f"preferences for {patient.user_id} exist ({len(existing)}) — skip")
        return

    contents = [p["content"] for p in prefs]
    vecs = await embed_batch(contents)
    for p, vec in zip(prefs, vecs, strict=True):
        session.add(PatientPreference(
            patient_id=patient.id,
            preference_type=p["type"],
            content=p["content"],
            boost_phrases=p["boost"],
            embedding=vec,
        ))
    await session.commit()
    print(f"preferences seeded for {patient.user_id}: {len(prefs)} rows")


async def reset_demo_data(session) -> None:
    user_ids = [p["user_id"] for p in DEMO_PATIENTS]
    await session.execute(sql_text("""
        DELETE FROM phrase_log
        WHERE patient_id IN (SELECT id FROM patients WHERE user_id = ANY(:uids))
    """), {"uids": user_ids})
    await session.execute(sql_text("""
        DELETE FROM patient_preference
        WHERE patient_id IN (SELECT id FROM patients WHERE user_id = ANY(:uids))
    """), {"uids": user_ids})
    await session.commit()
    print(f"RESET: phrase_log + patient_preference wiped for {user_ids}")


# ────────── main ──────────


async def main() -> None:
    do_reset = "--reset" in sys.argv

    async with AsyncSessionLocal() as session:
        if do_reset:
            print("=== RESET demo data ===")
            await reset_demo_data(session)

        print("=== Seeding phrase_catalog ===")
        await seed_catalog(session)

        print("\n=== Seeding patients ===")
        patients = await seed_patients(session)

        print("\n=== Seeding phrase_log ===")
        for p in DEMO_PATIENTS:
            await seed_phrase_logs_for_patient(session, patients[p["user_id"]], p["profile"])

        print("\n=== Seeding patient_preference ===")
        for p in DEMO_PATIENTS:
            await seed_preferences_for_patient(session, patients[p["user_id"]], p["profile"])

        print("\n=== Seed complete ===")
        print(f"  catalog phrases: {len(CATALOG_PHRASES)}")
        print(f"  demo patients  : {len(DEMO_PATIENTS)}")


if __name__ == "__main__":
    asyncio.run(main())
