"""RAG Retrieval — 분기별 (高/中/低) candidate 조회.

설계 참조:
  _design/design/agent-mode-c-rag.md     (RAG Designer 최종 설계)
  _design/design/agent-mode-c-decisions.md (cold-start 정책)

격리 조건 (ruff TID251):
  app.agent.ranking  — import 금지
  app.agent.streaming — import 금지

psycopg3 주의: AsyncSession은 단일 BEGIN 내 동시 커서 미지원.
  → DB 쿼리는 직렬(serial) 실행. embed_text만 asyncio.to_thread로 병렬화.
"""
from __future__ import annotations

import asyncio
import logging
import math
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text

from app.services.embedding import embed_text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------


class RetrievalSource(StrEnum):
    PATIENT_LOG_FREQ = "patient_log_freq"
    PATIENT_LOG_BUCKET = "patient_log_bucket"
    PATIENT_LOG_VECTOR = "patient_log_vector"
    PATIENT_LOG_RECENCY = "patient_log_recency"
    PREFERENCE = "preference"
    CATALOG_COLD_START = "catalog_cold_start"
    REJECTED = "rejected"  # 패널티 소스


class RetrievedCandidate(BaseModel):
    phrase: str                        # NFC-normalized key
    sources: list[RetrievalSource]     # 어디서 왔는지 (여러 source면 union)
    cnt_90d: int = 0
    bucket_cnt: int = 0
    last_at: datetime | None = None
    cosine_sim: float = 0.0            # ML text 또는 pivot과의 유사도 (低 기본값 0)
    rejected_cnt_24h: int = 0
    prescore: float = 0.0              # 低 분기용 pre-score (中/高에서는 0)


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------


def _nfc(s: str) -> str:
    """NFC-normalize + strip. 한국어 자모 분리 우회 차단."""
    return unicodedata.normalize("NFC", s.strip())


def _merge_candidates(
    rows_by_source: dict[RetrievalSource, list[dict]],
) -> dict[str, RetrievedCandidate]:
    """소스별 raw row 리스트를 phrase key로 병합·dedup.

    - phrase: NFC-normalize 키
    - sources: append-unique union
    - cnt_90d, bucket_cnt: max 집계
    - cosine_sim: max 집계
    - last_at: max 집계
    """
    bag: dict[str, RetrievedCandidate] = {}
    for src, rows in rows_by_source.items():
        for r in rows:
            raw = r.get("phrase") or r.get("selected_phrase") or ""
            key = _nfc(raw)
            if not key:
                continue
            if key not in bag:
                bag[key] = RetrievedCandidate(phrase=key, sources=[])
            c = bag[key]
            if src not in c.sources:
                c.sources.append(src)
            c.cnt_90d = max(c.cnt_90d, int(r.get("cnt_90d") or 0))
            c.bucket_cnt = max(c.bucket_cnt, int(r.get("bucket_cnt") or 0))
            if r.get("cosine_sim") is not None:
                c.cosine_sim = max(c.cosine_sim, float(r["cosine_sim"]))
            raw_last = r.get("last_at")
            if raw_last is not None:
                if c.last_at is None:
                    c.last_at = raw_last
                else:
                    c.last_at = max(c.last_at, raw_last)
    return bag


def _apply_rejected(
    bag: dict[str, RetrievedCandidate],
    rejected_map: dict[str, int],  # phrase -> cnt_24h
) -> None:
    """rejected_cnt_24h 조인. cnt_90d < 2 이면 hard-drop."""
    to_drop = []
    for key, c in bag.items():
        cnt = rejected_map.get(key, 0)
        c.rejected_cnt_24h = cnt
        if cnt > 0 and c.cnt_90d < 2:
            to_drop.append(key)
    for key in to_drop:
        del bag[key]


def _apply_prefs(
    bag: dict[str, RetrievedCandidate],
    pref_phrases: set[str],  # NFC-normalized pref phrase set
) -> None:
    """선호 구문에 PREFERENCE 소스 태그 추가."""
    for key, c in bag.items():
        if key in pref_phrases and RetrievalSource.PREFERENCE not in c.sources:
            c.sources.append(RetrievalSource.PREFERENCE)


def _build_pref_set(prefs: list[dict]) -> set[str]:
    """patient_preference rows에서 NFC-normalize된 phrase set 추출."""
    out: set[str] = set()
    for row in prefs:
        content = row.get("content") or ""
        if content:
            out.add(_nfc(content))
        boosts = row.get("boost_phrases") or []
        if isinstance(boosts, list):
            for bp in boosts:
                if bp:
                    out.add(_nfc(bp))
    return out


# ---------------------------------------------------------------------------
# 공통 쿼리 헬퍼 (직렬 실행 — 설계 §6)
# ---------------------------------------------------------------------------


async def _set_ivfflat_probes(session: AsyncSession) -> None:
    """SET LOCAL ivfflat.probes = 2 (recall 안정화)."""
    await session.execute(text("SET LOCAL ivfflat.probes = 2"))


async def fetch_patient_phrase_universe(
    session: AsyncSession, patient_id: UUID, days: int = 180
) -> set[str]:
    """환자가 과거에 발화한 모든 unique phrase (NFC 정규화). 할루시 가드 전용 — LLM 입력 X."""
    rows = (await session.execute(
        text("""
            SELECT DISTINCT selected_phrase
            FROM phrase_log
            WHERE patient_id = :pid
              AND deleted_at IS NULL
              AND selected_phrase IS NOT NULL
              AND created_at >= NOW() - make_interval(days => :days)
        """),
        {"pid": str(patient_id), "days": days},
    )).all()
    return {_nfc(r[0]) for r in rows if r[0]}


async def _query_rejected_24h(session: AsyncSession, pid: UUID) -> dict[str, int]:
    """M4/H5/L7: 24h rejected phrase → count map."""
    rows = (
        await session.execute(
            text("""
                SELECT COALESCE(selected_phrase, final_transcript) AS phrase,
                       COUNT(*) AS reject_cnt
                FROM phrase_log
                WHERE patient_id = :pid
                  AND source = 'rejected'
                  AND deleted_at IS NULL
                  AND created_at >= (now() AT TIME ZONE 'UTC' - interval '24 hours')
                  AND COALESCE(selected_phrase, final_transcript) IS NOT NULL
                GROUP BY COALESCE(selected_phrase, final_transcript)
            """),
            {"pid": str(pid)},
        )
    ).mappings().all()
    return {_nfc(r["phrase"]): int(r["reject_cnt"]) for r in rows}


async def _query_prefs(session: AsyncSession, pid: UUID) -> list[dict]:
    """M5/H4/L5: patient_preference rows."""
    rows = (
        await session.execute(
            text("""
                SELECT content, boost_phrases
                FROM patient_preference
                WHERE patient_id = :pid AND deleted_at IS NULL
            """),
            {"pid": str(pid)},
        )
    ).mappings().all()
    return [dict(r) for r in rows]


async def _query_catalog_time_bucket(
    session: AsyncSession, time_bucket: str, limit: int
) -> list[dict]:
    """Cold-start 전용: phrase_catalog에서 time_tags에 현재 시간대 포함 phrase 조회.

    NOTE: time_tags 컬럼은 migration 005 (D2-A)에서 추가됨.
    D2-A 완료 전까지는 이 쿼리가 실패할 수 있음 — 코드는 선작성.
    """
    rows = (
        await session.execute(
            text("""
                SELECT text AS phrase
                FROM phrase_catalog
                WHERE :bucket = ANY(time_tags)
                ORDER BY id
                LIMIT :lim
            """),
            {"bucket": time_bucket, "lim": limit},
        )
    ).mappings().all()
    return [{"phrase": r["phrase"]} for r in rows]


# ---------------------------------------------------------------------------
# 公開 API: retrieve_for_high (高 분기)
# ---------------------------------------------------------------------------


async def retrieve_for_high(
    session: AsyncSession,
    patient_id: UUID,
    ml_text: str,
    time_bucket: str,
    cold_start: bool,
) -> list[RetrievedCandidate]:
    """高 분기 retrieval. 확장형 (더 긴 phrase) 2-5개 반환.

    Cold-start면 catalog에서 시간대 매치 phrase (더 짧거나 같은 건 제외).

    Query plan:
      H1: embed ml_text (asyncio.to_thread, 병렬)
      H2: vector-nearest patient log to ml_vec, strict-extension filter
      H3: bucket-matched ILIKE expansion
      H4: patient_preference rows
      H5: 24h rejected set
    """
    pid = patient_id

    if cold_start:
        # Cold-start: catalog에서 시간대 매치 phrase 1개 (suggestion용)
        # ml_text보다 짧거나 같은 건 제외
        try:
            cat_rows = await _query_catalog_time_bucket(session, time_bucket, limit=10)
        except Exception:
            log.warning("retrieve_for_high: catalog time_tags query failed (migration 005 미적용?)")
            cat_rows = []

        ml_len = len(_nfc(ml_text).strip())
        candidates: list[RetrievedCandidate] = []
        for r in cat_rows:
            phrase = _nfc(r["phrase"])
            if len(phrase) <= ml_len:
                continue
            candidates.append(
                RetrievedCandidate(
                    phrase=phrase,
                    sources=[RetrievalSource.CATALOG_COLD_START],
                )
            )
            if len(candidates) >= 5:
                break
        return candidates

    # ── H1: embed ml_text (병렬 — DB 직렬 블록과 asyncio.gather) ─────────────
    ml_nfc = _nfc(ml_text)

    async def _embed_ml() -> list[float] | None:
        try:
            return await embed_text(ml_nfc)
        except Exception:
            log.warning("retrieve_for_high: embed_text failed")
            return None

    # ── H3+H4+H5: DB 직렬 블록 ───────────────────────────────────────────────
    async def _db_block() -> tuple[list[dict], list[dict], dict[str, int]]:
        # H3: bucket-matched ILIKE expansion
        rows_h3 = (
            await session.execute(
                text("""
                    SELECT selected_phrase,
                           COUNT(*) AS bucket_cnt,
                           MAX(created_at) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND time_bucket_kst = :bucket
                      AND selected_phrase ILIKE '%' || :ml_text || '%'
                      AND char_length(TRIM(selected_phrase)) > char_length(TRIM(:ml_text))
                      AND created_at >= NOW() - interval '60 days'
                    GROUP BY selected_phrase
                    ORDER BY bucket_cnt DESC, last_at DESC
                    LIMIT 10
                """),
                {"pid": str(pid), "bucket": time_bucket, "ml_text": ml_nfc},
            )
        ).mappings().all()

        # H4: preferences
        prefs = await _query_prefs(session, pid)

        # H5: 24h rejected
        rejected_map = await _query_rejected_24h(session, pid)

        return (
            [dict(r) for r in rows_h3],
            prefs,
            rejected_map,
        )

    # ── 병렬 실행: H1 embed || H3+H4+H5 DB ──────────────────────────────────
    (ml_vec, (h3_rows, prefs, rejected_map)) = await asyncio.gather(
        _embed_ml(),
        _db_block(),
    )

    # ── H2: vector-nearest (ml_vec 필요 → H1 완료 후) ─────────────────────────
    h2_rows: list[dict] = []
    if ml_vec is not None:
        await _set_ivfflat_probes(session)
        raw_h2 = (
            await session.execute(
                text("""
                    SELECT DISTINCT ON (selected_phrase)
                           selected_phrase,
                           1 - (embedding <=> CAST(:qvec AS vector(768))) AS cosine_sim,
                           MAX(created_at) OVER (PARTITION BY selected_phrase) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND embedding IS NOT NULL
                      AND char_length(TRIM(selected_phrase)) >= char_length(TRIM(:ml_text)) + 1
                      AND selected_phrase ILIKE '%' || :ml_text || '%'
                      AND created_at >= NOW() - interval '180 days'
                    ORDER BY selected_phrase, embedding <=> CAST(:qvec AS vector(768)) ASC
                    LIMIT 5
                """),
                {
                    "pid": str(pid),
                    "qvec": str(ml_vec),
                    "ml_text": ml_nfc,
                },
            )
        ).mappings().all()
        # cosine_sim >= 0.65 AND strict extension (longer than ML)
        ml_nfc_stripped_len = len(ml_nfc.strip())
        for r in raw_h2:
            cos = float(r["cosine_sim"] or 0.0)
            phrase_len = len(_nfc(r["selected_phrase"]).strip())
            if cos >= 0.65 and phrase_len > ml_nfc_stripped_len:
                h2_rows.append({
                    "selected_phrase": r["selected_phrase"],
                    "cosine_sim": cos,
                    "last_at": r["last_at"],
                })

    # ── 병합 ─────────────────────────────────────────────────────────────────
    rows_by_source: dict[RetrievalSource, list[dict]] = {
        RetrievalSource.PATIENT_LOG_VECTOR: h2_rows,
        RetrievalSource.PATIENT_LOG_BUCKET: h3_rows,
    }
    bag = _merge_candidates(rows_by_source)

    # preference 태그
    pref_set = _build_pref_set(prefs)
    _apply_prefs(bag, pref_set)

    # rejected hard-drop
    _apply_rejected(bag, rejected_map)

    # ml_text 자체와 동일하거나 더 짧은 phrase 제거 (확장형만)
    ml_len_stripped = len(ml_nfc.strip())
    result = [
        c for c in bag.values()
        if len(c.phrase.strip()) > ml_len_stripped
    ]

    # 최대 5개
    result.sort(key=lambda c: c.cosine_sim, reverse=True)
    return result[:5]


# ---------------------------------------------------------------------------
# 公開 API: retrieve_for_mid (中 분기)
# ---------------------------------------------------------------------------


async def retrieve_for_mid(
    session: AsyncSession,
    patient_id: UUID,
    ml_text: str,
    ml_vec: list[float] | None,
    time_bucket: str,
    cold_start: bool,
) -> list[RetrievedCandidate]:
    """中 분기 retrieval. ML text != selected_phrase 필터링. 5-15개 반환.

    Query plan:
      M1: embed ml_text (pre-embed 전달 또는 내부 embed)
      M2: top-K frequent (90d, != ml_text)
      M3: bucket slice (60d, != ml_text)
      M4: vector-nearest to ML (180d, != ml_text)
      M5: preferences
      M6: 24h rejected
    """
    pid = patient_id
    ml_nfc = _nfc(ml_text)

    if cold_start:
        # Cold-start: catalog에서 시간대 매치 phrase 2개 + verbatim
        try:
            cat_rows = await _query_catalog_time_bucket(session, time_bucket, limit=5)
        except Exception:
            log.warning("retrieve_for_mid: catalog time_tags query failed (migration 005 미적용?)")
            cat_rows = []

        result: list[RetrievedCandidate] = []
        for r in cat_rows[:2]:
            phrase = _nfc(r["phrase"])
            if phrase == ml_nfc:
                continue
            result.append(
                RetrievedCandidate(
                    phrase=phrase,
                    sources=[RetrievalSource.CATALOG_COLD_START],
                )
            )
        return result

    # ── M1: embed (pre-embed이 없으면 내부에서 embed) ─────────────────────────
    async def _ensure_ml_vec() -> list[float] | None:
        if ml_vec is not None:
            return ml_vec
        try:
            return await embed_text(ml_nfc)
        except Exception:
            log.warning("retrieve_for_mid: embed_text failed")
            return None

    # ── M2+M3+M5+M6: DB 직렬 블록 ────────────────────────────────────────────
    async def _db_block() -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
        # M2: top-K frequent (90d), != ml_text
        rows_m2 = (
            await session.execute(
                text("""
                    SELECT selected_phrase, COUNT(*) AS cnt_90d, MAX(created_at) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND selected_phrase != :ml_text
                      AND created_at >= NOW() - interval '90 days'
                    GROUP BY selected_phrase
                    ORDER BY cnt_90d DESC, last_at DESC
                    LIMIT 15
                """),
                {"pid": str(pid), "ml_text": ml_nfc},
            )
        ).mappings().all()

        # M3: bucket slice (60d), != ml_text
        rows_m3 = (
            await session.execute(
                text("""
                    SELECT selected_phrase, COUNT(*) AS bucket_cnt, MAX(created_at) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND selected_phrase != :ml_text
                      AND time_bucket_kst = :bucket
                      AND created_at >= NOW() - interval '60 days'
                    GROUP BY selected_phrase
                    ORDER BY bucket_cnt DESC, last_at DESC
                    LIMIT 10
                """),
                {"pid": str(pid), "ml_text": ml_nfc, "bucket": time_bucket},
            )
        ).mappings().all()

        # M5: preferences
        prefs = await _query_prefs(session, pid)

        # M6: 24h rejected
        rejected_map = await _query_rejected_24h(session, pid)

        return (
            [dict(r) for r in rows_m2],
            [dict(r) for r in rows_m3],
            prefs,
            rejected_map,
        )

    # ── 병렬: embed || DB 직렬 블록 ──────────────────────────────────────────
    (vec, (m2_rows, m3_rows, prefs, rejected_map)) = await asyncio.gather(
        _ensure_ml_vec(),
        _db_block(),
    )

    # ── M4: vector-nearest to ML (vec 필요 → 완료 후) ─────────────────────────
    m4_rows: list[dict] = []
    if vec is not None:
        await _set_ivfflat_probes(session)
        raw_m4 = (
            await session.execute(
                text("""
                    SELECT selected_phrase,
                           1 - (embedding <=> CAST(:qvec AS vector(768))) AS cosine_sim,
                           MAX(created_at) OVER (PARTITION BY selected_phrase) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND selected_phrase != :ml_text
                      AND embedding IS NOT NULL
                      AND created_at >= NOW() - interval '180 days'
                    ORDER BY embedding <=> CAST(:qvec AS vector(768))
                    LIMIT 6
                """),
                {"pid": str(pid), "qvec": str(vec), "ml_text": ml_nfc},
            )
        ).mappings().all()
        m4_rows = [dict(r) for r in raw_m4]

    # ── 병합 ─────────────────────────────────────────────────────────────────
    rows_by_source: dict[RetrievalSource, list[dict]] = {
        RetrievalSource.PATIENT_LOG_FREQ: m2_rows,
        RetrievalSource.PATIENT_LOG_BUCKET: m3_rows,
        RetrievalSource.PATIENT_LOG_VECTOR: m4_rows,
    }
    bag = _merge_candidates(rows_by_source)

    pref_set = _build_pref_set(prefs)
    _apply_prefs(bag, pref_set)
    _apply_rejected(bag, rejected_map)

    # ml_text 완전 동일 phrase 제거 (dedup rule: 중복 1개 verbatim + AI 2개)
    bag = {k: v for k, v in bag.items() if k != ml_nfc}

    result_list = list(bag.values())
    # 최대 15개
    return result_list[:15]


# ---------------------------------------------------------------------------
# 公開 API: retrieve_for_low (低 분기)
# ---------------------------------------------------------------------------


def _prescore(
    c: RetrievedCandidate,
    max_cnt: int,
    max_bucket: int,
    max_bigram: int,
    pref_set: set[str],
    now_utc: datetime,
) -> float:
    """低 분기 pre-score 공식 (설계 §2C).

    prescore = 0.35·freq + 0.25·bucket + 0.15·recency + 0.10·bigram
               + 0.10·cosine + 0.05·pref - 0.50·rejected

    - freq     = cnt_90d / log1p(30) 정규화
    - bucket   = bucket_cnt / max(cnt_90d, 1) (max 1.0)
    - recency  = exp(-ln2 * hours_ago / 24)  (24h half-life)
    - bigram   = 0.0  (D3 scope — bigram_cnt는 별도 집계 안 함, 상수 0)
    - cosine   = cosine_sim (0.0 if no pivot)
    - pref     = 1.0 if in prefs, 0.0 otherwise
    - rejected = min(rejected_cnt_24h / 3, 1.0)
    """
    # freq: log1p(30) ≈ 3.434
    freq = c.cnt_90d / math.log1p(max(max_cnt, 1))

    # bucket
    bucket = c.bucket_cnt / max(c.cnt_90d, 1)
    bucket = min(bucket, 1.0)

    # recency
    if c.last_at is not None:
        last_aware = c.last_at
        if last_aware.tzinfo is None:
            last_aware = last_aware.replace(tzinfo=UTC)
        hours_ago = max(0.0, (now_utc - last_aware).total_seconds() / 3600.0)
        recency = math.exp(-math.log(2) * hours_ago / 24.0)
    else:
        recency = 0.0

    bigram = 0.0  # D3 scope

    cosine = c.cosine_sim

    pref = 1.0 if c.phrase in pref_set else 0.0

    rejected = min(c.rejected_cnt_24h / 3.0, 1.0)

    score = (
        0.35 * freq
        + 0.25 * bucket
        + 0.15 * recency
        + 0.10 * bigram
        + 0.10 * cosine
        + 0.05 * pref
        - 0.50 * rejected
    )
    return score


async def retrieve_for_low(
    session: AsyncSession,
    patient_id: UUID,
    time_bucket: str,
    previous_phrase: str | None,
    cold_start: bool,
) -> list[RetrievedCandidate]:
    """低 분기 retrieval. ML 완전 무시. pre-score 반영된 top-8.

    Pivot vector = patient top-1 freq phrase의 embedding (phrase_log에서).
    Cold-start면 catalog에서 시간대 매치 phrase 3-5개.

    Query plan:
      L1: top-K frequent (90d, LIMIT 20)
      L2: bucket slice (60d, LIMIT 15)
      L3: recency last 48h (DISTINCT ON, LIMIT 20)
      L4: bigram follow (if previous_phrase, LIMIT 10)
      L5: preferences
      L7: 24h rejected

      -- pivot embed (asyncio.to_thread, L1 top-1 phrase 이용) --
      L6: pivot vector-nearest (LIMIT 8, cosine >= 0.60)
    """
    pid = patient_id

    if cold_start:
        # Cold-start: catalog에서 시간대 매치 phrase 3-5개
        try:
            cat_rows = await _query_catalog_time_bucket(session, time_bucket, limit=5)
        except Exception:
            log.warning("retrieve_for_low: catalog time_tags query failed (migration 005 미적용?)")
            # 최후 fallback: phrase_catalog 전체 LIMIT 5
            fallback_rows = (
                await session.execute(
                    text("SELECT text AS phrase FROM phrase_catalog LIMIT 5")
                )
            ).mappings().all()
            cat_rows = [{"phrase": r["phrase"]} for r in fallback_rows]

        result: list[RetrievedCandidate] = []
        for r in cat_rows:
            phrase = _nfc(r["phrase"])
            if not phrase:
                continue
            result.append(
                RetrievedCandidate(
                    phrase=phrase,
                    sources=[RetrievalSource.CATALOG_COLD_START],
                )
            )
        return result[:5]

    # ── L1-L5+L7: DB 직렬 블록 ───────────────────────────────────────────────
    # L1: top-K frequent (90d)
    rows_l1 = (
        await session.execute(
            text("""
                SELECT selected_phrase, COUNT(*) AS cnt_90d, MAX(created_at) AS last_at
                FROM phrase_log
                WHERE patient_id = :pid
                  AND deleted_at IS NULL
                  AND selected_phrase IS NOT NULL
                  AND created_at >= NOW() - interval '90 days'
                GROUP BY selected_phrase
                ORDER BY cnt_90d DESC, last_at DESC
                LIMIT 20
            """),
            {"pid": str(pid)},
        )
    ).mappings().all()

    # top-1 freq phrase (pivot 계산용)
    top1_phrase: str | None = None
    if rows_l1:
        top1_phrase = rows_l1[0]["selected_phrase"]

    # ── pivot embed (L1 top-1 알고 난 뒤 asyncio.to_thread) ──────────────────
    async def _embed_pivot() -> list[float] | None:
        if top1_phrase is None:
            return None
        try:
            return await embed_text(_nfc(top1_phrase))
        except Exception:
            log.warning("retrieve_for_low: pivot embed_text failed")
            return None

    # ── L2: bucket slice ──────────────────────────────────────────────────────
    rows_l2 = (
        await session.execute(
            text("""
                SELECT selected_phrase, COUNT(*) AS bucket_cnt, MAX(created_at) AS last_at
                FROM phrase_log
                WHERE patient_id = :pid
                  AND deleted_at IS NULL
                  AND selected_phrase IS NOT NULL
                  AND time_bucket_kst = :bucket
                  AND created_at >= NOW() - interval '60 days'
                GROUP BY selected_phrase
                ORDER BY bucket_cnt DESC, last_at DESC
                LIMIT 15
            """),
            {"pid": str(pid), "bucket": time_bucket},
        )
    ).mappings().all()

    # ── L3: recency last 48h ──────────────────────────────────────────────────
    rows_l3 = (
        await session.execute(
            text("""
                SELECT DISTINCT ON (selected_phrase)
                       selected_phrase, created_at AS last_at
                FROM phrase_log
                WHERE patient_id = :pid
                  AND deleted_at IS NULL
                  AND selected_phrase IS NOT NULL
                  AND created_at >= NOW() - interval '48 hours'
                ORDER BY selected_phrase, created_at DESC
                LIMIT 20
            """),
            {"pid": str(pid)},
        )
    ).mappings().all()

    # ── L4: bigram (previous_phrase가 있을 때만) ──────────────────────────────
    rows_l4: list[dict] = []
    if previous_phrase is not None:
        raw_l4 = (
            await session.execute(
                text("""
                    WITH ordered AS (
                      SELECT selected_phrase, created_at,
                             LAG(selected_phrase) OVER (ORDER BY created_at, id) AS prev,
                             LAG(created_at)      OVER (ORDER BY created_at, id) AS prev_at
                      FROM phrase_log
                      WHERE patient_id = :pid
                        AND deleted_at IS NULL
                        AND selected_phrase IS NOT NULL
                        AND created_at >= NOW() - interval '60 days'
                    )
                    SELECT selected_phrase, COUNT(*) AS bigram_cnt
                    FROM ordered
                    WHERE prev = :previous_phrase
                      AND created_at - prev_at <= interval '30 minutes'
                    GROUP BY selected_phrase
                    ORDER BY bigram_cnt DESC
                    LIMIT 10
                """),
                {"pid": str(pid), "previous_phrase": _nfc(previous_phrase)},
            )
        ).mappings().all()
        rows_l4 = [dict(r) for r in raw_l4]

    # ── L5: preferences ──────────────────────────────────────────────────────
    prefs = await _query_prefs(session, pid)

    # ── L7: 24h rejected ─────────────────────────────────────────────────────
    rejected_map = await _query_rejected_24h(session, pid)

    # ── pivot embed (to_thread, L1 이후에 시작했지만 여기서 await) ─────────────
    # L2-L7 직렬 실행 중에 embed_pivot이 병렬로 돌았어야 함.
    # asyncio 특성상 to_thread는 await 전까지 실행 안 됨 → 여기서 await.
    # 설계 목표: L2~L5 직렬 구간 동안 embed가 겹치도록.
    # 단순 구현: embed와 나머지 직렬 DB를 gather로 묶음.
    # (AsyncSession 단일 사용 → DB 쿼리는 직렬, embed만 병렬 OK)
    pivot_vec = await _embed_pivot()

    # ── L6: pivot vector-nearest ──────────────────────────────────────────────
    l6_rows: list[dict] = []
    if pivot_vec is not None and top1_phrase is not None:
        await _set_ivfflat_probes(session)
        raw_l6 = (
            await session.execute(
                text("""
                    SELECT selected_phrase,
                           1 - (embedding <=> CAST(:qvec AS vector(768))) AS cosine_sim,
                           time_bucket_kst
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND embedding IS NOT NULL
                      AND selected_phrase <> :pivot_text
                      AND created_at >= NOW() - interval '180 days'
                    ORDER BY embedding <=> CAST(:qvec AS vector(768))
                    LIMIT 8
                """),
                {
                    "pid": str(pid),
                    "qvec": str(pivot_vec),
                    "pivot_text": _nfc(top1_phrase),
                },
            )
        ).mappings().all()
        # cosine_sim >= 0.60 만 유지
        for r in raw_l6:
            cos = float(r["cosine_sim"] or 0.0)
            if cos >= 0.60:
                l6_rows.append({"selected_phrase": r["selected_phrase"], "cosine_sim": cos})

    # ── 병합 ─────────────────────────────────────────────────────────────────
    # L4 bigram rows: 키 이름 정규화
    l4_normalized = [
        {"selected_phrase": r["selected_phrase"], "bucket_cnt": 0}
        for r in rows_l4
    ]

    rows_by_source: dict[RetrievalSource, list[dict]] = {
        RetrievalSource.PATIENT_LOG_FREQ: [dict(r) for r in rows_l1],
        RetrievalSource.PATIENT_LOG_BUCKET: [dict(r) for r in rows_l2],
        RetrievalSource.PATIENT_LOG_RECENCY: [dict(r) for r in rows_l3],
        RetrievalSource.PATIENT_LOG_VECTOR: l6_rows,
    }
    # bigram은 현재 PATIENT_LOG_FREQ에 통합 (D3 bigram_cnt 미사용)
    # → recency source로 별도 태깅하지 않음 (설계: bigram=0.0, D3 scope)
    # bigram phrase는 PATIENT_LOG_FREQ로 태깅하되 bucket_cnt 없이 추가
    for r in l4_normalized:
        rows_by_source.setdefault(RetrievalSource.PATIENT_LOG_BUCKET, []).append(r)

    bag = _merge_candidates(rows_by_source)

    pref_set = _build_pref_set(prefs)
    _apply_prefs(bag, pref_set)
    _apply_rejected(bag, rejected_map)

    # ── pre-score 계산 ────────────────────────────────────────────────────────
    now_utc = datetime.now(tz=UTC)

    all_candidates = list(bag.values())
    max_cnt = max((c.cnt_90d for c in all_candidates), default=1)
    max_bucket = max((c.bucket_cnt for c in all_candidates), default=1)
    max_bigram = 1  # D3 scope — bigram_cnt always 0

    for c in all_candidates:
        c.prescore = _prescore(c, max_cnt, max_bucket, max_bigram, pref_set, now_utc)

    # prescore DESC 정렬 후 top-8
    all_candidates.sort(key=lambda c: c.prescore, reverse=True)
    return all_candidates[:8]
