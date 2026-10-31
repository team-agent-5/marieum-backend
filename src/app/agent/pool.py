"""Candidate pool builder — UNION with source-priority deduplication.

F5 semantic (vector queries) is deferred; embedding is carried through as-is.
"""
from __future__ import annotations

from sqlalchemy import text

from app.services.embedding import parse_pgvector

from .contracts import PoolCandidate, RankingContext


async def gather_candidates(session, ctx: RankingContext) -> list[PoolCandidate]:
    """Build a candidate pool of up to 30 entries.

    Sources (priority order):
      1. patient_log  — 90-day distinct selected_phrase (highest priority)
      2. phrase_catalog — cold-start fallback (~25 rows)

    Deduplication: patient_log wins over catalog on same phrase text.
    F5 vector queries are NOT run here — deferred to S2.
    """
    # 1. patient_log — 90d distinct selected_phrase, carrying most-recent embedding
    # TODO: add `AND source != 'voice_bypass'` guard before ranker.
    # 이관(agent/ranking/pool.py)과 함께 추가. 데모엔 영향 ≈ 0이나,
    # 운영 phrase_log에 bypass 로그 쌓이면 patient_frequency·recency feature 오염.
    log_rows = (
        await session.execute(
            text("""
                SELECT DISTINCT ON (selected_phrase)
                       selected_phrase AS phrase,
                       cnt_90d, last_at, embedding
                FROM (
                    SELECT selected_phrase, embedding, created_at,
                           COUNT(*) OVER (PARTITION BY selected_phrase) AS cnt_90d,
                           MAX(created_at) OVER (PARTITION BY selected_phrase) AS last_at
                    FROM phrase_log
                    WHERE patient_id = :pid
                      AND deleted_at IS NULL
                      AND selected_phrase IS NOT NULL
                      AND created_at >= now() - interval '90 days'
                ) t
                ORDER BY selected_phrase, created_at DESC
                LIMIT 50
            """),
            {"pid": ctx.patient_id},
        )
    ).mappings().all()

    # 2. phrase_catalog — ALL rows (cold-start fallback; typically ~25 rows)
    cat_rows = (
        await session.execute(
            text("""
                SELECT id AS catalog_id, text AS phrase, embedding
                FROM phrase_catalog
            """)
        )
    ).mappings().all()

    # 3. Dedupe: patient_log wins over catalog on same phrase text
    seen: set[str] = set()
    out: list[PoolCandidate] = []

    for r in log_rows:
        if r["phrase"] in seen:
            continue
        seen.add(r["phrase"])
        out.append(
            PoolCandidate(
                phrase=r["phrase"],
                source="patient_log",
                cnt_90d=r["cnt_90d"] or 0,
                last_at=r["last_at"],
                catalog_id=None,
                embedding=parse_pgvector(r["embedding"]),
            )
        )

    for r in cat_rows:
        if r["phrase"] in seen:
            continue
        seen.add(r["phrase"])
        out.append(
            PoolCandidate(
                phrase=r["phrase"],
                source="catalog",
                cnt_90d=0,
                last_at=None,
                catalog_id=r["catalog_id"],
                embedding=parse_pgvector(r["embedding"]),
            )
        )

    return out[:30]
