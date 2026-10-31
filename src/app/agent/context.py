"""RankingContext builder — single async transaction, 4 parallel queries."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.agent.contracts import AgentInput, RankingContext
from app.services.embedding import parse_pgvector

log = logging.getLogger(__name__)


def _kst_bucket(now: datetime) -> str:
    """Compute time_bucket_kst from a datetime.  Matches migration 004 expression."""
    kst_hour = now.astimezone(ZoneInfo("Asia/Seoul")).hour
    if 6 <= kst_hour <= 11:
        return "morning"
    elif 12 <= kst_hour <= 17:
        return "afternoon"
    elif 18 <= kst_hour <= 22:
        return "evening"
    else:
        return "night"


async def _embed_prefix(prefix: str) -> list[float] | None:
    """Return embedding for prefix if len >= 3, else None."""
    if len(prefix) < 3:
        return None
    try:
        from app.services.embedding import embed_text  # type: ignore[import]

        vec = await embed_text(prefix)
    except ImportError:
        from app.services.embedding import get_embedder

        vec = await asyncio.to_thread(get_embedder().encode, prefix)
    return list(vec) if vec is not None else None


async def build_ranking_context(session, input: AgentInput) -> RankingContext:
    """
    Build a RankingContext for one ranking request.

    All DB queries run in a single async transaction via asyncio.gather.
    Query C (bigrams) is skipped entirely when input.previous_phrase is None.
    Query E (rejected_counts, Phase B3) feeds features/rejected_penalty.compute.
    저신뢰 분기 전용 — ranker.py 가 고신뢰/재질의 분기에서는 ctx 를 아예 빌드하지 않는다.
    """
    pid = input.patient_id
    time_bucket_kst = _kst_bucket(input.now)

    # NOTE: AsyncSession doesn't support concurrent cursors — queries A/B/C/D run
    # sequentially (await). Only _embed_prefix (CPU-bound via to_thread) runs in
    # parallel via asyncio.gather below.

    # ── Query A — 90d patient_frequency + recency MAX ─────────────────────────
    # TODO(M3, Phase C): add `AND source != 'voice_bypass'` guard before ranker.
    # 이관(agent/ranking/pool.py)과 함께 추가. 데모엔 영향 ≈ 0이나,
    # 운영 phrase_log에 bypass 로그 쌓이면 patient_frequency·recency feature 오염.
    # 참고: docs/design/streaming/02-agent-llm-pipeline.md §4.2
    # 참고: docs/design/streaming/09-senior-approval.md §1.1
    rows_a = (await session.execute(
        text("""
            SELECT selected_phrase, COUNT(*) AS cnt, MAX(created_at) AS last_at
            FROM phrase_log
            WHERE patient_id = :pid
              AND deleted_at IS NULL
              AND selected_phrase IS NOT NULL
              AND created_at >= now() - interval '90 days'
            GROUP BY selected_phrase
            """),
        {"pid": pid},
    )).mappings().all()
    freq_counts: dict[str, int] = {}
    last_at_map: dict[str, datetime] = {}
    for row in rows_a:
        freq_counts[row["selected_phrase"]] = int(row["cnt"])
        last_at_map[row["selected_phrase"]] = row["last_at"]
    freq_total = sum(freq_counts.values())

    # ── Query B — 60d time_bucket_kst ─────────────────────────────────────────
    # TODO(M3, Phase C): add `AND source != 'voice_bypass'` guard before ranker.
    # 이관(agent/ranking/pool.py)과 함께 추가. 데모엔 영향 ≈ 0이나,
    # 운영 phrase_log에 bypass 로그 쌓이면 patient_frequency·recency feature 오염.
    # 참고: docs/design/streaming/02-agent-llm-pipeline.md §4.2
    # 참고: docs/design/streaming/09-senior-approval.md §1.1
    rows_b = (await session.execute(
        text("""
            SELECT selected_phrase, time_bucket_kst, COUNT(*) AS cnt
            FROM phrase_log
            WHERE patient_id = :pid
              AND deleted_at IS NULL
              AND selected_phrase IS NOT NULL
              AND created_at >= now() - interval '60 days'
            GROUP BY selected_phrase, time_bucket_kst
            """),
        {"pid": pid},
    )).mappings().all()
    time_bucket_counts: dict[str, dict[str, int]] = {}
    for row in rows_b:
        phrase = row["selected_phrase"]
        bucket = row["time_bucket_kst"]
        cnt = int(row["cnt"])
        time_bucket_counts.setdefault(phrase, {})[bucket] = cnt
    time_bucket_total: dict[str, int] = {
        phrase: sum(buckets.values())
        for phrase, buckets in time_bucket_counts.items()
    }

    # ── Query C — 60d bigrams (only if previous_phrase is set) ────────────────
    bigram_counts: dict[str, int] = {}
    bigram_total = 0
    if input.previous_phrase is not None:
        # TODO(M3, Phase C): add `AND source != 'voice_bypass'` guard before ranker.
        # 이관(agent/ranking/pool.py)과 함께 추가. 데모엔 영향 ≈ 0이나,
        # 운영 phrase_log에 bypass 로그 쌓이면 patient_frequency·recency feature 오염.
        # 참고: docs/design/streaming/02-agent-llm-pipeline.md §4.2
        # 참고: docs/design/streaming/09-senior-approval.md §1.1
        rows_c = (await session.execute(
            text("""
                WITH ordered AS (
                  SELECT selected_phrase, created_at,
                         LAG(selected_phrase) OVER (ORDER BY created_at, id) AS prev,
                         LAG(created_at) OVER (ORDER BY created_at, id) AS prev_at
                  FROM phrase_log
                  WHERE patient_id = :pid
                    AND deleted_at IS NULL
                    AND selected_phrase IS NOT NULL
                    AND created_at >= now() - interval '60 days'
                )
                SELECT selected_phrase, COUNT(*) AS cnt
                FROM ordered
                WHERE prev = :previous_phrase
                  AND created_at - prev_at <= interval '30 minutes'
                GROUP BY selected_phrase
                """),
            {"pid": pid, "previous_phrase": input.previous_phrase},
        )).mappings().all()
        for row in rows_c:
            bigram_counts[row["selected_phrase"]] = int(row["cnt"])
        bigram_total = sum(bigram_counts.values())

    # ── Query D — patient_preference rows ─────────────────────────────────────
    rows_d = (await session.execute(
        text("""
            SELECT id, preference_type, content, boost_phrases, embedding
            FROM patient_preference
            WHERE patient_id = :pid AND deleted_at IS NULL
            """),
        {"pid": pid},
    )).mappings().all()
    prefs: list[dict] = []
    for row in rows_d:
        prefs.append(
            {
                "id": row["id"],
                "preference_type": row["preference_type"],
                "content": row["content"],
                "boost_phrases": row["boost_phrases"],
                "embedding": parse_pgvector(row["embedding"]),
            }
        )

    # ── Query E — 24h rejected 로그 (Phase B3, §6.2 rejected_penalty 입력) ────
    # action='rejected' 로 저장된 phrase_log 를 phrase 별로 묶는다.
    # 저장 규칙: `api/phrase.py` 가 rejected 시 `source="rejected"` (API source enum)
    # 로 저장 — 설계상 `selected_phrase` 는 null 이지만 운영 데이터에선 반영 안 될 수 있어
    # final_transcript fallback 포함. `candidates_shown` JSONB 에 담긴 후보들을
    # 저수준 집계용으로 쓸 수도 있지만, 1차 MVP 는 phrase 매칭 간소화를 위해
    # final_transcript / selected_phrase 중 non-null 1건으로 phrase 를 식별.
    #
    # 24h 창은 설계 §6.2 그대로 — HALF_LIFE 24h 와 동일한 창.
    # created_at 은 naive UTC (models.py L121) 이지만, migration 004 가 KST
    # casting 유틸을 쓰므로 "now() - interval '24 hours'" 비교로 안전.
    rows_e = (await session.execute(
        text("""
            SELECT
                COALESCE(selected_phrase, final_transcript) AS phrase,
                created_at AS reject_at
            FROM phrase_log
            WHERE patient_id = :pid
              AND source = 'rejected'
              AND deleted_at IS NULL
              AND created_at >= (now() AT TIME ZONE 'UTC' - interval '24 hours')
              AND COALESCE(selected_phrase, final_transcript) IS NOT NULL
            """),
        {"pid": pid},
    )).mappings().all()
    rejected_counts: dict[str, list[tuple[datetime, int]]] = {}
    for row in rows_e:
        phrase = row["phrase"]
        reject_at = row["reject_at"]
        if phrase is None or reject_at is None:
            continue
        rejected_counts.setdefault(phrase, []).append((reject_at, 1))

    # Prefix embed is CPU-bound (not DB) — run in parallel with any post-work if needed.
    prefix_vec = await _embed_prefix(input.prefix)

    return RankingContext(
        patient_id=pid,
        prefix=input.prefix,
        prefix_vec=prefix_vec,
        now=input.now,
        previous_phrase=input.previous_phrase,
        accuracy_mode=input.accuracy_mode,
        time_bucket_kst=time_bucket_kst,
        freq_counts=freq_counts,
        freq_total=freq_total,
        time_bucket_counts=time_bucket_counts,
        time_bucket_total=time_bucket_total,
        bigram_counts=bigram_counts,
        bigram_total=bigram_total,
        last_at=last_at_map,
        prefs=prefs,
        rejected_counts=rejected_counts,
    )
