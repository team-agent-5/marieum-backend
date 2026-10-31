"""Rejected feedback penalty.

phrase_log 에 `action='rejected'` 로 쌓인 24h 내 로그를 **음수 penalty** 로 환산한다.
`RankingContext` 에 해당 필드가 **없을 수도 있다** — getattr 로 defensive 하게 접근한다.

공식 (사용자 지침 — 간이화):
    penalty(phrase) = Σ (-0.15 × exp(-ln2 × hours_ago_i / 24))
                       for each rejected_log_i WHERE phrase_i == phrase
    penalty_clip    = max(-0.5, penalty(phrase))   # 누적 상한 (하한 -0.5)
    반환은 **음수 또는 0** — ranker 가중합에서 `base + rejected_penalty` 로 더해진다.

NOTE: 원문 수식은 F1/F2 를 **곱셈 감쇠** (`s_f1 * (1 - rejected_signal)`) 하지만,
사용자 지침은 "선형 -0.15 × decay 를 가중합에 **add**" 로 간이화한다 — 1차 MVP 범위.
F1/F2 감쇠 방식으로 재조정할 여지 있음 (`FeatureScores.rejected_penalty` 필드만 유지하면 호환).

사용자 지침 "24h 이내 3회 이상 거부 → blacklist 플래그" 는 **본 모듈에서 미구현** —
Q (결정 보류) 상태이므로 주석만 남긴다.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Iterable

# -----------------------------------------------------------------------------
# 상수 — 사용자 지침 명시 값
# -----------------------------------------------------------------------------

name = "rejected_penalty"
weight = 1.0  # 가중합에서 add 되므로 w=1 (기존 feature 와 성격 다름)

# 1회 rejected 당 base penalty (음수 방향 상한).
PENALTY_PER_REJECT = 0.15
# 반감기 24h — recency 와 동일.
HALF_LIFE_HOURS = 24.0
# 누적 penalty 하한 (음수 방향 최대값, 즉 반환값은 >= -CUMULATIVE_FLOOR 반대 부호).
CUMULATIVE_FLOOR = 0.5

# TODO: context.py 보강 — `RankingContext` 에 아래 필드를 추가해야 완전한 집계가 가능하다.
#   `RankingContext` 에 아래 필드를 추가해야 완전한 집계가 가능하다.
#     rejected_counts: dict[str, list[tuple[datetime, int]]]
#         phrase -> [(reject_time_utc, count), ...]
#         (count 는 동일 bucket 다중 rejected 를 1건 로그로 압축했을 때의 다중도.
#          단순 1건=1 로 저장해도 공식은 동일하게 작동.)
#   집계 쿼리:
#     SELECT candidates_shown->X->>'text' AS phrase, created_at
#     FROM phrase_log
#     WHERE patient_id = :pid AND action = 'rejected'
#       AND created_at >= now() - interval '24 hours'
#       AND deleted_at IS NULL
#   현재는 ctx 에 필드가 없으면 빈 dict 로 간주 → penalty = 0.0 반환.


# -----------------------------------------------------------------------------
# 핵심 유틸
# -----------------------------------------------------------------------------


def exp_decay(hours_since: float, half_life_h: float = HALF_LIFE_HOURS) -> float:
    """반감기 `half_life_h` 의 지수 감쇠 가중치.

    exp(-ln(2) * hours_since / half_life_h)

    - `hours_since <= 0` 이면 1.0 (방금 발생한 거부 = 전액 반영).
    - 음수 입력은 clamp (시계 역행 방어).
    """
    if hours_since <= 0.0:
        return 1.0
    return math.exp(-math.log(2) * hours_since / half_life_h)


def _hours_between(now: datetime, past: datetime) -> float:
    """UTC naive/aware 혼재를 흡수하고 시 단위 차이 반환."""
    if past.tzinfo is None:
        past = past.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now - past).total_seconds() / 3600.0)


def _iter_rejects(rejected_counts: object, phrase: str) -> Iterable[tuple[datetime, int]]:
    """ctx.rejected_counts 에서 특정 phrase 의 reject 이벤트 iterator.

    기대 shape:
        dict[str, list[tuple[datetime, int]]]
    실제 shape 가 달라도(예: list[datetime]) best-effort 로 흡수.
    """
    if not isinstance(rejected_counts, dict):
        return []
    events = rejected_counts.get(phrase)
    if not events:
        return []

    normalized: list[tuple[datetime, int]] = []
    for item in events:
        if isinstance(item, tuple) and len(item) == 2:
            ts, cnt = item
            try:
                cnt_int = int(cnt)
            except (TypeError, ValueError):
                cnt_int = 1
            if isinstance(ts, datetime):
                normalized.append((ts, max(1, cnt_int)))
        elif isinstance(item, datetime):
            normalized.append((item, 1))
    return normalized


# -----------------------------------------------------------------------------
# feature API
# -----------------------------------------------------------------------------


def compute(cand: object, ctx: object) -> float:  # type: ignore[override]
    """해당 phrase 의 rejected penalty 계산.

    공식:
        penalty = Σ (-PENALTY_PER_REJECT × exp_decay(hours_since_reject_i))   (24h 내 로그)
        clip    = max(-CUMULATIVE_FLOOR, penalty)
        → 반환은 **음수 또는 0** (ranker 가중합에서 `base` 에 add).

    주의:
    - 24h 밖 로그는 상위 집계 (context.py) 가 걸러주는 전제. 방어적으로 다시 걸러도 됨.
    - ctx 에 `rejected_counts` 가 없거나 비면 **0.0** 반환.
    - 24h 이내 3회 이상 rejected → blacklist 전환 (score=0 강제) 플래그는
      본 모듈에서 미구현 (Q 결정 보류). branches/*.py 가 ctx 직접 조회해 처리할 수 있음.
    """
    phrase: str = cand.phrase  # type: ignore[attr-defined]
    now: datetime = ctx.now  # type: ignore[attr-defined]
    rejected_counts = getattr(ctx, "rejected_counts", None)
    if rejected_counts is None:
        return 0.0

    total = 0.0
    for reject_ts, multiplicity in _iter_rejects(rejected_counts, phrase):
        hours = _hours_between(now, reject_ts)
        if hours > HALF_LIFE_HOURS * 24:  # 24h 창 보호 (half_life 가 아니라 실제 창 24h)
            # 실제 창은 24h — 반감기와 동일. 과도한 과거는 context.py 에서 자르는 게 정석.
            continue
        total += -PENALTY_PER_REJECT * exp_decay(hours) * max(1, multiplicity)

    # 누적 상한 (음수 방향).
    if total < -CUMULATIVE_FLOOR:
        total = -CUMULATIVE_FLOOR

    # 양수 방어 (연산 실수 보호) — 반환은 항상 음수 or 0.
    if total > 0.0:
        return 0.0
    return total
