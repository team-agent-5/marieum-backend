"""rejected_penalty — cold-user confidence scaling 테스트.

버그: freq_total(90일 총 발화 선택 수)에 무관하게 PENALTY_PER_REJECT=0.15가 동일 적용.
신규 유저(freq_total 소량)가 거부 2~3회만 해도 하루 종일 해당 문구가 억제됨.

수정: confidence = freq_total / (freq_total + COLD_START_K) 로 패널티 스케일.
     CUMULATIVE_FLOOR도 같이 스케일하여 하한도 유저 이력에 비례.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agent.features.rejected_penalty import (
    COLD_START_K,
    CUMULATIVE_FLOOR,
    HALF_LIFE_HOURS,
    PENALTY_PER_REJECT,
    compute,
    exp_decay,
)

# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

NOW = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)


def _cand(phrase: str = "물 주세요") -> SimpleNamespace:
    return SimpleNamespace(phrase=phrase)


def _ctx(
    phrase: str = "물 주세요",
    reject_hours_ago: list[float] | None = None,
    freq_total: int = 100,
) -> SimpleNamespace:
    """rejected_counts와 freq_total을 가진 최소 ctx 생성."""
    rejected_counts: dict = {}
    if reject_hours_ago:
        events = [(NOW - timedelta(hours=h), 1) for h in reject_hours_ago]
        rejected_counts[phrase] = events
    return SimpleNamespace(
        now=NOW,
        rejected_counts=rejected_counts,
        freq_total=freq_total,
    )


# ─────────────────────────────────────────────
# 경계값 테스트
# ─────────────────────────────────────────────


class TestColdUserBoundary:

    def test_freq_total_zero_returns_zero(self):
        """신규 유저(freq_total=0): 거부 이력 있어도 penalty=0."""
        ctx = _ctx(reject_hours_ago=[0.5, 1.0, 2.0], freq_total=0)
        result = compute(_cand(), ctx)
        assert result == 0.0, f"freq_total=0 이면 penalty=0 이어야 함, got {result}"

    def test_freq_total_equals_k_gives_half_penalty(self):
        """freq_total=COLD_START_K → confidence=0.5 → penalty가 이론 풀 패널티의 절반."""
        # 방금 1회 거부 (hours=0 → decay=1.0) → 이론 풀 패널티 = -PENALTY_PER_REJECT
        ctx_cold = _ctx(reject_hours_ago=[0.0], freq_total=COLD_START_K)
        result_cold = compute(_cand(), ctx_cold)

        # 이론값: -PENALTY_PER_REJECT * decay(0) * confidence(K) = -0.15 * 1.0 * 0.5
        expected_cold = -PENALTY_PER_REJECT * exp_decay(0.0) * 0.5
        assert abs(result_cold - expected_cold) < 1e-9, (
            f"freq_total=K 에서 패널티가 이론 풀 패널티의 0.5배여야 함. "
            f"expected={expected_cold:.4f}, got {result_cold:.4f}"
        )

    def test_large_freq_total_confidence_approaches_one(self):
        """freq_total=10000 → confidence≈1.0, 기존 로직과 거의 동일."""
        ctx = _ctx(reject_hours_ago=[0.0], freq_total=10_000)
        result = compute(_cand(), ctx)
        expected = -PENALTY_PER_REJECT  # 방금 1회 거부, decay=1.0
        assert abs(result - expected) < 1e-3, (
            f"freq_total 충분히 클 때 기존 패널티와 거의 동일해야 함. "
            f"expected≈{expected:.4f}, got {result:.4f}"
        )


# ─────────────────────────────────────────────
# 회귀 테스트 (기존 동작 유지)
# ─────────────────────────────────────────────


class TestRegression:

    def test_no_rejected_counts_field_returns_zero(self):
        """ctx에 rejected_counts 없으면 0.0 반환 (기존 방어 로직 유지)."""
        ctx = SimpleNamespace(now=NOW, freq_total=100)
        result = compute(_cand(), ctx)
        assert result == 0.0

    def test_phrase_not_in_rejected_counts_returns_zero(self):
        """거부된 phrase가 없으면 0.0."""
        ctx = _ctx(reject_hours_ago=[1.0], freq_total=100)
        result = compute(SimpleNamespace(phrase="다른 문장"), ctx)
        assert result == 0.0

    def test_cumulative_floor_clipping_applied(self):
        """많은 거부 → effective_floor 이하로 내려가지 않음."""
        ctx = _ctx(reject_hours_ago=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], freq_total=10_000)
        result = compute(_cand(), ctx)
        effective_floor = -CUMULATIVE_FLOOR  # freq_total 충분히 커서 confidence≈1
        assert result >= effective_floor - 1e-9, (
            f"floor clipping 미작동: result={result:.4f} < floor={effective_floor:.4f}"
        )

    def test_result_always_non_positive(self):
        """반환값은 항상 음수 또는 0."""
        ctx = _ctx(reject_hours_ago=[1.0], freq_total=50)
        result = compute(_cand(), ctx)
        assert result <= 0.0, f"패널티는 음수 또는 0이어야 함, got {result}"

    def test_multiplicity_amplifies_penalty(self):
        """multiplicity=3인 거부 이벤트 → 1회 거부보다 패널티 3배."""
        phrase = "물 주세요"
        ctx_single = SimpleNamespace(
            now=NOW,
            freq_total=10_000,
            rejected_counts={phrase: [(NOW - timedelta(hours=0.1), 1)]},
        )
        ctx_triple = SimpleNamespace(
            now=NOW,
            freq_total=10_000,
            rejected_counts={phrase: [(NOW - timedelta(hours=0.1), 3)]},
        )
        r_single = compute(_cand(phrase), ctx_single)
        r_triple = compute(_cand(phrase), ctx_triple)
        assert abs(r_triple - r_single * 3) < 1e-9, (
            f"multiplicity=3 → 패널티 3배 기대. single={r_single:.4f}, triple={r_triple:.4f}"
        )


# ─────────────────────────────────────────────
# floor도 함께 스케일되는지 검증
# ─────────────────────────────────────────────


class TestFloorScaling:

    def test_floor_scales_with_confidence(self):
        """confidence=0.5(freq_total=K)일 때 effective_floor = -CUMULATIVE_FLOOR * 0.5."""
        many_rejects = [0.01 * i for i in range(1, 50)]  # 많은 거부로 floor 도달 유도
        ctx = _ctx(reject_hours_ago=many_rejects, freq_total=COLD_START_K)
        result = compute(_cand(), ctx)

        confidence = COLD_START_K / (COLD_START_K + COLD_START_K)  # = 0.5
        expected_floor = -CUMULATIVE_FLOOR * confidence
        assert abs(result - expected_floor) < 1e-9, (
            f"floor도 confidence 비례로 스케일되어야 함. "
            f"expected={expected_floor:.4f}, got={result:.4f}"
        )

    def test_new_user_floor_is_zero(self):
        """freq_total=0 → effective_floor=0, 아무리 거부해도 penalty=0."""
        many_rejects = [0.01 * i for i in range(1, 50)]
        ctx = _ctx(reject_hours_ago=many_rejects, freq_total=0)
        result = compute(_cand(), ctx)
        assert result == 0.0, f"신규 유저는 floor=0이므로 penalty=0, got {result}"


# ─────────────────────────────────────────────
# 단조성 프로퍼티 (monotonicity)
# ─────────────────────────────────────────────


class TestMonotonicity:

    def test_penalty_increases_monotonically_with_freq_total(self):
        """freq_total 증가 → |penalty| 단조 증가 (방향 뒤집힘 버그 방지)."""
        reject_hours_ago = [0.5, 1.0]
        freq_totals = [0, 1, 2, 5, 10, 20, 50, 100, 500]
        penalties = [
            compute(_cand(), _ctx(reject_hours_ago=reject_hours_ago, freq_total=ft))
            for ft in freq_totals
        ]
        for i in range(len(penalties) - 1):
            assert penalties[i] >= penalties[i + 1], (
                f"단조성 위반: freq_total={freq_totals[i]} → penalty={penalties[i]:.4f} "
                f"but freq_total={freq_totals[i+1]} → penalty={penalties[i+1]:.4f} "
                "(더 커야 함 — 음수이므로 더 작은 값)"
            )


# ─────────────────────────────────────────────
# 시나리오 테스트
# ─────────────────────────────────────────────


class TestScenarios:

    def test_cold_user_3_rejects_much_less_than_full(self):
        """신규 유저(freq_total=2) 패널티가 파워 유저(freq_total=100) 대비 현저히 작아야 함."""
        ctx_cold = _ctx(reject_hours_ago=[0.1, 0.5, 1.0], freq_total=2)
        ctx_power = _ctx(reject_hours_ago=[0.1, 0.5, 1.0], freq_total=100)

        result_cold = compute(_cand(), ctx_cold)
        result_power = compute(_cand(), ctx_power)

        # freq_total=2 → confidence=2/7≈0.286, freq_total=100 → confidence=100/105≈0.952
        # 비율 ≈ 0.286/0.952 ≈ 0.3 — 신규 유저 패널티가 파워 유저의 35% 이하여야 함
        ratio = result_cold / result_power
        assert ratio > 0.0, "both should be negative"
        assert ratio < 0.35, (
            f"신규 유저(freq=2) 패널티가 파워 유저(freq=100) 대비 35% 미만이어야 함. "
            f"ratio={ratio:.3f} (cold={result_cold:.4f}, power={result_power:.4f})"
        )

    def test_power_user_3_rejects_strong_suppression(self):
        """파워 유저(freq_total=100), 3회 거부 → 강한 패널티 (< -0.3)."""
        ctx = _ctx(reject_hours_ago=[0.1, 0.5, 1.0], freq_total=100)
        result = compute(_cand(), ctx)
        assert result < -0.3, (
            f"파워 유저는 3회 거부 시 강한 패널티여야 함 (< -0.3), got {result:.4f}"
        )

    def test_negative_freq_total_treated_as_zero(self):
        """freq_total 음수 → 0으로 취급, penalty=0."""
        ctx = _ctx(reject_hours_ago=[0.5], freq_total=-5)
        result = compute(_cand(), ctx)
        assert result == 0.0, f"음수 freq_total → 신규 유저 취급, penalty=0, got {result}"
