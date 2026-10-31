"""Confidence Router 유닛 테스트 (Phase A2).

경계값 테스트 + None/음수 D3 fallback 검증.
설계 참조: ``marieum-plan/design/agent-algorithm.md`` §2.
"""
from __future__ import annotations

import pytest

from app.agent.confidence_router import (
    classify_branch,
    expected_candidate_count,
    is_llm_correction_enabled,
)


# ══════════════════════════════════════════════════════════════════════════════
# classify_branch — 경계값
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "conf,expected",
    [
        # retry 분기 — conf < 0.40
        (0.0, "retry"),
        (0.1, "retry"),
        (0.39, "retry"),
        (0.3999, "retry"),
        # 경계 (0.40) — low 시작
        (0.40, "low"),
        (0.5, "low"),
        (0.7, "low"),
        (0.89, "low"),
        (0.8999, "low"),
        # 경계 (0.90) — mid 시작
        (0.90, "mid"),
        (0.95, "mid"),
        (0.97, "mid"),
        (0.9799, "mid"),
        # 경계 (0.98) — high 시작
        (0.98, "high"),
        (0.99, "high"),
        (1.0, "high"),
    ],
)
def test_classify_branch_boundaries(conf: float, expected: str) -> None:
    """임계값 경계 분기 분류."""
    assert classify_branch(conf) == expected


def test_classify_branch_none_fallback_to_high() -> None:
    """D3: asr_confidence 미제공 → 1.0 간주 → high."""
    assert classify_branch(None) == "high"


def test_classify_branch_negative_fallback_to_high() -> None:
    """D3: 음수 (방어적) → high."""
    assert classify_branch(-0.1) == "high"
    assert classify_branch(-1.0) == "high"


# ══════════════════════════════════════════════════════════════════════════════
# is_llm_correction_enabled
# ══════════════════════════════════════════════════════════════════════════════


def test_llm_correction_enabled_for_low_and_mid() -> None:
    """low·mid 분기만 교정 LLM 호출."""
    assert is_llm_correction_enabled("low") is True
    assert is_llm_correction_enabled("mid") is True


def test_llm_correction_disabled_for_high_and_retry() -> None:
    """high·retry 분기는 LLM skip."""
    assert is_llm_correction_enabled("high") is False
    assert is_llm_correction_enabled("retry") is False


# ══════════════════════════════════════════════════════════════════════════════
# expected_candidate_count
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "branch,count",
    [
        ("retry", 0),
        ("low", 3),
        ("mid", 3),
        ("high", 1),
    ],
)
def test_expected_candidate_count(branch: str, count: int) -> None:
    """UI 가 기대하는 분기별 후보 수."""
    assert expected_candidate_count(branch) == count  # type: ignore[arg-type]
