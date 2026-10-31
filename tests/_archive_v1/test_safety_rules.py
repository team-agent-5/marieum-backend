"""가드레일 §10.4 High-stakes 룰 단위 테스트."""

from __future__ import annotations

from app.orchestrator.safety_rules import check_high_stakes, is_nurse_direct


def test_dyspnea_keyword_detected() -> None:
    assert "숨이 차" in check_high_stakes("숨이 차요 너무 힘들어요")[0]


def test_fall_keyword_detected() -> None:
    assert check_high_stakes("넘어졌어요")


def test_normal_speech_no_high_stakes() -> None:
    assert check_high_stakes("물 좀 주세요") == []


def test_chest_pain_detected() -> None:
    assert check_high_stakes("가슴이 아파요")


def test_nurse_direct_mapping() -> None:
    assert is_nurse_direct("dyspnea")
    assert is_nurse_direct("fall")
    assert not is_nurse_direct("water")
    assert not is_nurse_direct("toilet")
