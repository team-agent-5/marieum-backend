"""시간대 기반 하드코딩 fallback — RAG retrieval 모두 실패 시."""
from __future__ import annotations

from typing import Literal

TimeBucket = Literal["morning", "afternoon", "evening", "night"]

# 각 시간대별 환자가 흔히 요청하는 phrase 2-3개
_FALLBACK: dict[TimeBucket, list[str]] = {
    "morning": ["물 주세요", "약 주세요", "일으켜 주세요"],
    "afternoon": ["물 주세요", "밥 먹을래요", "화장실 가고 싶어요"],
    "evening": ["미지근한 물 주세요", "밥 먹을래요", "눕고 싶어요"],
    "night": ["물 주세요", "약 주세요", "불 꺼주세요"],
}


def fallback_candidates(bucket: TimeBucket, limit: int = 3) -> list[str]:
    """시간대별 하드코딩 phrase 반환. DB 접근 없음.

    사용처: RAG retrieval 완전 실패 (DB down, 쿼리 실패, 결과 0건) 시
            orchestrator가 L2 fallback으로 호출.
    """
    return _FALLBACK.get(bucket, _FALLBACK["afternoon"])[:limit]
