"""Patient 라우터 (v4) — /patient/me, /patient/preferences."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Patient
from app.schemas import (
    PatientMeOut,
    PatientPreferences,
    PatientPrincipal,
    PreferencesPatch,
)
from app.security import require_patient

router = APIRouter(prefix="/patient", tags=["patient"])


def _to_me(row: Patient) -> PatientMeOut:
    return PatientMeOut(
        id=row.id,
        nickname=row.nickname,
        display_name=row.display_name,
        room=row.room,
        disease=row.disease,
        onboarding_completed=row.onboarding_completed,
        preferences=PatientPreferences(
            tts_rate=row.tts_rate,  # type: ignore[arg-type]
            tts_volume=row.tts_volume,  # type: ignore[arg-type]
            large_text=row.large_text,
            realtime_mode=row.realtime_mode,
            onboarding_completed=row.onboarding_completed,
        ),
        created_at=row.created_at,
    )


@router.get(
    "/me",
    response_model=PatientMeOut,
    summary="환자 프로필 로드 (M2)",
)
async def get_me(
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PatientMeOut:
    row = await session.get(Patient, me.patient_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="patient not found")
    return _to_me(row)


@router.patch(
    "/preferences",
    response_model=PatientPreferences,
    summary="환자 설정 미러 (S14·S15·S16)",
    description="로컬이 ground-truth. 서버는 다른 디바이스 복구용 스냅샷.",
)
async def patch_preferences(
    body: PreferencesPatch,
    me: Annotated[PatientPrincipal, Depends(require_patient)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PatientPreferences:
    row = await session.get(Patient, me.patient_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="patient not found")

    if body.tts_rate is not None:
        row.tts_rate = body.tts_rate
    if body.tts_volume is not None:
        row.tts_volume = body.tts_volume
    if body.large_text is not None:
        row.large_text = body.large_text
    if body.realtime_mode is not None:
        row.realtime_mode = body.realtime_mode
    if body.onboarding_completed is not None and body.onboarding_completed:
        # 한 번 true면 false로 되돌리지 않음
        row.onboarding_completed = True

    await session.commit()
    await session.refresh(row)
    return PatientPreferences(
        tts_rate=row.tts_rate,  # type: ignore[arg-type]
        tts_volume=row.tts_volume,  # type: ignore[arg-type]
        large_text=row.large_text,
        realtime_mode=row.realtime_mode,
        onboarding_completed=row.onboarding_completed,
    )
