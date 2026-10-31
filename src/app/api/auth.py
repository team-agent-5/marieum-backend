"""Auth 라우터 (v4) — signup / login / logout.

- 환자 자율 가입 (user_id/password/nickname). 관리자 등록 X.
- JWT 365일 TTL.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Patient
from app.schemas import (
    AuthResponse,
    LoginRequest,
    PatientBrief,
    PatientPrincipal,
    SignupRequest,
)
from app.security import PATIENT_TTL, hash_password, issue_token, require_patient, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="환자 회원가입 → JWT 발급 (M0, S0)",
    responses={
        409: {"description": "user_id 중복"},
        400: {"description": "검증 실패 (비밀번호 불일치 등)"},
    },
)
async def signup(
    body: SignupRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthResponse:
    # 중복 체크
    existing = await session.scalar(select(Patient).where(Patient.user_id == body.user_id))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "CONFLICT", "message": "이미 사용 중인 아이디입니다"},
        )

    patient = Patient(
        user_id=body.user_id,
        password_hash=hash_password(body.password),
        nickname=body.nickname,
    )
    session.add(patient)
    await session.commit()
    await session.refresh(patient)

    token, _ = issue_token(patient.id, patient.nickname)
    return AuthResponse(
        access_token=token,
        expires_in=int(PATIENT_TTL.total_seconds()),
        patient=PatientBrief(
            id=patient.id,
            nickname=patient.nickname,
            onboarding_completed=patient.onboarding_completed,
        ),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="환자 로그인 → JWT 발급 (M1, S1)",
    responses={401: {"description": "ID/비밀번호 불일치"}},
)
async def login(
    body: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthResponse:
    row = await session.scalar(select(Patient).where(Patient.user_id == body.user_id))
    if row is None or row.deleted_at is not None or not verify_password(body.password, row.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "UNAUTHORIZED",
                "message": "아이디 또는 비밀번호가 올바르지 않습니다",
            },
        )

    token, _ = issue_token(row.id, row.nickname)
    return AuthResponse(
        access_token=token,
        expires_in=int(PATIENT_TTL.total_seconds()),
        patient=PatientBrief(
            id=row.id,
            nickname=row.nickname,
            onboarding_completed=row.onboarding_completed,
        ),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="로그아웃 — stateless, 클라이언트가 토큰 삭제 (M1)",
)
async def logout(_me: Annotated[PatientPrincipal, Depends(require_patient)]) -> None:
    # JWT stateless. 서버 블랙리스트는 1차 MVP 미구현.
    return None
