"""JWT + 비밀번호 해시 helper (v4 — 환자 1인 앱).

- JWT HS256, 365일 TTL, 10s clock skew, require sub/exp/iat
- 비밀번호: scrypt (stdlib hashlib), compare_digest로 타이밍 공격 방지
- Depends(require_patient) → PatientPrincipal 반환
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.models import Patient
from app.schemas import PatientPrincipal

PATIENT_TTL = timedelta(days=365)
_JWT_LEEWAY_SEC = 10


# ────────── 비밀번호 해시 ──────────


def _hash_password(password: str, salt: bytes | None = None) -> str:
    """password hash. 반환 형식: 'scrypt$<salt_hex>$<hash_hex>'."""
    if salt is None:
        salt = uuid.uuid4().bytes
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def hash_password(password: str) -> str:
    return _hash_password(password)


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    recomputed = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    ).hex()
    # 타이밍 공격 방어: 상수 시간 비교.
    return hmac.compare_digest(recomputed, hash_hex)


# ────────── JWT 발급·검증 ──────────


def issue_token(patient_id: uuid.UUID, nickname: str) -> tuple[str, datetime]:
    now = datetime.now(tz=UTC)
    exp = now + PATIENT_TTL
    payload: dict = {
        "sub": str(patient_id),
        "nickname": nickname,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return token, exp


def decode_token(token: str) -> PatientPrincipal:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            leeway=_JWT_LEEWAY_SEC,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "토큰이 만료되었습니다. 다시 로그인해 주세요."},
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "유효하지 않은 토큰입니다."},
        ) from exc

    try:
        patient_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "토큰 payload가 올바르지 않습니다."},
        ) from exc

    return PatientPrincipal(patient_id=patient_id, nickname=payload.get("nickname", ""))


# ────────── FastAPI dependency ──────────


async def require_patient(
    session: Annotated[AsyncSession, Depends(get_session)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> PatientPrincipal:
    """Authorization: Bearer <JWT> 디코딩 + DB 활성 환자 확인."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "Bearer 토큰이 필요합니다."},
        )
    token = authorization.split(" ", 1)[1]
    principal = decode_token(token)

    row = await session.get(Patient, principal.patient_id)
    if row is None or row.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "환자 레코드를 찾을 수 없습니다."},
        )
    return principal


