"""가드레일 §10.6 — append-only 감사 로그."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog


async def audit_append(
    session: AsyncSession,
    *,
    trace_id: str,
    event: str,
    payload: dict[str, Any],
    patient_id: uuid.UUID | None = None,
) -> None:
    row = AuditLog(
        trace_id=trace_id,
        event=event,
        payload=payload,
        patient_id=patient_id,
    )
    session.add(row)
    await session.flush()
