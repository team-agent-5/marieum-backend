"""SQLAlchemy ORM 모델 (v4 — 환자 1인 앱 전용).

v1 돌봄·관리자 테이블 (`calls`, `caregivers`, `feedback`, `call_responses`, `documents`)은
전부 삭제됨. 상세 설계: `marieum-plan/design/data-model.md`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────
# Patient — 환자 (앱 1인 사용자)
# ──────────────────────────────────────────────────────────────


class Patient(Base):
    """환자. user_id/password 로그인(§M0·M1), 메타(display_name/room/disease)는 백엔드 사후 등록."""

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 인증
    user_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    nickname: Mapped[str] = mapped_column(String(16))

    # 메타 (nullable)
    display_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    room: Mapped[str | None] = mapped_column(String(16), nullable=True)
    disease: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 설정 (서버 미러, 로컬이 primary)
    tts_rate: Mapped[str] = mapped_column(String(8), default="normal")       # slow|normal|fast
    tts_volume: Mapped[str] = mapped_column(String(8), default="normal")      # low|normal|high
    large_text: Mapped[bool] = mapped_column(Boolean, default=False)
    realtime_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)


# ──────────────────────────────────────────────────────────────
# PhraseCatalog — cold-start 공통 문장 (seed-time, read-only)
# ──────────────────────────────────────────────────────────────


class PhraseCatalog(Base):
    """신규 환자 cold-start용 공통 문장 20~50건. Seed로만 insert, 환자 선택 쌓이면 가중치 자연 하락."""

    __tablename__ = "phrase_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)        # 일반/요청/감정/응답 등
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    source: Mapped[str] = mapped_column(String(16), default="manual")              # manual/zeroth/self-recorded
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# PhraseLog — 환자별 발화·선택 이력 (append-only, 핵심)
# ──────────────────────────────────────────────────────────────


class PhraseLog(Base):
    """환자별 발화·선택 이력. ranking engine이 매번 조회해 feature 계산."""

    __tablename__ = "phrase_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)

    # ASR 입력 (nullable — preset은 전사 없음)
    partial_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 선택 결과
    source: Mapped[str] = mapped_column(String(16))                                # voice|preset|fallback|voice_bypass|rejected
    candidates_shown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    selected_phrase: Mapped[str | None] = mapped_column(Text, nullable=True)        # rejected 시 null
    selected_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)   # action=rejected 전용

    # 피드백 측정
    selection_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 임베딩 (selected_phrase 또는 final_transcript. write-time 생성, rejected면 null 가능)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)

    # sequence_context 체인
    prev_phrase_log_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("phrase_log.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)


# ──────────────────────────────────────────────────────────────
# PatientPreference — 명시 등록 선호 (seed로만)
# ──────────────────────────────────────────────────────────────


class PatientPreference(Base):
    """관리자/백엔드가 환자 등록 시 명시 입력한 선호. 'registered_preference' feature용."""

    __tablename__ = "patient_preference"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), index=True
    )
    preference_type: Mapped[str] = mapped_column(String(32))                       # preferred_phrase | time_preference | topic_preference
    content: Mapped[str] = mapped_column(Text)                                     # 자유 텍스트
    boost_phrases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)  # 매칭 시 가중치 올릴 문장
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ──────────────────────────────────────────────────────────────
# AuditLog — append-only 이벤트 로그 (§7.6)
# ──────────────────────────────────────────────────────────────


class AuditLog(Base):
    """§10.6. 모든 발화 흐름 이벤트 append."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True, index=True
    )
    event: Mapped[str] = mapped_column(String(32))
    # utterance_start | transcript_interim | transcript_final | candidates_generated |
    # phrase_selected | phrase_rejected | preset_selected | bypass_logged | tts_played | fallback_used
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
