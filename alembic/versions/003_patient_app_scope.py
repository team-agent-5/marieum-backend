"""v4 patient-only scope: drop caregiver tables, rebuild patients, add phrase_log/catalog/preference

- 돌봄·관리자 테이블 전면 DROP: calls, caregivers, feedback, call_responses, documents
- Patient 재구성: 돌봄 FK 제거, 인증(user_id/password_hash/nickname) + 설정 컬럼 추가
- 신규 테이블: phrase_catalog, phrase_log, patient_preference
- AuditLog에 patient_id FK 추가

Revision ID: 003_patient_app_scope
Revises: 002_api_spec
Create Date: 2026-10-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "003_patient_app_scope"
down_revision = "002_api_spec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0. 필수 익스텐션 — trigram 인덱스 사용 전 선행 생성
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 1. 돌봄 측 테이블 DROP (FK 순서 역순)
    #    call_responses → calls, feedback → calls → caregivers → documents → patients FK → caregivers
    op.execute("DROP INDEX IF EXISTS ix_call_responses_created_at")
    op.execute("DROP INDEX IF EXISTS ix_call_responses_call_id")
    op.execute("DROP TABLE IF EXISTS call_responses CASCADE")

    op.execute("DROP INDEX IF EXISTS ix_feedback_trace_id")
    op.execute("DROP TABLE IF EXISTS feedback CASCADE")

    op.execute("DROP INDEX IF EXISTS ix_calls_intent")
    op.execute("DROP INDEX IF EXISTS ix_calls_created_at")
    op.execute("DROP INDEX IF EXISTS ix_calls_patient_id")
    op.execute("DROP INDEX IF EXISTS ix_calls_trace_id")
    op.execute("DROP TABLE IF EXISTS calls CASCADE")

    # documents: patient_note·clinical_guide 전부 v4 범위 외
    op.execute("DROP INDEX IF EXISTS ix_documents_embedding")
    op.execute("DROP INDEX IF EXISTS ix_documents_deleted_at")
    op.execute("DROP INDEX IF EXISTS ix_documents_patient_id")
    op.execute("DROP INDEX IF EXISTS ix_documents_source")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")

    # patients.default_caregiver_id → caregivers FK 제거 (CASCADE)
    with op.batch_alter_table("patients") as batch:
        batch.drop_column("default_caregiver_id")

    op.execute("DROP TABLE IF EXISTS caregivers CASCADE")

    # 2. Patient 재구성 — 인증·설정 컬럼 추가, swallow_level·meta는 유지(DB엔 남지만 코드는 참조 안함)
    with op.batch_alter_table("patients") as batch:
        batch.add_column(sa.Column("user_id", sa.String(32), nullable=True))
        batch.add_column(sa.Column("password_hash", sa.String(128), nullable=True))
        batch.add_column(sa.Column("nickname", sa.String(16), nullable=True))

        batch.add_column(sa.Column("tts_rate", sa.String(8), nullable=False, server_default="normal"))
        batch.add_column(sa.Column("tts_volume", sa.String(8), nullable=False, server_default="normal"))
        batch.add_column(sa.Column("large_text", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("realtime_mode", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("onboarding_completed", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("deleted_at", sa.DateTime(), nullable=True))

        # 기존 room/name/birthdate/disease/swallow_level은 유지 (nullable). display_name은 name 재사용.
        batch.alter_column("name", new_column_name="display_name", existing_type=sa.String(64), nullable=True)
        batch.alter_column("room", existing_type=sa.String(32), nullable=True)
        batch.alter_column("disease", existing_type=sa.String(64), nullable=True)
        # swallow_level은 v4 범위 밖. 컬럼은 남겨두되 코드에서 참조 안 함.

    op.create_index("ix_patients_user_id", "patients", ["user_id"], unique=True)
    op.create_index("ix_patients_created_at", "patients", ["created_at"])
    op.create_index("ix_patients_deleted_at", "patients", ["deleted_at"])

    # 3. phrase_catalog
    op.create_table(
        "phrase_catalog",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("category", sa.String(32), nullable=True),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.execute("CREATE INDEX ix_phrase_catalog_text_trgm ON phrase_catalog USING gin (text gin_trgm_ops)")
    op.execute(
        "CREATE INDEX ix_phrase_catalog_embedding ON phrase_catalog "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )

    # 4. phrase_log
    op.create_table(
        "phrase_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("partial_transcript", sa.Text, nullable=True),
        sa.Column("final_transcript", sa.Text, nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("candidates_shown", postgresql.JSONB, nullable=True),
        sa.Column("selected_phrase", sa.Text, nullable=True),
        sa.Column("selected_rank", sa.Integer, nullable=True),
        sa.Column("reject_reason", sa.String(32), nullable=True),
        sa.Column("selection_latency_ms", sa.Integer, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column(
            "prev_phrase_log_id",
            sa.BigInteger,
            sa.ForeignKey("phrase_log.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
    )
    op.create_index(
        "ix_phrase_log_patient_created",
        "phrase_log",
        ["patient_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_phrase_log_patient_selected",
        "phrase_log",
        ["patient_id", "selected_phrase"],
    )
    op.create_index("ix_phrase_log_patient_deleted", "phrase_log", ["patient_id", "deleted_at"])
    op.create_index("ix_phrase_log_trace_id", "phrase_log", ["trace_id"])
    op.execute(
        "CREATE INDEX ix_phrase_log_embedding ON phrase_log "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )
    op.execute(
        "CREATE INDEX ix_phrase_log_selected_trgm ON phrase_log "
        "USING gin (selected_phrase gin_trgm_ops)"
    )

    # 5. patient_preference
    op.create_table(
        "patient_preference",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column("preference_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("boost_phrases", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_patient_preference_patient", "patient_preference", ["patient_id", "deleted_at"])
    op.execute(
        "CREATE INDEX ix_patient_preference_embedding ON patient_preference "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )

    # 6. audit_log — patient_id FK 추가 (nullable, 소급 적용 가능하도록)
    with op.batch_alter_table("audit_log") as batch:
        batch.add_column(
            sa.Column(
                "patient_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("patients.id"),
                nullable=True,
            )
        )
    op.create_index("ix_audit_log_patient_id", "audit_log", ["patient_id"])


def downgrade() -> None:
    # 되돌리는 건 복잡하고 돌봄 테이블 데이터 복구 불가. 1차 MVP 범위에서 downgrade는 지원하지 않음.
    # 로컬 DB를 리셋하는 쪽이 안전: docker compose down -v && up -d && alembic upgrade head.
    raise RuntimeError(
        "003_patient_app_scope는 downgrade를 지원하지 않습니다. "
        "DB를 리셋하려면 `docker compose down -v && docker compose up -d && alembic upgrade head` 를 사용하세요."
    )
