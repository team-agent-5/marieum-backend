"""api-spec.md coverage: call_responses, feedback.note, calls.notify_attempts, documents.deleted_at

- calls.notify_attempts        : §3.6.7 재알림 attempt 횟수 persist
- calls.status                 : 'cancelled' 포함 (타입은 String 그대로, 코멘트만)
- feedback.note                : §3.6.5 자유 메모 (§10.8 편향 모니터링 문맥)
- call_responses (new)         : §3.6.6 돌봄 → 환자 응답 로그
- documents.deleted_at         : §10.7 메모 삭제 24h 가드레일 (soft delete)

Revision ID: 002_api_spec
Revises: 001_init
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "002_api_spec"
down_revision = "001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. calls.notify_attempts (default 1 — 최초 호출이 attempt 1)
    op.add_column(
        "calls",
        sa.Column("notify_attempts", sa.Integer(), nullable=False, server_default="1"),
    )
    # 서버 default 는 신규 insert 편의용. 코드에서 명시 설정하므로 server_default 유지해도 무해.

    # 2. feedback.note
    op.add_column("feedback", sa.Column("note", sa.Text(), nullable=True))

    # 3. documents.deleted_at (soft delete)
    op.add_column("documents", sa.Column("deleted_at", sa.DateTime(), nullable=True))
    op.create_index("ix_documents_deleted_at", "documents", ["deleted_at"])

    # 4. call_responses (신규 테이블)
    op.create_table(
        "call_responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "call_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("calls.id"),
            nullable=False,
        ),
        sa.Column(
            "sender_caregiver_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("caregivers.id"),
            nullable=True,
        ),
        sa.Column("mp3_key", sa.String(64), nullable=True),
        sa.Column("custom_text", sa.Text(), nullable=True),
        sa.Column("expect_answer", sa.String(16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_call_responses_call_id", "call_responses", ["call_id"])
    op.create_index("ix_call_responses_created_at", "call_responses", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_call_responses_created_at", table_name="call_responses")
    op.drop_index("ix_call_responses_call_id", table_name="call_responses")
    op.drop_table("call_responses")

    op.drop_index("ix_documents_deleted_at", table_name="documents")
    op.drop_column("documents", "deleted_at")

    op.drop_column("feedback", "note")

    op.drop_column("calls", "notify_attempts")
