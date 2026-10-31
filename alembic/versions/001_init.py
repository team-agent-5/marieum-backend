"""init schema with pgvector

Revision ID: 001_init
Revises:
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "caregivers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("device_id", sa.String(128), nullable=True),
        sa.Column("active_from", sa.String(8), nullable=True),
        sa.Column("active_to", sa.String(8), nullable=True),
        sa.Column("meta", postgresql.JSONB, nullable=False, server_default="{}"),
    )

    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("room", sa.String(32), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("birthdate", sa.DateTime, nullable=True),
        sa.Column("disease", sa.String(64), nullable=False),
        sa.Column("swallow_level", sa.String(16), nullable=False, server_default="normal"),
        sa.Column("default_caregiver_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("caregivers.id"), nullable=True),
        sa.Column("meta", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("corrected_text", sa.Text, nullable=False),
        sa.Column("intent", sa.String(32), nullable=False),
        sa.Column("intent_confidence", sa.Float, nullable=False),
        sa.Column("intent_source", sa.String(16), nullable=False),
        sa.Column("high_stakes", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("priority", sa.Integer, nullable=False, server_default="1"),
        sa.Column("assignee_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("caregivers.id"), nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_calls_trace_id", "calls", ["trace_id"])
    op.create_index("ix_calls_patient_id", "calls", ["patient_id"])
    op.create_index("ix_calls_created_at", "calls", ["created_at"])
    op.create_index("ix_calls_intent", "calls", ["intent"])

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("modified_fields", postgresql.JSONB, nullable=True),
        sa.Column("reject_reason", sa.Text, nullable=True),
        sa.Column("caregiver_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("caregivers.id"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_trace_id", "feedback", ["trace_id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patients.id"), nullable=True),
        sa.Column("intent_tags", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("title", sa.String(256), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source_url", sa.String(512), nullable=True),
        sa.Column("page", sa.Integer, nullable=True),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_documents_source", "documents", ["source"])
    op.create_index("ix_documents_patient_id", "documents", ["patient_id"])
    op.execute("CREATE INDEX ix_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops)")

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("event", sa.String(32), nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_audit_trace_id", "audit_log", ["trace_id"])
    op.create_index("ix_audit_created_at", "audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("documents")
    op.drop_table("feedback")
    op.drop_table("calls")
    op.drop_table("patients")
    op.drop_table("caregivers")
    op.execute("DROP EXTENSION IF EXISTS vector")
