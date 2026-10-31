"""Add time_tags column to phrase_catalog for time-based recommendation.

- phrase_catalog: add time_tags TEXT[] column with GIN index
- Enables filtering catalog phrases by time-of-day bucket (morning/afternoon/evening/night)

Revision ID: 005_catalog_time_tags
Revises: 004_agent_prep
Create Date: 2026-10-31
"""

from __future__ import annotations

from alembic import op

revision = "005_catalog_time_tags"
down_revision = "004_agent_prep"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add time_tags column as TEXT[] with empty-array default
    op.execute(
        """
        ALTER TABLE phrase_catalog
        ADD COLUMN time_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]
        """
    )

    # GIN index for efficient array-containment queries (e.g. time_tags @> '{morning}')
    op.execute(
        """
        CREATE INDEX ix_phrase_catalog_time_tags
        ON phrase_catalog USING GIN (time_tags)
        """
    )


def downgrade() -> None:
    raise RuntimeError(
        "005 adds time_tags required by agent time-based ranking; downgrade disabled in hackathon scope"
    )
