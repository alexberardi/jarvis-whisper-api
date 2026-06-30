"""Seed whisper.allow_model_autodownload setting (default false, fail closed)

Seeds a system-default (all scope cols NULL) row for the egress opt-in gate.
Default 'false' keeps the service fully offline: whisper.cpp will NOT
auto-download a model from huggingface.co unless an operator explicitly opts
in. Outbound internet is opt-in only.

Revision ID: 004
Revises: 003
Create Date: 2026-06-30 11:30:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


SETTING = {
    "key": "whisper.allow_model_autodownload",
    "value": "false",
    "value_type": "bool",
    "category": "whisper.model",
    "description": (
        "Allow whisper.cpp / pywhispercpp to auto-download a model from "
        "huggingface.co when no local file exists at whisper.model_path. "
        "Default false keeps the service fully offline (outbound internet is "
        "opt-in only)."
    ),
    "env_fallback": "WHISPER_ALLOW_MODEL_AUTODOWNLOAD",
    "requires_reload": True,
    "is_secret": False,
}


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    if is_postgres:
        conn.execute(
            sa.text("""
                INSERT INTO settings (key, value, value_type, category, description,
                                     env_fallback, requires_reload, is_secret,
                                     household_id, node_id, user_id)
                VALUES (:key, :value, :value_type, :category, :description,
                       :env_fallback, :requires_reload, :is_secret,
                       NULL, NULL, NULL)
                ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
            """),
            SETTING
        )
    else:
        conn.execute(
            sa.text("""
                INSERT OR IGNORE INTO settings (key, value, value_type, category, description,
                                               env_fallback, requires_reload, is_secret,
                                               household_id, node_id, user_id)
                VALUES (:key, :value, :value_type, :category, :description,
                       :env_fallback, :requires_reload, :is_secret,
                       NULL, NULL, NULL)
            """),
            SETTING
        )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            DELETE FROM settings
            WHERE key = :key
              AND household_id IS NULL
              AND node_id IS NULL
              AND user_id IS NULL
        """),
        {"key": SETTING["key"]}
    )
