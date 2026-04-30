"""Drop whisper.enable_cuda setting (build-time only, not a runtime toggle)

Revision ID: 003
Revises: 002
Create Date: 2026-04-30 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.get_bind().execute(
        sa.text("""
            DELETE FROM settings
            WHERE key = 'whisper.enable_cuda'
        """)
    )


def downgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    setting = {
        "key": "whisper.enable_cuda",
        "value": "false",
        "value_type": "bool",
        "category": "whisper.model",
        "description": "Enable CUDA acceleration for whisper.cpp",
        "env_fallback": "WHISPER_ENABLE_CUDA",
        "requires_reload": True,
        "is_secret": False,
    }

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
            setting
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
            setting
        )
