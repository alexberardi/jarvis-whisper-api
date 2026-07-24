"""Seed voice.emotion_enabled + voice.emotion_min_confidence (acoustic affect)

Seeds system-default (all scope cols NULL) rows so the acoustic-affect settings
show up and are togglable in admin. voice.emotion_enabled defaults 'false' (the
feature is opt-in; when off the transcript path is unchanged). The gate still
functions off the code default without these rows — this migration only makes
them visible/editable, matching the repo's seed-every-setting convention.

Revision ID: 005
Revises: 004
Create Date: 2026-07-23 21:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


SETTINGS = [
    {
        "key": "voice.emotion_enabled",
        "value": "false",
        "value_type": "bool",
        "category": "voice",
        "description": (
            "Analyze the acoustic affect (arousal/energy) of the command audio "
            "and return it as an `affect` block on the transcribe response. "
            "Default OFF: when off, the analysis never runs (the response still "
            "carries `affect: null`). Sensitive inference — kept local and not "
            "persisted."
        ),
        "env_fallback": None,
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "voice.emotion_min_confidence",
        "value": "0.45",
        "value_type": "float",
        "category": "voice",
        "description": (
            "Minimum affect confidence (0-1) before the read is surfaced on the "
            "response. Below this, `affect` is null — a shaky read is worse than "
            "none. Every read is still logged for threshold tuning."
        ),
        "env_fallback": None,
        "requires_reload": False,
        "is_secret": False,
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    pg_sql = sa.text("""
        INSERT INTO settings (key, value, value_type, category, description,
                             env_fallback, requires_reload, is_secret,
                             household_id, node_id, user_id)
        VALUES (:key, :value, :value_type, :category, :description,
               :env_fallback, :requires_reload, :is_secret,
               NULL, NULL, NULL)
        ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
    """)
    sqlite_sql = sa.text("""
        INSERT OR IGNORE INTO settings (key, value, value_type, category, description,
                                       env_fallback, requires_reload, is_secret,
                                       household_id, node_id, user_id)
        VALUES (:key, :value, :value_type, :category, :description,
               :env_fallback, :requires_reload, :is_secret,
               NULL, NULL, NULL)
    """)

    for setting in SETTINGS:
        conn.execute(pg_sql if is_postgres else sqlite_sql, setting)


def downgrade() -> None:
    conn = op.get_bind()
    for setting in SETTINGS:
        conn.execute(
            sa.text("""
                DELETE FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            """),
            {"key": setting["key"]},
        )
