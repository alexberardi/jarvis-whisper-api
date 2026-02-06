"""Seed default settings

Revision ID: 002
Revises: 001
Create Date: 2026-02-05 17:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


# Settings definitions from app/services/settings_definitions.py
# EXCLUDED: whisper.model_path, whisper.cli_path (paths - fallback to env)
SETTINGS = [
    # Whisper model configuration - only non-path settings
    {
        "key": "whisper.enable_cuda",
        "value": "false",
        "value_type": "bool",
        "category": "whisper.model",
        "description": "Enable CUDA acceleration for whisper.cpp",
        "env_fallback": "WHISPER_ENABLE_CUDA",
        "requires_reload": True,
        "is_secret": False,
    },
    # Transcription parameters
    {
        "key": "whisper.default_temperature",
        "value": "0.0",
        "value_type": "float",
        "category": "whisper.transcription",
        "description": "Default initial temperature for sampling (0.0-1.0)",
        "env_fallback": "WHISPER_DEFAULT_TEMPERATURE",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "whisper.default_temperature_inc",
        "value": "0.2",
        "value_type": "float",
        "category": "whisper.transcription",
        "description": "Default temperature increment on decode failure (0.0-1.0)",
        "env_fallback": "WHISPER_DEFAULT_TEMPERATURE_INC",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "whisper.default_beam_size",
        "value": "5",
        "value_type": "int",
        "category": "whisper.transcription",
        "description": "Default beam size for beam search (1-16)",
        "env_fallback": "WHISPER_DEFAULT_BEAM_SIZE",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "whisper.language",
        "value": "en",
        "value_type": "string",
        "category": "whisper.transcription",
        "description": "Default language for transcription",
        "env_fallback": "WHISPER_LANGUAGE",
        "requires_reload": False,
        "is_secret": False,
    },
    # Voice recognition
    {
        "key": "voice.recognition_enabled",
        "value": "false",
        "value_type": "bool",
        "category": "voice",
        "description": "Enable speaker identification using resemblyzer",
        "env_fallback": "USE_VOICE_RECOGNITION",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "voice.similarity_threshold",
        "value": "0.75",
        "value_type": "float",
        "category": "voice",
        "description": "Cosine similarity threshold for speaker matching",
        "env_fallback": "VOICE_SIMILARITY_THRESHOLD",
        "requires_reload": False,
        "is_secret": False,
    },
    # Server configuration
    {
        "key": "server.port",
        "value": "8012",
        "value_type": "int",
        "category": "server",
        "description": "API server port",
        "env_fallback": "PORT",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "server.log_console_level",
        "value": "INFO",
        "value_type": "string",
        "category": "server",
        "description": "Console logging level (DEBUG, INFO, WARNING, ERROR)",
        "env_fallback": "JARVIS_LOG_CONSOLE_LEVEL",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "server.log_remote_level",
        "value": "DEBUG",
        "value_type": "string",
        "category": "server",
        "description": "Remote logging level (DEBUG, INFO, WARNING, ERROR)",
        "env_fallback": "JARVIS_LOG_REMOTE_LEVEL",
        "requires_reload": False,
        "is_secret": False,
    },
    # Auth configuration
    {
        "key": "auth.cache_ttl_seconds",
        "value": "60",
        "value_type": "int",
        "category": "auth",
        "description": "Auth validation cache TTL in seconds",
        "env_fallback": "NODE_AUTH_CACHE_TTL",
        "requires_reload": False,
        "is_secret": False,
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    for setting in SETTINGS:
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
            {"key": setting["key"]}
        )
