"""Settings service for jarvis-whisper-api.

Provides runtime configuration that can be modified without restarting.
Settings are stored in the database with fallback to environment variables.
"""

import logging
from typing import Any

from jarvis_settings_client import SettingsService as BaseSettingsService

from app.services.settings_definitions import SETTINGS_DEFINITIONS

logger = logging.getLogger(__name__)


class WhisperSettingsService(BaseSettingsService):
    """Settings service for Whisper API with helper methods."""

    def get_model_config(self) -> dict[str, Any]:
        """Get whisper model configuration."""
        return {
            "model_path": self.get_str("whisper.model_path", "~/whisper.cpp/models/ggml-base.en.bin"),
            "cli_path": self.get_str("whisper.cli_path", ""),
            "enable_cuda": self.get_bool("whisper.enable_cuda", False),
        }

    def get_transcription_defaults(self) -> dict[str, Any]:
        """Get default transcription parameters."""
        return {
            "temperature": self.get_float("whisper.default_temperature", 0.0),
            "temperature_inc": self.get_float("whisper.default_temperature_inc", 0.2),
            "beam_size": self.get_int("whisper.default_beam_size", 5),
            "language": self.get_str("whisper.language", "en"),
        }

    def get_voice_config(self) -> dict[str, Any]:
        """Get voice recognition configuration."""
        return {
            "enabled": self.get_bool("voice.recognition_enabled", False),
            "similarity_threshold": self.get_float("voice.similarity_threshold", 0.75),
        }


# Global singleton
_settings_service: WhisperSettingsService | None = None


def get_settings_service() -> WhisperSettingsService:
    """Get the global SettingsService instance."""
    global _settings_service
    if _settings_service is None:
        from app.db.models import Setting
        from app.db.session import get_session_local

        SessionLocal = get_session_local()
        _settings_service = WhisperSettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=SessionLocal,
            setting_model=Setting,
        )
    return _settings_service


def reset_settings_service() -> None:
    """Reset the settings service singleton (for testing)."""
    global _settings_service
    _settings_service = None
