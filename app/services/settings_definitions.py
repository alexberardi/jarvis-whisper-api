"""Settings definitions for jarvis-whisper-api.

Defines all configurable settings with their types, defaults, and metadata.
"""

from jarvis_settings_client import SettingDefinition


SETTINGS_DEFINITIONS: list[SettingDefinition] = [
    # Whisper model configuration
    SettingDefinition(
        key="whisper.model_path",
        category="whisper.model",
        value_type="string",
        default="~/whisper.cpp/models/ggml-base.en.bin",
        description="Path to the Whisper GGML model file",
        env_fallback="WHISPER_MODEL",
        requires_reload=True,
    ),
    SettingDefinition(
        key="whisper.cli_path",
        category="whisper.model",
        value_type="string",
        default="",
        description="Path to whisper-cli binary (auto-detected if empty)",
        env_fallback="WHISPER_CLI",
        requires_reload=True,
    ),
    SettingDefinition(
        key="whisper.enable_cuda",
        category="whisper.model",
        value_type="bool",
        default=False,
        description="Enable CUDA acceleration for whisper.cpp",
        env_fallback="WHISPER_ENABLE_CUDA",
        requires_reload=True,
    ),

    # Transcription parameters
    SettingDefinition(
        key="whisper.default_temperature",
        category="whisper.transcription",
        value_type="float",
        default=0.0,
        description="Default initial temperature for sampling (0.0-1.0)",
        env_fallback="WHISPER_DEFAULT_TEMPERATURE",
    ),
    SettingDefinition(
        key="whisper.default_temperature_inc",
        category="whisper.transcription",
        value_type="float",
        default=0.2,
        description="Default temperature increment on decode failure (0.0-1.0)",
        env_fallback="WHISPER_DEFAULT_TEMPERATURE_INC",
    ),
    SettingDefinition(
        key="whisper.default_beam_size",
        category="whisper.transcription",
        value_type="int",
        default=5,
        description="Default beam size for beam search (1-16)",
        env_fallback="WHISPER_DEFAULT_BEAM_SIZE",
    ),
    SettingDefinition(
        key="whisper.language",
        category="whisper.transcription",
        value_type="string",
        default="en",
        description="Default language for transcription",
        env_fallback="WHISPER_LANGUAGE",
    ),

    # Voice recognition
    SettingDefinition(
        key="voice.recognition_enabled",
        category="voice",
        value_type="bool",
        default=False,
        description="Enable speaker identification using resemblyzer",
        env_fallback="USE_VOICE_RECOGNITION",
    ),
    SettingDefinition(
        key="voice.similarity_threshold",
        category="voice",
        value_type="float",
        default=0.75,
        description="Cosine similarity threshold for speaker matching",
        env_fallback="VOICE_SIMILARITY_THRESHOLD",
    ),

    # Server configuration
    SettingDefinition(
        key="server.port",
        category="server",
        value_type="int",
        default=8012,
        description="API server port",
        env_fallback="PORT",
        requires_reload=True,
    ),
    SettingDefinition(
        key="server.log_console_level",
        category="server",
        value_type="string",
        default="INFO",
        description="Console logging level (DEBUG, INFO, WARNING, ERROR)",
        env_fallback="JARVIS_LOG_CONSOLE_LEVEL",
    ),
    SettingDefinition(
        key="server.log_remote_level",
        category="server",
        value_type="string",
        default="DEBUG",
        description="Remote logging level (DEBUG, INFO, WARNING, ERROR)",
        env_fallback="JARVIS_LOG_REMOTE_LEVEL",
    ),

    # Auth configuration
    SettingDefinition(
        key="auth.cache_ttl_seconds",
        category="auth",
        value_type="int",
        default=60,
        description="Auth validation cache TTL in seconds",
        env_fallback="NODE_AUTH_CACHE_TTL",
    ),
]
