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
        description="Enable speaker identification",
    ),
    SettingDefinition(
        key="voice.encoder",
        category="voice",
        value_type="string",
        default="ecapa",
        description=(
            "Speaker recognition encoder. 'ecapa' (SpeechBrain ECAPA-TDNN) is "
            "the modern default, better on short utterances. 'resemblyzer' is "
            "the legacy GE2E encoder, kept as a rollback option."
        ),
        options=["ecapa", "resemblyzer"],
    ),
    SettingDefinition(
        key="voice.similarity_threshold",
        category="voice",
        value_type="float",
        default=0.5,
        description=(
            "Cosine similarity threshold for speaker matching on "
            "normal-length utterances (between voice.short_cutoff_seconds "
            "and voice.long_cutoff_seconds). Optimal value depends on the "
            "encoder — ECAPA typically wants ~0.50, resemblyzer ~0.75. "
            "Tune empirically using scripts/benchmark_speaker_encoders.py."
        ),
    ),
    SettingDefinition(
        key="voice.threshold_short",
        category="voice",
        value_type="float",
        default=0.65,
        description=(
            "Stricter threshold applied to short clips (duration < "
            "voice.short_cutoff_seconds). Defaults assume ECAPA; tune via "
            "the benchmark script."
        ),
    ),
    SettingDefinition(
        key="voice.threshold_long",
        category="voice",
        value_type="float",
        default=0.4,
        description=(
            "Relaxed threshold applied to long clips (duration > "
            "voice.long_cutoff_seconds). Defaults assume ECAPA."
        ),
    ),
    SettingDefinition(
        key="voice.short_cutoff_seconds",
        category="voice",
        value_type="float",
        default=1.0,
        description="Clips shorter than this use voice.threshold_short.",
    ),
    SettingDefinition(
        key="voice.long_cutoff_seconds",
        category="voice",
        value_type="float",
        default=3.0,
        description="Clips longer than this use voice.threshold_long.",
    ),

    # Server configuration
    SettingDefinition(
        key="server.port",
        category="server",
        value_type="int",
        default=7706,
        description="API server port",
        env_fallback="PORT",
        requires_reload=True,
    ),
    SettingDefinition(
        key="server.log_console_level",
        category="server",
        value_type="string",
        default="INFO",
        description="Console logging level",
        env_fallback="JARVIS_LOG_CONSOLE_LEVEL",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
    ),
    SettingDefinition(
        key="server.log_remote_level",
        category="server",
        value_type="string",
        default="DEBUG",
        description="Remote logging level",
        env_fallback="JARVIS_LOG_REMOTE_LEVEL",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
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
