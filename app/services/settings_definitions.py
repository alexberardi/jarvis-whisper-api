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
        key="whisper.allow_model_autodownload",
        category="whisper.model",
        value_type="bool",
        default=False,
        description=(
            "Allow whisper.cpp / pywhispercpp to auto-download a model from "
            "huggingface.co when no local file exists at whisper.model_path. "
            "Default false keeps the service fully offline: outbound internet "
            "is opt-in only. When false and no local model is present, the "
            "engine fails closed with an actionable error instead of egressing. "
            "Supply your own model via whisper.model_path / WHISPER_MODEL and "
            "leave this off, or set it true to permit the download."
        ),
        env_fallback="WHISPER_ALLOW_MODEL_AUTODOWNLOAD",
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

    # Embed-time audio preprocessing (applied symmetrically to enrollment
    # AND recognition audio at the single ECAPA embed choke point).
    SettingDefinition(
        key="voice.embed_preprocess_enabled",
        category="voice",
        value_type="bool",
        default=False,
        description=(
            "Apply symmetric preprocessing (DC removal + silence trim + RMS "
            "normalization) to BOTH enrollment and recognition audio before "
            "ECAPA embedding. Default OFF: an offline leave-one-out benchmark "
            "on real profiles showed it is score-neutral (ECAPA already "
            "mean-var-normalizes input), so it ships dormant pending "
            "evaluation against per-trial telemetry rather than changing "
            "production matching today. Flipping this re-embeds cached "
            "profiles on the next request. ECAPA only — resemblyzer already "
            "normalizes internally."
        ),
    ),
    SettingDefinition(
        key="voice.embed_target_rms_db",
        category="voice",
        value_type="float",
        default=-23.0,
        description=(
            "Target RMS level (dBFS) that recognition + enrollment audio is "
            "normalized to before ECAPA embedding when "
            "voice.embed_preprocess_enabled is true. Affects scoring "
            "consistency only, not the loudness of any audio output."
        ),
    ),
    SettingDefinition(
        key="voice.embed_trim_silence_db",
        category="voice",
        value_type="float",
        default=-40.0,
        description=(
            "Silence threshold (dBFS) for trimming leading/trailing silence "
            "before ECAPA embedding when voice.embed_preprocess_enabled is "
            "true. Conservative by default so speech is never removed; an "
            "all-silence clip falls back to the untrimmed signal."
        ),
    ),
    SettingDefinition(
        key="voice.long_cutoff_seconds",
        category="voice",
        value_type="float",
        default=3.0,
        description="Clips longer than this use voice.threshold_long.",
    ),

    # Acoustic affect (mood) — a prosodic read of HOW something was said, for the
    # LLM to fuse with WHAT was said. Opt-in, off by default; runs concurrently
    # with the speaker pass so it adds no wall-clock. See app/affect.py.
    SettingDefinition(
        key="voice.emotion_enabled",
        category="voice",
        value_type="bool",
        default=False,
        description=(
            "Analyze the acoustic affect (arousal/energy) of the command audio "
            "and return it as an `affect` block on the transcribe response. "
            "Default OFF: when off, the analysis never runs (the response still "
            "carries `affect: null`). Sensitive inference — kept local, "
            "per-household opt-in, and not persisted."
        ),
    ),
    SettingDefinition(
        key="voice.emotion_min_confidence",
        category="voice",
        value_type="float",
        default=0.45,
        description=(
            "Minimum affect confidence (0-1) before the read is surfaced on the "
            "response. Below this, `affect` is null — a shaky read is worse than "
            "none, so we bias toward silence. Every read is still logged for "
            "threshold tuning regardless of this gate."
        ),
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
