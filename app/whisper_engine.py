"""In-process whisper.cpp engine via pywhispercpp.

Replaces the per-call `whisper-cli` subprocess. The model is loaded once
into VRAM (or RAM, on CPU builds) and reused for every request, eliminating
the fork/exec/load cost that dominated request latency on the GPU build.

Model selection is read from the settings service on every call, mirroring
the TTS provider-manager fingerprint-reload pattern: when the admin changes
`whisper.model_path` (or any other tracked setting), the next call notices
the fingerprint change and rebuilds the model. Failed rebuilds keep the
existing model so transcription doesn't break on a bad path.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pywhispercpp.model import Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EngineFingerprint:
    """Settings state that should trigger a model reload when changed."""
    model_path: str
    n_threads: int


def _read_fingerprint() -> _EngineFingerprint:
    """Read the active model config from settings, with env fallback.

    Pulls `whisper.model_path` from the settings service (DB → env → default
    resolution is handled by jarvis-settings-client). `n_threads` is taken
    from `WHISPER_N_THREADS` since it's not exposed as a setting yet.
    """
    from app.services.settings_service import get_settings_service
    settings = get_settings_service()
    model_path = settings.get_str("whisper.model_path", "")
    if not model_path:
        # Last-ditch — a missing path means the operator hasn't configured
        # whisper at all; fail loudly rather than load a default that
        # probably isn't on disk.
        raise RuntimeError(
            "whisper.model_path is not set and WHISPER_MODEL env var is empty"
        )
    n_threads = int(os.getenv("WHISPER_N_THREADS", "4"))
    return _EngineFingerprint(model_path=model_path, n_threads=n_threads)


def _build_model(fp: _EngineFingerprint) -> "Model":
    """Construct a pywhispercpp Model from the given fingerprint."""
    from pywhispercpp.model import Model
    logger.info(
        "Loading whisper model from %s (n_threads=%d)",
        fp.model_path,
        fp.n_threads,
    )
    return Model(
        model=fp.model_path,
        n_threads=fp.n_threads,
        print_progress=False,
        print_realtime=False,
    )


class WhisperEngine:
    """Caches a single Model and reloads when the settings fingerprint changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: "Model | None" = None
        self._fingerprint: _EngineFingerprint | None = None

    def get(self) -> "Model":
        fp = _read_fingerprint()
        with self._lock:
            if self._model is not None and self._fingerprint == fp:
                return self._model

            try:
                new_model = _build_model(fp)
            except Exception as e:
                if self._model is not None:
                    logger.warning(
                        "Failed to reload whisper model for %s: %s. "
                        "Keeping previous model.",
                        fp,
                        e,
                        exc_info=True,
                    )
                    return self._model
                raise

            self._model = new_model
            self._fingerprint = fp
            logger.info("Whisper model loaded")
            return new_model

    def reset(self) -> None:
        """Drop the cached model. Next `get()` rebuilds from settings."""
        with self._lock:
            self._model = None
            self._fingerprint = None


_engine: WhisperEngine | None = None


def get_engine() -> WhisperEngine:
    global _engine
    if _engine is None:
        _engine = WhisperEngine()
    return _engine


def get_model() -> "Model":
    """Return the active whisper Model, reloading from settings if needed."""
    return get_engine().get()


def reset_model_for_tests() -> None:
    """Test-only helper: reset the engine singleton + cached model."""
    global _engine
    _engine = None
