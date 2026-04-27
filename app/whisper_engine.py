"""In-process whisper.cpp engine via pywhispercpp.

Replaces the per-call `whisper-cli` subprocess. The model is loaded once
into VRAM (or RAM, on CPU builds) and reused for every request, eliminating
the fork/exec/load cost that dominated request latency on the GPU build.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pywhispercpp.model import Model

logger = logging.getLogger(__name__)

_model: "Model | None" = None
_lock = Lock()


def get_model() -> "Model":
    """Return the process-wide whisper Model, loading on first call.

    Double-checked locking so concurrent first-callers don't both pay the
    load cost. After init, callers see the cached instance with no lock.

    Reads `WHISPER_MODEL` for the GGML model path and `WHISPER_N_THREADS`
    (default 4) for CPU thread count. Thread count is harmless on CUDA
    builds (whisper.cpp ignores it for GPU work).
    """
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        from pywhispercpp.model import Model

        model_path = os.getenv("WHISPER_MODEL")
        if not model_path:
            raise RuntimeError(
                "WHISPER_MODEL env var is not set; cannot load whisper model"
            )

        n_threads = int(os.getenv("WHISPER_N_THREADS", "4"))
        logger.info(
            "Loading whisper model from %s (n_threads=%d)", model_path, n_threads
        )
        _model = Model(
            model=model_path,
            n_threads=n_threads,
            print_progress=False,
            print_realtime=False,
        )
        logger.info("Whisper model loaded")
        return _model


def reset_model_for_tests() -> None:
    """Reset the cached model (test-only helper)."""
    global _model
    with _lock:
        _model = None
