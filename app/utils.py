from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.exceptions import WhisperTranscriptionError
from app.whisper_engine import get_model

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:
    VoiceEncoder = None  # type: ignore[assignment,misc]
    preprocess_wav = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class SpeakerResult:
    """Result of speaker recognition."""

    user_id: int | None
    confidence: float


def run_whisper(
    wav_path: str,
    prompt: str | None = None,
    temperature: float = 0.0,
    temperature_inc: float = 0.2,
    beam_size: int = 5,
) -> str:
    """Transcribe audio using the in-process whisper.cpp model.

    Args:
        wav_path: Path to WAV file to transcribe.
        prompt: Optional initial prompt to guide transcription.
        temperature: Initial temperature for sampling (default 0.0).
        temperature_inc: Temperature increment on decode failure (default 0.2).
        beam_size: Beam size for beam search (default 5).

    Returns:
        Transcribed text.

    Raises:
        WhisperTranscriptionError: If transcription fails.
    """
    try:
        model = get_model()
        segments = model.transcribe(
            wav_path,
            language="en",
            initial_prompt=prompt or "",
            temperature=temperature,
            temperature_inc=temperature_inc,
            beam_search={"beam_size": beam_size, "patience": -1.0},
        )
    except Exception as e:
        raise WhisperTranscriptionError(
            f"Whisper transcription failed: {type(e).__name__}: {e}",
            stderr=str(e),
        ) from e

    return " ".join(seg.text for seg in segments).strip()


PROFILE_DIR = Path("voice_profiles")
_encoder: VoiceEncoder | None = None

# Cache loaded embeddings keyed by household_id, then user_id
_household_profiles_cache: dict[str, dict[int, np.ndarray]] = {}


def _resolve_voice_device() -> str:
    """Pick the torch device for the speaker-recognition encoder.

    Honors ``JARVIS_VOICE_DEVICE`` env (``auto`` | ``cuda`` | ``cpu``).
    Falls back to CPU if CUDA is requested or auto-selected but the
    runtime doesn't support it — never raises so the service stays up.
    """
    pref = os.getenv("JARVIS_VOICE_DEVICE", "auto").lower()
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if pref == "cuda":
        if not cuda_ok:
            logger.warning(
                "JARVIS_VOICE_DEVICE=cuda but torch.cuda.is_available() is False — falling back to CPU"
            )
            return "cpu"
        return "cuda"
    return "cuda" if cuda_ok else "cpu"


def _get_encoder() -> "VoiceEncoder":
    """Lazy-load the VoiceEncoder. Device chosen via JARVIS_VOICE_DEVICE."""
    global _encoder
    if _encoder is None:
        if VoiceEncoder is None:
            raise ImportError("resemblyzer is required for voice recognition")
        device = _resolve_voice_device()
        logger.info("Loading VoiceEncoder on device=%s", device)
        _encoder = VoiceEncoder(device=device, verbose=False)
    return _encoder


def hash_user_id(user_id: int) -> str:
    """Hash a user_id to a 16-character filename-safe string.

    Args:
        user_id: The user ID to hash.

    Returns:
        A 16-character hex string (SHA256 truncated).
    """
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]


def load_household_profiles(
    household_id: str, member_ids: list[int]
) -> dict[int, np.ndarray]:
    """Load voice profiles for household members.

    Profiles are stored at: voice_profiles/{household_id}/{hash(user_id)}.wav

    Args:
        household_id: The household UUID.
        member_ids: List of user IDs in the household.

    Returns:
        Dictionary mapping user_id to voice embedding.
    """
    # Check cache first
    if household_id in _household_profiles_cache:
        return _household_profiles_cache[household_id]

    profiles: dict[int, np.ndarray] = {}
    household_dir = PROFILE_DIR / household_id

    if not household_dir.exists():
        logger.debug(f"Household directory not found: {household_dir}")
        _household_profiles_cache[household_id] = profiles
        return profiles

    for user_id in member_ids:
        filename = hash_user_id(user_id) + ".wav"
        filepath = household_dir / filename
        if filepath.exists():
            try:
                wav = preprocess_wav(filepath)
                profiles[user_id] = _get_encoder().embed_utterance(wav)
                logger.debug(f"Loaded voice profile for user {user_id}")
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to load profile for user {user_id}: {type(e).__name__}: {e}")

    _household_profiles_cache[household_id] = profiles
    return profiles


def invalidate_household_cache(household_id: str | None = None) -> None:
    """Invalidate cached household profiles.

    Args:
        household_id: Specific household to invalidate, or None to clear all.
    """
    global _household_profiles_cache
    if household_id is None:
        _household_profiles_cache = {}
    elif household_id in _household_profiles_cache:
        del _household_profiles_cache[household_id]


def recognize_speaker(
    audio_path: str,
    household_id: str,
    member_ids: list[int],
    threshold: float = 0.75,
) -> SpeakerResult:
    """Recognize speaker from audio file within a household.

    Args:
        audio_path: Path to WAV file.
        household_id: The household UUID to scope the search.
        member_ids: List of user IDs in the household.
        threshold: Minimum confidence score for a match (default 0.75).

    Returns:
        SpeakerResult with user_id (int or None) and confidence.
        user_id is None if no match exceeds threshold or if no profiles exist.
    """
    profiles = load_household_profiles(household_id, member_ids)

    if not profiles:
        logger.warning(f"No speaker profiles for household {household_id}")
        return SpeakerResult(user_id=None, confidence=0.0)

    try:
        wav = preprocess_wav(audio_path)
        embed = _get_encoder().embed_utterance(wav)
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to process input audio: {type(e).__name__}: {e}")
        return SpeakerResult(user_id=None, confidence=0.0)

    # Compare to each profile
    scores: dict[int, float] = {
        user_id: float(np.inner(embed, ref_embed))
        for user_id, ref_embed in profiles.items()
    }

    # Choose best match above threshold
    best_user_id = max(scores, key=lambda k: scores[k])
    best_score = scores[best_user_id]

    matched = best_score > threshold
    logger.info(
        "Speaker match: household=%s best_user=%s score=%.3f threshold=%.2f → %s",
        household_id,
        best_user_id,
        best_score,
        threshold,
        "MATCHED" if matched else "no match",
    )

    if matched:
        return SpeakerResult(user_id=best_user_id, confidence=best_score)

    return SpeakerResult(user_id=None, confidence=best_score)
