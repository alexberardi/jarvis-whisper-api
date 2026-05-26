from __future__ import annotations

import hashlib
import logging
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from app.audio import load_audio
from app.exceptions import WhisperTranscriptionError
from app.whisper_engine import get_model

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:
    VoiceEncoder = None  # type: ignore[assignment,misc]
    preprocess_wav = None  # type: ignore[assignment]

try:
    from speechbrain.inference.speaker import EncoderClassifier as _SbEncoderClassifier
except ImportError:
    _SbEncoderClassifier = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class SpeakerResult:
    """Result of speaker recognition."""

    user_id: int | None
    confidence: float


@dataclass
class SpeakerEncoder:
    """Uniform facade over different speaker-embedding backends.

    Both encoders return L2-normalized embedding vectors (different
    dimensionality — 256 for resemblyzer, 192 for ECAPA — but cosine
    similarity is dimension-agnostic). Profiles on disk are stored as
    WAVs and re-embedded at load time, so switching the encoder doesn't
    invalidate existing enrollments — it just forces a cache re-load.
    """

    name: str
    embed: Callable[[Path], np.ndarray]


WHISPER_TARGET_SR = 16000


def _load_for_whisper(wav_path: str) -> np.ndarray:
    """Load a WAV as 16 kHz mono float32 ready for pywhispercpp.

    pywhispercpp's Model.transcribe rejects WAV files that aren't already
    at 16 kHz with `Exception: WAV file must be 16000 Hz`. The old
    whisper-cli subprocess auto-resampled inside the binary; the
    in-process call does not. Resample here to restore parity.
    """
    audio, sr = load_audio(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != WHISPER_TARGET_SR:
        g = math.gcd(sr, WHISPER_TARGET_SR)
        audio = resample_poly(audio, WHISPER_TARGET_SR // g, sr // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


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
        audio = _load_for_whisper(wav_path)
        model = get_model()
        segments = model.transcribe(
            audio,
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
_encoder: SpeakerEncoder | None = None
_encoder_fingerprint: str | None = None

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


def _wav_duration_seconds(path: Path) -> float:
    """Return clip duration in seconds. Returns 0.0 if unreadable."""
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate) if rate else 0.0
    except (OSError, wave.Error):
        return 0.0


def _load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load a WAV as 16 kHz mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)

    if sr != 16000:
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / (norm + 1e-9)


def _load_resemblyzer_encoder(device: str) -> SpeakerEncoder:
    if VoiceEncoder is None or preprocess_wav is None:
        raise ImportError("resemblyzer is not installed")
    inner = VoiceEncoder(device=device, verbose=False)

    def embed(path: Path) -> np.ndarray:
        wav = preprocess_wav(path)
        # resemblyzer returns L2-normalized embeddings already
        return inner.embed_utterance(wav)

    return SpeakerEncoder(name="resemblyzer", embed=embed)


def _load_ecapa_encoder(device: str) -> SpeakerEncoder:
    if _SbEncoderClassifier is None:
        raise ImportError("speechbrain is not installed")
    import torch

    classifier = _SbEncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.getenv("JARVIS_ECAPA_CACHE", "/tmp/ecapa-spkrec"),
    )

    def embed(path: Path) -> np.ndarray:
        signal = _load_wav_mono_16k(path)
        with torch.no_grad():
            tensor = torch.from_numpy(signal).unsqueeze(0)
            if device == "cuda":
                tensor = tensor.to(device)
            emb = classifier.encode_batch(tensor)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    return SpeakerEncoder(name="ecapa", embed=embed)


def _desired_encoder_name() -> str:
    """Resolve the currently-selected encoder name from settings."""
    try:
        from app.services.settings_service import get_settings_service
        return get_settings_service().get_string("voice.encoder", "ecapa") or "ecapa"
    except Exception:
        return os.getenv("VOICE_ENCODER", "ecapa")


def _get_encoder() -> SpeakerEncoder:
    """Lazy-load the speaker encoder. Honors voice.encoder setting.

    Reloads if the setting or device changes. Falls back gracefully:
    requested ECAPA but speechbrain unavailable → try resemblyzer.
    Requested resemblyzer but unavailable → ImportError (caller handles).
    """
    global _encoder, _encoder_fingerprint
    desired = _desired_encoder_name()
    device = _resolve_voice_device()
    fingerprint = f"{desired}:{device}"

    if _encoder is not None and _encoder_fingerprint == fingerprint:
        return _encoder

    logger.info("Loading speaker encoder name=%s device=%s", desired, device)
    try:
        if desired == "ecapa":
            new_encoder = _load_ecapa_encoder(device)
        elif desired == "resemblyzer":
            new_encoder = _load_resemblyzer_encoder(device)
        else:
            logger.warning("Unknown voice.encoder=%r, falling back to ecapa", desired)
            new_encoder = _load_ecapa_encoder(device)
    except ImportError as e:
        # Graceful fallback if the requested encoder's dep is missing
        if desired == "ecapa":
            logger.warning(
                "ECAPA encoder unavailable (%s) — falling back to resemblyzer",
                e,
            )
            new_encoder = _load_resemblyzer_encoder(device)
        else:
            raise

    _encoder = new_encoder
    if _encoder_fingerprint is not None:
        # Encoder switched — wipe cached profile embeddings so they get re-embedded
        invalidate_household_cache()
    _encoder_fingerprint = fingerprint
    return _encoder


def hash_user_id(user_id: int) -> str:
    """Hash a user_id to a 16-character filename-safe string.

    Args:
        user_id: The user ID to hash.

    Returns:
        A 16-character hex string (SHA256 truncated).
    """
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]


def user_profile_dir(household_id: str, user_id: int) -> Path:
    """Path to the per-user directory under voice_profiles/."""
    return PROFILE_DIR / household_id / hash_user_id(user_id)


def legacy_profile_path(household_id: str, user_id: int) -> Path:
    """Path to the legacy single-file profile location."""
    return PROFILE_DIR / household_id / (hash_user_id(user_id) + ".wav")


def migrate_legacy_profile_to_directory(household_id: str, user_id: int) -> None:
    """Move a legacy single .wav into the new per-user directory as sample_000.

    No-op if there's no legacy file. Safe to call repeatedly. Does NOT
    invalidate the in-memory cache — call sites that change the layout
    should invalidate explicitly.
    """
    legacy = legacy_profile_path(household_id, user_id)
    if not legacy.exists():
        return
    user_dir = user_profile_dir(household_id, user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    target = user_dir / "sample_000.wav"
    if target.exists():
        # Conflict: directory already has sample_000, drop the legacy file
        # (the directory is canonical; this only happens if someone
        # manually re-introduced a legacy file).
        logger.warning(
            "Legacy profile collision for user %s — dropping legacy file in favor of directory",
            user_id,
        )
        legacy.unlink()
        return
    legacy.rename(target)
    logger.info(
        "Migrated legacy profile for user %s to %s", user_id, target.name
    )


def next_sample_index(household_id: str, user_id: int) -> int:
    """Return the next free sample index for a user (>= existing max + 1)."""
    user_dir = user_profile_dir(household_id, user_id)
    if not user_dir.is_dir():
        return 0
    indices: list[int] = []
    for p in user_dir.glob("sample_*.wav"):
        stem = p.stem  # 'sample_007'
        try:
            indices.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0


def user_profile_paths(household_id: str, user_id: int) -> list[Path]:
    """Return all WAV sample paths for a user, in deterministic order.

    Supports two on-disk layouts:

    - Legacy single-file: ``voice_profiles/{household}/{hash}.wav``
    - Multi-sample directory: ``voice_profiles/{household}/{hash}/sample_NNN.wav``

    Directory layout takes precedence if both exist (it's the canonical form
    after the first multi-sample enrollment). Returns an empty list if no
    samples are found.
    """
    household_dir = PROFILE_DIR / household_id
    user_dir = household_dir / hash_user_id(user_id)
    legacy_path = household_dir / (hash_user_id(user_id) + ".wav")

    if user_dir.is_dir():
        return sorted(user_dir.glob("sample_*.wav"))
    if legacy_path.exists():
        return [legacy_path]
    return []


def load_household_profiles(
    household_id: str, member_ids: list[int]
) -> dict[int, np.ndarray]:
    """Load voice profiles for household members.

    Reads one or more WAV samples per user (see ``user_profile_paths`` for
    the on-disk layout) and averages their embeddings to produce a single
    reference vector per user. Multi-sample enrollment reduces variance
    from any single bad take.

    Args:
        household_id: The household UUID.
        member_ids: List of user IDs in the household.

    Returns:
        Dictionary mapping user_id to L2-normalized voice embedding.
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

    encoder = _get_encoder()
    for user_id in member_ids:
        paths = user_profile_paths(household_id, user_id)
        if not paths:
            continue
        try:
            embeddings = [encoder.embed(p) for p in paths]
            avg = np.mean(np.stack(embeddings), axis=0)
            profiles[user_id] = _l2_normalize(avg)
            logger.debug(
                f"Loaded voice profile for user {user_id} (samples={len(paths)})"
            )
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


def _pick_threshold(duration_s: float) -> float:
    """Pick a length-adaptive similarity threshold based on clip duration.

    Reads settings: voice.threshold_short / voice.similarity_threshold /
    voice.threshold_long, gated by voice.short_cutoff_seconds and
    voice.long_cutoff_seconds. Falls back to environment defaults if the
    settings service is unavailable.
    """
    try:
        from app.services.settings_service import get_settings_service
        svc = get_settings_service()
        normal = svc.get_float("voice.similarity_threshold", 0.5)
        short = svc.get_float("voice.threshold_short", 0.65)
        long_ = svc.get_float("voice.threshold_long", 0.4)
        short_cut = svc.get_float("voice.short_cutoff_seconds", 1.0)
        long_cut = svc.get_float("voice.long_cutoff_seconds", 3.0)
    except Exception:
        normal = float(os.getenv("VOICE_SIMILARITY_THRESHOLD", "0.5"))
        short = float(os.getenv("VOICE_THRESHOLD_SHORT", "0.65"))
        long_ = float(os.getenv("VOICE_THRESHOLD_LONG", "0.4"))
        short_cut = float(os.getenv("VOICE_SHORT_CUTOFF_SECONDS", "1.0"))
        long_cut = float(os.getenv("VOICE_LONG_CUTOFF_SECONDS", "3.0"))

    if duration_s > 0 and duration_s < short_cut:
        return short
    if duration_s > long_cut:
        return long_
    return normal


def recognize_speaker(
    audio_path: str,
    household_id: str,
    member_ids: list[int],
    threshold: float | None = None,
) -> SpeakerResult:
    """Recognize speaker from audio file within a household.

    Args:
        audio_path: Path to WAV file.
        household_id: The household UUID to scope the search.
        member_ids: List of user IDs in the household.
        threshold: Optional explicit threshold override. When None (the
            default), a length-adaptive threshold is picked from settings:
            stricter for short clips, looser for long ones. See
            ``_pick_threshold`` for the curve.

    Returns:
        SpeakerResult with user_id (int or None) and confidence.
        user_id is None if no match exceeds threshold or if no profiles exist.
    """
    profiles = load_household_profiles(household_id, member_ids)

    if not profiles:
        logger.warning(f"No speaker profiles for household {household_id}")
        return SpeakerResult(user_id=None, confidence=0.0)

    audio_path_p = Path(audio_path)
    duration_s = _wav_duration_seconds(audio_path_p)
    if threshold is None:
        threshold = _pick_threshold(duration_s)

    try:
        embed = _get_encoder().embed(audio_path_p)
    except (OSError, ValueError, RuntimeError, ImportError) as e:
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
        "Speaker match: household=%s best_user=%s score=%.3f threshold=%.2f duration=%.2fs → %s",
        household_id,
        best_user_id,
        best_score,
        threshold,
        duration_s,
        "MATCHED" if matched else "no match",
    )

    if matched:
        return SpeakerResult(user_id=best_user_id, confidence=best_score)

    return SpeakerResult(user_id=None, confidence=best_score)
