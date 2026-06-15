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

from app.audio import load_audio, normalize_audio, trim_silence
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
) -> tuple[str, list[dict]]:
    """Transcribe audio using the in-process whisper.cpp model.

    Args:
        wav_path: Path to WAV file to transcribe.
        prompt: Optional initial prompt to guide transcription.
        temperature: Initial temperature for sampling (default 0.0).
        temperature_inc: Temperature increment on decode failure (default 0.2).
        beam_size: Beam size for beam search (default 5).

    Returns:
        ``(text, segments)`` where ``text`` is the joined transcript and
        ``segments`` is a list of ``{"t0_ms", "t1_ms", "text"}`` dicts.
        whisper.cpp reports ``t0``/``t1`` in centiseconds (1/100 s); we
        convert to milliseconds here so the API speaks one unit.
        Callers use the gap between adjacent ``[t0, t1]`` ranges as a
        narration-vs-command shape signal (no inter-segment pauses + lots
        of words = ambient speech).

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

    seg_dicts = [
        {"t0_ms": int(seg.t0) * 10, "t1_ms": int(seg.t1) * 10, "text": seg.text}
        for seg in segments
    ]
    text = " ".join(seg.text for seg in segments).strip()
    return text, seg_dicts


PROFILE_DIR = Path("voice_profiles")
_encoder: SpeakerEncoder | None = None
_encoder_fingerprint: str | None = None

# Cache speaker embeddings keyed on (household_id, user_id) — NOT a whole-
# household result dict. Caching per member is what makes an empty or partial
# member scope unable to poison a later, fuller scope: each member is resolved
# independently and load_household_profiles assembles the result per request.
# A value of None is a negative cache ("no profile on disk for this user"),
# scoped to the exact user requested, so it can never block a different member.
_member_embedding_cache: dict[tuple[str, int], np.ndarray | None] = {}

# Fingerprint of the embed-preprocessing settings the cached centroids were
# built with. When the settings change at runtime, the cached embeddings are
# stale (they were pooled under the old loudness/trim regime) and must be
# re-embedded — see load_household_profiles.
_cache_prep_fingerprint: str | None = None


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


# 0.1s @ 16 kHz — never embed a sliver if silence-trim is over-aggressive.
_MIN_PREP_SAMPLES = 1600


def _embed_prep_settings() -> tuple[bool, float, float]:
    """Read the symmetric embed-preprocessing settings.

    Returns ``(enabled, target_rms_db, trim_silence_db)``. Defensive: any
    failure reading settings (e.g. settings DB momentarily unavailable, or
    no DB at all under unit tests) falls back to the documented defaults
    rather than breaking the embedding path.
    """
    try:
        from app.services.settings_service import get_settings_service
        svc = get_settings_service()
        return (
            svc.get_bool("voice.embed_preprocess_enabled", False),
            svc.get_float("voice.embed_target_rms_db", -23.0),
            svc.get_float("voice.embed_trim_silence_db", -40.0),
        )
    except Exception:  # noqa: BLE001 — settings unavailability must not break embedding
        return (False, -23.0, -40.0)


def _embed_prep_fingerprint() -> str:
    """Fingerprint of the prep settings, for profile-cache invalidation."""
    enabled, target_rms_db, trim_db = _embed_prep_settings()
    return f"{enabled}:{target_rms_db}:{trim_db}"


def _prep_for_embed(path: Path) -> np.ndarray:
    """Load + symmetrically normalize a clip for speaker embedding.

    Loads the WAV as 16 kHz mono float32, then — when
    ``voice.embed_preprocess_enabled`` — removes DC, trims leading/trailing
    silence, and RMS-normalizes to a fixed target. Both the enrollment
    samples (``load_household_profiles``) and the runtime query
    (``recognize_speaker``) pass through here, so a centroid built from raw
    far-field enrollment audio (~-27 dBFS in the field) and a live command
    that the node gain-boosted toward ~-18 dBFS land in the SAME loudness
    domain before ECAPA sees them. ECAPA cosine is sensitive to level and
    spectral tilt, so removing that enroll↔runtime asymmetry is what lifts
    the genuine-speaker score back above threshold.
    """
    audio = _load_wav_mono_16k(path)
    enabled, target_rms_db, trim_db = _embed_prep_settings()
    if not enabled:
        return audio

    # DC removal — a constant offset shifts the waveform and perturbs ECAPA.
    audio = audio - float(np.mean(audio))

    # Symmetric silence trim so enrollment (a long, pause-heavy read) and a
    # short command pool over comparable speech fractions. Fall back to the
    # untrimmed signal if the clip is (near) all-silence so we never embed a
    # sliver.
    trimmed = trim_silence(audio, 16000, threshold_db=trim_db)
    if len(trimmed) >= _MIN_PREP_SAMPLES:
        audio = trimmed

    return normalize_audio(audio, target_db=target_rms_db)


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
        signal = _prep_for_embed(path)
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
    from app.services.settings_service import get_settings_service
    return get_settings_service().get_str("voice.encoder", "ecapa") or "ecapa"


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


def _load_member_embedding(household_id: str, user_id: int) -> np.ndarray | None:
    """Load + average one member's enrolled samples into a single embedding.

    Reads all of the user's WAV samples (see ``user_profile_paths`` for the
    on-disk layout) and averages their embeddings into one L2-normalized
    reference vector. Multi-sample enrollment reduces variance from any single
    bad take. Returns None when the user has no profile on disk or embedding
    fails — callers treat that as "not enrolled", never as an error.
    """
    paths = user_profile_paths(household_id, user_id)
    if not paths:
        return None
    encoder = _get_encoder()
    try:
        embeddings = [encoder.embed(p) for p in paths]
        avg = np.mean(np.stack(embeddings), axis=0)
        logger.debug(
            "Loaded voice profile for user %s (samples=%d)", user_id, len(paths)
        )
        return _l2_normalize(avg)
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(
            "Failed to load profile for user %s: %s: %s",
            user_id,
            type(e).__name__,
            e,
        )
        return None


def load_household_profiles(
    household_id: str, member_ids: list[int]
) -> dict[int, np.ndarray]:
    """Load voice profiles for the requested household members.

    Embeddings are cached per ``(household_id, user_id)`` and the returned dict
    is assembled fresh on every call from exactly ``member_ids``. This is
    deliberate and load-bearing: a request carrying an empty or partial member
    scope (e.g. mobile push-to-talk, which knows the speaker from the JWT and
    never resolves the household roster) can no longer poison a later, fuller
    scope. The earlier design cached one dict per household keyed on
    household_id alone, so the first empty-scope request after a restart cached
    ``{}`` and every subsequent node command got that empty result back without
    ever running recognition.

    Args:
        household_id: The household UUID.
        member_ids: User IDs in the household to score against.

    Returns:
        Mapping of user_id to L2-normalized voice embedding, for the requested
        members that have a profile on disk.
    """
    # If the embed-preprocessing settings changed since the cache was built,
    # the centroids were pooled under the old loudness/trim regime — drop the
    # whole cache so every member re-embeds with the current settings.
    global _cache_prep_fingerprint
    fingerprint = _embed_prep_fingerprint()
    if fingerprint != _cache_prep_fingerprint:
        invalidate_household_cache()
        _cache_prep_fingerprint = fingerprint

    profiles: dict[int, np.ndarray] = {}
    for user_id in member_ids:
        key = (household_id, user_id)
        if key not in _member_embedding_cache:
            _member_embedding_cache[key] = _load_member_embedding(household_id, user_id)
        embedding = _member_embedding_cache[key]
        if embedding is not None:
            profiles[user_id] = embedding
    return profiles


def invalidate_household_cache(household_id: str | None = None) -> None:
    """Invalidate cached member embeddings.

    Args:
        household_id: Specific household to invalidate (drops every cached
            member embedding under it), or None to clear the entire cache.
    """
    global _member_embedding_cache
    if household_id is None:
        _member_embedding_cache = {}
        return
    for key in [k for k in _member_embedding_cache if k[0] == household_id]:
        del _member_embedding_cache[key]


def _pick_threshold(duration_s: float) -> float:
    """Pick a length-adaptive similarity threshold based on clip duration.

    Reads settings: voice.threshold_short / voice.similarity_threshold /
    voice.threshold_long, gated by voice.short_cutoff_seconds and
    voice.long_cutoff_seconds.
    """
    from app.services.settings_service import get_settings_service
    svc = get_settings_service()
    normal = svc.get_float("voice.similarity_threshold", 0.5)
    short = svc.get_float("voice.threshold_short", 0.65)
    long_ = svc.get_float("voice.threshold_long", 0.4)
    short_cut = svc.get_float("voice.short_cutoff_seconds", 1.0)
    long_cut = svc.get_float("voice.long_cutoff_seconds", 3.0)

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
        # Distinguish an empty member scope (R6: CC sent no/empty member_ids,
        # or no one is enrolled) from a genuine below-threshold no-match — they
        # look identical downstream but have different fixes.
        logger.warning(
            "speaker_profiles_empty household=%s member_ids=%s — no profiles loaded "
            "(check enrollment + household member scope)",
            household_id,
            member_ids,
        )
        return SpeakerResult(user_id=None, confidence=0.0)

    audio_path_p = Path(audio_path)
    duration_s = _wav_duration_seconds(audio_path_p)
    if threshold is None:
        threshold = _pick_threshold(duration_s)

    try:
        encoder = _get_encoder()
        embed = encoder.embed(audio_path_p)
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

    # Telemetry: the runner-up and the full per-member score vector let us
    # measure genuine-vs-impostor separation and false-accept headroom on real
    # traffic — today only the winner is logged, so thresholds can't be tuned
    # from data. Within-household margin is the precision signal that gates any
    # threshold-loosening change (a wrong match mis-scopes per-user secrets).
    ranked = sorted(scores.values(), reverse=True)
    second_best = ranked[1] if len(ranked) > 1 else 0.0
    margin = best_score - second_best
    scores_str = ",".join(
        f"{uid}:{scores[uid]:.3f}"
        for uid in sorted(scores, key=lambda k: scores[k], reverse=True)
    )

    matched = best_score > threshold
    logger.info(
        "Speaker match: household=%s best_user=%s score=%.3f second=%.3f margin=%.3f "
        "threshold=%.2f duration=%.2fs encoder=%s members=%d scores=[%s] → %s",
        household_id,
        best_user_id,
        best_score,
        second_best,
        margin,
        threshold,
        duration_s,
        encoder.name,
        len(profiles),
        scores_str,
        "MATCHED" if matched else "no match",
    )

    if matched:
        return SpeakerResult(user_id=best_user_id, confidence=best_score)

    return SpeakerResult(user_id=None, confidence=best_score)


def _cached_profile_counts() -> dict[str, int]:
    """Per-household count of members with a loaded (non-None) profile.

    Preserves the historical ``{household_id: count}`` shape of /health now
    that the cache is keyed on ``(household_id, user_id)``: a household whose
    requested members are all negative-cached reports 0, same as before.
    """
    counts: dict[str, int] = {}
    for (household_id, _user_id), embedding in _member_embedding_cache.items():
        counts.setdefault(household_id, 0)
        if embedding is not None:
            counts[household_id] += 1
    return counts


def speaker_recognition_status() -> dict[str, object]:
    """Observable speaker-recognition state for ``/health``.

    Surfaces whether recognition is enabled, the selected encoder, the
    embed-preprocess flag, and the per-household profile counts currently
    cached — so an operator (or jarvis-mcp ``debug_health``) can tell a
    disabled flag or an empty member scope apart from a genuine no-match at a
    glance, without grepping logs.
    """
    from app.services.settings_service import get_settings_service
    svc = get_settings_service()
    return {
        "recognition_enabled": svc.get_bool("voice.recognition_enabled", False),
        "encoder": _desired_encoder_name(),
        "embed_preprocess_enabled": svc.get_bool("voice.embed_preprocess_enabled", False),
        "cached_profile_counts": _cached_profile_counts(),
    }
