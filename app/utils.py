import hashlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from app.exceptions import WhisperTranscriptionError

logger = logging.getLogger(__name__)


@dataclass
class SpeakerResult:
    """Result of speaker recognition."""

    user_id: int | None
    confidence: float

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin"),
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_WHISPER_CLI = _REPO_ROOT / "bin" / "whisper-cli"
WHISPER_CLI = os.getenv("WHISPER_CLI")


def _resolve_whisper_cli() -> str:
    if WHISPER_CLI:
        return WHISPER_CLI

    system_cli = shutil.which("whisper-cli")
    if system_cli:
        return system_cli

    if _LOCAL_WHISPER_CLI.exists():
        return str(_LOCAL_WHISPER_CLI)

    return "whisper-cli"


def _build_subprocess_env(cli_path: str) -> dict[str, str]:
    env = os.environ.copy()
    if cli_path == str(_LOCAL_WHISPER_CLI):
        bin_dir = str(_LOCAL_WHISPER_CLI.parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
    return env


def run_whisper(
    wav_path: str,
    prompt: str | None = None,
    temperature: float = 0.0,
    temperature_inc: float = 0.2,
    beam_size: int = 5,
) -> str:
    """Run whisper-cli to transcribe audio.

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
    cli_path = _resolve_whisper_cli()
    env = _build_subprocess_env(cli_path)

    args = [
        cli_path,
        "-m",
        WHISPER_MODEL,
        "-f",
        wav_path,
        "--language",
        "en",
        "--output-txt",
        "--temperature",
        str(temperature),
        "--temperature-inc",
        str(temperature_inc),
        "--beam-size",
        str(beam_size),
    ]

    if prompt:
        args.extend(["--prompt", prompt])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise WhisperTranscriptionError(
            f"Whisper transcription failed with exit code {result.returncode}",
            stderr=result.stderr,
        )

    # Read the output .txt file
    txt_path = f"{wav_path}.txt"
    with open(txt_path) as f:
        return f.read().strip()


PROFILE_DIR = Path("voice_profiles")
encoder = VoiceEncoder()

# Cache loaded embeddings keyed by household_id, then user_id
_household_profiles_cache: dict[str, dict[int, np.ndarray]] = {}


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
                profiles[user_id] = encoder.embed_utterance(wav)
                logger.debug(f"Loaded voice profile for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to load profile for user {user_id}: {e}")

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
        embed = encoder.embed_utterance(wav)
    except Exception as e:
        logger.error(f"Failed to process input audio: {e}")
        return SpeakerResult(user_id=None, confidence=0.0)

    # Compare to each profile
    scores: dict[int, float] = {
        user_id: float(np.inner(embed, ref_embed))
        for user_id, ref_embed in profiles.items()
    }

    # Choose best match above threshold
    best_user_id = max(scores, key=lambda k: scores[k])
    best_score = scores[best_user_id]

    if best_score > threshold:
        return SpeakerResult(user_id=best_user_id, confidence=best_score)

    return SpeakerResult(user_id=None, confidence=best_score)
