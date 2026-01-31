import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from app.exceptions import WhisperTranscriptionError


@dataclass
class SpeakerResult:
    """Result of speaker recognition."""

    name: str
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


def run_whisper(wav_path: str, prompt: str | None = None) -> str:
    """Run whisper-cli to transcribe audio.

    Args:
        wav_path: Path to WAV file to transcribe.
        prompt: Optional initial prompt to guide transcription.

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

# Cache loaded embeddings
_loaded_profiles: dict[str, np.ndarray] = {}


def load_speaker_profiles() -> None:
    global _loaded_profiles
    _loaded_profiles = {}

    for file in PROFILE_DIR.glob("*.wav"):
        name = file.stem
        wav = preprocess_wav(file)
        embed = encoder.embed_utterance(wav)
        _loaded_profiles[name] = embed

def recognize_speaker(audio_path: str, threshold: float = 0.75) -> SpeakerResult:
    """Recognize speaker from audio file.

    Args:
        audio_path: Path to WAV file.
        threshold: Minimum confidence score for a match (default 0.75).

    Returns:
        SpeakerResult with name and confidence. Name is "unknown" if no match
        exceeds threshold or if no profiles are loaded.
    """
    if not _loaded_profiles:
        load_speaker_profiles()

    if not _loaded_profiles:
        print("[recognizer] No speaker profiles loaded.")
        return SpeakerResult(name="unknown", confidence=0.0)

    try:
        wav = preprocess_wav(audio_path)
        embed = encoder.embed_utterance(wav)
    except Exception as e:
        print(f"[recognizer] Failed to process input audio: {e}")
        return SpeakerResult(name="unknown", confidence=0.0)

    # Compare to each profile
    scores: dict[str, float] = {
        name: float(np.inner(embed, ref_embed))
        for name, ref_embed in _loaded_profiles.items()
    }

    # Choose best match above threshold
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    if best_score > threshold:
        return SpeakerResult(name=best_match, confidence=best_score)

    return SpeakerResult(name="unknown", confidence=best_score)
