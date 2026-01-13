import subprocess
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin"))

def run_whisper(wav_path):
    result = subprocess.run([
        "whisper-cli",
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "--language", "en",
        "--output-txt",
        ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    # Read the output .txt file
    txt_path = f"{wav_path}.txt"
    print(wav_path)
    print(txt_path)
    with open(txt_path) as f:
        return f.read().strip()


PROFILE_DIR = Path("voice_profiles")
encoder = VoiceEncoder()

# Cache loaded embeddings
_loaded_profiles = {}


def load_speaker_profiles():
    global _loaded_profiles
    _loaded_profiles = {}

    for file in PROFILE_DIR.glob("*.wav"):
        name = file.stem
        wav = preprocess_wav(file)
        embed = encoder.embed_utterance(wav)
        _loaded_profiles[name] = embed

def recognize_speaker(audio_path: str) -> str:
    if not _loaded_profiles:
        load_speaker_profiles()

    if not _loaded_profiles:
        print("[recognizer] No speaker profiles loaded.")
        return "unknown"

    try:
        wav = preprocess_wav(audio_path)
        embed = encoder.embed_utterance(wav)
    except Exception as e:
        print(f"[recognizer] Failed to process input audio: {e}")
        return "unknown"

    # Compare to each profile
    scores = {
        name: np.inner(embed, ref_embed)
        for name, ref_embed in _loaded_profiles.items()
    }

    # Choose best match above threshold
    best_match = max(scores, key=scores.get)
    if scores[best_match] > 0.75:  # adjustable threshold
        return best_match

    return "unknown"
