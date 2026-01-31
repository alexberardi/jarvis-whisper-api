"""Audio preprocessing functions for jarvis-whisper-api."""

import numpy as np
from scipy.io import wavfile

from app.exceptions import AudioProcessingError


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load WAV file as numpy array.

    Args:
        path: Path to WAV file.

    Returns:
        Tuple of (samples as float32 in [-1, 1], sample_rate).

    Raises:
        AudioProcessingError: If file cannot be loaded.
    """
    try:
        sample_rate, samples = wavfile.read(path)
    except FileNotFoundError as e:
        raise AudioProcessingError(f"File not found: {path}", operation="load") from e
    except ValueError as e:
        raise AudioProcessingError(f"Invalid WAV format: {e}", operation="load") from e
    except IOError as e:
        raise AudioProcessingError(f"Failed to read audio file: {e}", operation="load") from e

    # Convert to float32 in [-1, 1] range
    if samples.dtype == np.int16:
        audio = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        audio = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.float32:
        audio = samples
    elif samples.dtype == np.float64:
        audio = samples.astype(np.float32)
    else:
        audio = samples.astype(np.float32)

    return audio, sample_rate


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Save numpy array as WAV file.

    Args:
        path: Output file path.
        audio: Audio samples as float32 in [-1, 1] range.
        sr: Sample rate.

    Raises:
        AudioProcessingError: If file cannot be saved.
    """
    try:
        # Convert float32 to int16
        samples_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(path, sr, samples_int16)
    except (IOError, OSError) as e:
        raise AudioProcessingError(f"Failed to save audio: {e}", operation="save") from e


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS normalization to target dB level.

    Args:
        audio: Audio samples as float32.
        target_db: Target RMS level in dB (default -20.0).

    Returns:
        Normalized audio samples as float32, clipped to [-1, 1].
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))

    # Avoid division by zero for silent audio
    if rms < 1e-10:
        return audio.copy()

    # Convert target dB to linear scale
    target_linear = 10 ** (target_db / 20)

    # Calculate gain
    gain = target_linear / rms

    # Apply gain and clip to valid range
    normalized = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    return normalized


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_ms: int = 100,
) -> np.ndarray:
    """Remove leading/trailing silence below threshold.

    Args:
        audio: Audio samples as float32.
        sr: Sample rate.
        threshold_db: Silence threshold in dB (default -40.0).
        min_silence_ms: Minimum silence duration in ms to consider (default 100).

    Returns:
        Trimmed audio samples.
    """
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate frame size for silence detection
    frame_size = int(sr * min_silence_ms / 1000)
    if frame_size < 1:
        frame_size = 1

    # Find first non-silent sample
    start_idx = 0
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i : i + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > threshold_linear:
            start_idx = max(0, i - frame_size)  # Keep a small buffer
            break
    else:
        # All silent - return minimal audio
        return audio[:max(1, frame_size)]

    # Find last non-silent sample
    end_idx = len(audio)
    for i in range(len(audio) - frame_size, start_idx, -frame_size):
        frame = audio[i : i + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > threshold_linear:
            end_idx = min(len(audio), i + 2 * frame_size)  # Keep a small buffer
            break

    return audio[start_idx:end_idx]


def preprocess_audio(
    input_path: str,
    output_path: str,
    normalize: bool = True,
    trim: bool = True,
    target_db: float = -20.0,
    silence_threshold_db: float = -40.0,
) -> None:
    """Full preprocessing pipeline: load → normalize → trim → save.

    Args:
        input_path: Path to input WAV file.
        output_path: Path to output WAV file.
        normalize: Whether to apply RMS normalization (default True).
        trim: Whether to trim leading/trailing silence (default True).
        target_db: Target RMS level in dB for normalization (default -20.0).
        silence_threshold_db: Silence threshold in dB for trimming (default -40.0).

    Raises:
        AudioProcessingError: If preprocessing fails.
    """
    audio, sr = load_audio(input_path)

    if normalize:
        audio = normalize_audio(audio, target_db=target_db)

    if trim:
        audio = trim_silence(audio, sr, threshold_db=silence_threshold_db)

    save_audio(output_path, audio, sr)
