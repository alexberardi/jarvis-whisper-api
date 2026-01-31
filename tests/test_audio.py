"""Tests for audio preprocessing module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from app.audio import (
    load_audio,
    normalize_audio,
    preprocess_audio,
    save_audio,
    trim_silence,
)
from app.exceptions import AudioProcessingError


class TestLoadAudio:
    """Test load_audio function."""

    def test_load_audio_returns_samples_and_sample_rate(self) -> None:
        """load_audio should return (samples, sample_rate) tuple."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a simple test WAV file
            sr = 16000
            duration = 0.5
            samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
            samples_int16 = (samples * 32767).astype(np.int16)
            wavfile.write(f.name, sr, samples_int16)

            audio, sample_rate = load_audio(f.name)

            assert sample_rate == 16000
            assert len(audio) == int(sr * duration)
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32

            Path(f.name).unlink()

    def test_load_audio_normalizes_to_float32(self) -> None:
        """load_audio should convert int16 samples to float32 in [-1, 1] range."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sr = 16000
            # Create samples at full scale
            samples_int16 = np.array([32767, -32768, 0], dtype=np.int16)
            wavfile.write(f.name, sr, samples_int16)

            audio, _ = load_audio(f.name)

            assert audio.dtype == np.float32
            assert np.max(audio) <= 1.0
            assert np.min(audio) >= -1.0

            Path(f.name).unlink()

    def test_load_audio_file_not_found(self) -> None:
        """load_audio should raise AudioProcessingError for missing file."""
        with pytest.raises(AudioProcessingError) as exc_info:
            load_audio("/nonexistent/path/file.wav")

        assert exc_info.value.operation == "load"

    def test_load_audio_invalid_format(self) -> None:
        """load_audio should raise AudioProcessingError for invalid file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"not a wav file")
            f.flush()

            with pytest.raises(AudioProcessingError) as exc_info:
                load_audio(f.name)

            assert exc_info.value.operation == "load"

            Path(f.name).unlink()


class TestSaveAudio:
    """Test save_audio function."""

    def test_save_audio_creates_wav_file(self) -> None:
        """save_audio should create a valid WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sr = 16000
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32)

            save_audio(f.name, audio, sr)

            # Verify file can be read back
            sr_read, samples_read = wavfile.read(f.name)
            assert sr_read == sr
            assert len(samples_read) == len(audio)

            Path(f.name).unlink()

    def test_save_audio_converts_to_int16(self) -> None:
        """save_audio should convert float32 to int16."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sr = 16000
            audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)

            save_audio(f.name, audio, sr)

            sr_read, samples = wavfile.read(f.name)
            assert samples.dtype == np.int16
            assert samples[0] == 32767  # 1.0 -> max int16
            assert samples[1] == -32767  # -1.0 * 32767 = -32767

            Path(f.name).unlink()

    def test_save_audio_invalid_path(self) -> None:
        """save_audio should raise AudioProcessingError for invalid path."""
        audio = np.array([0.0], dtype=np.float32)

        with pytest.raises(AudioProcessingError) as exc_info:
            save_audio("/nonexistent/dir/file.wav", audio, 16000)

        assert exc_info.value.operation == "save"


class TestNormalizeAudio:
    """Test normalize_audio function."""

    def test_normalize_audio_to_target_db(self) -> None:
        """normalize_audio should adjust RMS to target dB level."""
        # Create quiet audio (low amplitude)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.01

        normalized = normalize_audio(audio, target_db=-20.0)

        # Calculate RMS in dB
        rms = np.sqrt(np.mean(normalized**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Should be close to target (within 1 dB tolerance)
        assert abs(rms_db - (-20.0)) < 1.0

    def test_normalize_audio_preserves_shape(self) -> None:
        """normalize_audio should preserve array shape."""
        audio = np.random.randn(16000).astype(np.float32)

        normalized = normalize_audio(audio)

        assert normalized.shape == audio.shape
        assert normalized.dtype == np.float32

    def test_normalize_audio_clips_to_valid_range(self) -> None:
        """normalize_audio should clip output to [-1, 1] range."""
        # Create audio that would exceed range after normalization
        audio = np.array([0.001, 0.001, 0.001], dtype=np.float32)

        # Target very loud level
        normalized = normalize_audio(audio, target_db=-3.0)

        assert np.max(np.abs(normalized)) <= 1.0

    def test_normalize_audio_silent_input(self) -> None:
        """normalize_audio should handle silent/zero audio gracefully."""
        audio = np.zeros(16000, dtype=np.float32)

        # Should not raise, should return zeros
        normalized = normalize_audio(audio)

        assert np.allclose(normalized, 0.0)

    def test_normalize_audio_custom_target(self) -> None:
        """normalize_audio should respect custom target_db."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

        normalized_loud = normalize_audio(audio, target_db=-10.0)
        normalized_quiet = normalize_audio(audio, target_db=-30.0)

        rms_loud = np.sqrt(np.mean(normalized_loud**2))
        rms_quiet = np.sqrt(np.mean(normalized_quiet**2))

        # Louder normalization should have higher RMS
        assert rms_loud > rms_quiet


class TestTrimSilence:
    """Test trim_silence function."""

    def test_trim_silence_removes_leading_silence(self) -> None:
        """trim_silence should remove leading silence."""
        sr = 16000
        silence = np.zeros(8000, dtype=np.float32)  # 0.5s silence
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
        audio = np.concatenate([silence, signal])

        trimmed = trim_silence(audio, sr)

        # Trimmed audio should be shorter (removed most of leading silence)
        assert len(trimmed) < len(audio)
        # Most of leading silence should be removed (allow some buffer)
        assert len(trimmed) < len(audio) - 6000

    def test_trim_silence_removes_trailing_silence(self) -> None:
        """trim_silence should remove trailing silence."""
        sr = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
        silence = np.zeros(8000, dtype=np.float32)
        audio = np.concatenate([signal, silence])

        trimmed = trim_silence(audio, sr)

        # Trimmed audio should be shorter (removed most of trailing silence)
        assert len(trimmed) < len(audio)
        # Most of trailing silence should be removed (allow some buffer)
        assert len(trimmed) < len(audio) - 6000

    def test_trim_silence_removes_both_ends(self) -> None:
        """trim_silence should remove silence from both ends."""
        sr = 16000
        silence = np.zeros(4000, dtype=np.float32)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
        audio = np.concatenate([silence, signal, silence])

        trimmed = trim_silence(audio, sr)

        # Should be approximately signal length
        assert len(trimmed) < len(audio)
        assert len(trimmed) >= len(signal) * 0.8  # Allow some tolerance

    def test_trim_silence_preserves_middle_silence(self) -> None:
        """trim_silence should preserve silence in the middle of audio."""
        sr = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.25, 4000)).astype(np.float32) * 0.5
        silence = np.zeros(4000, dtype=np.float32)
        audio = np.concatenate([signal, silence, signal])

        trimmed = trim_silence(audio, sr)

        # Middle silence should be preserved, length should be similar
        assert len(trimmed) >= len(signal) * 2

    def test_trim_silence_custom_threshold(self) -> None:
        """trim_silence should respect custom threshold_db."""
        sr = 16000
        # Very quiet "signal"
        quiet = np.ones(8000, dtype=np.float32) * 0.001
        loud = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
        audio = np.concatenate([quiet, loud, quiet])

        # With default threshold (-40 dB), quiet parts might be kept
        trimmed_default = trim_silence(audio, sr, threshold_db=-40.0)

        # With stricter threshold (-20 dB), quiet parts should be trimmed
        trimmed_strict = trim_silence(audio, sr, threshold_db=-20.0)

        # Stricter threshold should result in shorter audio
        assert len(trimmed_strict) <= len(trimmed_default)

    def test_trim_silence_all_silent(self) -> None:
        """trim_silence should handle all-silent audio gracefully."""
        sr = 16000
        audio = np.zeros(16000, dtype=np.float32)

        trimmed = trim_silence(audio, sr)

        # Should return minimal audio (not empty to avoid errors downstream)
        assert len(trimmed) >= 1

    def test_trim_silence_no_silence(self) -> None:
        """trim_silence should return audio unchanged if no silence to trim."""
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5

        trimmed = trim_silence(audio, sr)

        # Should be approximately same length
        assert abs(len(trimmed) - len(audio)) < sr * 0.05  # Within 50ms


class TestPreprocessAudio:
    """Test preprocess_audio function."""

    def test_preprocess_audio_creates_output_file(self) -> None:
        """preprocess_audio should create output file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                # Create input file
                sr = 16000
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32) * 0.1
                samples_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(f_in.name, sr, samples_int16)

                preprocess_audio(f_in.name, f_out.name)

                assert Path(f_out.name).exists()

                Path(f_in.name).unlink()
                Path(f_out.name).unlink()

    def test_preprocess_audio_with_normalization(self) -> None:
        """preprocess_audio should normalize audio when enabled."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                # Create quiet input file
                sr = 16000
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32) * 0.01
                samples_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(f_in.name, sr, samples_int16)

                preprocess_audio(f_in.name, f_out.name, normalize=True, trim=False)

                # Read output and check RMS is higher
                sr_out, samples_out = wavfile.read(f_out.name)
                output_audio = samples_out.astype(np.float32) / 32768.0
                rms_out = np.sqrt(np.mean(output_audio**2))

                rms_in = np.sqrt(np.mean(audio**2))
                assert rms_out > rms_in

                Path(f_in.name).unlink()
                Path(f_out.name).unlink()

    def test_preprocess_audio_with_trimming(self) -> None:
        """preprocess_audio should trim silence when enabled."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                sr = 16000
                silence = np.zeros(8000, dtype=np.float32)
                signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
                audio = np.concatenate([silence, signal, silence])
                samples_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(f_in.name, sr, samples_int16)

                preprocess_audio(f_in.name, f_out.name, normalize=False, trim=True)

                sr_out, samples_out = wavfile.read(f_out.name)
                assert len(samples_out) < len(audio)

                Path(f_in.name).unlink()
                Path(f_out.name).unlink()

    def test_preprocess_audio_disabled_processing(self) -> None:
        """preprocess_audio should pass through when both options disabled."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                sr = 16000
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32) * 0.5
                samples_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(f_in.name, sr, samples_int16)

                preprocess_audio(f_in.name, f_out.name, normalize=False, trim=False)

                sr_out, samples_out = wavfile.read(f_out.name)
                # Should be essentially unchanged
                assert len(samples_out) == len(audio)

                Path(f_in.name).unlink()
                Path(f_out.name).unlink()

    def test_preprocess_audio_invalid_input(self) -> None:
        """preprocess_audio should raise AudioProcessingError for invalid input."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            with pytest.raises(AudioProcessingError):
                preprocess_audio("/nonexistent/file.wav", f_out.name)

            Path(f_out.name).unlink(missing_ok=True)

    def test_preprocess_audio_custom_parameters(self) -> None:
        """preprocess_audio should accept custom normalization and trim parameters."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                sr = 16000
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32) * 0.1
                samples_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(f_in.name, sr, samples_int16)

                # Should not raise with custom parameters
                preprocess_audio(
                    f_in.name,
                    f_out.name,
                    normalize=True,
                    trim=True,
                    target_db=-15.0,
                    silence_threshold_db=-35.0,
                )

                assert Path(f_out.name).exists()

                Path(f_in.name).unlink()
                Path(f_out.name).unlink()
