"""Tests for utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.exceptions import WhisperTranscriptionError
from app.utils import (
    SpeakerResult,
    _load_for_whisper,
    hash_user_id,
    invalidate_household_cache,
    load_household_profiles,
    recognize_speaker,
    run_whisper,
)


def _make_segment(text: str) -> MagicMock:
    """Build a fake pywhispercpp Segment with a .text attribute."""
    seg = MagicMock()
    seg.text = text
    return seg


class TestSpeakerResult:
    """Test SpeakerResult dataclass."""

    def test_speaker_result_creation(self) -> None:
        """SpeakerResult should store user_id and confidence."""
        result = SpeakerResult(user_id=42, confidence=0.92)
        assert result.user_id == 42
        assert result.confidence == 0.92

    def test_speaker_result_unknown(self) -> None:
        """SpeakerResult should support unknown speaker with None user_id."""
        result = SpeakerResult(user_id=None, confidence=0.0)
        assert result.user_id is None
        assert result.confidence == 0.0

    def test_speaker_result_equality(self) -> None:
        """SpeakerResult should support equality comparison."""
        result1 = SpeakerResult(user_id=123, confidence=0.85)
        result2 = SpeakerResult(user_id=123, confidence=0.85)
        assert result1 == result2


class TestHashUserId:
    """Test hash_user_id function."""

    def test_hash_user_id_deterministic(self) -> None:
        """hash_user_id should return same hash for same input."""
        hash1 = hash_user_id(42)
        hash2 = hash_user_id(42)
        assert hash1 == hash2

    def test_hash_user_id_different_ids_different_hashes(self) -> None:
        """hash_user_id should return different hashes for different inputs."""
        hash1 = hash_user_id(1)
        hash2 = hash_user_id(2)
        assert hash1 != hash2

    def test_hash_user_id_length(self) -> None:
        """hash_user_id should return 16-character string."""
        result = hash_user_id(12345)
        assert len(result) == 16

    def test_hash_user_id_is_hex(self) -> None:
        """hash_user_id should return valid hex string."""
        result = hash_user_id(999)
        # Should not raise ValueError
        int(result, 16)


class TestLoadHouseholdProfiles:
    """Test load_household_profiles function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    @patch("app.utils.PROFILE_DIR", Path("/nonexistent"))
    def test_load_household_profiles_missing_directory(self) -> None:
        """load_household_profiles should return empty dict for missing directory."""
        result = load_household_profiles("missing-household", [1, 2, 3])
        assert result == {}

    def test_load_household_profiles_caches_results(self) -> None:
        """load_household_profiles should cache results."""
        # Use a nonexistent household directory
        household_id = "cache-test-household"

        # First call - will check filesystem
        result1 = load_household_profiles(household_id, [1])
        # Second call - should use cache without filesystem access
        result2 = load_household_profiles(household_id, [1])

        # Both should return empty dict (no profiles exist)
        assert result1 == {}
        assert result2 == {}
        # Verify it's the same cached object
        assert result1 is result2

    def test_load_household_profiles_empty_member_list(self) -> None:
        """load_household_profiles should handle empty member list."""
        result = load_household_profiles("empty-member-household", [])
        assert result == {}


class TestInvalidateHouseholdCache:
    """Test invalidate_household_cache function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    def test_invalidate_household_cache_specific(self) -> None:
        """invalidate_household_cache should clear specific household."""
        # Add to cache manually
        from app import utils
        utils._household_profiles_cache["household-1"] = {1: np.array([1.0])}
        utils._household_profiles_cache["household-2"] = {2: np.array([2.0])}

        invalidate_household_cache("household-1")

        assert "household-1" not in utils._household_profiles_cache
        assert "household-2" in utils._household_profiles_cache

    def test_invalidate_household_cache_all(self) -> None:
        """invalidate_household_cache should clear all when no ID specified."""
        from app import utils
        utils._household_profiles_cache["household-1"] = {1: np.array([1.0])}
        utils._household_profiles_cache["household-2"] = {2: np.array([2.0])}

        invalidate_household_cache()

        assert utils._household_profiles_cache == {}

    def test_invalidate_household_cache_nonexistent(self) -> None:
        """invalidate_household_cache should not error for nonexistent household."""
        # Should not raise
        invalidate_household_cache("nonexistent-household")


class TestLoadForWhisper:
    """Test the resample/mono helper that feeds pywhispercpp."""

    def _write_wav(self, path: Path, sr: int, channels: int = 1, seconds: float = 0.5) -> None:
        from scipy.io import wavfile
        n = int(sr * seconds)
        # Simple sine so resampling has signal to chew on
        t = np.linspace(0.0, seconds, n, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        if channels == 2:
            sig = np.stack([sig, sig], axis=1)
        wavfile.write(str(path), sr, (sig * 32767).astype(np.int16))

    def test_resamples_48khz_to_16khz(self, tmp_path: Path) -> None:
        wav = tmp_path / "48k.wav"
        self._write_wav(wav, sr=48000, seconds=1.0)

        out = _load_for_whisper(str(wav))

        assert out.dtype == np.float32
        assert out.ndim == 1
        assert abs(len(out) - 16000) < 5  # 1 second @ 16 kHz, ±a sample of slop

    def test_passthrough_at_16khz(self, tmp_path: Path) -> None:
        wav = tmp_path / "16k.wav"
        self._write_wav(wav, sr=16000, seconds=0.5)

        out = _load_for_whisper(str(wav))

        assert out.dtype == np.float32
        assert out.ndim == 1
        assert abs(len(out) - 8000) < 5  # 0.5 second @ 16 kHz

    def test_mono_mixdown(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        self._write_wav(wav, sr=16000, channels=2, seconds=0.5)

        out = _load_for_whisper(str(wav))

        assert out.ndim == 1


class TestRunWhisper:
    """Test run_whisper function."""

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_success(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should join segment texts and return the trimmed result."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello "), _make_segment("world")]
        mock_get_model.return_value = model

        result = run_whisper("/tmp/test.wav")

        assert result == "Hello  world"
        model.transcribe.assert_called_once()

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_failure_raises_transcription_error(
        self, mock_get_model: MagicMock, mock_load: MagicMock
    ) -> None:
        """run_whisper should wrap underlying errors in WhisperTranscriptionError."""
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("Model file not found")
        mock_get_model.return_value = model

        with pytest.raises(WhisperTranscriptionError) as exc_info:
            run_whisper("/tmp/test.wav")

        assert "Model file not found" in str(exc_info.value)
        assert "Model file not found" in (exc_info.value.stderr or "")

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_prompt(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass prompt as initial_prompt."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Jarvis turn on lights")]
        mock_get_model.return_value = model

        result = run_whisper("/tmp/test.wav", prompt="Jarvis commands")

        assert result == "Jarvis turn on lights"
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == "Jarvis commands"

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_without_prompt(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass empty initial_prompt when prompt is None."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav")

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == ""

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_temperature(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass temperature to the model."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", temperature=0.3)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature"] == 0.3

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_temperature_inc(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass temperature_inc to the model."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", temperature_inc=0.1)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature_inc"] == 0.1

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_beam_size(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass beam_size inside the beam_search dict."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", beam_size=3)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["beam_search"] == {"beam_size": 3, "patience": -1.0}

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_default_params(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should use sane defaults for temperature/beam params."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav")

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["temperature_inc"] == 0.2
        assert kwargs["beam_search"] == {"beam_size": 5, "patience": -1.0}
        assert kwargs["language"] == "en"

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_all_params_together(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should forward all params together."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        result = run_whisper(
            "/tmp/test.wav",
            prompt="Test prompt",
            temperature=0.5,
            temperature_inc=0.15,
            beam_size=8,
        )

        assert result == "Hello world"
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == "Test prompt"
        assert kwargs["temperature"] == 0.5
        assert kwargs["temperature_inc"] == 0.15
        assert kwargs["beam_search"] == {"beam_size": 8, "patience": -1.0}


class TestRecognizeSpeaker:
    """Test recognize_speaker function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    @patch("app.utils.load_household_profiles")
    def test_recognize_speaker_no_profiles(
        self, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id when no profiles exist."""
        mock_load.return_value = {}

        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 2])

        assert result.user_id is None
        assert result.confidence == 0.0

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_matches_correct_user(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return matched user_id above threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very similar to user 42's profile
        mock_encoder.return_value.embed_utterance.return_value = np.array([0.95, 0.0, 0.0])

        result = recognize_speaker("/tmp/test.wav", "household-1", [42])

        assert result.user_id == 42
        assert result.confidence > 0.75

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_below_threshold_returns_none(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id when below threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very different from user's profile
        mock_encoder.return_value.embed_utterance.return_value = np.array([0.0, 1.0, 0.0])

        result = recognize_speaker("/tmp/test.wav", "household-1", [42])

        assert result.user_id is None
        # Confidence should still be reported (the best score found)
        assert result.confidence < 0.75

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    def test_recognize_speaker_processing_error(
        self, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id on processing error."""
        mock_load.return_value = {1: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.side_effect = RuntimeError("Audio file corrupted")

        result = recognize_speaker("/tmp/test.wav", "household-1", [1])

        assert result.user_id is None
        assert result.confidence == 0.0

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_custom_threshold(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should respect custom threshold."""
        mock_load.return_value = {99: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding with 0.8 similarity
        mock_encoder.return_value.embed_utterance.return_value = np.array([0.8, 0.6, 0.0])

        # With default threshold (0.75), should match
        result_default = recognize_speaker("/tmp/test.wav", "household-1", [99])
        assert result_default.user_id == 99

        # With higher threshold (0.9), should not match
        result_strict = recognize_speaker(
            "/tmp/test.wav", "household-1", [99], threshold=0.9
        )
        assert result_strict.user_id is None

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_multiple_users_best_match(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return best matching user among multiple."""
        mock_load.return_value = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
        }
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding most similar to user 2's profile
        mock_encoder.return_value.embed_utterance.return_value = np.array([0.1, 0.95, 0.1])

        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 2, 3])

        assert result.user_id == 2
        assert result.confidence > 0.75
