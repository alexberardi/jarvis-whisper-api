"""Tests for utils module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from app.exceptions import WhisperTranscriptionError
from app.utils import (
    SpeakerResult,
    hash_user_id,
    invalidate_household_cache,
    load_household_profiles,
    recognize_speaker,
    run_whisper,
)


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


class TestRunWhisper:
    """Test run_whisper function."""

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_success(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should return transcribed text on success."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        result = run_whisper("/tmp/test.wav")

        assert result == "Hello world"
        mock_subprocess.assert_called_once()

    @patch("app.utils.subprocess.run")
    def test_run_whisper_failure_raises_transcription_error(
        self, mock_subprocess: MagicMock
    ) -> None:
        """run_whisper should raise WhisperTranscriptionError on failure."""
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="Model file not found"
        )

        with pytest.raises(WhisperTranscriptionError) as exc_info:
            run_whisper("/tmp/test.wav")

        assert "exit code 1" in str(exc_info.value)
        assert exc_info.value.stderr == "Model file not found"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Jarvis turn on lights"))
    def test_run_whisper_with_prompt(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should pass prompt to whisper-cli."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        result = run_whisper("/tmp/test.wav", prompt="Jarvis commands")

        assert result == "Jarvis turn on lights"

        # Verify --prompt was passed in the args
        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]  # First positional arg is the command list
        assert "--prompt" in args_list
        prompt_index = args_list.index("--prompt")
        assert args_list[prompt_index + 1] == "Jarvis commands"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello"))
    def test_run_whisper_without_prompt(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should not include --prompt when not provided."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        run_whisper("/tmp/test.wav")

        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]
        assert "--prompt" not in args_list

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_with_temperature(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should pass temperature to whisper-cli."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        run_whisper("/tmp/test.wav", temperature=0.3)

        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]
        assert "--temperature" in args_list
        temp_idx = args_list.index("--temperature")
        assert args_list[temp_idx + 1] == "0.3"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_with_temperature_inc(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should pass temperature-inc to whisper-cli."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        run_whisper("/tmp/test.wav", temperature_inc=0.1)

        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]
        assert "--temperature-inc" in args_list
        temp_inc_idx = args_list.index("--temperature-inc")
        assert args_list[temp_inc_idx + 1] == "0.1"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_with_beam_size(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should pass beam-size to whisper-cli."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        run_whisper("/tmp/test.wav", beam_size=3)

        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]
        assert "--beam-size" in args_list
        beam_idx = args_list.index("--beam-size")
        assert args_list[beam_idx + 1] == "3"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_default_temperature_params(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should use default temperature params."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        run_whisper("/tmp/test.wav")

        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]
        # Default values should be passed
        temp_idx = args_list.index("--temperature")
        assert args_list[temp_idx + 1] == "0.0"
        temp_inc_idx = args_list.index("--temperature-inc")
        assert args_list[temp_inc_idx + 1] == "0.2"
        beam_idx = args_list.index("--beam-size")
        assert args_list[beam_idx + 1] == "5"

    @patch("app.utils.subprocess.run")
    @patch("builtins.open", mock_open(read_data="Hello world"))
    def test_run_whisper_all_params_together(self, mock_subprocess: MagicMock) -> None:
        """run_whisper should handle all parameters together."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

        result = run_whisper(
            "/tmp/test.wav",
            prompt="Test prompt",
            temperature=0.5,
            temperature_inc=0.15,
            beam_size=8,
        )

        assert result == "Hello world"
        call_args = mock_subprocess.call_args
        args_list = call_args[0][0]

        # Verify all params
        assert "--prompt" in args_list
        assert "--temperature" in args_list
        assert "--temperature-inc" in args_list
        assert "--beam-size" in args_list


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
    @patch("app.utils.encoder")
    def test_recognize_speaker_matches_correct_user(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return matched user_id above threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very similar to user 42's profile
        mock_encoder.embed_utterance.return_value = np.array([0.95, 0.0, 0.0])

        result = recognize_speaker("/tmp/test.wav", "household-1", [42])

        assert result.user_id == 42
        assert result.confidence > 0.75

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils.encoder")
    def test_recognize_speaker_below_threshold_returns_none(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id when below threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very different from user's profile
        mock_encoder.embed_utterance.return_value = np.array([0.0, 1.0, 0.0])

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
        mock_preprocess.side_effect = Exception("Audio file corrupted")

        result = recognize_speaker("/tmp/test.wav", "household-1", [1])

        assert result.user_id is None
        assert result.confidence == 0.0

    @patch("app.utils.load_household_profiles")
    @patch("app.utils.preprocess_wav")
    @patch("app.utils.encoder")
    def test_recognize_speaker_custom_threshold(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should respect custom threshold."""
        mock_load.return_value = {99: np.array([1.0, 0.0, 0.0])}
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding with 0.8 similarity
        mock_encoder.embed_utterance.return_value = np.array([0.8, 0.6, 0.0])

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
    @patch("app.utils.encoder")
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
        mock_encoder.embed_utterance.return_value = np.array([0.1, 0.95, 0.1])

        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 2, 3])

        assert result.user_id == 2
        assert result.confidence > 0.75
