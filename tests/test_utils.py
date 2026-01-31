"""Tests for utils module."""

import subprocess
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from app.exceptions import WhisperTranscriptionError
from app.utils import SpeakerResult, recognize_speaker, run_whisper


class TestSpeakerResult:
    """Test SpeakerResult dataclass."""

    def test_speaker_result_creation(self) -> None:
        """SpeakerResult should store name and confidence."""
        result = SpeakerResult(name="alice", confidence=0.92)
        assert result.name == "alice"
        assert result.confidence == 0.92

    def test_speaker_result_unknown(self) -> None:
        """SpeakerResult should support unknown speaker."""
        result = SpeakerResult(name="unknown", confidence=0.0)
        assert result.name == "unknown"
        assert result.confidence == 0.0

    def test_speaker_result_equality(self) -> None:
        """SpeakerResult should support equality comparison."""
        result1 = SpeakerResult(name="bob", confidence=0.85)
        result2 = SpeakerResult(name="bob", confidence=0.85)
        assert result1 == result2


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

    @patch("app.utils._loaded_profiles", {})
    @patch("app.utils.load_speaker_profiles")
    def test_recognize_speaker_no_profiles(
        self, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return unknown when no profiles exist."""
        result = recognize_speaker("/tmp/test.wav")

        assert result.name == "unknown"
        assert result.confidence == 0.0

    @patch("app.utils._loaded_profiles", {"alice": np.array([1.0, 0.0, 0.0])})
    @patch("app.utils.preprocess_wav")
    @patch("app.utils.encoder")
    def test_recognize_speaker_match_above_threshold(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock
    ) -> None:
        """recognize_speaker should return matched speaker above threshold."""
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very similar to alice's profile
        mock_encoder.embed_utterance.return_value = np.array([0.95, 0.0, 0.0])

        result = recognize_speaker("/tmp/test.wav")

        assert result.name == "alice"
        assert result.confidence > 0.75

    @patch("app.utils._loaded_profiles", {"alice": np.array([1.0, 0.0, 0.0])})
    @patch("app.utils.preprocess_wav")
    @patch("app.utils.encoder")
    def test_recognize_speaker_below_threshold(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock
    ) -> None:
        """recognize_speaker should return unknown when below threshold."""
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding very different from alice's profile
        mock_encoder.embed_utterance.return_value = np.array([0.0, 1.0, 0.0])

        result = recognize_speaker("/tmp/test.wav")

        assert result.name == "unknown"
        # Confidence should still be reported (the best score found)
        assert result.confidence < 0.75

    @patch("app.utils._loaded_profiles", {"alice": np.array([1.0, 0.0, 0.0])})
    @patch("app.utils.preprocess_wav")
    def test_recognize_speaker_processing_error(
        self, mock_preprocess: MagicMock
    ) -> None:
        """recognize_speaker should return unknown on processing error."""
        mock_preprocess.side_effect = Exception("Audio file corrupted")

        result = recognize_speaker("/tmp/test.wav")

        assert result.name == "unknown"
        assert result.confidence == 0.0

    @patch("app.utils._loaded_profiles", {"alice": np.array([1.0, 0.0, 0.0])})
    @patch("app.utils.preprocess_wav")
    @patch("app.utils.encoder")
    def test_recognize_speaker_custom_threshold(
        self, mock_encoder: MagicMock, mock_preprocess: MagicMock
    ) -> None:
        """recognize_speaker should respect custom threshold."""
        mock_preprocess.return_value = np.array([1.0])
        # Return embedding with 0.8 similarity
        mock_encoder.embed_utterance.return_value = np.array([0.8, 0.6, 0.0])

        # With default threshold (0.75), should match
        result_default = recognize_speaker("/tmp/test.wav")
        assert result_default.name == "alice"

        # With higher threshold (0.9), should not match
        result_strict = recognize_speaker("/tmp/test.wav", threshold=0.9)
        assert result_strict.name == "unknown"
