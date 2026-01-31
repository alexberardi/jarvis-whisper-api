"""Tests for custom exceptions module."""

import pytest

from app.exceptions import (
    AudioProcessingError,
    SpeakerRecognitionError,
    WhisperError,
    WhisperTranscriptionError,
)


class TestWhisperExceptions:
    """Test Whisper-related exceptions."""

    def test_whisper_error_is_exception(self) -> None:
        """WhisperError should be a base Exception."""
        error = WhisperError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_whisper_transcription_error_inherits_from_whisper_error(self) -> None:
        """WhisperTranscriptionError should inherit from WhisperError."""
        error = WhisperTranscriptionError("transcription failed")
        assert isinstance(error, WhisperError)
        assert isinstance(error, Exception)
        assert str(error) == "transcription failed"

    def test_whisper_transcription_error_with_details(self) -> None:
        """WhisperTranscriptionError should support additional context."""
        error = WhisperTranscriptionError(
            "transcription failed", stderr="model not found"
        )
        assert error.stderr == "model not found"


class TestAudioProcessingError:
    """Test AudioProcessingError exception."""

    def test_audio_processing_error_is_exception(self) -> None:
        """AudioProcessingError should be a base Exception."""
        error = AudioProcessingError("normalization failed")
        assert isinstance(error, Exception)
        assert str(error) == "normalization failed"

    def test_audio_processing_error_with_operation(self) -> None:
        """AudioProcessingError should support operation context."""
        error = AudioProcessingError("failed", operation="normalize")
        assert error.operation == "normalize"


class TestSpeakerRecognitionError:
    """Test SpeakerRecognitionError exception."""

    def test_speaker_recognition_error_is_exception(self) -> None:
        """SpeakerRecognitionError should be a base Exception."""
        error = SpeakerRecognitionError("embedding failed")
        assert isinstance(error, Exception)
        assert str(error) == "embedding failed"
