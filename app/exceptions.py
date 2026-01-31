"""Custom exceptions for jarvis-whisper-api."""


class WhisperError(Exception):
    """Base exception for whisper transcription errors."""


class WhisperTranscriptionError(WhisperError):
    """Transcription failed (non-zero exit code)."""

    def __init__(self, message: str, stderr: str | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr


class AudioProcessingError(Exception):
    """Audio preprocessing failed (normalization, noise reduction, etc.)."""

    def __init__(self, message: str, operation: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation


class SpeakerRecognitionError(Exception):
    """Speaker recognition failed."""
