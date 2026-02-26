"""Tests for the main FastAPI application."""

import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from jarvis_auth_client.models import AppAuthResult, AppValidationResult, RequestContext


def _make_auth_result(**ctx_kwargs) -> AppAuthResult:
    """Helper to create an AppAuthResult for tests."""
    return AppAuthResult(
        app=AppValidationResult(valid=True, app_id="test-app"),
        context=RequestContext(
            household_id=ctx_kwargs.get("household_id"),
            node_id=ctx_kwargs.get("node_id", "test-node"),
            user_id=ctx_kwargs.get("user_id"),
            household_member_ids=ctx_kwargs.get("household_member_ids", []),
        ),
    )


@pytest.fixture()
def client():
    """Create a test client with mocked auth."""
    # Must set env var before importing main (module-level code uses it)
    with patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"}):
        from app.main import app, verify_app_auth

        app.dependency_overrides[verify_app_auth] = lambda: _make_auth_result()
        yield TestClient(app)
        app.dependency_overrides.clear()


class TestPingEndpoint:
    """Test GET /ping."""

    def test_ping_returns_pong(self, client: TestClient) -> None:
        """GET /ping should return pong."""
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}


class TestHealthEndpoint:
    """Test GET /health."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """GET /health should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestTranscribeEndpoint:
    """Test POST /transcribe."""

    @patch("app.main.run_whisper", return_value="Hello world")
    def test_transcribe_success(self, mock_whisper: MagicMock, client: TestClient) -> None:
        """POST /transcribe should return transcribed text."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world"
        assert data["speaker"]["user_id"] is None

    @patch("app.main.run_whisper", return_value="Hello world")
    def test_transcribe_with_prompt(self, mock_whisper: MagicMock, client: TestClient) -> None:
        """POST /transcribe should pass prompt to whisper."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe?prompt=Jarvis",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        mock_whisper.assert_called_once()
        call_kwargs = mock_whisper.call_args
        assert call_kwargs.kwargs.get("prompt") == "Jarvis" or call_kwargs[1].get("prompt") == "Jarvis"

    @patch("app.main.preprocess_audio")
    @patch("app.main.run_whisper", return_value="Preprocessed text")
    def test_transcribe_with_preprocessing(
        self, mock_whisper: MagicMock, mock_preprocess: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe with preprocess=true should preprocess audio."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe?preprocess=true",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "Preprocessed text"
        mock_preprocess.assert_called_once()

    @patch(
        "app.main.preprocess_audio",
        side_effect=__import__("app.exceptions", fromlist=["AudioProcessingError"]).AudioProcessingError(
            "normalization failed"
        ),
    )
    @patch("app.main.run_whisper", return_value="Original text")
    def test_transcribe_preprocessing_failure_falls_back(
        self, mock_whisper: MagicMock, mock_preprocess: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe should fall back to original on preprocessing failure."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe?preprocess=true",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "Original text"

    @patch(
        "app.main.run_whisper",
        side_effect=__import__("app.exceptions", fromlist=["WhisperTranscriptionError"]).WhisperTranscriptionError(
            "exit code 1", stderr="model not found"
        ),
    )
    def test_transcribe_whisper_error_returns_500(
        self, mock_whisper: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe should return 500 on transcription error."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["stderr"] == "model not found"

    @patch("app.main.run_whisper", side_effect=RuntimeError("unexpected error"))
    def test_transcribe_runtime_error_returns_500(
        self, mock_whisper: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe should return 500 on runtime error."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 500
        assert "error" in response.json()

    @patch("app.main.recognize_speaker")
    @patch("app.main.run_whisper", return_value="Hello")
    @patch.dict(os.environ, {"USE_VOICE_RECOGNITION": "true"})
    def test_transcribe_with_voice_recognition(
        self, mock_whisper: MagicMock, mock_recognize: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe should include speaker info when recognition enabled."""
        from app.utils import SpeakerResult

        mock_recognize.return_value = SpeakerResult(user_id=42, confidence=0.92)
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["speaker"]["user_id"] == 42
        assert data["speaker"]["confidence"] == 0.92

    @patch("app.main.run_whisper", return_value="Hello")
    def test_transcribe_with_custom_params(
        self, mock_whisper: MagicMock, client: TestClient
    ) -> None:
        """POST /transcribe should accept temperature, temperature_inc, beam_size."""
        wav_data = io.BytesIO(b"RIFF" + b"\x00" * 100)
        response = client.post(
            "/transcribe?temperature=0.5&temperature_inc=0.1&beam_size=8",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["temperature_inc"] == 0.1
        assert call_kwargs["beam_size"] == 8


class TestSetupRemoteLogging:
    """Test _setup_remote_logging function."""

    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"})
    def test_setup_remote_logging_no_app_key(self) -> None:
        """_setup_remote_logging should skip when JARVIS_APP_KEY not set."""
        from app.main import _setup_remote_logging

        with patch.dict(os.environ, {}, clear=False):
            # Remove JARVIS_APP_KEY if present
            os.environ.pop("JARVIS_APP_KEY", None)
            # Should not raise
            _setup_remote_logging()

    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"})
    def test_setup_remote_logging_import_error(self) -> None:
        """_setup_remote_logging should handle missing jarvis-log-client."""
        from app.main import _setup_remote_logging

        with patch.dict(os.environ, {"JARVIS_APP_KEY": "test-key"}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                # Should not raise (catches ImportError)
                pass
