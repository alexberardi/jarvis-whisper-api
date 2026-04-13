"""Tests for voice profile enrollment endpoints."""
import struct

import pytest
from fastapi.testclient import TestClient
from jarvis_auth_client.models import AppAuthResult, AppValidationResult, RequestContext

from app.utils import hash_user_id


@pytest.fixture(autouse=True)
def temp_profile_dir(tmp_path, monkeypatch):
    """Use a temporary directory for voice profiles."""
    monkeypatch.setattr("app.utils.PROFILE_DIR", tmp_path)
    monkeypatch.setattr("app.api.voice_profiles.PROFILE_DIR", tmp_path)
    yield tmp_path


@pytest.fixture()
def client():
    """Create test client with auth dependency overridden."""
    from app.deps import verify_app_auth
    from app.main import app

    mock_result = AppAuthResult(
        app=AppValidationResult(valid=True, app_id="command-center", name="Command Center"),
        context=RequestContext(
            household_id="test-household",
            node_id="kitchen-pi",
        ),
    )

    async def _mock_auth():
        return mock_result

    app.dependency_overrides[verify_app_auth] = _mock_auth
    yield TestClient(app)
    app.dependency_overrides.clear()


def _wav_bytes() -> bytes:
    """Return minimal WAV header for testing."""
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36, b'WAVE',
        b'fmt ', 16, 1, 1, 16000, 32000, 2, 16,
        b'data', 0,
    )


class TestEnrollVoiceProfile:
    def test_enroll_creates_file(self, client, temp_profile_dir):
        response = client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "enrolled"
        assert body["user_id"] == 42

        # Verify file exists
        expected_path = temp_profile_dir / "h1" / (hash_user_id(42) + ".wav")
        assert expected_path.exists()

    def test_enroll_overwrites_existing(self, client, temp_profile_dir):
        # Enroll twice
        for _ in range(2):
            response = client.post(
                "/voice-profiles/enroll",
                params={"user_id": 42, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )
            assert response.status_code == 200

        # Should only have one file
        h_dir = temp_profile_dir / "h1"
        wav_files = list(h_dir.glob("*.wav"))
        assert len(wav_files) == 1


class TestDeleteVoiceProfile:
    def test_delete_removes_file(self, client, temp_profile_dir):
        # First enroll
        client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )

        # Then delete
        response = client.delete(
            "/voice-profiles/42",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # File should be gone
        expected_path = temp_profile_dir / "h1" / (hash_user_id(42) + ".wav")
        assert not expected_path.exists()

    def test_delete_not_found(self, client):
        response = client.delete(
            "/voice-profiles/999",
            params={"household_id": "h1"},
        )
        assert response.status_code == 404


class TestListVoiceProfiles:
    def test_list_empty(self, client):
        response = client.get(
            "/voice-profiles",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        assert response.json()["profiles"] == []

    def test_list_with_profiles(self, client, temp_profile_dir):
        # Enroll two profiles
        for uid in [1, 2]:
            client.post(
                "/voice-profiles/enroll",
                params={"user_id": uid, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )

        response = client.get(
            "/voice-profiles",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        profiles = response.json()["profiles"]
        assert len(profiles) == 2
