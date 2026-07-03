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
    def test_enroll_creates_sample_in_user_dir(self, client, temp_profile_dir):
        response = client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "enrolled"
        assert body["user_id"] == 42
        assert body["sample_index"] == 0
        assert body["total_samples"] == 1

        # File should land in the per-user directory at sample_000.wav
        expected_path = temp_profile_dir / "h1" / hash_user_id(42) / "sample_000.wav"
        assert expected_path.exists()

    def test_enroll_appends_additional_samples(self, client, temp_profile_dir):
        # Three takes
        for i in range(3):
            response = client.post(
                "/voice-profiles/enroll",
                params={"user_id": 42, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )
            assert response.status_code == 200
            assert response.json()["sample_index"] == i

        user_dir = temp_profile_dir / "h1" / hash_user_id(42)
        samples = sorted(user_dir.glob("sample_*.wav"))
        assert [p.name for p in samples] == [
            "sample_000.wav",
            "sample_001.wav",
            "sample_002.wav",
        ]

    def test_enroll_with_explicit_index_overwrites(self, client, temp_profile_dir):
        # First take
        client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        # Overwrite sample 0
        response = client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1", "sample_index": 0},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.json()["sample_index"] == 0
        user_dir = temp_profile_dir / "h1" / hash_user_id(42)
        assert sorted(p.name for p in user_dir.glob("sample_*.wav")) == ["sample_000.wav"]

    def test_legacy_single_file_migrated_on_new_enroll(self, client, temp_profile_dir):
        # Seed a legacy single-file profile
        household_dir = temp_profile_dir / "h1"
        household_dir.mkdir(parents=True, exist_ok=True)
        legacy = household_dir / (hash_user_id(42) + ".wav")
        legacy.write_bytes(_wav_bytes())

        # New enrollment should migrate the legacy file to sample_000 and
        # land the new take at sample_001.
        response = client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        body = response.json()
        assert body["sample_index"] == 1
        assert body["total_samples"] == 2

        user_dir = household_dir / hash_user_id(42)
        assert (user_dir / "sample_000.wav").exists()
        assert (user_dir / "sample_001.wav").exists()
        assert not legacy.exists()  # migrated away


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


class TestListUserSamples:
    def test_list_returns_enrolled_samples(self, client, temp_profile_dir):
        for _ in range(3):
            client.post(
                "/voice-profiles/enroll",
                params={"user_id": 42, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )

        response = client.get(
            "/voice-profiles/42/samples",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == 42
        indices = sorted(s["index"] for s in body["samples"])
        assert indices == [0, 1, 2]
        for s in body["samples"]:
            assert s["filename"].startswith("sample_")
            assert s["size_bytes"] > 0

    def test_list_empty_for_unknown_user(self, client):
        response = client.get(
            "/voice-profiles/999/samples",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        assert response.json()["samples"] == []


class TestDeleteUserSample:
    def test_delete_existing_sample(self, client, temp_profile_dir):
        # Enroll two samples
        for _ in range(2):
            client.post(
                "/voice-profiles/enroll",
                params={"user_id": 42, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )

        response = client.delete(
            "/voice-profiles/42/samples/0",
            params={"household_id": "h1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "deleted"
        assert body["sample_index"] == 0
        assert body["remaining_samples"] == 1

        user_dir = temp_profile_dir / "h1" / hash_user_id(42)
        assert not (user_dir / "sample_000.wav").exists()
        assert (user_dir / "sample_001.wav").exists()

    def test_delete_missing_sample_returns_404(self, client, temp_profile_dir):
        client.post(
            "/voice-profiles/enroll",
            params={"user_id": 42, "household_id": "h1"},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        response = client.delete(
            "/voice-profiles/42/samples/99",
            params={"household_id": "h1"},
        )
        assert response.status_code == 404


class TestCheckVoiceProfile:
    def test_check_reports_existence_and_count(self, client, temp_profile_dir):
        # Not enrolled yet
        response = client.get(
            "/voice-profiles/check",
            params={"user_id": 42, "household_id": "h1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["exists"] is False
        assert body["sample_count"] == 0

        # Enroll twice, then re-check
        for _ in range(2):
            client.post(
                "/voice-profiles/enroll",
                params={"user_id": 42, "household_id": "h1"},
                files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
            )

        response = client.get(
            "/voice-profiles/check",
            params={"user_id": 42, "household_id": "h1"},
        )
        body = response.json()
        assert body["exists"] is True
        assert body["sample_count"] == 2


class TestDeleteAllUserProfiles:
    """DELETE /voice-profiles/user/{user_id} — account-deletion biometric purge."""

    def _enroll(self, client, user_id, household_id):
        r = client.post(
            "/voice-profiles/enroll",
            params={"user_id": user_id, "household_id": household_id},
            files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200

    def test_deletes_user_across_all_households(self, client, temp_profile_dir):
        self._enroll(client, 42, "h1")
        self._enroll(client, 42, "h2")
        self._enroll(client, 99, "h1")  # a different user, must survive

        resp = client.delete("/voice-profiles/user/42")

        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 42
        assert set(body["households"]) == {"h1", "h2"}

        assert not (temp_profile_dir / "h1" / hash_user_id(42)).exists()
        assert not (temp_profile_dir / "h2" / hash_user_id(42)).exists()
        assert (temp_profile_dir / "h1" / hash_user_id(99)).exists()

    def test_removes_legacy_single_file_profile(self, client, temp_profile_dir):
        legacy = temp_profile_dir / "h1" / (hash_user_id(7) + ".wav")
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(_wav_bytes())

        resp = client.delete("/voice-profiles/user/7")

        assert resp.status_code == 200
        assert resp.json()["households"] == ["h1"]
        assert not legacy.exists()

    def test_no_profiles_is_idempotent(self, client):
        resp = client.delete("/voice-profiles/user/12345")
        assert resp.status_code == 200
        assert resp.json()["households"] == []
