"""/transcribe must reject oversized uploads (one big clip freezes STT).

The size guard runs before any whisper work, so these tests need no model.
"""
import io
import os

# Importing app.main resolves the auth URL at module load; give the discovery
# fallback a value so the import doesn't require a live config-service in CI.
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jarvis_auth_client.models import AppAuthResult, AppValidationResult, RequestContext

import app.main as main


class _FakeUpload:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


class TestSpoolUpload:
    def test_rejects_over_cap_and_cleans_up(self, monkeypatch):
        monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
        with pytest.raises(HTTPException) as exc:
            main._spool_upload(_FakeUpload(b"x" * 5000))
        assert exc.value.status_code == 413

    def test_under_cap_returns_temp_path(self, monkeypatch):
        monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1_000_000)
        path = main._spool_upload(_FakeUpload(b"hello wav"))
        try:
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == b"hello wav"
        finally:
            os.unlink(path)


@pytest.fixture()
def client(monkeypatch):
    from app.deps import verify_app_auth

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)

    async def _mock_auth():
        return AppAuthResult(
            app=AppValidationResult(valid=True, app_id="command-center", name="CC"),
            context=RequestContext(household_id="h1", node_id="n1"),
        )

    main.app.dependency_overrides[verify_app_auth] = _mock_auth
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_transcribe_rejects_oversized_upload(client):
    resp = client.post(
        "/transcribe",
        files={"file": ("big.wav", b"x" * 5000, "audio/wav")},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()
