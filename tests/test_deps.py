"""Tests for authentication dependencies.

Phase 5 Migration: Node auth -> App-to-app auth
- Services now authenticate via X-Jarvis-App-Id + X-Jarvis-App-Key headers
- Context headers (X-Context-Household-Id, X-Context-Node-Id, etc.) are extracted

Tests use a minimal FastAPI app with TestClient so that Header() defaults
are resolved by FastAPI's dependency injection (not passed as raw Header objects).
"""

import os
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from jarvis_auth_client import init
from jarvis_auth_client.models import AppAuthResult, RequestContext


def _create_test_app():
    """Create a minimal FastAPI app that uses verify_app_auth."""
    with patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"}):
        from app.deps import verify_app_auth

    test_app = FastAPI()

    @test_app.get("/test-auth")
    async def test_auth_endpoint(auth: AppAuthResult = Depends(verify_app_auth)):
        return {
            "app_id": auth.app.app_id,
            "valid": auth.app.valid,
            "household_id": auth.context.household_id,
            "node_id": auth.context.node_id,
            "user_id": auth.context.user_id,
            "household_member_ids": auth.context.household_member_ids,
        }

    return test_app


class TestVerifyAppAuth:
    """Test cases for verify_app_auth dependency (app-to-app authentication)."""

    def setup_method(self) -> None:
        """Initialize auth client before each test."""
        init(auth_base_url="http://localhost:7701")
        self.app = _create_test_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        import jarvis_auth_client.fastapi as _fastapi_mod

        _fastapi_mod._http_client = None
        _fastapi_mod.clear_validation_cache()

    def test_missing_app_credentials_raises_401(self) -> None:
        """Should raise HTTPException when app credentials are missing."""
        response = self.client.get("/test-auth")

        assert response.status_code == 401
        assert "credentials" in response.json()["detail"].lower()

    def test_invalid_app_credentials_raises_401(self, httpx_mock: HTTPXMock) -> None:
        """Should raise HTTPException when app credentials are invalid."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=401,
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "invalid-app",
                "X-Jarvis-App-Key": "invalid-key",
            },
        )

        assert response.status_code == 401

    def test_valid_app_credentials_returns_auth_result(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Should return AppAuthResult with valid credentials."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["app_id"] == "command-center"

    def test_context_headers_are_extracted(self, httpx_mock: HTTPXMock) -> None:
        """Should extract context headers into RequestContext."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
                "X-Context-Household-Id": "household-123",
                "X-Context-Node-Id": "kitchen-pi",
                "X-Context-User-Id": "42",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["household_id"] == "household-123"
        assert data["node_id"] == "kitchen-pi"
        assert data["user_id"] == 42

    def test_context_headers_optional(self, httpx_mock: HTTPXMock) -> None:
        """Should work without context headers (they're optional)."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["household_id"] is None
        assert data["node_id"] is None
        assert data["user_id"] is None

    def test_household_member_ids_extracted(self, httpx_mock: HTTPXMock) -> None:
        """Should extract household_member_ids from context header."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
                "X-Context-Household-Id": "household-123",
                "X-Context-Household-Member-Ids": "1,2,42",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["household_member_ids"] == [1, 2, 42]

    def test_household_member_ids_default_empty(self, httpx_mock: HTTPXMock) -> None:
        """Should default household_member_ids to empty list when not provided."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["household_member_ids"] == []

    def test_auth_service_unavailable_raises_401(self, httpx_mock: HTTPXMock) -> None:
        """Should raise 401 when auth service is unreachable."""
        import httpx as httpx_lib

        httpx_mock.add_exception(
            httpx_lib.ConnectError("Connection refused"),
            url="http://localhost:7701/internal/app-ping",
        )

        response = self.client.get(
            "/test-auth",
            headers={
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret-key",
            },
        )

        assert response.status_code == 401
