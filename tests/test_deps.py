"""Tests for authentication dependencies.

Phase 5 Migration: Node auth -> App-to-app auth
- Services now authenticate via X-Jarvis-App-Id + X-Jarvis-App-Key headers
- Context headers (X-Context-Household-Id, X-Context-Node-Id, etc.) are extracted
"""

import pytest
from fastapi import HTTPException
from pytest_httpx import HTTPXMock

from jarvis_auth_client import init
from jarvis_auth_client.models import AppAuthResult, RequestContext


class TestVerifyAppAuth:
    """Test cases for verify_app_auth dependency (app-to-app authentication)."""

    def setup_method(self) -> None:
        """Initialize auth client before each test."""
        init(auth_base_url="http://localhost:7701")

    def teardown_method(self) -> None:
        """Clean up after each test."""
        import jarvis_auth_client.fastapi as _fastapi_mod

        _fastapi_mod._http_client = None

    @pytest.mark.asyncio
    async def test_missing_app_credentials_raises_401(self) -> None:
        """Should raise HTTPException when app credentials are missing."""
        from app.deps import verify_app_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_app_auth()

        assert exc_info.value.status_code == 401
        assert "credentials" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_invalid_app_credentials_raises_401(self, httpx_mock: HTTPXMock) -> None:
        """Should raise HTTPException when app credentials are invalid."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=401,
        )

        from app.deps import verify_app_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_app_auth(
                x_jarvis_app_id="invalid-app",
                x_jarvis_app_key="invalid-key",
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_app_credentials_returns_auth_result(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Should return AppAuthResult with valid credentials."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        from app.deps import verify_app_auth

        result = await verify_app_auth(
            x_jarvis_app_id="command-center",
            x_jarvis_app_key="secret-key",
        )

        assert isinstance(result, AppAuthResult)
        assert result.app.valid is True
        assert result.app.app_id == "command-center"

    @pytest.mark.asyncio
    async def test_context_headers_are_extracted(self, httpx_mock: HTTPXMock) -> None:
        """Should extract context headers into RequestContext."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        from app.deps import verify_app_auth

        result = await verify_app_auth(
            x_jarvis_app_id="command-center",
            x_jarvis_app_key="secret-key",
            x_context_household_id="household-123",
            x_context_node_id="kitchen-pi",
            x_context_user_id=42,
        )

        assert isinstance(result.context, RequestContext)
        assert result.context.household_id == "household-123"
        assert result.context.node_id == "kitchen-pi"
        assert result.context.user_id == 42

    @pytest.mark.asyncio
    async def test_context_headers_optional(self, httpx_mock: HTTPXMock) -> None:
        """Should work without context headers (they're optional)."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        from app.deps import verify_app_auth

        result = await verify_app_auth(
            x_jarvis_app_id="command-center",
            x_jarvis_app_key="secret-key",
        )

        assert result.context.household_id is None
        assert result.context.node_id is None
        assert result.context.user_id is None

    @pytest.mark.asyncio
    async def test_household_member_ids_extracted(self, httpx_mock: HTTPXMock) -> None:
        """Should extract household_member_ids from context header."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        from app.deps import verify_app_auth

        result = await verify_app_auth(
            x_jarvis_app_id="command-center",
            x_jarvis_app_key="secret-key",
            x_context_household_id="household-123",
            x_context_household_member_ids="1,2,42",
        )

        assert result.context.household_member_ids == [1, 2, 42]

    @pytest.mark.asyncio
    async def test_household_member_ids_default_empty(self, httpx_mock: HTTPXMock) -> None:
        """Should default household_member_ids to empty list when not provided."""
        httpx_mock.add_response(
            url="http://localhost:7701/internal/app-ping",
            status_code=200,
            json={"app_id": "command-center"},
        )

        from app.deps import verify_app_auth

        result = await verify_app_auth(
            x_jarvis_app_id="command-center",
            x_jarvis_app_key="secret-key",
        )

        assert result.context.household_member_ids == []

    @pytest.mark.asyncio
    async def test_auth_service_unavailable_raises_401(self, httpx_mock: HTTPXMock) -> None:
        """Should raise 401 when auth service is unreachable."""
        import httpx as httpx_lib

        httpx_mock.add_exception(
            httpx_lib.ConnectError("Connection refused"),
            url="http://localhost:7701/internal/app-ping",
        )

        from app.deps import verify_app_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_app_auth(
                x_jarvis_app_id="command-center",
                x_jarvis_app_key="secret-key",
            )

        assert exc_info.value.status_code == 401
