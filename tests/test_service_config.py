"""Tests for service_config module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app import service_config


class TestInit:
    """Test init() function."""

    def teardown_method(self) -> None:
        """Reset module state after each test."""
        service_config._initialized = False

    @patch.object(service_config, "_has_config_client", False)
    def test_init_without_config_client(self) -> None:
        """init should return False and set initialized when config client not installed."""
        result = service_config.init()
        assert result is False
        assert service_config._initialized is True

    @patch.object(service_config, "_has_config_client", True)
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_config_url(self) -> None:
        """init should return False when JARVIS_CONFIG_URL not set."""
        result = service_config.init()
        assert result is False
        assert service_config._initialized is True

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(service_config, "config_init", return_value=True, create=True)
    @patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"})
    def test_init_with_config_client_success(self, mock_config_init: MagicMock) -> None:
        """init should return True when config client initializes successfully."""
        result = service_config.init()
        assert result is True
        assert service_config._initialized is True
        mock_config_init.assert_called_once_with(
            config_url="http://localhost:7700", refresh_interval_seconds=300
        )

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(
        service_config, "config_init", side_effect=RuntimeError("Connection failed"), create=True
    )
    @patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"})
    def test_init_with_config_client_failure(self, mock_config_init: MagicMock) -> None:
        """init should return False when config client fails to initialize."""
        result = service_config.init()
        assert result is False
        assert service_config._initialized is True


class TestShutdown:
    """Test shutdown() function."""

    def teardown_method(self) -> None:
        """Reset module state after each test."""
        service_config._initialized = False

    @patch.object(service_config, "_has_config_client", False)
    def test_shutdown_without_config_client(self) -> None:
        """shutdown should reset initialized state."""
        service_config._initialized = True
        service_config.shutdown()
        assert service_config._initialized is False

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(service_config, "config_shutdown", create=True)
    def test_shutdown_with_config_client(self, mock_shutdown: MagicMock) -> None:
        """shutdown should call config_shutdown when client is available."""
        service_config._initialized = True
        service_config.shutdown()
        mock_shutdown.assert_called_once()
        assert service_config._initialized is False


class TestGetUrl:
    """Test _get_url() and get_auth_url() functions."""

    def teardown_method(self) -> None:
        """Reset module state after each test."""
        service_config._initialized = False

    @patch.object(service_config, "_has_config_client", False)
    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"})
    def test_get_url_env_fallback(self) -> None:
        """_get_url should fall back to env var when config client not available."""
        url = service_config._get_url("jarvis-auth")
        assert url == "http://localhost:7701"

    @patch.object(service_config, "_has_config_client", False)
    @patch.dict(os.environ, {}, clear=True)
    def test_get_url_missing_service_raises(self) -> None:
        """_get_url should raise ValueError when service cannot be discovered."""
        with pytest.raises(ValueError, match="Cannot discover"):
            service_config._get_url("jarvis-auth")

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(
        service_config, "get_service_url", return_value="http://discovered:7701", create=True
    )
    def test_get_url_config_client(self, mock_get_url: MagicMock) -> None:
        """_get_url should use config client when initialized."""
        service_config._initialized = True
        url = service_config._get_url("jarvis-auth")
        assert url == "http://discovered:7701"

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(service_config, "get_service_url", side_effect=Exception("fail"), create=True)
    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://fallback:7701"})
    def test_get_url_config_client_error_falls_back(self, mock_get_url: MagicMock) -> None:
        """_get_url should fall back to env var on config client error."""
        service_config._initialized = True
        url = service_config._get_url("jarvis-auth")
        assert url == "http://fallback:7701"

    @patch.object(service_config, "_has_config_client", True)
    @patch.object(service_config, "get_service_url", return_value=None, create=True)
    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://fallback:7701"})
    def test_get_url_config_client_returns_none_falls_back(
        self, mock_get_url: MagicMock
    ) -> None:
        """_get_url should fall back to env var when config client returns None."""
        service_config._initialized = True
        url = service_config._get_url("jarvis-auth")
        assert url == "http://fallback:7701"

    @patch.object(service_config, "_has_config_client", False)
    @patch.dict(os.environ, {"JARVIS_AUTH_BASE_URL": "http://localhost:7701"})
    def test_get_auth_url(self) -> None:
        """get_auth_url should return auth service URL."""
        url = service_config.get_auth_url()
        assert url == "http://localhost:7701"

    @patch.object(service_config, "_has_config_client", False)
    @patch.dict(os.environ, {}, clear=True)
    def test_get_url_unknown_service(self) -> None:
        """_get_url should raise ValueError for unknown service without env fallback."""
        with pytest.raises(ValueError, match="Cannot discover unknown-service"):
            service_config._get_url("unknown-service")
