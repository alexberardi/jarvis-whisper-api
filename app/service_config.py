"""Service URL discovery via jarvis-config-client.

Falls back to environment variables when the config client is not initialized.
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from jarvis_config_client import (
        init as config_init,
        shutdown as config_shutdown,
        get_service_url,
    )
    _has_config_client = True
except ImportError:
    _has_config_client = False

_initialized: bool = False

# Legacy env var fallbacks
_ENV_VAR_FALLBACKS: dict[str, str] = {
    "jarvis-auth": "JARVIS_AUTH_BASE_URL",
}


def init() -> bool:
    """Initialize service discovery. Call at startup."""
    global _initialized
    if not _has_config_client:
        logger.info("jarvis-config-client not installed, using env var fallbacks")
        _initialized = True
        return False

    config_url = os.getenv("JARVIS_CONFIG_URL")
    if not config_url:
        logger.warning("JARVIS_CONFIG_URL not set, using env var fallbacks")
        _initialized = True
        return False

    try:
        success = config_init(config_url=config_url, refresh_interval_seconds=300)
        _initialized = True
        if success:
            logger.info("Service discovery initialized")
        return success
    except RuntimeError as e:
        logger.error("Failed to initialize service discovery: %s", e)
        _initialized = True
        return False


def shutdown() -> None:
    """Shutdown service discovery."""
    global _initialized
    if _has_config_client:
        config_shutdown()
    _initialized = False


def _get_url(service_name: str) -> str:
    """Get URL for a service with fallback chain."""
    if _has_config_client and _initialized:
        try:
            url = get_service_url(service_name)
            if url:
                return url
        except Exception:
            pass

    env_var = _ENV_VAR_FALLBACKS.get(service_name)
    if env_var:
        env_url = os.getenv(env_var)
        if env_url:
            logger.warning(
                "Using legacy env var %s for %s. "
                "Consider registering in config-service instead.",
                env_var, service_name,
            )
            return env_url

    raise ValueError(
        f"Cannot discover {service_name}. "
        f"Set JARVIS_CONFIG_URL or {_ENV_VAR_FALLBACKS.get(service_name, 'N/A')}"
    )


def get_auth_url() -> str:
    """Get auth service URL."""
    return _get_url("jarvis-auth")
