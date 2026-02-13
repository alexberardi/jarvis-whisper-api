"""Service URL discovery via jarvis-config-client."""

import logging

from jarvis_config_client import (
    init as config_init,
    shutdown as config_shutdown,
    get_auth_url,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
get_auth_url = get_auth_url


def init() -> bool:
    """Initialize service discovery. Call at startup."""
    try:
        success = config_init()
        if success:
            logger.info("Service discovery initialized")
        return success
    except RuntimeError as e:
        logger.error("Failed to initialize service discovery: %s", e)
        raise


def shutdown() -> None:
    """Shutdown service discovery."""
    config_shutdown()
