"""
Node authentication dependency for jarvis-whisper-api.

Validates node credentials against jarvis-auth service.
"""

import os
import time

import httpx
from fastapi import Header, HTTPException

JARVIS_AUTH_BASE_URL = os.getenv("JARVIS_AUTH_BASE_URL", "http://localhost:8007")
JARVIS_AUTH_APP_ID = os.getenv("JARVIS_APP_ID", "jarvis-whisper")
JARVIS_AUTH_APP_KEY = os.getenv("JARVIS_APP_KEY")

# Cache for node validation results
_node_validation_cache: dict[str, tuple[dict, float]] = {}
NODE_AUTH_CACHE_TTL = int(os.getenv("NODE_AUTH_CACHE_TTL", "60"))


def _get_cached_validation(api_key: str) -> dict | None:
    """Get cached validation result if not expired."""
    if api_key in _node_validation_cache:
        result, timestamp = _node_validation_cache[api_key]
        if time.time() - timestamp < NODE_AUTH_CACHE_TTL:
            return result
        del _node_validation_cache[api_key]
    return None


def _cache_validation(api_key: str, result: dict) -> None:
    """Cache a validation result."""
    _node_validation_cache[api_key] = (result, time.time())


def _validate_node_with_auth_service(node_id: str, node_key: str) -> dict:
    """Validate node credentials with jarvis-auth service."""
    if not JARVIS_AUTH_APP_KEY:
        return {"valid": False, "reason": "JARVIS_APP_KEY not configured"}

    validate_url = JARVIS_AUTH_BASE_URL.rstrip("/") + "/internal/validate-node"

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                validate_url,
                headers={
                    "X-Jarvis-App-Id": JARVIS_AUTH_APP_ID,
                    "X-Jarvis-App-Key": JARVIS_AUTH_APP_KEY,
                },
                json={
                    "node_id": node_id,
                    "node_key": node_key,
                    "service_id": "jarvis-whisper",
                },
            )
    except httpx.RequestError as exc:
        return {"valid": False, "reason": f"Auth service unavailable: {exc}"}

    if resp.status_code != 200:
        return {"valid": False, "reason": f"Auth service error: {resp.status_code}"}

    return resp.json()


def verify_node_auth(x_api_key: str = Header(...)) -> str:
    """
    Verify node API key against jarvis-auth service.

    The x_api_key header should be in format "node_id:node_key".

    Returns:
        The node_id if authentication succeeds.

    Raises:
        HTTPException: If authentication fails.
    """
    # Check cache first
    cached = _get_cached_validation(x_api_key)
    if cached is not None:
        if not cached.get("valid"):
            raise HTTPException(status_code=401, detail=cached.get("reason", "Invalid API Key"))
        return cached.get("node_id", "unknown")

    # Validate format
    if ":" not in x_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key format. Expected 'node_id:node_key'")

    node_id, node_key = x_api_key.split(":", 1)
    result = _validate_node_with_auth_service(node_id, node_key)

    # Add node_id to result for caching
    if result.get("valid"):
        result["node_id"] = node_id

    _cache_validation(x_api_key, result)

    if not result.get("valid"):
        raise HTTPException(status_code=401, detail=result.get("reason", "Invalid API Key"))

    return node_id
