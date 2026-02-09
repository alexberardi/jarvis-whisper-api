"""
App-to-app authentication dependency for jarvis-whisper-api.

Phase 5 Migration: Node auth -> App-to-app auth
- Services authenticate via X-Jarvis-App-Id + X-Jarvis-App-Key headers
- Context headers (X-Context-Household-Id, X-Context-Node-Id, etc.) provide request origin
- X-Context-Household-Member-Ids provides member list for voice recognition
"""

from jarvis_auth_client.fastapi import require_app_auth as _require_app_auth
from jarvis_auth_client.models import AppAuthResult

# Create the app auth dependency
_app_auth_dep = _require_app_auth()


async def verify_app_auth(
    x_jarvis_app_id: str | None = None,
    x_jarvis_app_key: str | None = None,
    x_context_household_id: str | None = None,
    x_context_node_id: str | None = None,
    x_context_user_id: int | None = None,
    x_context_household_member_ids: str | None = None,
) -> AppAuthResult:
    """
    Verify app-to-app credentials against jarvis-auth service.

    This dependency validates that the calling service (e.g., command-center)
    has valid app credentials. Context headers provide information about
    the original request (household, node, user, household members).

    The household_member_ids are critical for voice recognition - they specify
    which voice profiles to load when identifying speakers.

    Returns:
        AppAuthResult containing app validation and request context.

    Raises:
        HTTPException: If authentication fails.
    """
    return await _app_auth_dep(
        x_jarvis_app_id=x_jarvis_app_id,
        x_jarvis_app_key=x_jarvis_app_key,
        x_context_household_id=x_context_household_id,
        x_context_node_id=x_context_node_id,
        x_context_user_id=x_context_user_id,
        x_context_household_member_ids=x_context_household_member_ids,
    )
