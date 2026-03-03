"""
App-to-app authentication dependency for jarvis-whisper-api.

Phase 5 Migration: Node auth -> App-to-app auth
- Services authenticate via X-Jarvis-App-Id + X-Jarvis-App-Key headers
- Context headers (X-Context-Household-Id, X-Context-Node-Id, etc.) provide request origin
- X-Context-Household-Member-Ids provides member list for voice recognition
"""

from jarvis_auth_client.fastapi import require_app_auth as _require_app_auth
from jarvis_auth_client.models import AppAuthResult

# App auth dependency — use directly in Depends().
# The library dependency reads X-Jarvis-App-Id / X-Jarvis-App-Key from headers
# and validates against jarvis-auth. Also extracts X-Context-* headers.
verify_app_auth = _require_app_auth()

__all__ = ["verify_app_auth", "AppAuthResult"]
