#!/usr/bin/env bash
# Native production run script (launchd entry point on macOS).
#
# Idempotent: creates .venv and installs deps on first run, then reuses them
# until pyproject.toml changes. Exec's uvicorn at the end so launchd tracks the
# server PID directly.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-.env}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-7706}"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
SENTINEL="$VENV/.deps_installed"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Avoid Objective-C fork-safety abort when uvicorn workers exec under launchd
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

# Native macOS service env: the Docker compose injects service-discovery URLs +
# DATABASE_URL per container; native services must derive them from the shared
# .env. config-service + postgres are published on the host (localhost).
#
# Service discovery: without JARVIS_CONFIG_URL the service can't find jarvis-auth
# and refuses to start. config-service is on the host at localhost:CONFIG_SERVICE_PORT.
if [[ -z "${JARVIS_CONFIG_URL:-}" ]]; then
    export JARVIS_CONFIG_URL="http://localhost:${CONFIG_SERVICE_PORT:-7700}"
fi

# Native macOS runs OFF-container, so config-service's default host.docker.internal
# URLs are unreachable here. Request the "external" URL style (localhost) so that
# per-request auth resolution (app/deps.py's require_app_auth(), which resolves the
# auth URL on every call — unlike main.py which captures it at import) reaches
# jarvis-auth. Without this, /transcribe 401s with "Auth service unavailable"
# (host.docker.internal unresolvable) even though /health is 200. Requires
# jarvis-config-client >= 0.2.1 — 0.2.0 silently ignores the style param.
export JARVIS_CONFIG_URL_STYLE="${JARVIS_CONFIG_URL_STYLE:-external}"

# DB: postgres runs in Docker, published on the host at 127.0.0.1:${POSTGRES_PORT}.
# Build DATABASE_URL from the shared .env creds (DB_USER + POSTGRES_PASSWORD).
if [[ -z "${DATABASE_URL:-}" ]]; then
    export DATABASE_URL="postgresql+psycopg2://${DB_USER:-jarvis}:${POSTGRES_PASSWORD:-}@127.0.0.1:${POSTGRES_PORT:-5432}/jarvis_whisper"
fi
if [[ -z "${MIGRATIONS_DATABASE_URL:-}" ]]; then
    export MIGRATIONS_DATABASE_URL="$DATABASE_URL"
fi

if [[ ! -x "$PY" ]]; then
    # Pick a Python >=3.11 (this project requires it). macOS system python3 is
    # usually 3.9, and `brew install python@3.11` provides `python3.11` — NOT a
    # bare `python3` — so prefer explicitly-versioned interpreters, and only
    # accept `python3` when it is new enough. Fail loudly instead of building a
    # 3.9 venv that pip then rejects dependency-by-dependency.
    BASE_PY=""
    for _cand in python3.13 python3.12 python3.11; do
        if command -v "$_cand" >/dev/null 2>&1; then BASE_PY="$_cand"; break; fi
    done
    if [[ -z "$BASE_PY" ]] && command -v python3 >/dev/null 2>&1 \
        && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
        BASE_PY="python3"
    fi
    if [[ -z "$BASE_PY" ]]; then
        echo "[whisper-native] ERROR: Python >=3.11 is required but was not found." >&2
        echo "[whisper-native]        Install it (e.g. 'brew install python@3.11') and retry." >&2
        exit 1
    fi
    echo "[whisper-native] creating venv at $VENV using $BASE_PY ($("$BASE_PY" --version 2>&1))"
    "$BASE_PY" -m venv "$VENV"
fi

# Install / refresh deps only when pyproject changes (or first run)
if [[ ! -f "$SENTINEL" || "$ROOT/pyproject.toml" -nt "$SENTINEL" ]]; then
    echo "[whisper-native] installing deps (this can take several minutes the first time — pywhispercpp builds whisper.cpp from source)"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -e "$ROOT"
    touch "$SENTINEL"
fi

echo "[whisper-native] running alembic migrations"
"$PY" -m alembic upgrade head

echo "[whisper-native] starting uvicorn on ${SERVER_HOST}:${PORT}"
exec "$VENV/bin/uvicorn" app.main:app --host "$SERVER_HOST" --port "$PORT"
