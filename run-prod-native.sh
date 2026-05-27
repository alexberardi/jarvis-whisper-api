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

if [[ ! -x "$PY" ]]; then
    echo "[whisper-native] creating venv at $VENV"
    python3 -m venv "$VENV"
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
