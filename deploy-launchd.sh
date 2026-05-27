#!/usr/bin/env bash
# (Re)deploy the launchd agent that runs jarvis-whisper-api natively on login.
#
# On macOS, whisper.cpp is built from source by pywhispercpp via pip — Metal
# acceleration is automatic. Running natively (vs in Docker) lets the model
# see the Apple GPU.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$SCRIPT_DIR"

LABEL="${LAUNCHD_LABEL:-com.jarvis.whisper-api}"
PORT="${WHISPER_PORT:-7706}"
ENV_FILE_PATH="${ENV_FILE_PATH:-$HOME/.jarvis/compose/.env}"

PLIST_TEMPLATE="$ROOT/scripts/launchd/$LABEL.plist"
AGENTS_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$AGENTS_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/jarvis-whisper-api"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "❌ deploy-launchd.sh only supports macOS (detected $(uname -s))"
    exit 1
fi

if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "❌ launchd template not found at $PLIST_TEMPLATE"
    exit 1
fi

mkdir -p "$AGENTS_DIR" "$LOG_DIR"

sed -e "s#__ROOT__#$ROOT#g" \
    -e "s#__USER__#$USER#g" \
    -e "s#__PORT__#$PORT#g" \
    -e "s#__ENV_FILE__#$ENV_FILE_PATH#g" \
    "$PLIST_TEMPLATE" > "$TARGET_PLIST"

echo "📄 Installed launchd plist to $TARGET_PLIST"

echo "🔄 Reloading launchd service $LABEL..."
launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$TARGET_PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "✅ LaunchAgent ready. Check status with: launchctl print gui/$(id -u)/$LABEL"
echo "📜 Logs: $LOG_DIR/{out,err}.log"
