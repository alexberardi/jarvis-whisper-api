#!/bin/bash
# Development server with hot reload
# Usage: ./run.sh [--docker]

set -e
cd "$(dirname "$0")"

if [[ "$1" == "--docker" ]]; then
    # Docker development mode
    BUILD_FLAGS=""
    if [[ "$2" == "--rebuild" ]]; then
        docker compose --env-file .env -f docker-compose.dev.yaml build --no-cache
        BUILD_FLAGS="--build"
    elif [[ "$2" == "--build" ]]; then
        BUILD_FLAGS="--build"
    fi
    docker compose --env-file .env -f docker-compose.dev.yaml up $BUILD_FLAGS
else
    # Local development mode
    export $(grep -v '^#' .env | xargs)

    # Activate venv if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Install jarvis client libraries
    JARVIS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    source "${JARVIS_ROOT}/scripts/install-clients.sh"
    install_jarvis_clients log-client config-client auth-client settings-client

    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8012} --reload
fi
