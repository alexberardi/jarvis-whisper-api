#!/bin/bash
export $(grep -v '^#' .env | xargs)

source venv/bin/activate

# Install jarvis-log-client from local path
pip install -q -e ../jarvis-log-client 2>/dev/null || echo "Note: jarvis-log-client not found, remote logging disabled"

uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8012} --reload
