#!/bin/bash
export $(grep -v '^#' .env | xargs)

source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-9999}
