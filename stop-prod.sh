#!/bin/bash

source venv/bin/activate
echo "🛑 Stopping all running uvicorn processes..."
pkill -f "uvicorn main:app"
pkill -f "/Users/jarvis/jarvis-whisper-api/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 9999"
echo "✅ Stopped."

