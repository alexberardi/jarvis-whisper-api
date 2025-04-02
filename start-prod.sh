#!/bin/bash

source venv/bin/activate
echo "🚀 Starting Jarvis Whisper API in background..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 9999 > uvicorn.log 2>&1 &
echo "✅ Running in background. Logs: uvicorn.log"

