#!/bin/bash

source venv/bin/activate
echo "🛑 Stopping all running uvicorn processes..."
pkill -f "uvicorn app.main:app" || true
echo "✅ Stopped."

