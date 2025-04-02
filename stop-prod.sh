#!/bin/bash

source venv/bin/activate
echo "🛑 Stopping all running uvicorn processes..."
pkill -f "uvicorn main:app"
echo "✅ Stopped."

