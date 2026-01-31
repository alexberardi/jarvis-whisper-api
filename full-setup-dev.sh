#!/bin/bash
set -euo pipefail

./setup-python.sh
./setup-whisper-cpp.sh
./run-dev.sh
