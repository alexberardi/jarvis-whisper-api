# jarvis-whisper-api
A Linux-friendly REST API wrapper for `whisper.cpp` with optional speaker recognition.

## Quick start (Ubuntu)
```bash
./setup-python.sh
./setup-whisper-cpp.sh
./run-dev.sh
```

## Environment variables
- `WHISPER_MODEL`: path to a GGML model file (default: `~/whisper.cpp/models/ggml-base.en.bin`)
- `WHISPER_CLI`: optional path to `whisper-cli` (useful if not on `PATH`)
- `WHISPER_ENABLE_CUDA`: set to `true` or `1` to build with CUDA
- `USE_VOICE_RECOGNITION`: set to `true` to enable speaker recognition
- `PORT`: override the default port (9999)

## Production
```bash
./full-setup-prod.sh
```

## Notes
- `setup-whisper-cpp.sh` installs build deps via `apt` and builds `whisper.cpp`.
- If you run `whisper-cli` from the repo `bin/` directory, `LD_LIBRARY_PATH` is handled automatically.
