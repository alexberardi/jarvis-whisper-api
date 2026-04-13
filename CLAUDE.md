# jarvis-whisper-api

REST API wrapper for whisper.cpp with optional speaker recognition.

## Quick Reference

```bash
# Setup
./setup-python.sh && ./setup-whisper-cpp.sh

# Run (Docker dev with hot reload + logging)
./run-docker-dev.sh

# Or direct (local dev)
./run-dev.sh

# Test (requires valid node auth)
curl -X POST -F "file=@jfk.wav" \
  -H "X-API-Key: node_id:node_key" \
  http://localhost:7706/transcribe
```

## Architecture

```
app/
├── main.py          # FastAPI app, routes: /ping, /transcribe
├── deps.py          # App-to-app authentication via jarvis-auth
├── utils.py         # run_whisper(), recognize_speaker(), hash_user_id()
├── exceptions.py
└── api/
    └── voice_profiles.py  # Voice enrollment CRUD endpoints
```

- **Transcription**: Shells out to `whisper-cli` from whisper.cpp
- **Speaker recognition**: Uses resemblyzer (optional, via USE_VOICE_RECOGNITION)
- **Authentication**: Nodes authenticate via jarvis-auth service

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 7706 | API port |
| `WHISPER_MODEL` | `~/whisper.cpp/models/ggml-base.en.bin` | GGML model path |
| `WHISPER_CLI` | auto-detected | Path to whisper-cli binary |
| `WHISPER_ENABLE_CUDA` | false | Build whisper.cpp with CUDA |
| `USE_VOICE_RECOGNITION` | false | Enable speaker identification |
| `JARVIS_AUTH_BASE_URL` | http://localhost:7701 | Auth service URL |
| `JARVIS_APP_ID` | jarvis-whisper | App ID for auth |
| `JARVIS_APP_KEY` | - | App key (required for auth) |
| `NODE_AUTH_CACHE_TTL` | 60 | Cache TTL for auth validation |

## API Endpoints

- `GET /ping` → `{"message": "pong"}` (no auth required)
- `GET /health` → `{"status": "healthy"}` (no auth required)
- `POST /transcribe` → `{"text": "...", "speaker": {"user_id": 42, "confidence": 0.87}}` (app-to-app auth)
  - Accepts: WAV file as multipart form data
  - Returns speaker user_id + confidence if voice recognition enabled

### Voice Profile Enrollment (app-to-app auth)

- `POST /voice-profiles/enroll?user_id=&household_id=` — Upload WAV voice sample
- `DELETE /voice-profiles/{user_id}?household_id=` — Remove a voice profile
- `GET /voice-profiles?household_id=` — List enrolled profiles

Profiles stored at `voice_profiles/{household_id}/{hash(user_id)}.wav`. Command-center proxies these via `/api/v0/media/whisper/voice-profiles/*`.

## Dependencies

**Python Libraries:**
- Python 3.12, FastAPI, uvicorn, resemblyzer, httpx
- jarvis-log-client (for remote logging)

**External Dependencies:**
- whisper.cpp (built via setup-whisper-cpp.sh)
- System: libopenblas, libsndfile1, ffmpeg

**Service Dependencies:**
- ✅ **Required**: `jarvis-auth` (7701) - Node authentication validation
- ⚠️ **Optional**: `jarvis-logs` (7702) - Centralized logging (degrades to console if unavailable)
- ⚠️ **Optional**: `jarvis-config-service` (7700) - Service discovery

**Used By:**
- `jarvis-command-center` - Speech-to-text transcription (optional)

**Impact if Down:**
- ⚠️ Speech transcription unavailable (if command-center uses this service)
- ✅ Command-center may have alternative transcription methods

## Logging

Uses jarvis-log-client for remote logging to jarvis-logs service.
Configure with `JARVIS_LOG_CONSOLE_LEVEL` and `JARVIS_LOG_REMOTE_LEVEL`.

## Speaker Recognition

To enable:
1. Set `USE_VOICE_RECOGNITION=true`
2. Enroll voice profiles via the API (`POST /voice-profiles/enroll`) or place WAV files in `voice_profiles/{household_id}/` directory
3. Threshold for match: 0.75 cosine similarity
4. Profiles are hashed by user_id (`hash_user_id()` in `app/utils.py`)

## Testing

```bash
pytest tests/test_voice_profiles.py -v  # 6 tests, no external deps needed
```

## Notes

- Input must be WAV format
- Temp files are cleaned up after transcription
- whisper-cli outputs to `{input}.txt`, which is read and deleted
