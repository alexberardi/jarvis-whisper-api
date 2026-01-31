# jarvis-whisper-api

REST API wrapper for whisper.cpp with optional speaker recognition.

## Quick Reference

```bash
# Setup
./setup-python.sh && ./setup-whisper-cpp.sh

# Run dev server (port 9999)
./run-dev.sh

# Test
curl -X POST -F "file=@jfk.wav" http://localhost:9999/transcribe
```

## Architecture

```
app/
├── main.py      # FastAPI routes: /ping, /transcribe
└── utils.py     # run_whisper(), recognize_speaker()
```

- **Transcription**: Shells out to `whisper-cli` from whisper.cpp
- **Speaker recognition**: Uses resemblyzer (optional, via USE_VOICE_RECOGNITION)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 9999 | API port |
| `WHISPER_MODEL` | `~/whisper.cpp/models/ggml-base.en.bin` | GGML model path |
| `WHISPER_CLI` | auto-detected | Path to whisper-cli binary |
| `WHISPER_ENABLE_CUDA` | false | Build whisper.cpp with CUDA |
| `USE_VOICE_RECOGNITION` | false | Enable speaker identification |

## API Endpoints

- `GET /ping` → `{"message": "pong"}`
- `POST /transcribe` → `{"text": "...", "speaker": "..."}`
  - Accepts: WAV file as multipart form data
  - Returns speaker as "unknown" if recognition disabled

## Dependencies

- **Runtime**: Python 3.12, FastAPI, uvicorn, resemblyzer
- **External**: whisper.cpp (built via setup-whisper-cpp.sh)
- **System**: libopenblas, libsndfile1, ffmpeg

## Speaker Recognition

To enable:
1. Set `USE_VOICE_RECOGNITION=true`
2. Add WAV files to `voice_profiles/` directory (filename = speaker name)
3. Threshold for match: 0.75 cosine similarity

## Notes

- Input must be WAV format
- Temp files are cleaned up after transcription
- whisper-cli outputs to `{input}.txt`, which is read and deleted
