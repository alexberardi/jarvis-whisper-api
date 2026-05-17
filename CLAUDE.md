# jarvis-whisper-api

Speech-to-text + speaker identification. **In-process whisper.cpp** (via `pywhispercpp`) + optional `resemblyzer` voice embeddings for speaker recognition. Authenticated app-to-app; called primarily by command-center as a proxy.

> **Why in-process:** earlier versions shelled out to `whisper-cli` per request. The fork/exec/model-load cycle dominated request latency on GPU builds. The current engine loads the model **once at startup** and reuses it for every request. Model swaps happen via a fingerprint-reload pattern (same as TTS): when the admin changes `whisper.model_path`, the next call notices and rebuilds.

---

## What this service is (and isn't)

| Responsibility | Endpoint | Auth |
|---|---|---|
| **Transcribe an audio clip → text** (+ optional speaker ID) | `POST /transcribe` | App-to-app |
| **Enroll a user's voice profile** | `POST /voice-profiles/enroll` | App-to-app |
| **Manage voice profiles** | `GET` / `DELETE /voice-profiles/...` | App-to-app |
| **Its own runtime settings** | `/settings/*` | Combined (read), superuser JWT (write) |

**Not** a:
- Streaming STT service. Single shot WAV in, text out.
- Voice activity detector. Audio must already be a chunk of speech; the node handles VAD.
- Diarization service (multi-speaker segmentation). It picks one speaker per clip.

---

## Quick Reference

```bash
# First-time setup (Python deps + build pywhispercpp + download model)
./setup-python.sh && ./setup-whisper-cpp.sh

# Dev (Docker with hot reload + logging)
./run-docker-dev.sh

# Bare metal
./run.sh

# Smoke test
curl -X POST -F "file=@jfk.wav" \
  -H "X-Jarvis-App-Id: $APP_ID" -H "X-Jarvis-App-Key: $APP_KEY" \
  http://localhost:7706/transcribe

# Test (no model needed)
pytest tests/test_voice_profiles.py -v
```

---

## Dependency graph

**Upstream (whisper depends on):**
- **whisper.cpp model file** — `.bin` (GGML) at the path in `whisper.model_path` setting. Required.
- **jarvis-auth** (port 7701) — app-to-app validation per request (cached, default 60s TTL)
- **jarvis-config-service** (port 7700) — service discovery (auth URL)
- **jarvis-logs** (port 7702, optional) — remote logging
- **PostgreSQL** (required) — own settings table (`multitenant_settings`)
- **resemblyzer** (Python lib, optional) — only loaded if `voice.recognition_enabled=true`. Pulls a torch + VoiceEncoder ~250MB.

**Downstream (depends on whisper):**
- **jarvis-command-center** — calls via `/api/v0/media/whisper/transcribe` proxy; nodes never hit whisper directly. CC passes `X-Context-Household-Id`, `X-Context-Node-Id`, `X-Context-Household-Member-Ids` so speaker recognition can scope to known household users.

**Impact if down:**
- Voice commands silently work *to* command-center but fail STT → CC returns error
- Voice profile enrollment from mobile fails (mobile → CC proxy → whisper)

---

## How it actually runs

### Startup
1. `service_config.init()` — reads `JARVIS_CONFIG_URL`, discovers other services
2. `_setup_remote_logging()` — best-effort hookup to jarvis-logs
3. **Whisper model pre-warm** — `get_model()` loads the GGML model into VRAM/RAM. ~3s on GPU build. **Failure is fatal** — health check will 500 rather than silently failing every request.
4. **VoiceEncoder pre-warm** (if `voice.recognition_enabled`) — `_get_encoder()` initializes resemblyzer + moves it to GPU. ~5-7s on CUDA cold start. **Failure is non-fatal** — the first transcription request pays the cost instead.

### Per request (`POST /transcribe`)
1. Save uploaded WAV to a temp file.
2. (Optional) `preprocess_audio` — normalize + silence-trim if `?preprocess=true`. Falls back to original on failure.
3. `run_whisper(audio, prompt, temperature, temperature_inc, beam_size)` — calls the in-process `pywhispercpp.Model.transcribe()`. The model fingerprint is checked first; if `whisper.model_path` or `n_threads` changed, model rebuilds before the call.
4. If `voice.recognition_enabled`: `recognize_speaker(audio, household_id, member_ids)` — embeds the clip, compares against each member's profile, returns `(user_id, confidence)` if max similarity >= `voice.similarity_threshold` (default 0.75).
5. Clean up temp files (including whisper's `.txt` output sibling).
6. Return `{"text": "...", "speaker": {"user_id": N or null, "confidence": 0.0-1.0}}`.

**Phase timings** are logged at INFO: `save / preproc / whisper / speaker / total` in ms. Use these for triage.

---

## "How to..." recipes

### Add a transcription parameter (e.g. `vad_filter`)

1. Add a query param to `transcribe()` in `app/main.py` with a default and Pydantic validation.
2. Thread it through `run_whisper()` in `app/utils.py`.
3. Map to `pywhispercpp.Model.transcribe()` kwargs in `app/whisper_engine.py`.
4. If it should also be configurable globally, add a `SettingDefinition` in `app/services/settings_definitions.py` and read it via `get_settings_service().get_*`.

### Change the model

Update setting `whisper.model_path` via `PUT /settings/whisper.model_path` (superuser JWT). The next request sees the fingerprint change and rebuilds. **No restart needed** — that's the point of the fingerprint pattern.

If a rebuild **fails** (path doesn't exist, file corrupted), the engine **keeps the previous model** loaded and the request that triggered the rebuild fails. Subsequent requests retry the rebuild. Restore the setting to a known-good path to recover.

### Add a new voice profile from outside command-center

Don't. Command-center is the canonical proxy and ensures auth/household context is consistent. If you need a direct path, call `POST /voice-profiles/enroll?user_id=&household_id=` with app-to-app credentials and a WAV body. Profiles store at `voice_profiles/{household_id}/{hash(user_id)}.wav`.

### Enable speaker recognition

`PUT /settings/voice.recognition_enabled` → `true`. Restart **not** needed for the flag itself; the per-request code reads it dynamically. But the VoiceEncoder lazy-loads on first use (~5-7s on GPU cold) — restart the service first if you want the pre-warm.

### Tune the speaker match threshold

`voice.similarity_threshold`, default 0.75. Lower → more false positives. Higher → more false negatives. Empirically 0.70-0.80 is the working band.

---

## Invariants & gotchas

1. **Authentication is app-to-app, NOT raw node X-API-Key.** Older docs say "X-API-Key: node_id:node_key". Current code uses `X-Jarvis-App-Id` + `X-Jarvis-App-Key` validated against jarvis-auth, with `X-Context-*` headers carrying household/node/member info. Command-center adds the context headers before proxying. **Don't add `verify_node_auth` here — it's not the model.**
2. **`USE_VOICE_RECOGNITION` env var is now a settings fallback.** Canonical config is `voice.recognition_enabled` in the settings DB. Env var still works as a fallback for existing installs but new deployments should set the setting.
3. **Model load failure at startup is fatal — by design.** If `whisper.model_path` is wrong, the service refuses to start cleanly. Better to fail loud than silently 500 every transcription request.
4. **Fingerprint reload only catches `whisper.model_path` and `n_threads`.** Other inference params (temperature, beam size, language) flow through the request directly; they don't need rebuild. If you add a new setting that affects model construction, add it to `_EngineFingerprint` in `app/whisper_engine.py`.
5. **VoiceEncoder is a heavy import.** It pulls torch + a ResNet. **Only import inside the conditional** (`if voice.recognition_enabled`). Lazy import is in `app/utils.py:_get_encoder`. Don't move it to module-level.
6. **`hash_user_id()` is intentionally one-way for privacy.** Profile filenames are `{household_id}/{hash(user_id)}.wav` so casual inspection of disk doesn't reveal who's who. Member IDs from `X-Context-Household-Member-Ids` are hashed at lookup time to find the matching profile.
7. **Speaker recognition is scoped to `household_member_ids`, not all enrolled profiles.** This is set by command-center on every request. If CC sends an empty list, speaker recognition returns `(None, 0.0)`. Don't bypass — it's how multi-tenant isolation works.
8. **whisper.cpp writes a `<input>.txt` sibling.** The transcription wrapper reads and deletes it. If you change the file naming, also change the cleanup logic in the `finally` block.
9. **Three Dockerfiles for three GPU paths.** `Dockerfile` (CPU), `Dockerfile.gpu` (NVIDIA), `Dockerfile.rocm` (AMD). Pick the right one via the compose file (`docker-compose.gpu.yaml` vs the default). Build flags differ.
10. **`/health` is shallow.** It returns `{"status": "healthy"}` even if the model failed to load mid-life — though the startup pre-warm makes that unlikely. If you need real liveness, hit `/transcribe` with a small WAV.

---

## API surface

### Public health
| Method | Path |
|---|---|
| GET | `/ping` |
| GET | `/health` |

### Transcribe (app-to-app)
| Method | Path | Notes |
|---|---|---|
| POST | `/transcribe` | Multipart WAV upload. Optional query: `prompt`, `preprocess`, `temperature`, `temperature_inc`, `beam_size` |

### Voice profiles (app-to-app, `app/api/voice_profiles.py`)
| Method | Path | Notes |
|---|---|---|
| POST | `/voice-profiles/enroll?user_id=&household_id=` | WAV body |
| DELETE | `/voice-profiles/{user_id}?household_id=` | |
| GET | `/voice-profiles?household_id=` | Lists enrolled user IDs |

### Settings (library mount)
- Reads: superuser JWT OR app credentials
- Writes: superuser JWT only

---

## Data model

| Where | What |
|---|---|
| **Filesystem** | Voice profiles: `voice_profiles/{household_id}/{hash(user_id)}.wav`. Mount as a volume in Docker — these are user-tied artifacts and should outlive the container. |
| **Postgres** | `multitenant_settings` table only — same schema as the rest of the stack. |

The service has **no DB tables of its own** beyond settings. Profiles intentionally live on disk for simplicity and direct WAV inspection during debugging.

---

## Config surface

**Env vars (bootstrap + secrets only):**

| Variable | Required | Purpose |
|---|---|---|
| `JARVIS_CONFIG_URL` | yes | Service discovery |
| `JARVIS_APP_ID` / `JARVIS_APP_KEY` | yes | App credential for logging + auth |
| `DATABASE_URL` | yes | Postgres |
| `WHISPER_MODEL` | fallback only | Falls back to this if `whisper.model_path` setting is empty |
| `WHISPER_N_THREADS` | optional (4) | n_threads passed to pywhispercpp |
| `JARVIS_LOG_CONSOLE_LEVEL` / `JARVIS_LOG_REMOTE_LEVEL` | optional | Logging |
| `PORT` | optional (7706) | API bind |

**DB-backed settings (preferred for non-secret config):**
| Key | Default | Notes |
|---|---|---|
| `whisper.model_path` | `~/whisper.cpp/models/ggml-base.en.bin` | Triggers reload via fingerprint |
| `whisper.default_temperature` | 0.0 | Per-request override available |
| `whisper.default_temperature_inc` | 0.2 | |
| `whisper.default_beam_size` | 5 | |
| `whisper.language` | `en` | |
| `voice.recognition_enabled` | false | Loads VoiceEncoder when flipped |
| `voice.similarity_threshold` | 0.75 | Cosine similarity match threshold |
| `auth.cache_ttl_seconds` | 60 | App-auth result cache |
| `server.port` / `server.log_*` | — | Requires reload for port |

---

## Architecture

```
jarvis-whisper-api/
├── app/
│   ├── main.py                          # FastAPI app, /transcribe, startup pre-warm
│   ├── whisper_engine.py                # pywhispercpp wrapper + fingerprint reload
│   ├── audio.py                         # preprocess_audio (normalize + trim)
│   ├── utils.py                         # run_whisper, recognize_speaker, hash_user_id, _get_encoder
│   ├── exceptions.py                    # WhisperTranscriptionError, AudioProcessingError
│   ├── deps.py                          # verify_app_auth = require_app_auth()
│   ├── service_config.py                # jarvis-config-client wrapper
│   ├── api/voice_profiles.py            # /voice-profiles/* CRUD
│   ├── db/                              # SQLAlchemy session for settings
│   └── services/
│       ├── settings_definitions.py      # SettingDefinitions list
│       └── settings_service.py
├── voice_profiles/                      # Disk storage — mount as volume
├── alembic/                             # Settings migrations
├── bin/                                 # Helper scripts
├── setup-whisper-cpp.sh                 # Build whisper.cpp + pull model
├── setup-python.sh                      # venv + pip install
├── docker-compose.{dev,gpu,prod}.yaml
├── Dockerfile / Dockerfile.gpu / Dockerfile.rocm
└── tests/                               # Unit tests (no model needed for voice_profiles tests)
```

---

## Testing

- **Unit tests only.** `tests/test_voice_profiles.py` runs without a model loaded (6 tests). Other test files require a real model and are excluded from CI.
- No fixtures spin up real whisper — engine is mocked.

```bash
pytest tests/test_voice_profiles.py -v
```

---

## Failure modes

| Failure | Behavior |
|---|---|
| Model file missing at startup | Service fails to start — pre-warm raises |
| Settings DB down | `whisper.model_path` falls back to `WHISPER_MODEL` env var; voice recognition flag falls back to `USE_VOICE_RECOGNITION` |
| Auth down | All `/transcribe` and `/voice-profiles/*` calls return 401/503 (settings reads still work if app-creds cached) |
| pywhispercpp build broken | Transcription raises `WhisperTranscriptionError` → 500 |
| Voice profile missing for a known user | Speaker returns `(None, 0.0)` — not an error |
| GPU OOM on cold-start | Pre-warm fails fatally on startup |
| Disk full on voice profile write | Enroll returns 500 |

---

## Out of scope / explicitly not here

- **VAD (voice activity detection)** — the node handles this before sending audio
- **Diarization (multi-speaker)** — single speaker per clip
- **Streaming STT** — single-shot only
- **Multi-language detection** — language is set globally via `whisper.language`
- **Profile training UX** — admin/mobile UIs own enrollment flow
- **Direct node access** — nodes never hit this service; CC proxies
