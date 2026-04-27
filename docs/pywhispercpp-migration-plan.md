# pywhispercpp migration plan

Status: not started — designed in conversation 2026-04-26 after standing up GPU + large-v3-turbo on prod.

## Why

After the GPU rebuild (v0.1.x of jarvis-whisper-api with `Dockerfile.gpu` + `large-v3-turbo`), the JFK 11 s benchmark on a 3090 returned in 0.86 s warm / 1.41 s cold. The actual GPU inference is ~100–200 ms; **the dominant cost is the per-call `whisper-cli` subprocess load** — model gets reloaded into VRAM every request, then freed on process exit. Verified by watching GPU memory: it doesn't budge between calls because the process holding the model dies.

Goal: eliminate the per-call model reload by keeping the model resident in the FastAPI process. Expected payoff: 0.4–0.7 s commands → ~50–150 ms end-to-end. That recovers the missing 5–10× the GPU is leaving on the table.

## Approach

Replace `subprocess.run([whisper-cli, ...])` in `app/utils.py::run_whisper` with a module-level `pywhispercpp.model.Model` instance, lazily initialized on first call (or in a startup hook) and reused for every subsequent request. Same whisper.cpp engine underneath — same accuracy, same model file, same CUDA backend — just no fork/exec/load per call.

## Current state (so a fresh session has the baseline)

- Repo: `/Users/alexanderberardi/jarvis/jarvis-whisper-api`. Last commit when this plan was written: latest `main`.
- Image on prod: `jarvis-whisper-api:gpu` (locally built from `Dockerfile.gpu`). Compose at `/home/jarvis/.jarvis/compose/docker-compose.yml` already has `runtime: nvidia`, GPU device reservation, and mounts `/home/jarvis/whisper-models:/models:ro`.
- Model on prod: `/home/jarvis/whisper-models/ggml-large-v3-turbo.bin` (1.6 GB).
- Container env: `WHISPER_MODEL=/models/ggml-large-v3-turbo.bin`, `JARVIS_VOICE_DEVICE=cuda`.
- Whisper.cpp init confirmed GPU: `use gpu = 1`, `flash attn = 1`, `CUDA : ARCHS = 860`, `CUDA0 total size = 1623.92 MB`.

The single call site to replace: `app/utils.py::run_whisper(wav_path, prompt, temperature, temperature_inc, beam_size)` — currently shells out to `whisper-cli` and reads back `<wav_path>.txt`.

## Implementation

### 1. Dependency

Add to `pyproject.toml` (or `requirements.txt` if that's what the build uses):

```
pywhispercpp>=1.3.0,<2.0
```

The pip wheel ships **CPU-only** binaries. To get CUDA, install with `--no-binary` so it builds against system libwhisper / cuBLAS:

```
GGML_CUDA=1 pip install --no-binary pywhispercpp pywhispercpp
```

This needs the same CUDA dev libs already in `Dockerfile.gpu` (we have `nvidia/cuda:12.4.1-devel-ubuntu22.04` as the base — should be sufficient). Verify by checking that `pywhispercpp.constants.WHISPER_CPP_VERSION` reports a build with cuBLAS enabled, or by running a transcribe and watching `nvidia-smi` show persistent VRAM use.

Pin the version. pywhispercpp lags whisper.cpp HEAD by weeks/months — known-good versions for large-v3-turbo are 1.3.x.

### 2. Model lifecycle

In a new module `app/whisper_engine.py`:

```python
from threading import Lock
from pywhispercpp.model import Model

_model: Model | None = None
_lock = Lock()

def get_model() -> Model:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = Model(
                    model=os.getenv("WHISPER_MODEL"),
                    n_threads=int(os.getenv("WHISPER_N_THREADS", "4")),
                    print_progress=False,
                    print_realtime=False,
                )
    return _model
```

Lazy init keeps app startup fast and avoids loading the model in test runs / CLI invocations. Lock guards the double-checked init. The model itself is thread-safe for inference per pywhispercpp docs but verify — if not, also serialize transcribe calls behind the lock or a separate inference lock (GPU-bound work serializes itself anyway, no parallelism win).

Add a startup hook in `app/main.py` (FastAPI `lifespan`) that calls `get_model()` so the first real request doesn't pay the load cost. ~3 s warm-up at boot is fine.

### 3. Replace `run_whisper`

Rewrite `app/utils.py::run_whisper`:

```python
def run_whisper(
    wav_path: str,
    prompt: str | None = None,
    temperature: float = 0.0,
    temperature_inc: float = 0.2,
    beam_size: int = 5,
) -> str:
    model = get_model()
    segments = model.transcribe(
        wav_path,
        language="en",
        initial_prompt=prompt or "",
        temperature=temperature,
        temperature_inc=temperature_inc,
        beam_size=beam_size,
    )
    return " ".join(seg.text for seg in segments).strip()
```

Drop:
- `_resolve_whisper_cli()`, `_LOCAL_WHISPER_CLI`, `_build_subprocess_env()` if unused elsewhere
- The `--output-txt` + `<wav_path>.txt` round-trip
- `_resolve_whisper_cli` settings and the `whisper-cli` path setting in `settings_definitions.py`

Keep the function signature the same so callers don't change.

### 4. Failure handling

- **Model load fails** (file missing, CUDA error, OOM): log and raise. Don't silently fall back to CPU — that masks regressions. Existing health check already 500s if FastAPI startup fails.
- **Transcribe error** (corrupt audio, etc): wrap in `WhisperTranscriptionError` like the current code does, preserve `stderr` semantics in the message even though we no longer have a subprocess stderr.

### 5. Dockerfile.gpu changes

Add the source-build step before the existing `pip install`:

```dockerfile
RUN GGML_CUDA=1 pip install --no-binary pywhispercpp pywhispercpp==1.3.<pinned>
```

Place AFTER `nvidia/cuda` libs are available and BEFORE the project install so the wheel cache is hot.

The whisper-cli binary can stay in the image for now — useful for ad-hoc benchmarking and as a fallback if we ever need to bisect a pywhispercpp regression.

## Test plan

### Unit / smoke
- Existing pytest suite. Update any test that mocks subprocess to mock `get_model()` instead.
- Add a test that imports `whisper_engine`, doesn't crash without a real model file (via env override to a stub), and that `run_whisper` raises `WhisperTranscriptionError` when transcription fails.

### Integration on dev (.103 or local docker)
1. Build the new image with `Dockerfile.gpu` locally on a CUDA host (ideally the prod box .107 since dev mac doesn't have NVIDIA).
2. Bring up the container.
3. Hit `/transcribe` with `samples/jfk.wav` → expect text returned.
4. `nvidia-smi` should show persistent ~2 GB VRAM allocation on the chosen device — that's the whole point.
5. Time 10 sequential requests with the same audio — first should be ~3 s (warm-up), rest should be ≤200 ms each.

### Production cutover on .107
1. Rebuild image: `cd /home/jarvis/.jarvis/compose && docker compose up -d --build jarvis-whisper-api` (the build context is already `/home/jarvis/jarvis-whisper-api` per the existing edit).
2. Force recreate so the new image takes effect.
3. Watch `docker logs jarvis-whisper-api` for the startup hook's model-load line.
4. Verify `nvidia-smi` shows the model resident.
5. Hit a real voice command from the kitchen pi → check `query_logs` for the transcribe latency in CC.

## Risks / open questions

- **pywhispercpp version pin.** Pick a version where `model.transcribe()` actually returns segments with `.text`. The 1.3.x line is the right ballpark but exact API may have shifted; verify against the source for whichever version we pin.
- **Thread-safety of `Model.transcribe`.** If pywhispercpp's `Model` is not thread-safe under concurrent calls, FastAPI's threadpool can race. Quick fix: serialize behind the same lock used for init. GPU-bound work serializes at the CUDA layer anyway, so no real throughput loss.
- **Memory.** The model stays resident — 2 GB on GPU0. Combined with the 22 GB background LLM and ~20 GB live LLM, we should still have headroom on the dual 3090s but verify after deploy.
- **Cold start.** Container restart now costs ~3 s of model load before the first request. Should be invisible since the lifespan hook runs before the server accepts connections.
- **VAD / chunking.** Long clips may need chunking — pywhispercpp handles this internally, but verify behavior matches what whisper-cli was doing before. Voice commands are short (≤7 s), so unlikely to be an issue in practice.

## Out of scope (for this change)

- Switching to faster-whisper / CTranslate2 — different runtime entirely, not whisper.cpp. Worth considering separately but biggest unknown is whether CT2 model conversions of large-v3-turbo are stable.
- Streaming transcription. Not needed for command-mode (we wait for the full clip).
- Per-household model selection. Single global model is fine until we have a real reason.
