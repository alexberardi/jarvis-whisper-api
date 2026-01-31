# PRD: Speech Recognition Enhancements

## Overview

Enhance `app/utils.py` to improve transcription accuracy, speaker recognition reliability, and overall robustness for real-world conditions (varying distances from mic, cheap USB microphones, noisy environments like parties).

## Goals

1. **Accuracy** - Better transcription in noisy environments and with domain-specific vocabulary
2. **Reliability** - Consistent performance regardless of recording conditions
3. **Observability** - Confidence scores and specific errors for better debugging
4. **Performance** - Reduced latency through preprocessing and async operations

## Non-Goals

- Word-level timestamps (short commands don't need this)
- Speaker enrollment API (backlogged for later)
- VAD preprocessing (not needed for short commands)
- Adaptive per-speaker thresholds (backlogged)

---

## Enhancements (Priority Order)

### 1. Specific Exception Classes
**Complexity:** Very Low

Replace generic `RuntimeError` with typed exceptions for better error handling.

```python
class WhisperError(Exception):
    """Base exception for whisper transcription errors"""

class WhisperTranscriptionError(WhisperError):
    """Transcription failed (non-zero exit code)"""

class AudioProcessingError(Exception):
    """Audio preprocessing failed (normalization, noise reduction, etc.)"""

class SpeakerRecognitionError(Exception):
    """Speaker recognition failed"""
```

**Files:** `app/utils.py`, new `app/exceptions.py`

---

### 2. Speaker Recognition Confidence Scores
**Complexity:** Very Low

Return confidence alongside speaker name so callers can decide how to handle uncertain matches.

```python
@dataclass
class SpeakerResult:
    name: str        # "alex" or "unknown"
    confidence: float  # 0.0 to 1.0 (cosine similarity)
```

**Current:** `recognize_speaker(audio_path) -> str`
**New:** `recognize_speaker(audio_path) -> SpeakerResult`

**Files:** `app/utils.py`

---

### 3. Initial Prompt/Context Support
**Complexity:** Low

Add optional prompt parameter to prime whisper with domain-specific vocabulary.

```python
def run_whisper(wav_path: str, prompt: str | None = None) -> str:
    # If prompt provided, add: "--prompt", prompt
```

**Example prompt:** `"Jarvis, set timer, play music, turn on lights, check weather, add to shopping list"`

**Benefit:** Significantly improves accuracy for known command patterns. "Travis" -> "Jarvis"

**Files:** `app/utils.py`

---

### 4. Adaptive Beam Size Based on SNR
**Complexity:** Low-Medium

Automatically detect audio quality and adjust beam size for speed/accuracy tradeoff.

```python
def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate signal-to-noise ratio in dB"""
    # Compare energy in speech vs non-speech segments
    ...

def get_optimal_beam_size(snr: float) -> int:
    if snr > 20:   # Clean audio
        return 1   # Fast greedy decoding
    elif snr > 10: # Moderate noise
        return 3
    else:          # Noisy (party, music)
        return 5   # More candidates for accuracy

def run_whisper(wav_path: str, beam_size: int | None = None) -> str:
    if beam_size is None:
        beam_size = get_optimal_beam_size(estimate_snr(audio))
    # Add: "--beam-size", str(beam_size)
```

**Files:** `app/utils.py`

---

### 5. Temperature Fallback
**Complexity:** Low

Add temperature parameter for handling uncertain audio segments.

```python
def run_whisper(
    wav_path: str,
    temperature: float = 0.0,
    temperature_fallback: list[float] | None = None
) -> str:
    # --temperature 0.0
    # If fallback: --temperature-inc for retry on low confidence
```

**Default:** `temperature=0.0` (deterministic)
**Fallback:** `[0.2, 0.4, 0.6]` for uncertain segments

**Files:** `app/utils.py`

---

### 6. Audio Normalization
**Complexity:** Low-Medium

Normalize audio levels before processing for consistent input regardless of:
- User distance from microphone
- Speaking volume (whisper vs shout)
- USB microphone sensitivity variations

```python
def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS normalization to target dB level"""
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / (rms + 1e-10))
```

**Method:** RMS normalization (better for speech than peak normalization)

**Files:** `app/utils.py`
**Dependencies:** numpy (already present)

---

### 7. Silence Trimming
**Complexity:** Low-Medium

Trim leading/trailing silence to reduce processing time and potential hallucination.

```python
def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_ms: int = 100
) -> np.ndarray:
    """Remove leading/trailing silence below threshold"""
    ...
```

**Benefit:** Faster transcription, reduced hallucination on silent segments

**Files:** `app/utils.py`
**Dependencies:** numpy (already present), possibly scipy

---

### 8. Noise Reduction
**Complexity:** Medium

Apply spectral gating to reduce background noise (music, HVAC, crowd noise).

```python
import noisereduce as nr

def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction via spectral gating"""
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
```

**Use case:** Party with music, HVAC noise, outdoor environments

**Files:** `app/utils.py`
**Dependencies:** `noisereduce` (new)

---

### 9. Multi-Embedding Speaker Profiles
**Complexity:** Medium-High

Support multiple voice embeddings per speaker for recognition across varying conditions.

**Current structure:**
```
voice_profiles/
├── alex.wav
└── jordan.wav
```

**New structure:**
```
voice_profiles/
├── alex/
│   ├── normal.wav
│   ├── morning.wav
│   └── tired.wav
└── jordan/
    └── normal.wav
```

```python
def load_speaker_profiles():
    """Load all embeddings, multiple per speaker"""
    for speaker_dir in PROFILE_DIR.iterdir():
        if speaker_dir.is_dir():
            embeddings = []
            for wav_file in speaker_dir.glob("*.wav"):
                embed = encoder.embed_utterance(preprocess_wav(wav_file))
                embeddings.append(embed)
            _loaded_profiles[speaker_dir.name] = embeddings

def recognize_speaker(audio_path: str) -> SpeakerResult:
    """Match against best embedding per speaker"""
    embed = encoder.embed_utterance(preprocess_wav(audio_path))

    best_match = "unknown"
    best_score = 0.0

    for name, embeddings in _loaded_profiles.items():
        # Take highest similarity across all embeddings for this speaker
        score = max(np.inner(embed, ref) for ref in embeddings)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > 0.75:
        return SpeakerResult(name=best_match, confidence=best_score)
    return SpeakerResult(name="unknown", confidence=best_score)
```

**Backward compatible:** Single WAV files still work (auto-wrapped as single-item list)

**Files:** `app/utils.py`

---

### 10. Async Subprocess
**Complexity:** Medium-High

Convert whisper subprocess to async for non-blocking concurrent requests.

```python
import asyncio

async def run_whisper_async(wav_path: str, **kwargs) -> str:
    cli_path = _resolve_whisper_cli()
    env = _build_subprocess_env(cli_path)

    cmd = [cli_path, "-m", WHISPER_MODEL, "-f", wav_path, "--language", "en", "--output-txt"]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise WhisperTranscriptionError(stderr.decode())

    txt_path = f"{wav_path}.txt"
    async with aiofiles.open(txt_path) as f:
        return (await f.read()).strip()
```

**Note:** Requires updating callers to use async/await

**Files:** `app/utils.py`, `app/main.py`
**Dependencies:** `aiofiles` (new)

---

## Audio Processing Pipeline

Final processing order for `run_whisper()`:

```
Input WAV
    │
    ▼
┌─────────────────┐
│ Load Audio      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize (RMS) │  ← Consistent levels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Noise Reduction │  ← Remove background
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trim Silence    │  ← Faster processing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Estimate SNR    │  ← Auto beam size
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Temp WAV   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ whisper-cli     │  ← With prompt, beam, temp
└────────┬────────┘
         │
         ▼
    Transcription
```

---

## New Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `noisereduce` | Spectral noise reduction | ~50KB |
| `aiofiles` | Async file I/O | ~20KB |
| `scipy` | Audio processing utilities | Already indirect dep |

---

## Testing Strategy

1. **Unit tests** for each preprocessing function
2. **Integration tests** with sample audio files:
   - Clean audio
   - Noisy audio (music, crowd)
   - Quiet speaker (far from mic)
   - Loud speaker (close to mic)
3. **Benchmark** latency before/after preprocessing

---

## Backlog (Future Consideration)

- Speaker enrollment API
- Adaptive per-speaker thresholds
- VAD preprocessing for longer recordings
- Word-level timestamps
