import logging
import os
import shutil
import tempfile
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app.audio import preprocess_audio
from app.deps import verify_app_auth
from app.exceptions import AudioProcessingError, WhisperTranscriptionError
from app.utils import recognize_speaker, run_whisper, speaker_recognition_status
from jarvis_auth_client.models import AppAuthResult

load_dotenv()

# Set up logging
console_level = os.getenv("JARVIS_LOG_CONSOLE_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, console_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("uvicorn")

# Remote logging handler (initialized in startup event)
_jarvis_handler = None

# Throttle for the "speaker recognition disabled" warning so a busy node
# doesn't spam the logs once per request when the flag is off.
_throttle: dict[str, float] = {"disabled_log": 0.0}


def _setup_remote_logging() -> None:
    """Set up remote logging to jarvis-logs server."""
    global _jarvis_handler
    try:
        from jarvis_log_client import init as init_log_client, JarvisLogHandler

        app_id = os.getenv("JARVIS_APP_ID", "jarvis-whisper")
        app_key = os.getenv("JARVIS_APP_KEY")
        if not app_key:
            logger.warning("JARVIS_APP_KEY not set, remote logging disabled")
            return

        init_log_client(app_id=app_id, app_key=app_key)

        remote_level = os.getenv("JARVIS_LOG_REMOTE_LEVEL", "DEBUG")
        _jarvis_handler = JarvisLogHandler(
            service="jarvis-whisper",
            level=getattr(logging, remote_level.upper(), logging.DEBUG),
        )

        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            logging.getLogger(logger_name).addHandler(_jarvis_handler)

        logger.info("Remote logging enabled to jarvis-logs")
    except ImportError:
        logger.debug("jarvis-log-client not installed, remote logging disabled")


app = FastAPI(title="Jarvis Whisper API", version="1.0.0")

# Add settings router from shared library
from jarvis_settings_client import create_settings_router, create_combined_auth, create_superuser_auth
from app.services.settings_service import get_settings_service

from app import service_config

_settings_router = create_settings_router(
    service=get_settings_service(),
    auth_dependency=create_combined_auth(service_config.get_auth_url()),
    write_auth_dependency=create_superuser_auth(service_config.get_auth_url()),
)
app.include_router(_settings_router, prefix="/settings", tags=["settings"])

# Voice profile enrollment endpoints
from app.api.voice_profiles import router as voice_profiles_router
app.include_router(voice_profiles_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on app startup."""
    service_config.init()
    _setup_remote_logging()

    # Pre-warm the whisper model so the first transcription request doesn't
    # pay the ~3s model load. Failure here is fatal — without the model the
    # service can't transcribe, so let the health check 500 rather than
    # silently degrading on every request.
    from app.whisper_engine import get_model
    t0 = time.perf_counter()
    get_model()
    logger.info(
        "Whisper model pre-warmed in %d ms",
        int((time.perf_counter() - t0) * 1000),
    )

    # Pre-warm VoiceEncoder if speaker recognition is on, so the first
    # transcription request doesn't pay the ~5-7s CUDA-init + model→GPU
    # cold-start cost. Failure is non-fatal — first request will pay it.
    if get_settings_service().get_bool("voice.recognition_enabled", False):
        try:
            from app.utils import _get_encoder
            t0 = time.perf_counter()
            _get_encoder()
            logger.info(
                "VoiceEncoder pre-warmed in %d ms",
                int((time.perf_counter() - t0) * 1000),
            )
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning(f"VoiceEncoder pre-warm failed: {type(e).__name__}: {e}")
    else:
        logger.warning(
            "SPEAKER RECOGNITION DISABLED (voice.recognition_enabled=false) — every "
            "/transcribe returns speaker.user_id=None. Set it true to enable."
        )

    logger.info("Jarvis Whisper API service started")


@app.get("/ping")
def pong():
    return {"message": "pong"}


@app.get("/health")
def health():
    status: dict[str, object] = {"status": "healthy"}
    try:
        status["speaker"] = speaker_recognition_status()
    except Exception as e:  # noqa: BLE001 — observability must never break liveness
        logger.debug("speaker status unavailable for /health: %s", e)
    return status


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    speaker_audio: UploadFile | None = File(default=None),
    prompt: str | None = Query(default=None, description="Initial prompt to guide transcription"),
    preprocess: bool = Query(default=False, description="Apply audio normalization and silence trimming"),
    temperature: float = Query(default=0.0, ge=0.0, le=1.0, description="Initial temperature for sampling (0.0-1.0)"),
    temperature_inc: float = Query(default=0.2, ge=0.0, le=1.0, description="Temperature increment on decode failure (0.0-1.0)"),
    beam_size: int = Query(default=5, ge=1, le=16, description="Beam size for beam search (1-16)"),
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Transcribe audio + (optionally) identify the speaker.

    Args:
        file: Required. The audio to transcribe (typically the user's
            command utterance after wake-word).
        speaker_audio: Optional. When provided, this audio is used for
            **speaker recognition only** while ``file`` is still used for
            transcription. Lets the caller concatenate wake-word audio in
            front of a short command (e.g. "delete it") so the speaker
            pass has enough signal even when the transcribed utterance is
            sub-1-second. Passing wake-word audio in ``file`` itself
            would confuse Whisper's text output, so it's a separate
            stream.
    """
    logger.debug(
        f"Transcription request from {auth.app.app_id} "
        f"for household {auth.context.household_id}, node {auth.context.node_id}"
    )

    t_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    speaker_audio_path: str | None = None
    if speaker_audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_sp:
            shutil.copyfileobj(speaker_audio.file, tmp_sp)
            speaker_audio_path = tmp_sp.name
    t_saved = time.perf_counter()

    processed_path: str | None = None

    try:
        # Apply preprocessing if requested
        if preprocess:
            processed_path = f"{tmp_path}.processed.wav"
            try:
                preprocess_audio(tmp_path, processed_path)
                logger.debug(f"Preprocessed audio for node {auth.context.node_id}")
            except AudioProcessingError as e:
                logger.warning(f"Preprocessing failed, using original: {e}")
                processed_path = None
        t_preproc = time.perf_counter()

        # Use processed file if available, otherwise original
        whisper_input = processed_path if processed_path else tmp_path

        text, segments = run_whisper(
            whisper_input,
            prompt=prompt,
            temperature=temperature,
            temperature_inc=temperature_inc,
            beam_size=beam_size,
        )
        t_whisper = time.perf_counter()
        logger.info(f"Transcribed {len(text)} chars, {len(segments)} segments for node {auth.context.node_id}")

        speaker_response: dict[str, int | float | None] = {"user_id": None, "confidence": 0.0}
        speaker_enabled = get_settings_service().get_bool("voice.recognition_enabled", False)

        if speaker_enabled:
            # Use the dedicated speaker-pass audio if the caller supplied one
            # (typically wake-word + command, for better short-clip recognition);
            # otherwise fall back to the transcription audio.
            speaker_input_path = speaker_audio_path or tmp_path
            # household_member_ids comes from context headers, passed by command-center
            speaker_result = recognize_speaker(
                speaker_input_path,
                household_id=auth.context.household_id or "",
                member_ids=auth.context.household_member_ids,
            )
            speaker_response = {
                "user_id": speaker_result.user_id,
                "confidence": speaker_result.confidence,
            }
            logger.info(
                "Speaker match: user_id=%s confidence=%.3f node=%s household=%s",
                speaker_result.user_id,
                speaker_result.confidence,
                auth.context.node_id,
                auth.context.household_id,
            )
        else:
            # Recognition is off — make that visible (rate-limited) so a
            # disabled flag isn't mistaken for a model that just never matches.
            now = time.monotonic()
            if now - _throttle["disabled_log"] > 60.0:
                _throttle["disabled_log"] = now
                logger.warning(
                    "speaker_recognition_disabled node=%s household=%s "
                    "(voice.recognition_enabled=false)",
                    auth.context.node_id,
                    auth.context.household_id,
                )
        t_speaker = time.perf_counter()

        logger.info(
            "transcribe phases (ms): save=%d preproc=%d whisper=%d speaker=%d total=%d",
            int((t_saved - t_start) * 1000),
            int((t_preproc - t_saved) * 1000),
            int((t_whisper - t_preproc) * 1000),
            int((t_speaker - t_whisper) * 1000) if speaker_enabled else 0,
            int((t_speaker - t_start) * 1000),
        )

        return {
            "text": text,
            "segments": segments,
            "speaker": speaker_response,
        }

    except WhisperTranscriptionError as e:
        logger.error(f"Transcription failed for node {auth.context.node_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "stderr": e.stderr},
        )
    except OSError as e:
        logger.error(f"File I/O error for node {auth.context.node_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"File error: {e}"})
    except (ValueError, RuntimeError) as e:
        logger.error(f"Processing error for node {auth.context.node_id}: {type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f"{tmp_path}.txt"):
            os.remove(f"{tmp_path}.txt")
        if speaker_audio_path and os.path.exists(speaker_audio_path):
            os.remove(speaker_audio_path)
        if processed_path:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if os.path.exists(f"{processed_path}.txt"):
                os.remove(f"{processed_path}.txt")

