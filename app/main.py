import logging
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app.audio import preprocess_audio
from app.deps import verify_app_auth
from app.exceptions import AudioProcessingError, WhisperTranscriptionError
from app.utils import recognize_speaker, run_whisper
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


@app.on_event("startup")
async def startup_event():
    """Initialize services on app startup."""
    _setup_remote_logging()
    logger.info("Jarvis Whisper API service started")


@app.get("/ping")
def pong():
    return {"message": "pong"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str | None = Query(default=None, description="Initial prompt to guide transcription"),
    preprocess: bool = Query(default=False, description="Apply audio normalization and silence trimming"),
    temperature: float = Query(default=0.0, ge=0.0, le=1.0, description="Initial temperature for sampling (0.0-1.0)"),
    temperature_inc: float = Query(default=0.2, ge=0.0, le=1.0, description="Temperature increment on decode failure (0.0-1.0)"),
    beam_size: int = Query(default=5, ge=1, le=16, description="Beam size for beam search (1-16)"),
    auth: AppAuthResult = Depends(verify_app_auth),
):
    logger.debug(
        f"Transcription request from {auth.app.app_id} "
        f"for household {auth.context.household_id}, node {auth.context.node_id}"
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

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

        # Use processed file if available, otherwise original
        whisper_input = processed_path if processed_path else tmp_path

        text = run_whisper(
            whisper_input,
            prompt=prompt,
            temperature=temperature,
            temperature_inc=temperature_inc,
            beam_size=beam_size,
        )
        logger.info(f"Transcribed {len(text)} chars for node {auth.context.node_id}")

        speaker_response: dict[str, int | float | None] = {"user_id": None, "confidence": 0.0}

        if os.getenv("USE_VOICE_RECOGNITION", "false").lower() == "true":
            # Use original file for speaker recognition (raw audio may be better)
            # household_member_ids comes from context headers, passed by command-center
            speaker_result = recognize_speaker(
                tmp_path,
                household_id=auth.context.household_id or "",
                member_ids=auth.context.household_member_ids,
            )
            speaker_response = {
                "user_id": speaker_result.user_id,
                "confidence": speaker_result.confidence,
            }

        return {
            "text": text,
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
        if processed_path:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if os.path.exists(f"{processed_path}.txt"):
                os.remove(f"{processed_path}.txt")

