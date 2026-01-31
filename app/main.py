import logging
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app.deps import verify_node_auth
from app.exceptions import WhisperTranscriptionError
from app.utils import recognize_speaker, run_whisper

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
    node_id: str = Depends(verify_node_auth),
):
    logger.debug(f"Transcription request from node: {node_id}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = run_whisper(tmp_path, prompt=prompt)
        logger.info(f"Transcribed {len(text)} chars for node {node_id}")

        speaker_response: dict[str, str | float] = {"name": "unknown", "confidence": 0.0}

        if os.getenv("USE_VOICE_RECOGNITION", "false").lower() == "true":
            speaker_result = recognize_speaker(tmp_path)
            speaker_response = {
                "name": speaker_result.name,
                "confidence": speaker_result.confidence,
            }

        return {
            "text": text,
            "speaker": speaker_response,
        }

    except WhisperTranscriptionError as e:
        logger.error(f"Transcription failed for node {node_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "stderr": e.stderr},
        )
    except Exception as e:
        logger.error(f"Unexpected error for node {node_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f"{tmp_path}.txt"):
            os.remove(f"{tmp_path}.txt")

