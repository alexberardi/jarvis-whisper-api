import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app.exceptions import WhisperTranscriptionError
from app.utils import recognize_speaker, run_whisper

load_dotenv()

app = FastAPI()


@app.get("/ping")
def pong():
    return {"message": "pong"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str | None = Query(default=None, description="Initial prompt to guide transcription"),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = run_whisper(tmp_path, prompt=prompt)

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
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "stderr": e.stderr},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f"{tmp_path}.txt"):
            os.remove(f"{tmp_path}.txt")

