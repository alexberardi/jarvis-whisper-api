from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils import run_whisper, recognize_speaker
import tempfile
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/ping")
def pong():
	return {"message": "pong" }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = run_whisper(tmp_path)
        speaker = None

        if os.getenv("USE_VOICE_RECOGNITION", "false").lower() == "true":
            print("here")
            speaker = recognize_speaker(tmp_path)


        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f"{tmp_path}.txt"):
            os.remove(f"{tmp_path}.txt")
        return {
                "text": text,
                "speaker": speaker or "unknown"
                }

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f"{tmp_path}.txt"):
            os.remove(f"{tmp_path}.txt")
        return JSONResponse(status_code=500, content={"error": str(e)})

