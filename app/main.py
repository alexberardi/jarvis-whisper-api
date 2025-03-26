from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils import run_whisper
import tempfile
import shutil

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = run_whisper(tmp_path)
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

