import subprocess
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin"))


def run_whisper(wav_path):
    result = subprocess.run([
        "whisper-cli",
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "--language", "en",
        "--output-txt",
        ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    # Read the output .txt file
    txt_path = f"{wav_path}.txt"
    print(wav_path)
    print(txt_path)
    with open(txt_path) as f:
        return f.read().strip()



