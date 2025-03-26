import subprocess

WHISPER_MODEL = "models/ggml-base.en.bin"

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
    txt_path = wav_path.replace(".wav", ".txt").replace("/tmp", "audio")
    with open(txt_path) as f:
        return f.read().strip()



