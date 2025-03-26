FROM python:3.11-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    make && \
    cp main /app/main


RUN mkdir -p models audio && \
    curl -L -o models/ggml-base.en.bin https://huggingface.com/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

EXPOSE 9999

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999"]


