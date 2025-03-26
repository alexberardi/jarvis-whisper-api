# Dockerfile
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9999

# Copy pre-built binaries into container
COPY bin/whisper-cli /usr/bin/whisper-cli
COPY bin/libwhisper.so.1 /usr/lib/libwhisper.so.1


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999"]
