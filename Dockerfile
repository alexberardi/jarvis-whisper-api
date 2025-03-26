FROM python:3.11-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create working directories
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    make -C examples && \
    cp examples/main /app/main

# Download model (you can pre-cache or mount this volume in production)
RUN mkdir -p models audio && \
    curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Expose port
EXPOSE 9999

# Start API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999"]
